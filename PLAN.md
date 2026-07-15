# RAG Pipeline — Design Plan

A from-scratch rewrite of a local, single-user RAG (retrieval-augmented
generation) system that ingests personal data (emails, chats, misc files,
document repositories), turns it into clean Markdown, embeds it into a vector
index, and exposes a chat interface over it.

This document is self-contained: it fully specifies the intended architecture
without reference to any earlier prototype.

---

## 1. Goals & Principles

- **Modular pipeline**: four independent stages (Extract → Translate → Embed →
  Chat), each runnable on its own from the command line.
- **Plugin-based ingestion**: new data sources are added by dropping in a plugin,
  with no changes to the core.
- **Provenance & reversibility**: keep history of translations, record which
  program version / LLM / prompt produced each artifact, allow reverting.
- **Incremental & resumable**: re-run stages over subsets; only reprocess what
  changed.
- **Backup-friendly storage**: large binary content lives on the filesystem as
  immutable, content-addressed blobs (good for incremental backups); the
  database stays small and holds only metadata + references.
- **Concurrency where it matters**: chat can run while the pipeline runs (with
  one documented exception around the vector store, see §9).

**Language / conventions**: Python. Full type annotations on all functions and
module-level globals. Docstrings on private/helper functions as well as public
ones.

---

## 2. Technology Choices

| Concern            | Choice                          | Rationale |
|--------------------|---------------------------------|-----------|
| Metadata DB        | **SQLite** in **WAL** mode      | Single file, no server, backup-friendly. WAL gives many concurrent readers + one writer across processes. |
| Blob storage       | **Filesystem, content-addressed by hash** | Dedup, immutable, incremental-backup-friendly, lock-free reads. |
| Vector store       | **Qdrant (embedded, local path)** | Same family as prototype, good metadata filtering. Single-process lock — see §9 for the concurrency policy. |
| LLM / embeddings   | **OpenAI-compatible endpoint**  | Works with local (Ollama / LM Studio / vLLM / llama.cpp) or hosted. |
| Config             | **Central YAML** file           | Everything except secrets. |
| Secrets            | **Desktop keyring** (+ file/env fallback) | No secrets in the YAML or repo. |

SQLite WAL settings: `PRAGMA journal_mode=WAL;` once at DB creation, plus a
per-connection `PRAGMA busy_timeout=<ms>;` so a blocked writer waits instead of
erroring with `SQLITE_BUSY`.

---

## 3. Storage Model

### 3.1 Blob store (filesystem)

- Content-addressed: path derived from the content hash (e.g. SHA-256),
  sharded, e.g. `blobs/ab/cd/abcd…`.
- Immutable: writing the same content twice is a no-op (natural dedup).
- The DB never stores blob bytes, only the hash.
- All blob reads are lock-free (files never mutate); deletion is only via GC
  (§8).

### 3.2 Metadata database (SQLite)

Core tables: `documents` (extraction output), `translation_sets` +
`translation_parts` (processing output), plus bookkeeping tables `embeddings`
and `schema_version`.

**`documents`** — output of extraction (module 1). One row per live document.

| Column        | Type    | Notes |
|---------------|---------|-------|
| source_type   | TEXT    | Plugin id, e.g. `email`. |
| doc_id        | TEXT    | Unique within a source_type. |
| version       | INTEGER | Starts at 1, increments when the source content or **stable** metadata changes. |
| blob_hash     | TEXT    | Reference to the raw source blob in the blob store. |
| stable_metadata | JSONB | Content-defining metadata: general keys (date, title, mime-type, …) + arbitrary plugin-specific keys. A change here **bumps the version** and retriggers downstream pipelines. |
| transient_metadata | JSONB | Informational, mutable metadata that does **not** affect the version: e.g. the IMAP folder a message currently sits in, whether it is currently read/unread. Updated freely; never retriggers translation/embedding. |
| updated_at    | TEXT    | Timestamp of last extraction touch. |

- Primary key: `(source_type, doc_id)`.
- **Two metadata sets.** `stable_metadata` is content-defining: it is an input to
  translation, so changing it bumps `version` and causes reprocessing.
  `transient_metadata` is purely informational state that changes often and must
  *not* trigger the pipeline (folder location, read flag, …). Both are exposed to
  downstream consumers (e.g. the chat model may read transient state for
  reasoning); only stable changes propagate.
- Both are JSONB and the single source of truth for their respective metadata.
  Commonly-filtered stable keys (date, title, mime-type) are exposed as
  **`GENERATED ... VIRTUAL` columns** (`stable_metadata ->> 'key'`) with indexes —
  always in sync by construction, giving indexed filtering. Arbitrary keys remain
  queryable via `->>` / `json_extract` (unindexed scan; fine at this scale,
  promotable to a generated column later).

A single translation run of a document may produce **several output blobs**
("parts") — e.g. translating an email yields the message body plus one
translation per attachment. Each part has its own blob and its own metadata, but
all parts of one run are **tied together**: they become active or inactive as a
unit. This is modelled as two tables — a *translation set* (the run) and its
*parts*.

**`translation_sets`** — one row per translation run of a document. History is
kept.

| Column         | Type    | Notes |
|----------------|---------|-------|
| id             | INTEGER | Surrogate primary key (the "set id"). |
| source_type    | TEXT    | FK part → `documents`. |
| doc_id         | TEXT    | FK part → `documents`. |
| source_version | INTEGER | The `documents.version` this run was made from (staleness detection). |
| metadata       | JSONB   | Run-level pipeline info: which LLM, quick vs. context mode, prompt version, extraction-derived data, etc. |
| translated_at  | TEXT    | Timestamp. |
| git_rev        | TEXT    | Program git revision at translation time. |
| is_active      | INTEGER | 1 = the currently-used translation set for this document. Applies to the **whole set**. |

- **At most one active set per document**, enforced by a partial unique index:
  `CREATE UNIQUE INDEX ... ON translation_sets(source_type, doc_id) WHERE is_active=1`.
- Activating a new set deactivates the previous one in the same transaction.
- `is_active` lives only here (on the set), so all parts of a set are
  active/inactive together by construction — there is no per-part active flag to
  get out of sync.

**`translation_parts`** — one row per output blob within a set.

| Column     | Type    | Notes |
|------------|---------|-------|
| id         | INTEGER | Surrogate primary key. |
| set_id     | INTEGER | FK → `translation_sets.id`. |
| part_key   | TEXT    | Stable identifier of the part within its set, e.g. `body`, `attachment:1`, `attachment:invoice.pdf`. |
| blob_hash  | TEXT    | Reference to this part's translated Markdown blob. |
| metadata   | JSONB   | Per-part metadata (may differ between parts, e.g. attachment filename / mime-type / OCR flag). |

- Unique `(set_id, part_key)`.
- Deleting a set cascades to its parts.

**`embeddings`** (bookkeeping for module 3) — tracks what is currently indexed
in the vector store so staleness can be detected. Granularity is the **part**
(each part is chunked and embedded independently):

| Column          | Type    | Notes |
|-----------------|---------|-------|
| source_type     | TEXT    | |
| doc_id          | TEXT    | |
| set_id          | INTEGER | Which `translation_sets.id` is currently embedded. |
| part_id         | INTEGER | Which `translation_parts.id` these points came from. |
| embed_model     | TEXT    | Model + dim used (invalidates on change). |
| embedded_at     | TEXT    | Timestamp. |

Re-embedding is required when a document's active `set_id` differs from the
embedded one (the whole document is re-embedded across all parts of the new
set), or the embed model/dim changed.

**`schema_version`** — a single integer row holding the DB's current schema
version; used by the migration mechanism below.

### 3.3 Schema migrations

The schema *will* evolve (new metadata columns, new tables, new indexes). A
migration mechanism is built in **from day one**, before there is any real data
— retrofitting migrations onto a populated DB later is painful.

How it works:

- **Migrations are an ordered, numbered list** of forward steps, e.g.
  `0001_init`, `0002_add_translation_index`, … Each step is a small unit (SQL
  and/or Python) that transforms the schema from version *N-1* to version *N*.
  Migration `0001` creates the entire initial schema, so a database is *never*
  in an un-versioned state.
- **The `schema_version` row records the highest applied step.**
- **On every startup**, the runner reads `schema_version` and applies each
  pending step in order, up to the version the code expects.
- **Each step runs in its own transaction** and bumps `schema_version` within
  that same transaction. If a step fails, it **rolls back cleanly**, leaving the
  DB at the previous version — never half-migrated (ties into the
  "fail, don't corrupt" principle, §8).
- **Forward-only.** For a single-user local tool there is no need for
  down-migrations; reverting means restoring a backup.
- The code refuses to open a DB whose `schema_version` is *newer* than it
  understands (i.e. an older program build against a DB upgraded by a newer
  build), rather than risk misinterpreting it.
- If the on-disk **blob layout** ever needs to change, it is versioned and
  migrated by the same discipline.

---

## 4. Module 1 — Extraction

Pulls data from sources into the `documents` table + blob store.

**Structure**: a *general part* (owns the DB, blob store, CLI, plugin
discovery) and a *plugin system*.

- Each plugin is an implementation of an `Extractor` class defined by the
  general part, located in its own subdirectory of the extraction package
  (e.g. `extract/plugins/email/`). The
  **`source_type` is the plugin's subdirectory name** — the framework derives
  and enforces it from the directory; the plugin does not declare it. This keeps
  the id single-sourced (no field to drift from the directory) and guarantees
  uniqueness (directory names are unique).
- A plugin **never touches the DB directly**. It receives a `Store` handle
  (provided by the general part) which is the only interface to the system.
- **Isolation is enforced by the API**: the `Store` handed to a plugin is bound
  to that plugin's `source_type`; every operation is implicitly scoped to it, so
  a plugin can never read or affect another plugin's documents.

### 4.1 The `Extractor` / `Store` API

A plugin subclasses `Extractor` and implements `run`. The general part discovers
plugins by subdirectory, takes the **directory name as the `source_type`**,
opens a `Store` scoped to that id, and calls `run`.

```python
class Extractor(ABC):
    """A data-source plugin. One subclass per source type.

    The `source_type` is not declared here — the framework derives it from the
    plugin's subdirectory name (e.g. `extract/plugins/email/` -> "email") and
    enforces it. The `Store` handed to `run` is already bound to that id.
    """

    @abstractmethod
    def run(self, store: "Store", state_dir: "Path",
            config: "Mapping[str, object]", secrets: "Secrets",
            **options: object) -> None:
        """Fetch from the source and reconcile it into `store`.

        The plugin decides what "reconcile" means: upsert everything it finds
        and delete what has vanished. `options` are plugin-specific CLI args.

        `config` is this plugin's own subtree of the global YAML config
        (the framework selects the subkey for this `source_type` and hands over
        just that subtree, in easily accessible form). Plugin-specific settings
        (host, folder, filters, …) live there.

        `secrets` is the same secret-resolution mechanism the main program uses
        (desktop keyring + file/env fallback, §10) — so a plugin fetches its
        passwords / API keys exactly the way the core does, rather than reading
        them from `config`. Secrets never live in the YAML.

        `state_dir` is a private, persistent directory owned by this plugin,
        for its own bookkeeping across runs (e.g. an IMAP plugin remembering
        which message UIDs it has already processed, or an incremental sync
        cursor). The framework creates it and guarantees it is stable across
        runs and unique to this plugin; its contents are opaque to the system.
        The plugin is free to use it however it likes — a single JSON file, a
        SQLite DB, whatever fits its complexity.
        """


class Store(Protocol):
    """The system-facing API given to a plugin. Bound to one source_type;
    every method is implicitly scoped to it."""

    def upsert(self, doc_id: str, content: bytes,
               metadata: dict[str, object],
               *, transient: dict[str, object] | None = None,
               exist_ok: bool = True,
               missing_ok: bool = True) -> "UpsertResult":
        """Create or update a document.

        `metadata` is the **stable** (content-defining) metadata; `transient` is
        optional informational state (folder, read flag, …) that must not affect
        the version.

        The framework hashes `content`, compares content + stable `metadata`
        against the stored row, and: inserts at version 1 if new; bumps `version`
        if content or stable metadata changed; leaves the version alone if only
        the transient metadata (or nothing) changed. Transient metadata is always
        written. Writes the blob to the blob store. Returns what happened
        (created / updated / unchanged, w.r.t. the versioned content) and the
        resulting version.

        `exist_ok` / `missing_ok` are optional consistency checks:
        - `exist_ok=False` -> raise if the document already exists (the plugin
          asserts this is a fresh insert).
        - `missing_ok=False` -> raise if the document does not yet exist (the
          plugin asserts this is an update of a known document).
        Both default to True (permissive upsert). They let a plugin catch its
        own bugs — e.g. a duplicate doc_id, or an update to something it thought
        it had already created.
        """

    def delete(self, doc_id: str, *, missing_ok: bool = True) -> bool:
        """Delete a document that vanished at the source. Cascades immediately
        (see below). Returns whether a row existed.

        `missing_ok=False` -> raise if the document does not exist (the plugin
        asserts it should be there). Defaults to True (deleting an absent
        document is a no-op)."""

    def exists(self, doc_id: str) -> bool:
        """Whether a document with this id currently exists."""

    def list_ids(self) -> "Iterable[str]":
        """All current doc_ids for this source_type — used by the plugin to
        detect which documents vanished and should be deleted."""

    def get_metadata(self, doc_id: str) -> "dict[str, object] | None":
        """Stored **stable** metadata for a document, or None if absent."""

    def get_transient(self, doc_id: str) -> "dict[str, object] | None":
        """Stored **transient** metadata for a document, or None if absent."""
```

Note that **version handling lives entirely in the framework** (in `upsert`),
not in the plugin. The plugin just declares "here is the current content and
metadata for this doc_id"; the framework decides whether that constitutes a
change.

The per-plugin `state_dir` lives under the configured storage directory (e.g.
`storage/plugin_state/<source_type>/`), is never touched by GC or other plugins,
and should be included in backups (it is real state, not a cache — losing it may
force a plugin into a full re-scan or duplicate work).

### 4.2 Change detection

`upsert` bumps `version` when the content **or the stable metadata** changed
(both are inputs to translation, so either changing warrants reprocessing). It
compares the new content hash against the stored `blob_hash` *and* the new stable
metadata against the stored stable metadata, and skips the bump only when both
are identical.

**Transient metadata never bumps the version.** A change to `transient_metadata`
alone (e.g. a mail moved to another folder, or marked read) is written straight
to the row without incrementing `version` and without retriggering translation or
embedding. Every `upsert` refreshes the transient metadata; if content and stable
metadata are unchanged, the result is reported as *unchanged* (no reprocessing)
even though the transient state was updated. Bookkeeping fields (e.g.
`updated_at`) are likewise excluded from the change comparison.

The bump is **content-driven, not semantic**: if the plugin supplies changed
bytes or metadata, the version increases, even if the change is cosmetically
spurious. It is the **plugin's responsibility** not to introduce spurious
changes — e.g. don't inject the current date/time into the content or a
non-stable field ordering, or every run will look like a change and trigger
needless retranslation + re-embedding.

### 4.3 Deletion (cascade)

Deletion is **immediate and cascading**. When a document is deleted (via
`Store.delete`), in the same operation the framework removes: the `documents`
row, all its `translation_sets` and their `translation_parts`, its `embeddings`
bookkeeping rows, and the corresponding points in the vector store. (Blobs are
shared/immutable and are reclaimed separately by GC once unreferenced — see §8.)
Retrieval therefore never returns chunks whose source document no longer exists.

### 4.4 Example plugin: email (IMAP)

Extracts email from one or more IMAP accounts.

**Configuration** (its YAML subtree): a list of IMAP **accounts**. Each account:
- `account-id` — short, stable string; used in doc-ids.
- `description` — human-readable, for human consumption only.
- connection settings — host, username, folder(s), etc. The **password comes
  from the shared secrets mechanism** (§10), never from the YAML.

**Documents**:
- `doc_id = "<account-id>:<message-id>"`.
- Duplicate message-ids (within an account) do **not** raise — it doesn't matter
  which copy is kept (they are the same message). Implemented via `upsert` with
  the default `exist_ok=True`.
- **Blob** = the full raw email in RFC 5322 format, headers included.
- **Stable metadata** (content-defining; changes bump the version):
  - `date`
  - `sender` — cleaned `NAME <EMAIL>` form (NAME optional).
  - `recipients` — cleaned, comma-separated list of `NAME <EMAIL>` entries.
  - `subject`
  - `account` — the `account-id`.
- **Transient metadata** (informational; never bumps the version):
  - `folder` — the IMAP folder the message currently resides in.
  - `read` — whether the message is currently marked read/seen.
  - (other volatile IMAP flags as useful.)
  Moving a message between folders or reading it thus updates the row without
  retriggering translation/embedding.

**State dir**: stores per-folder IMAP sync state (e.g. `UIDVALIDITY` + the set /
high-water-mark of seen UIDs) so each run cheaply detects **new** messages
(UIDs above the mark) and **deleted** ones (known UIDs no longer present →
`store.delete`). On a `UIDVALIDITY` change the folder is resynced from scratch.

**`run` options**:
- `since=<timestamp>` — only messages at/after this time.
- `until=<timestamp>` — only messages at/before this time.
- `count=<N>` — max number of messages to extract in this run (bounds cost).
- `folder=<name>` — restrict to one IMAP folder (default: all folders).
- `account=<account-id>` — restrict to one configured account (default: all).

Timestamps are parsed by the shared timestamp utility (§10.1).

---

## 5. Module 2 — Processing / Translation

Takes a document from `documents` and produces one or more Markdown translation
blobs, stored as a **translation set** (§3.2) in the DB + blob store.

**One source → several parts.** A single translation run may emit multiple
output blobs, each a *part* with its own metadata. Example: translating an email
produces a `body` part (the message text) plus one part per attachment
(`attachment:invoice.pdf`, an OCR'd image, …). The parts are grouped into one
translation set and share a single `is_active` flag, so they are always
activated / deactivated / reverted together — a document is never left with a
half-active mix of old and new parts. A simple document (e.g. a note) just
produces a set with a single `body` part.

**Possible processing steps** (composable, selected by filters):
- Data-format conversion (e.g. email → text).
- Image processing (OCR / picture recognition).
- AI rewriting: an LLM rewrites terse content into a more self-contained form
  for better embedding.
- **Context-aware rewriting**: the translator may query the *already-built RAG*
  to resolve context (e.g. what a first name refers to, overall topic).
- Access to the raw extraction result and its metadata.
- Identity translation: if the source is already plain text / Markdown, pass it
  through.

**Selection**: which documents to (re)translate and how is driven by arbitrary
filters — document type, source, age of last translation, etc.

**Output** (see the `translation_sets` / `translation_parts` schema, §3.2): a set
row (source id, source_version, run-level pipeline metadata, timestamp, git
revision, `is_active`) plus one part row per output blob (part_key, blob,
per-part metadata).

**"is active"**: old translation sets are retained (for revert / comparison).
`is_active` is a **per-set** flag marking the set currently fed to embedding; all
its parts inherit it. **At most one** active set per document (partial unique
index, §3.2). **Zero** active is also a valid state: it means the document is
currently not translated — treated the same, for RAG purposes, as a document
that hasn't been translated yet (or isn't in the index). Embedding and stats
treat zero-active as normal, not an error.

**Transactions & idempotence**: each translation set — the set row *and all its
parts* — is committed in **one transaction**, even within a batch run. A set is
therefore all-or-nothing (never a partially-written set), an interrupted batch
never loses completed sets, and a crash mid-batch leaves a consistent DB. Idempotence is
*not* built into the engine; it comes from **how you phrase the selection**.
Typical queries are naturally idempotent, e.g. "translate all untranslated
documents", "…all translated more than 100 days ago", "…all translated with an
older prompt version". Choosing correct, convergent selection filters is the
**user's responsibility**.

**Git gating**: a translation may only be **committed to the DB when the repo is
fully committed** (so `git_rev` is meaningful). Two escape hatches that do *not*
require a clean repo:
- **Inspection mode**: write the translation result to a text file for review.
- **Dry-run mode**: run the translation but do not store it in the DB.

---

## 6. Module 3 — Embedding / RAG Index

Chunks the parts of each active translation set and embeds them into Qdrant.

- **All parts of the active set are embedded**; each part is chunked
  independently.
- Chunking: configurable size / overlap.
- Embeds each chunk; the Qdrant payload carries the source reference
  (`source_type`, `doc_id`) plus the `set_id` and `part_id` it came from — for
  filtering and cleanup.
- **Auto-update**: when a document's active set changes (detected via the
  `embeddings` table, §3.2), *all* points for that document (across every part)
  are deleted and the new set's parts are re-embedded — so parts never mix across
  sets.
- Selectable by filter (like translate).
- The collection records the embed model + dimension; changing either
  invalidates the collection and forces a full re-embed. The tooling detects a
  mismatch and refuses to mix vectors of different models/dims in one collection.
- **Future option** (only when actually needed): if the embed model is changed,
  the system may be refactored to run *two* vector collections in parallel — one
  for documents on the old model, one on the new — migrating incrementally rather
  than forcing an immediate full re-embed. Deferred until required.

---

## 7. Module 4 — Chat

Chat interface over the RAG, with selectable modes and LLM models.

- **Simple mode**: the LLM is given search results retrieved for the user's
  query and answers from them.
- **Complex mode**: the LLM can actively access the RAG, the raw documents, the
  translations, and perform web searches; it plans its own retrieval strategy.

Model and mode are selectable on the command line.

---

## 8. CLI Tools

A top-level dispatcher with subcommands (each also runnable as a package
`__main__`):

- **`extract`** — run individual source plugins; plugins may add their own
  options.
- **`translate`** — run translation over a selected subset:
  - filters for selecting documents,
  - max number of documents,
  - by age of last translation,
  - (more added over time),
  - **inspection mode**: write translation output to a text file,
  - **dry-run mode**: don't store in DB (allowed even on a dirty repo).
- **`embed`** — embed a filtered subset; auto-updates changed translations.
- **`chat`** — model and mode selectable on the command line.
- **`maintenance`**:
  - GC-like tasks (filter-driven): remove unreferenced blobs, prune inactive
    translations, drop orphaned embeddings.
  - stats.
  - (new tasks added over time).

**Safety principle — prefer failing over corrupting.** Every maintenance/GC
operation is built with all reasonable safeguards, following the rule *it is OK
to abort, it is never OK to corrupt*:
- Blob/embedding GC uses reference counting or a mark-and-sweep run **under the
  write lock**, so it never deletes content a concurrent writer still references.
- On any uncertainty (unexpected state, a reference it can't resolve), GC
  **aborts loudly** rather than deleting.
- Blobs are only ever deleted when provably unreferenced; deletion is the last
  step, after all references are confirmed gone.
- Destructive maintenance supports a dry-run/report mode so the intended
  deletions can be inspected first.

**Backups (SQLite + WAL).** WAL leaves `-wal`/`-shm` sidecar files; naively
copying just the `.db` file can capture an inconsistent state. Document and/or
provide a correct procedure: run a WAL checkpoint first, or copy all three files
together, or use the SQLite Online Backup API. Blobs, being immutable and
content-addressed, are safe to back up incrementally at any time.

---

## 9. Concurrency Model

- **Metadata DB (SQLite + WAL)**: many concurrent readers + one writer across
  processes. Chat (reader) runs freely while the pipeline (writer) works.
  `busy_timeout` makes the occasional second writer wait rather than error.
- **Blobs**: immutable, content-addressed → all reads are lock-free; only GC
  deletes, and only under the write lock.
- **Vector store (embedded Qdrant)**: single-process exclusive lock on the
  storage directory. Current policy:
  - **`embed`** acquires exclusive access for the *entire run*. While it runs,
    chat cannot query the vector store.
  - **`chat`** opens the Qdrant client on demand and **releases it after ~1s of
    idle**, so `embed` can acquire the lock.
  - This is an accepted limitation for now. If truly simultaneous embed + chat
    over vectors becomes necessary, migrate to a vector store that supports
    multi-process access (e.g. Qdrant in server mode, or `sqlite-vec`, which
    inherits the WAL reader/writer model).

---

## 10. Configuration & Secrets

- **Config**: a single central YAML file holds everything except secrets
  (endpoints, model names, chunking, storage paths, per-source settings, …).
  Each extraction plugin has its **own subtree** (keyed by its `source_type`,
  e.g. under `plugins:` / `extract:`); the framework hands a plugin just that
  subtree (§4.1), so plugins never parse the global file or see unrelated keys.
- **Secrets**: resolved from the desktop keyring, with file/env fallback for
  headless use. No secret ever lives in the YAML or the repo. Plugins use the
  **same secret-resolution mechanism** as the core (passed into `run`, §4.1),
  not the config.

### 10.1 Timestamp parsing utility

A general timestamp-parsing utility lives in the **core codebase** (not in any
plugin), so every module and plugin parses times the same way. It accepts:
- absolute forms (ISO 8601 dates / datetimes), and
- convenience relative forms, e.g. `1d ago`, `2h ago`, `30m ago`, `1w ago`,
  and keywords like `now` / `today` / `yesterday`.

It is used wherever the CLI or a plugin takes a time (e.g. the email plugin's
`since` / `until`, `translate`'s "older than" filters), giving one consistent
syntax across the whole tool.

---

## 11. Repository Layout (proposed)

```
rag/
  __init__.py
  config.py          # YAML load → typed config
  secrets.py         # keyring + fallback resolution
  db.py              # SQLite schema, connection, WAL, migrations
  blobs.py           # content-addressed FS blob store
  gitinfo.py         # git revision / clean-tree checks
  timestamps.py      # shared timestamp parsing ("1d ago", ISO 8601, …)
  cli.py             # top-level subcommand dispatch
  extract/
    __init__.py
    base.py          # Extractor ABC + DB-facing API (enforces source isolation)
    __main__.py
    plugins/
      email/         # the only source plugin for now (imap.py = IMAP I/O + sync state)
  translate/
    base.py          # processing steps, filters, translator API
    __main__.py
  embed/
    __init__.py
    __main__.py
  chat/
    __init__.py
    __main__.py
  maintenance/
    __init__.py
    __main__.py
tests/
```

---

## 12. Suggested Build Order

1. **Foundations**: `config`, `secrets`, `db` (schema + WAL + migrations),
   `blobs`, `gitinfo`. Tests for each.
2. **Module 1 (extract)**: general part + `Extractor` API + isolation
   enforcement; the email (IMAP) plugin end-to-end. (Work email-only until the
   system is proven; other source plugins deferred.)
3. **Module 2 (translate)**: identity/pass-through translation first, then the
   git gating, inspection & dry-run modes, then AI rewriting.
4. **Module 3 (embed)**: chunk + embed active translations, staleness-driven
   re-embed.
5. **Module 4 (chat)**: simple mode first, then complex mode.
6. **Additional extraction plugins** (notes, chats, …) and **maintenance**
   (GC, stats), once email has proven the design.
7. Iterate: more plugins, more filters, context-aware translation.

---

## 13. Feedback & Resolved Decisions

Original review points, each with the decision taken. Points marked **[open]**
are accepted risks worth revisiting later; the rest are resolved and folded into
the body above.

1. **Vector-store concurrency is the weakest part. [open]** The embedded-Qdrant
   policy (embed takes an exclusive lock for its whole run; chat releases on
   idle) means chat's retrieval is unavailable during embedding, and requires
   hand-written retry/backoff plus an idle-timer lifecycle on the chat side.
   Residual risks: lock races, a **stale lock if `embed` crashes mid-run**
   (needs cleanup), and up-to-1s chat stalls. *Decision:* keep embedded Qdrant
   with this policy for now (§9). `sqlite-vec` (same single-file model, real WAL
   reader/writer concurrency, no custom locking) remains the recommended
   fallback if simultaneous embed + chat over vectors becomes necessary.

2. **Zero active translations. Resolved.** A partial unique index enforces *at
   most one* active; *zero* active is now defined as a valid state — the document
   is simply currently untranslated, equivalent for RAG purposes to not being
   inserted. Embedding and stats treat it as normal (§5).

3. **Deletion / orphan propagation. Resolved.** Deletion is immediate and
   cascading: removing a document removes its translations, embeddings, and
   vector-store points in the same operation; blobs are reclaimed by GC once
   unreferenced (§4.3). Retrieval never returns chunks for a deleted document.

4. **Context-aware translation is circular / bootstrapping-sensitive.
   Accepted.** The circularity (RAG built from translations that may query the
   RAG) and non-uniform quality on cold start are by design. The CLI lets you
   re-run and improve translations in later passes; choosing a sensible order of
   passes is the **user's responsibility** (§5).

5. **Resumability & idempotence. Resolved.** Each translation is its own
   transaction (even in a batch), so an interruption never loses completed work.
   Idempotence is not engine-enforced; it emerges from convergent selection
   queries ("translate all untranslated", "…older than 100 days", "…with an
   older prompt"). Correct queries are the **user's responsibility** (§5).

6. **Trustworthy versioning. Resolved.** Version checking is done by the
   framework, not the plugin: if the plugin-supplied source or metadata changes,
   the version increases — even if the change is spurious. It is the **plugin's
   responsibility** to avoid injecting volatile data (e.g. current date) that
   would cause spurious bumps (§4.2).

7. **Git-clean gating. Kept.** Hard-blocking DB writes on a dirty repo is kept
   for now (inspection mode covers dev-time needs; §5). Easy to relax later
   (e.g. a `dirty` flag + `--force`) if it proves too rigid.

8. **Embed model/dim change is a full-collection event. Resolved.** The tooling
   records model + dim and refuses to mix them in one collection (§6). A future
   two-collection scheme (old-model + new-model side by side) is noted for when a
   model change actually happens — deferred until needed.

9. **Metadata storage. Resolved.** `metadata` is JSONB (source of truth); common
   keys are indexed `GENERATED ... VIRTUAL` columns; arbitrary keys are
   scan-queryable via `json_extract`, promotable later (§3.2).

10. **Migrations from day one. Resolved (detailed).** Numbered, forward-only,
    transactional migration steps tracked by `schema_version`, applied
    automatically on startup, with migration `0001` creating the initial schema.
    Full mechanism specified in §3.3.

11. **GC & backup safety. Resolved.** All reasonable safeguards added under the
    principle *OK to fail, never OK to corrupt*: GC runs under the write lock,
    aborts on uncertainty, deletes blobs only when provably unreferenced, offers
    dry-run; plus a documented WAL-safe backup procedure (§8).

12. **Testing strategy. Adopted.** Fixture-based tests per module (fake source
    dir, tmp SQLite, tmp blob dir, fake embedding endpoint) are part of the build
    order (§12), written alongside each module rather than after.




