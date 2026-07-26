# Module 1 — Extraction

Pulls data from sources into the `documents` table + blob store.

**Structure**: a *general part* (owns the DB, blob store, CLI, plugin discovery)
and a *plugin system*.

- Each plugin is an implementation of an `Extractor` class defined by the general
  part, located in its own subdirectory of the extraction package (e.g.
  `extract/plugins/email/`). The **`source_type` is the plugin's subdirectory
  name** — the framework derives and enforces it from the directory; the plugin
  does not declare it. This keeps the id single-sourced (no field to drift from
  the directory) and guarantees uniqueness (directory names are unique).
- A plugin **never touches the DB directly**. It receives a `Store` handle
  (provided by the general part) which is the only interface to the system.
- **Isolation is enforced by the API**: the `Store` handed to a plugin is bound to
  that plugin's `source_type`; every operation is implicitly scoped to it, so a
  plugin can never read or affect another plugin's documents.

## The `Extractor` / `Store` API

A plugin subclasses `Extractor` and implements `run`. The general part discovers
plugins by subdirectory, takes the **directory name as the `source_type`**, opens
a `Store` scoped to that id, and calls `run`.

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
        (desktop keyring + file fallback) — so a plugin fetches its passwords /
        API keys exactly the way the core does, rather than reading them from
        `config`. Secrets never live in the YAML.

        `state_dir` is a private, persistent directory owned by this plugin,
        for its own bookkeeping across runs (e.g. an IMAP plugin remembering
        which message UIDs it has already processed, or an incremental sync
        cursor). The framework creates it and guarantees it is stable across
        runs and unique to this plugin; its contents are opaque to the system.
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
        - `exist_ok=False` -> raise if the document already exists.
        - `missing_ok=False` -> raise if the document does not yet exist.
        Both default to True (permissive upsert). They let a plugin catch its
        own bugs — e.g. a duplicate doc_id, or an update to something it thought
        it had already created.
        """

    def delete(self, doc_id: str, *, missing_ok: bool = True) -> bool:
        """Delete a document that vanished at the source. Cascades immediately
        (see below). Returns whether a row existed.

        `missing_ok=False` -> raise if the document does not exist. Defaults to
        True (deleting an absent document is a no-op)."""

    def exists(self, doc_id: str) -> bool:
        """Whether a document with this id currently exists."""

    def list_ids(self, *, stable: dict[str, object] | None = None,
                 transient: dict[str, object] | None = None) -> "Iterable[str]":
        """Current doc_ids for this source_type, optionally filtered by metadata.

        `stable` / `transient` map a metadata key to a scalar (equality) or a
        list (membership); all conditions AND together. Unfiltered, returns every
        id (used to detect vanished documents); filtered, finds documents by
        metadata (e.g. all mail whose transient `folder` is now excluded)."""

    def get_metadata(self, doc_id: str) -> "dict[str, object] | None":
        """Stored **stable** metadata for a document, or None if absent."""

    def get_transient(self, doc_id: str) -> "dict[str, object] | None":
        """Stored **transient** metadata for a document, or None if absent."""
```

Note that **version handling lives entirely in the framework** (in `upsert`), not
in the plugin. The plugin just declares "here is the current content and metadata
for this doc_id"; the framework decides whether that constitutes a change.

The per-plugin `state_dir` lives under the configured storage directory (e.g.
`storage/plugin_state/<source_type>/`), is never touched by GC or other plugins,
and should be included in backups (it is real state, not a cache — losing it may
force a plugin into a full re-scan or duplicate work).

## Change detection

`upsert` bumps `version` when the content **or the stable metadata** changed (both
are inputs to translation, so either changing warrants reprocessing). It compares
the new content hash against the stored `blob_hash` *and* the new stable metadata
against the stored stable metadata, and skips the bump only when both are
identical.

**Transient metadata never bumps the version.** A change to `transient_metadata`
alone (e.g. a mail moved to another folder, or marked read) is written straight to
the row without incrementing `version` and without retriggering translation or
embedding. Every `upsert` refreshes the transient metadata; if content and stable
metadata are unchanged, the result is reported as *unchanged* (no reprocessing)
even though the transient state was updated. Bookkeeping fields (e.g. `updated_at`)
are likewise excluded from the change comparison.

The bump is **content-driven, not semantic**: if the plugin supplies changed bytes
or metadata, the version increases, even if the change is cosmetically spurious. It
is the **plugin's responsibility** not to introduce spurious changes — e.g. don't
inject the current date/time into the content or a non-stable field ordering, or
every run will look like a change and trigger needless retranslation + re-embedding.

## Deletion (cascade)

Deletion is **immediate and cascading**. When a document is deleted (via
`Store.delete`), in the same operation the framework removes: the `documents` row,
all its `translation_sets` and their `translation_parts`, its `embeddings`
bookkeeping rows, and the corresponding points in the vector store. (Blobs are
shared/immutable and are reclaimed separately by GC once unreferenced.) Retrieval
therefore never returns chunks whose source document no longer exists.

## Example plugin: email (IMAP)

Extracts email from one or more IMAP accounts.

**Configuration** (its YAML subtree): a list of IMAP **accounts**. Each account:
- `account-id` — short, stable string; used in doc-ids.
- `description` — human-readable, for human consumption only.
- connection settings — host, username, optional `folders` (default: all) and
  `exclude_folders` (skip-list, e.g. Spam/Trash), etc. The **password comes from
  the shared secrets mechanism** ([configuration.md](configuration.md)), never
  from the YAML.
- `auth` — authentication mechanism: `login` (IMAP `LOGIN`, default), `plain`
  (SASL `PLAIN`, for servers that reject `LOGIN` but accept it over TLS), or
  `xoauth2` (OAuth2 bearer token, required by servers that no longer accept
  passwords — e.g. Microsoft 365). For `xoauth2` the account may carry an optional
  `oauth:` subtree (`tenant`, `client_id`, `scopes`, `client_secret`) overriding
  built-in defaults; see [configuration.md](configuration.md) for the token
  handling.

**Documents**:
- `doc_id = "<message-id>"` — the bare Message-ID, **not** scoped by account.
  Rationale: the same message moved between accounts is still the same message, so
  it keeps one identity. (If a message lacks a Message-ID, a UID-based id is
  synthesised.)
- Duplicate message-ids do **not** raise — it doesn't matter which copy is kept
  (they are the same message). Implemented via `upsert` with the default
  `exist_ok=True`.
- **Blob** = the full raw email in RFC 5322 format, headers included.
- **Stable / transient metadata**: see [metadata.md](metadata.md).

**State dir — the presence file** (`uidmap.json`, one joint file shared by *all*
accounts). For each account+folder it stores the `UIDVALIDITY` and a
`uid -> {"message-id", "extracted"}` map:

```json
{"accounts": {"<account-id>": {"folders": {"<folder>": {
  "uidvalidity": 12345,
  "uids": {"<uid>": {"message-id": "<doc_id>", "extracted": true}}}}}}}
```

`message-id` is the document identity (`doc_id`, shared across accounts); `extracted`
records whether **this uid's bytes** are the version currently stored for that
message-id. A message-id can appear under several uids (same message in several
folders/accounts), but only **one** copy is ever stored, so at most one of those
uids is `extracted:true`.

**Invariants** — each chosen so a stale or partial file degrades safely, never
destructively:

1. **Incomplete by design.** The file lists only uids some past pass has seen. A
   message-id / folder / account not in it means "not yet observed", never "gone".
   An additive run only touches the folders it actually swept.
2. **At most one `extracted:true` per message-id.** Exactly one document is stored
   per message-id; the flag marks which uid-version fills it. Every other uid
   sharing that message-id is `false`.
3. **`extracted:true` ⟹ the document exists** for that message-id (best-effort;
   self-heals by re-download if the store was wiped underneath the file).
4. **Extract fails safe.** A wrong or missing flag at worst makes extract re-fetch a
   message it already has — `upsert` is idempotent, so never corruption or loss.
5. **Delete ignores the flag.** Deletion decides purely on **message-id presence**
   (server message-ids vs. the file's, after a complete refresh). Even fully corrupt
   flags cannot cause a wrong deletion — only a wrong *message-id set* could, and the
   delete pass rebuilds that set in full before deciding.

**Extract (additive sync).** For each in-scope folder, over the uids the server
returns for the (optionally date-restricted) query:
- A uid is a **download candidate** unless the file already records its message-id as
  extracted — i.e. this uid is `extracted:true`, or a sibling uid with the same
  message-id is. Already-extracted message-ids are skipped without a fetch.
- On download: fetch → resolve message-id → `upsert`; set this uid `extracted:true`
  and flip any sibling with the same message-id to `false` (invariant 2). If the
  fetch reveals the message-id was already extracted elsewhere, record the uid
  `extracted:false` and skip the redundant `upsert`.
- Newly-seen uids are added; **completeness is never assumed** — folders not visited
  this run keep their existing entries.
- `full=1` starts from an empty extracted-set, so every in-scope message is re-fetched
  and re-stored (still bounded by `count`).

**Deletion (`delete=true`)** — an opt-in, expensive, plugin-internal pass, in two
ordered phases:
1. **Refresh the file completely** — scan **every** account's included folders and
   record all live `uid -> message-id` pairs (resolving an unknown uid via a cheap
   Message-ID-only header fetch). Existing `extracted` flags are carried forward;
   newly-seen uids get `extracted:false`. This phase stores no bodies, so it never
   sets a flag `true`.
2. **Reconcile** — delete every stored document whose message-id is a live uid
   **nowhere** across all accounts. Deletion is thus by **global presence** (delete
   only when the last holder is gone), not per-folder — so a move between folders or
   servers never looks like a deletion, and a message duplicated on two servers
   survives until removed from both. This subsumes an excluded-folder purge: a message
   living only in an `exclude_folders` folder is absent from the map and so is deleted.

Conservative on failure: an unreachable account keeps its prior entries (its documents
stay present), and an interrupted scan deletes nothing. `delete=true` is a standalone,
global mode and rejects the additive options below.

**`run` options**:
- `since=<timestamp>` — only messages at/after this time.
- `until=<timestamp>` — only messages at/before this time.
- `count=<N>` — max number of messages to extract in this run (bounds cost).
- `folder=<name>` — restrict to one IMAP folder (default: all folders).
- `account=<account-id>` — restrict to one configured account (default: all).
- `full=1` — force a full resync: invalidate the stored UID memory so every
  message is re-considered. Still bounded by `count`, so a large mailbox resyncs
  over several runs.
- `delete=true` — run the standalone cross-account deletion pass instead of an
  additive sync (see above).

Timestamps are parsed by the shared timestamp utility (see
[configuration.md](configuration.md)).
