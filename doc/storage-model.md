# Storage model

Metadata lives in SQLite (`storage/metadata.db`); large binary content lives on
the filesystem as content-addressed blobs (`storage/blobs/`). The DB never stores
blob bytes, only hashes.

## Blob store (filesystem)

- Content-addressed: path derived from the content hash (SHA-256), sharded, e.g.
  `blobs/ab/cd/abcd…`.
- Immutable: writing the same content twice is a no-op (natural dedup).
- The DB never stores blob bytes, only the hash.
- All blob reads are lock-free (files never mutate); deletion is only via GC.

## Metadata database (SQLite)

Core tables: `documents` (extraction output), `translation_sets` +
`translation_parts` (processing output), plus bookkeeping tables `embeddings`
and `schema_version`.

### `documents` — output of extraction (module 1). One row per live document.

| Column        | Type    | Notes |
|---------------|---------|-------|
| source_type   | TEXT    | Plugin id, e.g. `email`. |
| doc_id        | TEXT    | Unique within a source_type. |
| version       | INTEGER | Starts at 1, increments when the source content or **stable** metadata changes. |
| blob_hash     | TEXT    | Reference to the raw source blob in the blob store. |
| stable_metadata | JSONB | Content-defining metadata: general keys (date, title, mime-type, …) + arbitrary plugin-specific keys. A change here **bumps the version** and retriggers downstream pipelines. |
| transient_metadata | JSONB | Informational, mutable metadata that does **not** affect the version: e.g. the IMAP folder a message currently sits in, whether it is read/unread. Updated freely; never retriggers translation/embedding. |
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
  promotable to a generated column later). See [metadata.md](metadata.md) for the
  concrete keys.

A single translation run of a document may produce **several output blobs**
("parts") — e.g. translating an email yields the message body plus one
translation per attachment. Each part has its own blob and its own metadata, but
all parts of one run are **tied together**: they become active or inactive as a
unit. This is modelled as two tables — a *translation set* (the run) and its
*parts*.

### `translation_sets` — one row per translation run of a document. History is kept.

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

### `translation_parts` — one row per output blob within a set.

| Column      | Type    | Notes |
|-------------|---------|-------|
| id          | INTEGER | Surrogate primary key. |
| set_id      | INTEGER | FK → `translation_sets.id`. |
| sub_doc_id  | JSONB   | The part's `sub_doc_id` — a **list of strings** identifying the part within its document (e.g. `["body"]`, `["attachment:1","body"]`). Source of truth. |
| blob_hash   | TEXT    | Reference to this part's translated Markdown blob. |
| metadata    | JSONB   | Per-part metadata (may differ between parts, e.g. attachment filename / mime-type / OCR flag). |
| part_key    | TEXT    | **Generated** canonical text of `sub_doc_id` (`json(sub_doc_id)`) — used for the uniqueness constraint and lookups; always in sync by construction. |

- Unique `(set_id, part_key)`, i.e. one part per distinct `sub_doc_id` in a set.
- Deleting a set cascades to its parts.

### `embeddings` (bookkeeping for module 3)

The **join between parts and embedding models**: one row per `(part, model)`
actually embedded, so staleness can be detected per model. A part may have several
rows (one per model). See [embedding.md](embedding.md) for the multi-model design.

| Column       | Type    | Notes |
|--------------|---------|-------|
| source_type  | TEXT    | |
| doc_id       | TEXT    | |
| set_id       | INTEGER | Which `translation_sets.id` is embedded for this model. |
| part_id      | INTEGER | Which `translation_parts.id` these points came from. |
| model        | TEXT    | The embedding **model id** (from the registry; its own index/collection). |
| embedded_at  | TEXT    | Timestamp. |

- Primary key: `(part_id, model)` — a part is embedded at most once per model.
- Re-embedding for a given model is required when a document's active `set_id`
  differs from the row's embedded `set_id`; each model is refreshed independently
  on its own pass. Absence of a row for a `(part, model)` means "not yet embedded
  by that model" — a normal state, not an error.

### `schema_version`

A single integer row holding the DB's current schema version; used by the
migration mechanism below.

## Schema migrations

The schema *will* evolve (new metadata columns, new tables, new indexes). A
migration mechanism is built in **from day one**, before there is any real data —
retrofitting migrations onto a populated DB later is painful.

How it works:

- **Migrations are an ordered, numbered list** of forward steps, e.g. `0001_init`,
  `0002_add_translation_index`, … Each step is a small unit (SQL and/or Python)
  that transforms the schema from version *N-1* to version *N*. Migration `0001`
  creates the entire initial schema, so a database is *never* in an un-versioned
  state.
- **The `schema_version` row records the highest applied step.**
- **On every startup**, the runner reads `schema_version` and applies each pending
  step in order, up to the version the code expects.
- **Each step runs in its own transaction** and bumps `schema_version` within that
  same transaction. If a step fails, it **rolls back cleanly**, leaving the DB at
  the previous version — never half-migrated (ties into the "fail, don't corrupt"
  principle).
- **Forward-only.** For a single-user local tool there is no need for
  down-migrations; reverting means restoring a backup.
- The code refuses to open a DB whose `schema_version` is *newer* than it
  understands, rather than risk misinterpreting it.
- If the on-disk **blob layout** ever needs to change, it is versioned and migrated
  by the same discipline.
