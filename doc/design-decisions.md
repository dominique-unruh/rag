# Design decisions & rationale

Original design-review points, each with the decision taken. Points marked
**[open]** are accepted risks worth revisiting later (also tracked in
[`../PLAN.md`](../PLAN.md)); the rest are resolved and reflected in the other docs
here.

1. **Vector-store concurrency is the weakest part. [open]** The embedded-Qdrant
   policy (embed takes an exclusive lock for its whole run; chat releases on idle)
   means chat's retrieval is unavailable during embedding, and requires
   hand-written retry/backoff plus an idle-timer lifecycle on the chat side.
   Residual risks: lock races, a **stale lock if `embed` crashes mid-run** (needs
   cleanup), and up-to-1s chat stalls. *Decision:* keep embedded Qdrant with this
   policy for now (see [operations.md](operations.md)). `sqlite-vec` (same
   single-file model, real WAL reader/writer concurrency, no custom locking)
   remains the recommended fallback if simultaneous embed + chat over vectors
   becomes necessary.

2. **Zero active translations. Resolved.** A partial unique index enforces *at most
   one* active; *zero* active is a valid state — the document is simply currently
   untranslated, equivalent for RAG purposes to not being inserted. Embedding and
   stats treat it as normal.

3. **Deletion / orphan propagation. Resolved.** Deletion is immediate and cascading:
   removing a document removes its translations, embeddings, and vector-store points
   in the same operation; blobs are reclaimed by GC once unreferenced. Retrieval
   never returns chunks for a deleted document.

4. **Context-aware translation is circular / bootstrapping-sensitive. Accepted.** The
   circularity (RAG built from translations that may query the RAG) and non-uniform
   quality on cold start are by design. The CLI lets you re-run and improve
   translations in later passes; choosing a sensible order of passes is the user's
   responsibility.

5. **Resumability & idempotence. Resolved.** Each translation is its own transaction
   (even in a batch), so an interruption never loses completed work. Idempotence is
   not engine-enforced; it emerges from convergent selection queries. Correct queries
   are the user's responsibility.

6. **Trustworthy versioning. Resolved.** Version checking is done by the framework,
   not the plugin: if the plugin-supplied source or metadata changes, the version
   increases — even if the change is spurious. It is the plugin's responsibility to
   avoid injecting volatile data (e.g. current date) that would cause spurious bumps.

7. **Git-clean gating. Kept.** Hard-blocking DB writes on a dirty repo is kept for now
   (inspection mode covers dev-time needs). Easy to relax later (e.g. a `dirty` flag +
   `--force`) if it proves too rigid.

8. **Embed model/dim change is a full-collection event. Resolved.** The tooling records
   model + dim and refuses to mix them in one collection. A future two-collection
   scheme (old-model + new-model side by side) is noted for when a model change
   actually happens — deferred until needed.

9. **Metadata storage. Resolved.** `metadata` is JSONB (source of truth); common keys
   are indexed `GENERATED ... VIRTUAL` columns; arbitrary keys are scan-queryable via
   `json_extract`, promotable later.

10. **Migrations from day one. Resolved (detailed).** Numbered, forward-only,
    transactional migration steps tracked by `schema_version`, applied automatically
    on startup, with migration `0001` creating the initial schema. Full mechanism in
    [storage-model.md](storage-model.md).

11. **GC & backup safety. Resolved.** All reasonable safeguards added under the
    principle *OK to fail, never OK to corrupt*: GC runs under the write lock, aborts
    on uncertainty, deletes blobs only when provably unreferenced, offers dry-run;
    plus a documented WAL-safe backup procedure (see [operations.md](operations.md)).

12. **Testing strategy. Adopted.** Fixture-based tests per module (fake source dir,
    tmp SQLite, tmp blob dir, fake embedding endpoint), written alongside each module
    rather than after.

13. **Blob compression: per-blob zstd, no packing/dictionary. Resolved.** Measured a
    real 2200-blob email store (262 MiB): zstd -19 saves **~47% on disk**, while
    filesystem block-rounding waste was only **1.8%** (blobs average ~122 KiB — the
    store is byte-heavy, not file-count-heavy). So **packing** small blobs into
    containers would recover <2% for a lot of complexity (a blob index as a second
    source of truth, a repack tool, GC interaction) — rejected. A **trained
    dictionary** only helps small inputs, which are negligible here — rejected. zstd
    adapts within a single frame but there is no cross-blob online learning; each
    blob is an independent frame (required anyway for content-addressed random
    access). *Decision:* 1-byte codec marker (`0` raw / `1` zstd; 256 slots) +
    `min(raw, zstd)` payload, hash over uncompressed content; a one-time
    `BlobStore.migrate()` (run at pipeline start, guarded by a `.format_version`
    sentinel) upgrades legacy raw blobs, so the read path carries no legacy
    fallback. See [storage-model.md](storage-model.md).
