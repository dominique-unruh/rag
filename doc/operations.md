# Operations — CLI, concurrency, backups

## CLI tools

A top-level dispatcher (`rag`) with subcommands (each also runnable as a package
`__main__`):

- **`extract`** — run individual source plugins; plugins may add their own options
  (see the email options in [extraction.md](extraction.md)).
- **`translate`** — run translation over a selected subset:
  - filters for selecting documents,
  - max number of documents,
  - by age of last translation,
  - **inspection mode**: write translation output to a text file,
  - **dry-run mode**: don't store in DB (allowed even on a dirty repo).
- **`embed`** — embed a filtered subset (per model); auto-updates changed
  translations. See [embedding.md](embedding.md).
- **`maintenance <tool>`**:
  - `db-browser` — a local web UI for browsing the extraction database. Split
    screen: the `documents` rows with their most relevant fields on top; the
    selected document's full metadata (stable + transient) as a table below, plus a
    content preview and a download button (using the document's mime-type).
  - `delete` — cascading deletion via the store API.
  - `stats` — count everything in the metadata DB (read-only): documents (total +
    per `source_type`), translation sets (total/active), documents with no active
    translation, active sets gone stale (their `source_version` no longer matches
    the document's `version`), and translation parts (total + in active sets); plus,
    per embedding model, how many active parts are embedded vs still missing and the
    raw embedding-row count (flagging models in the DB but absent from config). Each
    extraction plugin also contributes its own cheap `statistics()` read from its
    state dir only (no network/IMAP/DB scans). It also reports **on-disk storage
    use**: the metadata DB (with WAL sidecars), the blob store (actual disk
    allocation via `du`), the vector store directory, and each plugin's state dir. `--source <type>` restricts to one source; `--json` emits
    machine-readable output.

The **`chat`** subcommand and GC maintenance tasks are not built yet — see
[`../PLAN.md`](../PLAN.md).

**Safety principle — prefer failing over corrupting.** Every maintenance/GC
operation is built with all reasonable safeguards, following the rule *it is OK to
abort, it is never OK to corrupt*:
- Blob/embedding GC uses reference counting or a mark-and-sweep run **under the
  write lock**, so it never deletes content a concurrent writer still references.
- On any uncertainty (unexpected state, a reference it can't resolve), GC **aborts
  loudly** rather than deleting.
- Blobs are only ever deleted when provably unreferenced; deletion is the last step,
  after all references are confirmed gone.
- Destructive maintenance supports a dry-run/report mode so the intended deletions
  can be inspected first.

## Concurrency model

- **Metadata DB (SQLite + WAL)**: many concurrent readers + one writer across
  processes. Chat (reader) runs freely while the pipeline (writer) works.
  `busy_timeout` makes the occasional second writer wait rather than error.
- **Blobs**: immutable, content-addressed → all reads are lock-free; only GC
  deletes, and only under the write lock.
- **Vector store (embedded Qdrant)**: single-process exclusive lock on the storage
  directory. Current policy:
  - **`embed`** acquires exclusive access for the *entire run*. While it runs, chat
    cannot query the vector store.
  - **`chat`** opens the Qdrant client on demand and **releases it after ~1s of
    idle**, so `embed` can acquire the lock.
  - This is an accepted limitation for now. If truly simultaneous embed + chat over
    vectors becomes necessary, migrate to a vector store that supports multi-process
    access (e.g. Qdrant in server mode, or `sqlite-vec`, which inherits the WAL
    reader/writer model).

## Backups (SQLite + WAL)

WAL leaves `-wal`/`-shm` sidecar files; naively copying just the `.db` file can
capture an inconsistent state. Use a correct procedure: run a WAL checkpoint first,
or copy all three files together, or use the SQLite Online Backup API. Blobs, being
immutable and content-addressed, are safe to back up incrementally at any time.
Per-plugin `state_dir` contents (e.g. the email `uidmap.json` and OAuth `secrets/`)
are real state, not caches, and belong in backups.
