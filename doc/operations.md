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
- **`chat`** — interactive chat over the RAG (simple / retrieve-then-read mode).
  `--model` picks the chat LLM (default `chat.default_model`), `--top-k` the number
  of retrieved chunks, `--models` restricts which embedding indices are queried.
  See [chat.md](chat.md).
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
  - `fsck` — check the storage-model invariants that are *assumed* but not
    enforced (a crash between the SQLite / vector / blob stores, a half-finished
    run, or a manual edit can break them). Read-only. Each invariant is one check:
    dangling translation sets (no document), parts (no set) and embeddings rows
    (no part/set/document); more than one active translation set per document;
    blob references with no blob on disk; **dangling vector-store points** (no
    backing embeddings row/document) and collections matching no configured model.
    It also calls each extractor's `fsck()` on its own state dir — the email plugin
    validates its presence file (`uidmap.json`): correct JSON/shape and no two uids
    flagged `extracted:true` for the same message-id. Problems print as they are
    found; exits non-zero if any *error*-severity problem is found. `--fix` applies
    repairs — `no` (report only, default), `ask` (confirm each), `yes` (apply all);
    each fix announces exactly what it changes. Fixes so far: **dangling points**
    are deleted from their collection; an email **invariant-2** violation clears
    `extracted` on all uids of that message-id (a later extract re-stores exactly
    one). A central check attaches a fix by setting `apply_fix`/`fix_description` on
    its `Problem`; a plugin fixes inside its own `fsck` — both via the shared
    `ask_fix` helper (`rag/fsck.py`). Extend either way.

The **agentic** chat mode and GC maintenance tasks are not built yet — see
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
- **Vector store (Qdrant)**: concurrency depends on the deployment mode (below).
  - **Embedded-local**: single-process exclusive lock on the storage directory.
    `embed` acquires it for the *entire run*; chat opens on demand and releases
    after ~1s idle so `embed` can acquire it. An accepted limitation of this mode.
  - **Server**: the server handles concurrency — `embed` and chat can talk to it
    at once, no directory lock. This removes the limitation above.

## Vector store (Qdrant): embedded-local vs server

The vector backend is selected by the top-level `vector_store` config section.

- **Embedded-local (default).** Omit `vector_store.url`. Qdrant runs in-process
  (`QdrantClient(path=…)`) against `storage/qdrant`. It loads the whole collection
  into **RAM** — simple and dependency-free, but RAM-bound and single-writer. Good
  for development and small corpora.
- **Server (recommended at scale).** Set `vector_store.url`. Vectors can live
  **on disk** with optional **quantization**, so RAM holds only the HNSW graph
  (+ quantized vectors), not every raw vector — required once the corpus outgrows
  RAM. Run the server separately (e.g. `docker run -p 6333:6333 qdrant/qdrant`).

```yaml
vector_store:
  url: "http://localhost:6333"    # set => server mode; omit => embedded-local
  prefer_grpc: false              # gRPC transport (port 6334) instead of HTTP
  timeout: 60                     # per-request timeout (seconds)
  on_disk: true                   # store vectors on disk (default true in server mode)
  quantization:                   # optional; omit for full-precision fp32
    type: scalar                  # int8 scalar quantization (~4x smaller)
    always_ram: false             # int8 copy on disk too (lowest RAM); true pins it in RAM
```

`on_disk` and `quantization` are **server-only** collection settings — the
embedded-local store is always in RAM and cannot honour them. Setting either
without a `url` is rejected with a `ConfigError` (rather than silently giving no
benefit).

**Create-only.** Both `on_disk` and `quantization` are read **only when a
collection is first created** (the `embed` pass creates a model's collection on
its first run). Editing them in `config.yaml` afterwards has **no effect** on an
existing collection — the tooling never re-applies them. To change the storage
profile of a populated model you must either:
- **recreate** the collection — drop it (or the whole model's index) and re-embed
  (cheap once an embedding cache exists), or re-run `migrate_qdrant_to_server.py`
  into a fresh collection; or
- issue a server-side **`update_collection`** (`quantization_config` / vectors
  `on_disk`), after which Qdrant's optimizer rebuilds the affected data in the
  background.

So decide `on_disk` + `quantization` **before the first `embed`** of each model
(or before running the migration).

**Authentication.** The API key is **not** a config option: when the server needs
one, store it in the keyring in the shared `rag` service (the same service the
email plugin uses), under a name **scoped to the server's host:port** —
`vector_store_api_key:<host>:<port>`. The host:port suffix (derived from
`vector_store.url`; default port 6333) ensures a key stored for one server is
never sent to a different one by accident. E.g. for `http://localhost:6333`:

```sh
keyring set rag vector_store_api_key:localhost:6333
```

The client resolves it from there (keyring/file) and never prompts — an absent key
simply means "no auth". Options, in the order you'd reach for them on a single-user box:
- **None + network isolation** — bind the server to localhost / a private network
  and set no key. Fine when nothing else can reach the port. Store nothing.
- **Static API key** — start the server with `QDRANT__SERVICE__API_KEY=…` and store
  that same value under `vector_store_api_key:<host>:<port>`. Sent as the `api-key`
  header. Use an `https://` URL so it isn't sent in clear. A separate
  `QDRANT__SERVICE__READ_ONLY_API_KEY` can gate read-only clients (db-browser /
  chat) from the writer (`embed`).
- **JWT / RBAC** (Qdrant ≥1.9, requires an API key set) — a signed token scoping
  access to specific collections, read-only, payload filters, and expiry. Store the
  token under `vector_store_api_key:<host>:<port>`; the client sends it identically.
- **TLS** — use `https://` in the URL; the client negotiates TLS. (Mutual-TLS /
  client certificates are possible server-side but not wired into the config here.)

**Migrating an existing embedded-local store to a server.** The two on-disk formats
are incompatible, so you copy the points rather than move files:
1. Start the server and add the `vector_store` section above to `config.yaml`.
2. Run `python scripts/migrate_qdrant_to_server.py --config config.yaml` once. It
   reads the local store at `storage/qdrant`, recreates each collection on the
   server with your `on_disk`/`quantization` profile, and copies every point
   (deterministic ids → re-runnable, no duplicates). It never deletes the local
   directory — remove it yourself once the server is verified.

> Note: `maintenance stats` measures `vector_store_bytes` from the local
> `storage/qdrant` directory only; in server mode the vectors live on the server
> host, so that figure reads 0 here.

## Backups (SQLite + WAL)

WAL leaves `-wal`/`-shm` sidecar files; naively copying just the `.db` file can
capture an inconsistent state. Use a correct procedure: run a WAL checkpoint first,
or copy all three files together, or use the SQLite Online Backup API. Blobs, being
immutable and content-addressed, are safe to back up incrementally at any time.
Per-plugin `state_dir` contents (e.g. the email `uidmap.json` and OAuth `secrets/`)
are real state, not caches, and belong in backups.
