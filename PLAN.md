# RAG Pipeline — Design Plan (remaining work)

A local, single-user RAG system that ingests personal data (email, notes, …),
turns it into clean Markdown, embeds it into a vector index, and exposes a chat
interface over it.

The pipeline has four stages — **Extract → Translate → Embed → Chat**. The first
three are **implemented**; their architecture is documented under
[`doc/`](doc/) (start at [`doc/README.md`](doc/README.md)). This file tracks only
what is **not yet built**.

---

## Module 4 — Chat (not built)

Chat interface over the RAG, with selectable modes and LLM models.

- **Simple mode**: the LLM is given search results retrieved for the user's query
  and answers from them.
- **Complex mode**: the LLM can actively access the RAG, the raw documents, the
  translations, and perform web searches; it plans its own retrieval strategy.

Model and mode are selectable on the command line. Chat LLMs use the same backend
options as embedding models (see [`doc/embedding.md`](doc/embedding.md)): an
**OpenAI-compatible** endpoint (hosted, or local via Ollama's `/v1` / LM Studio /
vLLM / llama.cpp) or **native Ollama**.

**Retrieval / fusion.** When several embedding models are populated, retrieval
queries each selected model's index and **fuses** the ranked lists (e.g.
Reciprocal Rank Fusion) into one result set — this is where a dense model and a
sparse/lexical model combine into hybrid search (semantic recall + exact token
matches). Which models participate is a query-time choice; a model with incomplete
coverage simply contributes what it has. (A cross-encoder reranker could later sit
after fusion as an optional query-time stage — no index change.)

The `chat` CLI subcommand is likewise not built yet.

---

## More extractors (not built)

New source plugins to add once email has proven the design (each a subdirectory
under `extract/plugins/`, `source_type` = directory name):

- **Zulip** — messages/streams from a Zulip account.
- **Other chat accounts** — further chat platforms (Signal, Messenger, WhatsApp, Telegram, …).
- **Notes (in Obsidian)**.
- **RWTH Intranet** — in particular regulations, *Rundschreiben*, and similar
  official documents.

---

## Maintenance — remaining tasks (not built)

The `maintenance` dispatcher exists with `db-browser`, `delete` and `stats` (see
[`doc/operations.md`](doc/operations.md)). Still to build:

- **GC-like tasks** (filter-driven): remove unreferenced blobs, prune inactive
  translations, drop orphaned embeddings — each under the write lock, aborting on
  uncertainty, with a dry-run/report mode (the safety principle is documented in
  [`doc/operations.md`](doc/operations.md)).

---

## Storage — blob compression (not built)

Compress the content-addressed blob store with **zstd**, per blob. Decided by
measurement (`scripts/estimate_zstd_savings.py`) over a real 2200-blob email
store (262 MiB): zstd -19 shrinks it **~47% on disk (~126 MiB saved)**.

- **Per-blob zstd, no packing, no dictionary.** Block-rounding waste was only
  **1.8%** of disk (blobs average ~122 KiB — the store is byte-heavy, not
  file-count-heavy), so packing many small files into containers would recover
  <2% for a lot of complexity (a blob index as a second source of truth, a repack
  tool, GC interaction). The 26% of blobs that fit in one block are a rounding
  error (~2 MiB total). A trained dictionary only helps small inputs, which are
  negligible here — skip it.
- **Identity unchanged.** Hash stays over the **uncompressed** content, so dedup
  and immutability are untouched; compression is purely a storage detail.
- **Format:** a 1-byte marker + payload (`0x00` raw, `0x01` zstd frame). `put()`
  writes `min(raw, zstd)` with the matching marker — attachments (jpeg/pdf/zip in
  base64) barely compress and must never be force-grown; the 47% average blends
  very-compressible text with near-incompressible attachments. `get()` reads the
  marker and decompresses as needed.
- **Migration:** existing blobs are unmarked raw bytes. Add a `maintenance
  recompress` pass that rewrites every blob into the marked format under the write
  lock (fail-don't-corrupt: temp + atomic rename per blob). Prefer an explicit
  migration over sniffing zstd magic bytes.
- **Level:** -19 is slow to compress but the store is write-once and zstd
  decompresses fast regardless; drop toward ~15 only if extraction throughput
  suffers (negligible ratio loss).
- **Dependency:** add `zstandard` to `pyproject`.

*Note on adaptive/online dictionaries:* zstd adapts **within a single frame**
(later bytes reuse earlier ones), but there is **no cross-blob online learning** —
each blob is an independent frame with no memory of the others. Dictionaries are
**batch-trained** offline over a sample; you can periodically **re-train** as the
corpus grows, but that is a full batch step, not incremental, and every blob must
then record which dictionary id it was compressed with so it can be decompressed.
That bookkeeping only pays off for the small-file/packing case, which the
measurement rules out here.

---

## Open risks

- **Vector-store concurrency [open].** The embedded-Qdrant policy (embed takes an
  exclusive lock for its whole run; chat releases on idle) leaves chat retrieval
  unavailable during embedding and needs retry/backoff plus an idle-timer
  lifecycle. Residual risks: lock races, a stale lock if `embed` crashes mid-run,
  and up-to-1s chat stalls. Fallback if this bites: `sqlite-vec` (real WAL
  reader/writer concurrency, no custom locking). Full rationale in
  [`doc/design-decisions.md`](doc/design-decisions.md) (item 1).

---

## Remaining build order

1. **Module 4 (chat)**: simple mode first, then complex mode; fuse multiple
   embedding indices when present.
2. **Additional extraction plugins** (see *More extractors* above), now that email
   has proven the design.
3. **Maintenance**: GC; blob zstd compression + `recompress` migration.
4. Iterate: more filters, context-aware translation, more source types.
