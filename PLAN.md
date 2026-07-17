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
3. **Maintenance**: GC.
4. Iterate: more filters, context-aware translation, more source types.
