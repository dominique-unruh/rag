# Module 3 — Embedding / RAG Index

Chunks the parts of each active translation set and embeds them into one or more
vector indices — **one index per embedding model**. The system supports several
embedding models side by side (e.g. a strong dense model plus a sparse/lexical one
for hybrid retrieval, or an old and a new dense model during a migration).

## Multiple models, one index each

- A **model registry** (in config) lists the enabled embedding models. Each entry
  has: a stable **model id** (treated like a version — changing the model or its
  parameters means a *new* id), a **type** (`dense` / `sparse`), a **backend**, the
  index/collection name, and for dense models the dimension + metric + any query
  instruction template.
- The **backend** selects how the model is reached:
  - `openai` — any OpenAI-compatible embeddings endpoint (`api_base` + optional api
    key from secrets); covers hosted providers and local servers including Ollama's
    `/v1`, LM Studio, vLLM, llama.cpp.
  - `ollama` — a native Ollama server (its own embeddings API; can also pull /
    manage models). Configured by its `host` (e.g. `http://localhost:11434`) and
    Ollama model name.
  - `bm25` — a local sparse backend: term-frequency vectors computed locally, no
    external service; Qdrant applies IDF at query time.
  Backends are per model, so e.g. a native-Ollama dense model and a local sparse
  model can coexist in the registry.
- Each model owns its **own index** (its own Qdrant collection, or a sparse index).
  Different models never share a space — dims/metrics may differ freely. They are
  combined only at query time (fusion — see the chat plan), never compared directly.
- Adding a model later (e.g. bolting sparse onto a dense-only system) is just a new
  registry entry plus its own embed passes. **No re-embedding of existing models, no
  schema change, no migration.**

## Per-(part, model) bookkeeping

The `embeddings` table (see [storage-model.md](storage-model.md)) is the **join**
between parts and models: one row per `(part, model)` actually embedded, recording
which translation `set_id` / `source_version` was embedded for that model and when.
A part may therefore have several embedding rows — one per model. "Not yet embedded
by model X" is a normal state (like a zero-active translation), never an error.

## Embedding a model (a "pass")

- `embed` runs **per model** (`--model <id>`, optionally filtered like translate). A
  pass embeds the parts that are **missing or stale for that model** and leaves other
  models untouched. Models converge independently; the corpus is fully covered once
  every model's passes are done. Verbose runs print `[n/total]` progress (the total
  is a cheap SQL precount of the documents the pass will process).
- Within a pass: all parts of each document's **active** set are chunked
  (configurable size / overlap) and embedded; each point's payload carries
  `source_type`, `doc_id`, `set_id`, `part_id` (and the model id) for filtering and
  cleanup.
- **Staleness** is per model: when a document's active set changes, the
  `(part, model)` rows whose embedded `set_id` differs from the active one are
  re-embedded on that model's next pass — the document's old points in *that model's*
  index are deleted and the new set's parts re-embedded, so parts never mix across
  sets.
- A dense model records its dimension; the tooling refuses to mix vectors of
  different dims/models within one index.
- A per-model **fingerprint** guards against incompatible config changes: if a
  model id's incompatibility-defining config changes, the pass refuses rather than
  polluting its index with mismatched vectors.

## Cross-index cleanup

Deleting a document (or replacing its active set) must remove/replace its points in
**every** model's index, and the corresponding `(part, model)` rows. The DB rows
cascade via foreign keys; the vector-store point removal loops over all model
indices. Retrieval never returns a chunk for a deleted document from any index.
