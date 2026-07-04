# ragproto — local RAG over email + notes

A small, local Retrieval-Augmented-Generation prototype over personal data — IMAP
email and Markdown notes — wired entirely from **stock LlamaIndex components**. No
custom retrievers, extractors, or index classes: everything is configuration and
glue. Vectors live in Qdrant, a property graph is built alongside, and you chat
over both from the CLI.

## How it works

```
IMAP (SINCE cutoff) ─┐                          ┌─► Qdrant (vectors)
                     ├─► dedup gate ─► insert ──┤
notes/*.md ──────────┘   (docstore)  (PropertyGraphIndex)
                                                └─► SimplePropertyGraphStore (triples)
                                                          │
                                              as_chat_engine(condense_plus_context)
                                                          │
                                                     CLI chat REPL
```

- **Incremental.** A docstore-backed `IngestionPipeline` (UPSERTS) compares content
  hashes and only lets new/changed documents through, so re-running is cheap and
  cron-safe. Stable IDs: absolute file path for notes, `Message-ID` for mail.
- **Email cutoff.** Only mail newer than the configured date/age is fetched
  (server-side IMAP `SINCE`), so old mail is never downloaded.
- **Any OpenAI-compatible endpoint** for chat + embeddings (Ollama, LM Studio,
  vLLM, llama.cpp, or a hosted gateway).

## Requirements

- Python ≥ 3.10
- [`uv`](https://docs.astral.sh/uv/) (used by the Makefile; `pip` works too)
- An OpenAI-compatible endpoint serving one chat model and one embedding model

## Setup

```sh
make install                       # create .venv and install deps
cp config.yaml.example config.yaml # then edit it
```

Edit `config.yaml`: set `models.api_base`, `chat_model`, `embed_model`, and
`embed_dim` (**must** equal the embedding model's output width — mismatch is
detected and rejected before anything is written to Qdrant). Point `notes.directory`
at your notes and/or fill in the `email` block.

### Secrets

There is **no** `secrets.yaml`. Credentials are resolved at load time:

- **API key** — in order: inline `models.api_key` → `models.api_key_file` → desktop
  keyring (service `ragproto`, user `api_key@<api_base>`) → interactive prompt (the
  entered key is cached in the keyring). Local servers: set `api_key: none`.
- **IMAP password** — in order: inline `email.password` → `email.password_file` →
  **desktop keyring** (service `ragproto`, user = the IMAP username) → **interactive
  prompt**. A prompted password is stored back into the keyring, so subsequent runs
  are non-interactive. In a non-interactive context (e.g. cron) with nothing found,
  ingestion fails fast with a clear message instead of hanging.

Keep secret files private: `chmod 600` them (the loader warns if they are
group/other-readable).

## Usage

```sh
make ingest    # incremental ingest — safe to re-run / cron
make chat      # interactive Q&A REPL
```

Or directly:

```sh
.venv/bin/python -m rag.ingest
.venv/bin/python -m rag.chat
```

In the chat REPL: ask questions; `/reset` clears history; `/exit` (or Ctrl-D) quits.

- `rag.chat --trace` — print every store round-trip (vector query → Qdrant,
  synonym query → graph) with the query and returned nodes, abbreviated. The LLM's
  own condensing/answering is not traced — only what hits the databases.
- `rag.ingest --verbose --delay=N` — per-item progress; pause `N`s between
  items to throttle the endpoint.

## Inspecting the stores

```sh
make inspect      # writes inspect.html — one self-contained report
```

Opens as a single page with the property graph (interactive), a sample of Qdrant
points (full text + payload, foldable), and the docstore dedup table. Embedded
Qdrant (`path:`) has no web dashboard; this report is how you see its contents.

## Configuration reference

| Section | Key | Notes |
|---|---|---|
| `models` | `api_base`, `chat_model`, `embed_model` | OpenAI-compatible endpoint + model names |
| | `embed_dim` | must match the embedding model's vector width |
| | `temperature` | low (≈0.1) keeps answers grounded |
| | `api_key` / `api_key_file` | see Secrets |
| `email` | `enabled`, `host`, `username`, `folder` | IMAP source |
| | `since_days` / `since_date` | server-side cutoff; `since_date` wins |
| | `password` / `password_file` | see Secrets |
| `notes` | `enabled`, `directory` | recursive `*.md` |
| `qdrant` | `path` **or** `url` (exactly one), `collection` | embedded dir or running server |
| `graph` | `llm_extraction` | one LLM call per chunk when on — the cost driver |
| | `max_triplets_per_chunk` | |
| `storage` | `dir` | `docstore.json` + `property_graph_store.json` |
| `chunking` | `chunk_size`, `chunk_overlap` | |

## Development

```sh
make test      # run the unit suite (no network; mocks + temp dirs)
make clean     # remove caches; make clean-storage to drop the index
```

The suite covers config/secret resolution (files, keyring, prompt), the dedup gate,
email ID assignment, store roundtrips, and the embedding-dimension guard.

## Notes & limits (prototype scope)

- Deletion is not handled: notes removed from disk or expired mail stay indexed, and
  a changed document leaves its old vectors behind (duplicates) since deletion is out
  of scope.
- Changing `embed_model`/`embed_dim` later means re-ingesting into a fresh
  `qdrant.collection`.
- Graph quality depends on the chat model's extraction ability; set
  `graph.llm_extraction: false` for cheap first runs (structural edges only). Vector
  retrieval works regardless.
- Embedded Qdrant (`path`) locks its directory — only one of ingest/chat at a time.
