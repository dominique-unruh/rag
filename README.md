# ragproto — local RAG over email + notes

A small, local Retrieval-Augmented-Generation prototype over personal data — IMAP
email and Markdown notes — wired entirely from **stock LlamaIndex components**. No
custom retrievers, extractors, or index classes: everything is configuration and
glue. Vectors live in Qdrant, a property graph is built alongside, and you chat
over both from the CLI.

## How it works

Three independent, idempotent pipelines around one canonical store:

```
 sources          1. extract          2. embed
 IMAP  ──┐                          ┌────────────► Qdrant (vectors)
 notes ──┴──►  Document Store  ─────┤
              (SQLite documents.db) └────────────► Kùzu graph (graph.kuzu)
                                       3. graph
                        │
                        └── chat (condense_plus_context) + viewer read Qdrant + Kùzu
```

- **Document Store is the source of truth.** `extract` scans sources and writes
  documents; `embed` and `graph` are pure functions of the Document Store — each
  diffs its own ledger against it (by content hash) to add / update / **delete**.
- **Deletion works.** Remove a note or let mail age past the cutoff → `extract`
  drops it from the Document Store, and `embed`/`graph` remove its vectors/graph
  data on their next run.
- **Incremental & crash-safe.** Every pipeline detects only new/changed/deleted
  work and commits per document.
- **Stable IDs.** `doc_id = "{extractor}:{local_id}"` — notes: file path;
  email: `Message-ID` (uid fallback).
- **Email cutoff.** Server-side IMAP `SINCE`; old mail never downloaded.
- **Any OpenAI-compatible endpoint** for chat + embeddings.

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
make sync      # run all three pipelines: extract -> embed -> graph
make chat      # interactive Q&A REPL
make view      # browse the stores in a local web app
```

Or a single pipeline at a time (each independently re-runnable):

```sh
.venv/bin/python -m rag.extract [--only email|notes]   # sources -> Document Store
.venv/bin/python -m rag.embed   [--delay N]            # Document Store -> Qdrant
.venv/bin/python -m rag.graph   [--delay N]            # Document Store -> Kùzu graph
.venv/bin/python -m rag.chat    [--trace]
.venv/bin/python -m rag.viewer  [--reload]
```

In the chat REPL: ask questions; `/reset` clears history; `/exit` (or Ctrl-D) quits.
`--trace` prints every store round-trip (vector query → Qdrant, synonym query →
graph) with the query and returned nodes — not the LLM's own thinking.

## Inspecting the stores — the viewer

```sh
make view      # http://127.0.0.1:5000  (auto-reloads on source edits)
```

A small Flask app that browses **one item at a time** — a document, entity,
relation, or chunk — with all properties in a table, references rendered as
clickable links, in/out relations, a back button, and a history panel. A search
box takes a text query (ranked graph matches) or a pasted id (jumps to the item),
with an optional **semantic** checkbox for a Qdrant vector search. Notes link out
to Obsidian (`obsidian://`), emails to `mid:`. Embedded Qdrant/Kùzu have no
dashboards of their own; this viewer is how you see their contents.

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
| `storage` | `dir` | `documents.db` + `qdrant/` + `graph.kuzu/` |
| `chunking` | `chunk_size`, `chunk_overlap` | |

## Development

```sh
make test      # run the unit suite (no network; mocks + temp dirs)
make clean     # remove caches; make clean-storage to drop the index
```

The suite covers config/secret resolution (files, keyring, prompt), the Document
Store (dedup, reconcile-deletion, ledgers, change planning), extraction,
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
