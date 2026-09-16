# CLAUDE.md

Guidance for working in this repository.

## What this is

A local, single-user **RAG pipeline** over personal data (email, notes, …). The
architecture of the implemented parts is documented under [`doc/`](doc/) (index:
[`doc/README.md`](doc/README.md)); [`PLAN.md`](PLAN.md) tracks only the remaining
unbuilt work (chat, GC/stats, more plugins). Read both before changing architecture.

Pipeline stages: **extract → translate → embed → chat**. **Extraction (module 1),
translation (module 2) and embedding (module 3) are implemented; chat is not yet.**
The previous prototype lives untouched under [`old/`](old/) for reference; do not
build on it — this is a from-scratch rewrite.

## Architecture (implemented parts)

- `rag/config.py` — central YAML config; each plugin gets its own subtree.
- `rag/secrets.py` — secret resolution: keyring → file → prompt. Never in YAML.
- `rag/timestamps.py` — shared timestamp parsing (`1d ago`, ISO 8601, `today`, …).
- `rag/mime.py` — shared libmagic-based mime detection (`MimeDetector`); used by the documents extractor and the archive translator.
- `rag/blobs.py` — content-addressed blob store on the filesystem (SHA-256, sharded, immutable).
- `rag/db.py` — SQLite (WAL) connection + schema + forward-only migrations (via `PRAGMA user_version`).
- `rag/extract/` — module 1:
  - `base.py` — the `Extractor` plugin base class and `Store` API.
  - `store.py` — `SqliteStore`: version handling + change detection.
  - `discovery.py` — finds plugins; **`source_type` == plugin subdirectory name** (enforced).
  - `runner.py` — wires config/secrets/db/blobs/state-dir and invokes a plugin.
  - `plugins/email/` — IMAP source plugin (the first, most-exercised one).
  - `plugins/documents/` — filesystem source plugin: ingests files from configured
    on-disk folders (Obsidian vaults, scraper mirrors with `.metadata.json`
    sidecars). Mime detection via libmagic (`python-magic`).
- `rag/gitinfo.py` — git revision + clean-tree check (translation git gating).
- `rag/translate/` — module 2:
  - `base.py` — `Document`, `Translation`, the stateless `Translator` ABC, `make_translation`.
  - `pipeline.py` — selection, invariant enforcement (doc_id, unique sub_doc_id), git gating, DB bookkeeping.
  - `store.py` — `TranslationStore`: select docs needing translation, write sets/parts.
  - `discovery.py` — loads a translator by module name (`--translator`, default `dispatch`).
  - `markdown.py` — HTML→Markdown (markdownify).
  - `translators/` — one file per translator (`dispatch`, `email`, `text`, `pdf`, `docx`, `xlsx`, `pptx`, `archive`); each exposes `translator = <Class>()`. `pptx` renders a PowerPoint presentation (deck/template/slideshow) as one `## Slide N` section each (shape text, tables, speaker notes) via python-pptx. `archive` unpacks ZIP/TAR/gzip/bzip2/xz (stdlib) and 7z/RAR (py7zr/rarfile) containers in memory, emits a Markdown table-of-contents part, then re-dispatches each contained file (one bad member is skipped, not fatal).
- `rag/embed/` — module 3:
  - `models.py` — embedding model registry (`ModelSpec`; dense backends `openai`/`ollama`, sparse backend `bm25`).
  - `backends.py` — dense embed via OpenAI-compatible or native Ollama (stdlib HTTP); `sparse.py` — local BM25 term-frequency vectors.
  - `chunk.py` — character chunking; `vectorstore.py` — embedded Qdrant, one collection per model (dense `VectorParams`, sparse IDF).
  - `store.py` — `EmbedStore`: select active sets needing embed (per model), record `embeddings` rows.
  - `pipeline.py` — `run_embed`: chunk + embed + replace points + bookkeeping.
- `rag/maintenance/` — `db-browser` (Flask web UI), `delete` (cascading store API).
- `rag/cli.py` — `rag` entry point (`extract`, `translate`, `embed`, `maintenance` subcommands).

### Storage model (see PLAN §3)

- **Metadata → SQLite** under `storage/metadata.db`; **blobs → filesystem** under
  `storage/blobs/` (referenced by hash). Plugin state → `storage/plugin_state/<source_type>/`.
- `documents` has **two** metadata columns: `stable_metadata` (content-defining;
  a change bumps `version` and retriggers the pipeline) and `transient_metadata`
  (volatile state like IMAP folder / read flag; **never** bumps the version).
- Translations are modelled as `translation_sets` + `translation_parts` (one
  source can yield several parts, activated together). `documents` deletion
  cascades to translations and embeddings via `ON DELETE CASCADE`.

## Conventions (follow these)

- **Type annotations on everything** — every function signature and module-level global.
- **Docstrings on everything** — all functions (including `_private` helpers),
  fields/attributes, and module globals. Add **explanatory inline comments** in
  function bodies (what + why).
- **Plugin isolation**: a plugin only ever uses the `Store` handle it is given
  (scoped to its `source_type`); it never touches the DB directly.
- **Fail, don't corrupt**: prefer aborting over risking data corruption (GC,
  migrations, atomic writes).
- **Ask before scope expansion**: confirm before design changes, reverts, or work
  beyond the stated task.

## Commands

```sh
pip install -e .[test]          # install (dev)
pytest                          # run tests (venv: .venv/bin/python -m pytest)
rag extract email since='1w ago' count=100 folder=INBOX
python -m rag.extract email count=100   # equivalent module invocation
```

Tests use a temp SQLite DB + temp blob dir (see `tests/conftest.py`); the email
plugin's parsing helpers are unit-tested without a live IMAP server.
