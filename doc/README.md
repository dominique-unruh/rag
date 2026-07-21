# Architecture documentation

Reference docs for the **implemented** parts of the RAG pipeline — a local,
single-user system that ingests personal data (email, notes, …), turns it into
clean Markdown, embeds it into a vector index, and chats over it. Work still to be
built (agentic chat mode, GC maintenance, further source plugins) lives in
[`../PLAN.md`](../PLAN.md).

Pipeline stages: **extract → translate → embed → chat**. Extraction, translation
and embedding are implemented, as is chat **simple mode**; agentic chat mode is
not yet built.

## Index

- [storage-model.md](storage-model.md) — blob store, SQLite schema, migrations.
- [extraction.md](extraction.md) — module 1: the `Extractor`/`Store` API, change
  detection, cascading deletion, and the email (IMAP) plugin.
- [translation.md](translation.md) — module 2: the `Translator` API, the
  pipeline's bookkeeping, and the built-in translators.
- [embedding.md](embedding.md) — module 3: the multi-model index, per-(part,
  model) bookkeeping, embed passes, cross-index cleanup.
- [chat.md](chat.md) — module 4 (simple mode): retrieval + RRF fusion, chunk-text
  reconstruction, the chat LLM backends, and the REPL.
- [configuration.md](configuration.md) — the central YAML config, secret
  resolution (keyring / OAuth2 tokens), and shared timestamp parsing.
- [operations.md](operations.md) — the CLI tools, concurrency model, and backups.
- [metadata.md](metadata.md) — the metadata keys each extractor writes.
- [design-decisions.md](design-decisions.md) — resolved design decisions and
  their rationale.

## Goals & principles

- **Modular pipeline**: four independent stages (Extract → Translate → Embed →
  Chat), each runnable on its own from the command line.
- **Plugin-based ingestion**: new data sources are added by dropping in a plugin,
  with no changes to the core.
- **Provenance & reversibility**: keep history of translations, record which
  program version / LLM / prompt produced each artifact, allow reverting.
- **Incremental & resumable**: re-run stages over subsets; only reprocess what
  changed.
- **Backup-friendly storage**: large binary content lives on the filesystem as
  immutable, content-addressed blobs (good for incremental backups); the database
  stays small and holds only metadata + references.
- **Concurrency where it matters**: chat can run while the pipeline runs (with one
  documented exception around the vector store — see [operations.md](operations.md)).

**Language / conventions**: Python. Full type annotations on all functions and
module-level globals. Docstrings on private/helper functions as well as public
ones.

## Technology choices

| Concern            | Choice                          | Rationale |
|--------------------|---------------------------------|-----------|
| Metadata DB        | **SQLite** in **WAL** mode      | Single file, no server, backup-friendly. WAL gives many concurrent readers + one writer across processes. |
| Blob storage       | **Filesystem, content-addressed by hash** | Dedup, immutable, incremental-backup-friendly, lock-free reads. |
| Vector store       | **Qdrant (embedded, local path)** | Good metadata filtering. Single-process lock — see the concurrency policy in [operations.md](operations.md). |
| LLM / embeddings   | **OpenAI-compatible endpoint** *or* **native Ollama** | OpenAI-compatible covers hosted + local servers (LM Studio / vLLM / llama.cpp, and Ollama's `/v1`); native Ollama is also supported directly. Selected per model. |
| Config             | **Central YAML** file           | Everything except secrets. |
| Secrets            | **Desktop keyring** (+ file fallback)   | No secrets in the YAML or repo. |

SQLite WAL settings: `PRAGMA journal_mode=WAL;` once at DB creation, plus a
per-connection `PRAGMA busy_timeout=<ms>;` so a blocked writer waits instead of
erroring with `SQLITE_BUSY`.

## Repository layout

```
rag/
  config.py          # YAML load → typed config
  secrets.py         # keyring + file/prompt secret resolution
  db.py              # SQLite schema, connection, WAL, migrations
  blobs.py           # content-addressed FS blob store
  gitinfo.py         # git revision / clean-tree checks
  timestamps.py      # shared timestamp parsing ("1d ago", ISO 8601, …)
  cli.py             # top-level subcommand dispatch
  extract/
    base.py          # Extractor ABC + Store API (enforces source isolation)
    store.py         # SqliteStore: version handling + change detection
    discovery.py     # plugin discovery (source_type == subdir name)
    runner.py        # wires config/secrets/db/blobs/state-dir; invokes a plugin
    plugins/
      email/         # the only source plugin for now (imap.py, oauth2.py)
  translate/
    base.py          # Document, Translation, Translator ABC, make_translation
    pipeline.py      # selection, doc_id enforcement, set/part DB bookkeeping
    store.py         # TranslationStore
    discovery.py     # loads a translator by module name
    markdown.py      # HTML → Markdown helper (markdownify)
    translators/     # one file per Translator; filename = --translator value
      dispatch.py    #   translator = Dispatch()  (default; routes by mime-type)
      email.py text.py pdf.py ignore.py
  embed/
    models.py backends.py sparse.py chunk.py vectorstore.py store.py pipeline.py
  chat/
    models.py        # chat LLM registry (ChatModelSpec, default selection)
    llm.py           # streaming chat backends (ollama / openai)
    retrieve.py      # multi-model search + RRF fusion + chunk-text reconstruction
    prompt.py        # simple-mode prompt construction (pure)
    terminal.py      # OSC 8 hyperlinks + citation parsing
    session.py       # readline REPL (retrieve-then-read loop)
  maintenance/       # db-browser (web UI), delete (cascading store API)
tests/
```
