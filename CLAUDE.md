# CLAUDE.md

Guidance for working in this repository.

## What this is

A local, single-user **RAG pipeline** over personal data (email, notes, …). The
full design is in [`PLAN.md`](PLAN.md) — read it before changing architecture.

Pipeline stages: **extract → translate → embed → chat**. **Only extraction
(module 1) is implemented so far.** The previous prototype lives untouched under
[`old/`](old/) for reference; do not build on it — this is a from-scratch rewrite.

## Architecture (implemented parts)

- `rag/config.py` — central YAML config; each plugin gets its own subtree.
- `rag/secrets.py` — secret resolution: env → keyring → file → prompt. Never in YAML.
- `rag/timestamps.py` — shared timestamp parsing (`1d ago`, ISO 8601, `today`, …).
- `rag/blobs.py` — content-addressed blob store on the filesystem (SHA-256, sharded, immutable).
- `rag/db.py` — SQLite (WAL) connection + schema + forward-only migrations (via `PRAGMA user_version`).
- `rag/extract/` — module 1:
  - `base.py` — the `Extractor` plugin base class and `Store` API.
  - `store.py` — `SqliteStore`: version handling + change detection.
  - `discovery.py` — finds plugins; **`source_type` == plugin subdirectory name** (enforced).
  - `runner.py` — wires config/secrets/db/blobs/state-dir and invokes a plugin.
  - `plugins/email/` — the only source plugin for now (IMAP). Working email-only
    until the system is proven; other sources come later.
- `rag/cli.py` — `rag` entry point (`rag extract <source_type> [key=value …]`).

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
