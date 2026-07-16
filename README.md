# rag

A local, single-user retrieval-augmented-generation pipeline over personal data
(email, notes, …). See [`PLAN.md`](PLAN.md) for the full design.

The pipeline has four stages: **extract** → **translate** → **embed** → **chat**.
Only **extraction** (module 1) is implemented so far.

## Install

```sh
pip install -e .[test]
```

## Configure

```sh
cp config.yaml.example config.yaml
# edit config.yaml; secrets go in the OS keyring (or a file), never the YAML
```

## Extract

Only the **email** (IMAP) source is wired up for now.

```sh
rag extract email since='1w ago' count=100 folder=INBOX
```

Extraction writes document metadata to a SQLite database and raw source bytes to
a content-addressed blob store, both under the configured `storage.dir`.

## Translate

Convert extracted documents to Markdown (module 2). Storing to the DB requires a
clean git tree; `--inspect` / `--dry-run` do not.

```sh
rag translate --inspect                 # preview, no DB write
rag translate --limit 100               # translate up to 100 documents
rag translate --translator text --force # force with a specific translator
```

## Browse the database

```sh
rag maintenance db-browser        # opens a local web UI in your browser
```

## Test

```sh
pytest
```
