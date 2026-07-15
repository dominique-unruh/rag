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
# edit config.yaml; secrets go in the OS keyring / env, never the YAML
```

## Extract

Only the **email** (IMAP) source is wired up for now.

```sh
rag extract email since='1w ago' count=100 folder=INBOX
```

Extraction writes document metadata to a SQLite database and raw source bytes to
a content-addressed blob store, both under the configured `storage.dir`.

## Test

```sh
pytest
```
