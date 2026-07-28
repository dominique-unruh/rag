"""Tests for the ``maintenance stats`` tool (DB counts + per-plugin state stats)."""

from __future__ import annotations

import json
from pathlib import Path

from rag.blobs import BlobStore
from rag.config import Config
from rag.db import Database
from rag.extract.store import SqliteStore
from rag.maintenance.stats import gather_stats
from rag.translate.store import TranslationStore

# A minimal embedding model so per-model stats have something to report. Models
# live in the top-level `models:` section; `embedding.models` lists the ids used.
_MODELS = {
    "test": {"type": "dense", "backend": "openai", "api_base": "http://x", "model": "m", "dim": 4},
}


def _config(storage: Path) -> Config:
    """A Config rooted at ``storage`` with the test embedding model."""
    return Config(
        {"storage": {"dir": str(storage)}, "models": _MODELS, "embedding": {"models": ["test"]}},
        storage / "config.yaml",
    )


def test_stats_empty_no_database(tmp_path: Path) -> None:
    """With no database yet, stats report zeros rather than erroring."""
    stats = gather_stats(_config(tmp_path / "s"))
    assert stats["documents"] == {"total": 0, "by_source": {}}
    assert stats["translations"]["sets_total"] == 0
    assert stats["embeddings"]["models"] == {}
    assert stats["plugins"] == {}


def test_stats_counts_documents_translations(tmp_path: Path) -> None:
    """Documents, translation sets/parts and the missing/stale figures are counted."""
    storage = tmp_path / "s"
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    store = SqliteStore(db.conn, blobs, "email")
    # Two documents; only one gets an active translation set (two parts).
    store.upsert("a", b"raw-a", {"mime-type": "message/rfc822", "title": "A"})
    store.upsert("b", b"raw-b", {"mime-type": "message/rfc822", "title": "B"})
    TranslationStore(db.conn, blobs).write_set(
        source_type="email", doc_id="a", source_version=1,
        translator_name="dispatch", git_rev="r",
        parts=[(["body"], "hello", {}), (["sig"], "bye", {})],
    )
    db.close()

    stats = gather_stats(_config(storage))
    assert stats["documents"] == {"total": 2, "by_source": {"email": 2}}
    tr = stats["translations"]
    assert tr["sets_total"] == 1 and tr["sets_active"] == 1
    assert tr["parts_total"] == 2 and tr["parts_active"] == 2
    # Doc "b" has no active translation; nothing is stale yet.
    assert tr["untranslated_documents"] == 1
    assert tr["stale_active_sets"] == 0


def test_stats_stale_after_version_bump(tmp_path: Path) -> None:
    """Re-extracting a document bumps its version, making its active set stale."""
    storage = tmp_path / "s"
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    store = SqliteStore(db.conn, blobs, "email")
    store.upsert("a", b"raw-a", {"mime-type": "message/rfc822", "title": "A"})
    TranslationStore(db.conn, blobs).write_set(
        source_type="email", doc_id="a", source_version=1,
        translator_name="dispatch", git_rev="r", parts=[(["body"], "hi", {})],
    )
    # Content change bumps the document to version 2; the active set stays at v1.
    store.upsert("a", b"raw-a-v2", {"mime-type": "message/rfc822", "title": "A2"})
    db.close()

    tr = gather_stats(_config(storage))["translations"]
    assert tr["stale_active_sets"] == 1


def test_stats_embeddings_missing_and_orphan(tmp_path: Path) -> None:
    """Per-model stats report embedded/missing active parts and flag orphan models."""
    storage = tmp_path / "s"
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    store = SqliteStore(db.conn, blobs, "email")
    store.upsert("a", b"raw", {"mime-type": "message/rfc822", "title": "A"})
    ts = TranslationStore(db.conn, blobs)
    ts.write_set(
        source_type="email", doc_id="a", source_version=1,
        translator_name="dispatch", git_rev="r",
        parts=[(["body"], "hello", {}), (["sig"], "bye", {})],
    )
    # Look up the part ids to record one embedding for the configured model and
    # one for an orphan model (present in DB, absent from config).
    parts = db.conn.execute(
        "SELECT id, set_id FROM translation_parts ORDER BY id"
    ).fetchall()
    p0, set_id = parts[0]["id"], parts[0]["set_id"]
    db.conn.execute(
        "INSERT INTO embeddings(source_type, doc_id, set_id, part_id, model, embedded_at)"
        " VALUES('email', 'a', ?, ?, 'test', 't')", (set_id, p0),
    )
    db.conn.execute(
        "INSERT INTO embeddings(source_type, doc_id, set_id, part_id, model, embedded_at)"
        " VALUES('email', 'a', ?, ?, 'gone', 't')", (set_id, p0),
    )
    db.conn.commit()
    db.close()

    models = gather_stats(_config(storage))["embeddings"]["models"]
    # Configured model: 1 of 2 active parts embedded -> 1 missing.
    assert models["test"]["configured"] is True
    assert models["test"]["embedded_active_parts"] == 1
    assert models["test"]["missing_active_parts"] == 1
    assert models["test"]["rows_total"] == 1
    # Orphan model still shows up, flagged as not configured.
    assert models["gone"]["configured"] is False
    assert models["gone"]["rows_total"] == 1


def test_stats_plugin_statistics_from_state_dir(tmp_path: Path) -> None:
    """The email plugin's statistics() is read from its state dir (uid map)."""
    storage = tmp_path / "s"
    # Minimal joint uid map in the email plugin's state dir.
    state_dir = storage / "plugin_state" / "email"
    state_dir.mkdir(parents=True)
    (state_dir / "uidmap.json").write_text(json.dumps({
        "accounts": {
            "acc": {"folders": {"INBOX": {"uidvalidity": 1, "uids": {"1": "mid-1", "2": "mid-2"}}}}
        }
    }))
    stats = gather_stats(_config(storage))
    email_stats = stats["plugins"]["email"]
    assert email_stats["tracked_uids_total"] == 2
    assert email_stats["distinct_message_ids"] == 2
    assert email_stats["accounts"]["acc"] == {"folders": 1, "tracked_uids": 2}


def test_stats_storage_use(tmp_path: Path) -> None:
    """Storage stats report DB (with WAL sidecars), vector store and plugin dirs."""
    storage = tmp_path / "s"
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    SqliteStore(db.conn, blobs, "email").upsert(
        "a", b"raw", {"mime-type": "message/rfc822", "title": "A"}
    )
    db.close()
    # A file in the email plugin's state dir gives it a non-zero footprint.
    state_dir = storage / "plugin_state" / "email"
    state_dir.mkdir(parents=True)
    (state_dir / "uidmap.json").write_text('{"accounts": {}}')

    st = gather_stats(_config(storage))["storage"]
    # The DB file exists and was counted (WAL sidecars included when present).
    assert st["database_bytes"] > 0
    # The upserted document's raw blob was written to the blob store.
    assert st["blobs_bytes"] > 0
    # No vector store was created in this test.
    assert st["vector_store_bytes"] == 0
    # The email plugin dir has our uidmap.json in it.
    assert st["plugins"]["email"] > 0


def test_stats_source_filter(tmp_path: Path) -> None:
    """A --source filter restricts DB counts and plugin stats to one source_type."""
    storage = tmp_path / "s"
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    SqliteStore(db.conn, blobs, "email").upsert(
        "a", b"raw", {"mime-type": "message/rfc822", "title": "A"}
    )
    db.close()
    # Filtering to a different source yields zero documents.
    stats = gather_stats(_config(storage), source_type="notes")
    assert stats["documents"]["total"] == 0
    assert stats["plugins"] == {}
    # Filtering to the real source yields the one document.
    stats = gather_stats(_config(storage), source_type="email")
    assert stats["documents"]["total"] == 1
