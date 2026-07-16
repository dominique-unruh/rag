"""Tests for the SQLite-backed extraction store: change detection, transient
metadata, consistency checks and cascading deletion."""

from __future__ import annotations

import pytest

from rag.db import Database
from rag.errors import DocumentExistsError, DocumentMissingError
from rag.extract.store import SqliteStore
from rag.extract.base import UpsertStatus


def test_create_then_unchanged(store: SqliteStore) -> None:
    """First upsert creates v1; an identical upsert reports unchanged."""
    r1 = store.upsert("a", b"body", {"title": "A"})
    assert (r1.status, r1.version) == (UpsertStatus.CREATED, 1)
    r2 = store.upsert("a", b"body", {"title": "A"})
    assert (r2.status, r2.version) == (UpsertStatus.UNCHANGED, 1)


def test_stable_metadata_change_bumps_version(store: SqliteStore) -> None:
    """Changing stable metadata bumps the version."""
    store.upsert("a", b"body", {"title": "A"})
    r = store.upsert("a", b"body", {"title": "B"})
    assert (r.status, r.version) == (UpsertStatus.UPDATED, 2)


def test_content_change_bumps_version(store: SqliteStore) -> None:
    """Changing the content bumps the version."""
    store.upsert("a", b"body", {"title": "A"})
    r = store.upsert("a", b"body2", {"title": "A"})
    assert (r.status, r.version) == (UpsertStatus.UPDATED, 2)


def test_transient_change_does_not_bump_version(store: SqliteStore) -> None:
    """A transient-only change updates state but leaves the version alone."""
    store.upsert("a", b"body", {"title": "A"}, transient={"folder": "INBOX", "read": False})
    r = store.upsert("a", b"body", {"title": "A"}, transient={"folder": "Archive", "read": True})
    # Reported UNCHANGED (no reprocessing) and version stays at 1 ...
    assert (r.status, r.version) == (UpsertStatus.UNCHANGED, 1)
    # ... but the transient metadata was actually updated.
    assert store.get_transient("a") == {"folder": "Archive", "read": True}


def test_metadata_key_order_is_not_a_change(store: SqliteStore) -> None:
    """Reordering metadata keys does not count as a change."""
    store.upsert("a", b"body", {"title": "A", "date": "2025"})
    r = store.upsert("a", b"body", {"date": "2025", "title": "A"})
    assert r.status == UpsertStatus.UNCHANGED


def test_exist_ok_false_raises_on_existing(store: SqliteStore) -> None:
    """``exist_ok=False`` rejects an already-present document."""
    store.upsert("a", b"body", {"title": "A"})
    with pytest.raises(DocumentExistsError):
        store.upsert("a", b"body", {"title": "A"}, exist_ok=False)


def test_missing_ok_false_raises_on_absent(store: SqliteStore) -> None:
    """``missing_ok=False`` rejects a not-yet-existing document."""
    with pytest.raises(DocumentMissingError):
        store.upsert("a", b"body", {"title": "A"}, missing_ok=False)


def test_get_metadata_and_list_ids(store: SqliteStore) -> None:
    """Stored stable metadata and id listing round-trip correctly."""
    store.upsert("a", b"body", {"title": "A"})
    store.upsert("b", b"body2", {"title": "B"})
    assert store.get_metadata("a") == {"title": "A"}
    assert set(store.list_ids()) == {"a", "b"}
    assert store.get_metadata("missing") is None


def test_list_ids_metadata_filters(store: SqliteStore) -> None:
    """list_ids filters by stable/transient metadata (equality and membership)."""
    store.upsert("a", b"x", {"subject": "A"}, transient={"account": "main", "folder": "Spam"})
    store.upsert("b", b"y", {"subject": "B"}, transient={"account": "main", "folder": "INBOX"})
    store.upsert("c", b"z", {"subject": "C"}, transient={"account": "work", "folder": "INBOX"})
    # Scalar equality on transient.
    assert list(store.list_ids(transient={"folder": "Spam"})) == ["a"]
    # Membership (IN) + AND across keys.
    assert set(store.list_ids(transient={"account": "main", "folder": ["Spam", "INBOX"]})) == {"a", "b"}
    # Stable-metadata filter.
    assert list(store.list_ids(stable={"subject": "C"})) == ["c"]
    # No filter -> all.
    assert set(store.list_ids()) == {"a", "b", "c"}
    # Empty membership set matches nothing.
    assert list(store.list_ids(transient={"folder": []})) == []


def test_delete(store: SqliteStore) -> None:
    """Delete removes the document and reports existence."""
    store.upsert("a", b"body", {"title": "A"})
    assert store.delete("a") is True
    assert not store.exists("a")
    # Deleting an absent document is a no-op by default ...
    assert store.delete("a") is False
    # ... but raises when the caller asserts it should exist.
    with pytest.raises(DocumentMissingError):
        store.delete("a", missing_ok=False)


def test_isolation_between_source_types(database: Database, blob_store) -> None:
    """Stores of different source types do not see each other's documents."""
    notes = SqliteStore(database.conn, blob_store, "notes")
    email = SqliteStore(database.conn, blob_store, "email")
    notes.upsert("x", b"n", {"title": "N"})
    # Same doc_id under a different source type is a distinct document.
    assert email.exists("x") is False
    email.upsert("x", b"e", {"title": "E"})
    assert notes.get_metadata("x") == {"title": "N"}
    assert email.get_metadata("x") == {"title": "E"}


def test_delete_cascades_to_translations_and_embeddings(
    store: SqliteStore, database: Database
) -> None:
    """Deleting a document cascades to translation sets, parts and embeddings."""
    store.upsert("a", b"body", {"title": "A"})
    conn = database.conn
    # Manually insert a downstream translation set + part + embedding row.
    conn.execute(
        "INSERT INTO translation_sets (id, source_type, doc_id, source_version, "
        "metadata, translated_at, git_rev, is_active) "
        "VALUES (1, 'email', 'a', 1, jsonb('{}'), 't', 'rev', 1)"
    )
    conn.execute(
        "INSERT INTO translation_parts (id, set_id, sub_doc_id, blob_hash, metadata) "
        "VALUES (1, 1, jsonb('[\"body\"]'), 'h', jsonb('{}'))"
    )
    conn.execute(
        "INSERT INTO embeddings (source_type, doc_id, set_id, part_id, model, embedded_at) "
        "VALUES ('email', 'a', 1, 1, 'm', 't')"
    )
    conn.commit()

    store.delete("a")

    # Every dependent row is gone via ON DELETE CASCADE.
    assert conn.execute("SELECT COUNT(*) FROM translation_sets").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM translation_parts").fetchone()[0] == 0
    assert conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0] == 0
