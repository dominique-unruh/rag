"""Tests for the db-browser maintenance tool helpers."""

from __future__ import annotations

import time
from pathlib import Path

from rag.db import Database
from rag.maintenance.dbbrowser import IdleDatabase, _download_name


def test_idle_database_releases_and_reopens(tmp_path: Path) -> None:
    """The connection is closed after the idle window and reopened on next use."""
    # Create an empty, migrated database file to connect to.
    Database.open(tmp_path / "m.db").close()
    idb = IdleDatabase(tmp_path / "m.db", idle_seconds=0.2)
    try:
        # A query opens the connection.
        assert idb.query("SELECT 1")[0][0] == 1
        assert idb._conn is not None
        # After more than the idle window (plus a poll tick) it is released.
        time.sleep(0.6)
        assert idb._conn is None
        # A further query transparently reopens it.
        assert idb.query("SELECT 2")[0][0] == 2
        assert idb._conn is not None
    finally:
        idb.close()


def test_download_name_uses_mime_extension() -> None:
    """Download filenames are sanitised and get a mime-derived extension."""
    # doc_id with @ and dots is sanitised; message/rfc822 -> .eml.
    assert _download_name("abc@example.org", "message/rfc822") == "abc_example.org.eml"
    # Markdown mapping.
    assert _download_name("note", "text/markdown") == "note.md"
    # Unknown/absent mime -> no extension added.
    assert _download_name("blob", None) == "blob"
