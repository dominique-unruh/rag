"""Shared pytest fixtures for the extraction test suite."""

from __future__ import annotations

from pathlib import Path
from typing import Iterator

import pytest

from rag.blobs import BlobStore
from rag.db import Database
from rag.extract.store import SqliteStore


@pytest.fixture
def blob_store(tmp_path: Path) -> BlobStore:
    """A blob store rooted in a temporary directory."""
    return BlobStore(tmp_path / "blobs")


@pytest.fixture
def database(tmp_path: Path) -> Iterator[Database]:
    """An open, migrated database on a temporary file, closed on teardown."""
    db = Database.open(tmp_path / "metadata.db")
    try:
        yield db
    finally:
        db.close()


@pytest.fixture
def store(database: Database, blob_store: BlobStore) -> SqliteStore:
    """A ``SqliteStore`` scoped to an ``email`` source type for tests."""
    return SqliteStore(database.conn, blob_store, "email")
