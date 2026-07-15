"""Tests for the content-addressed blob store."""

from __future__ import annotations

from rag.blobs import BlobStore


def test_put_get_roundtrip(blob_store: BlobStore) -> None:
    """Stored bytes are retrievable by their returned hash."""
    digest = blob_store.put(b"hello world")
    assert blob_store.get(digest) == b"hello world"


def test_put_is_content_addressed_and_dedup(blob_store: BlobStore) -> None:
    """Identical content yields the same hash and stores once."""
    d1 = blob_store.put(b"same")
    d2 = blob_store.put(b"same")
    assert d1 == d2
    assert blob_store.exists(d1)


def test_hash_matches_sha256(blob_store: BlobStore) -> None:
    """The digest is the SHA-256 hex of the content."""
    import hashlib

    data = b"abc"
    assert blob_store.put(data) == hashlib.sha256(data).hexdigest()


def test_delete(blob_store: BlobStore) -> None:
    """Delete removes the blob and reports whether it existed."""
    digest = blob_store.put(b"bye")
    assert blob_store.delete(digest) is True
    assert not blob_store.exists(digest)
    # Deleting again is a no-op returning False.
    assert blob_store.delete(digest) is False
