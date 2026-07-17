"""Tests for the content-addressed blob store."""

from __future__ import annotations

import hashlib

import pytest

from rag.blobs import BlobCorruptionError, BlobStore


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
    """The digest is the SHA-256 hex of the content (of the UNCOMPRESSED bytes)."""
    data = b"abc"
    assert blob_store.put(data) == hashlib.sha256(data).hexdigest()


def test_compressible_stored_compressed(blob_store: BlobStore) -> None:
    """Repetitive content is stored zstd-compressed (marker 1) and shrinks on disk."""
    data = b"From: a@b\nSubject: hi\n" * 500
    digest = blob_store.put(data)
    on_disk = blob_store.path_for(digest).read_bytes()
    # Leading marker byte is the zstd codec, and the file is far smaller than raw.
    assert on_disk[0] == 1
    assert len(on_disk) < len(data)
    # Round-trips back to the exact original.
    assert blob_store.get(digest) == data


def test_incompressible_stored_raw(blob_store: BlobStore) -> None:
    """Content zstd cannot shrink is stored raw (marker 0), never grown."""
    # Random-ish bytes: zstd yields no gain, so raw must win.
    data = bytes((i * 37 + 11) % 256 for i in range(64))
    digest = blob_store.put(data)
    on_disk = blob_store.path_for(digest).read_bytes()
    assert on_disk[0] == 0
    # Raw payload is exactly the content plus the single marker byte.
    assert on_disk == b"\x00" + data
    assert blob_store.get(digest) == data


def test_empty_blob_roundtrip(blob_store: BlobStore) -> None:
    """Empty content stores and retrieves cleanly."""
    digest = blob_store.put(b"")
    assert blob_store.get(digest) == b""


def _write_legacy(blob_store: BlobStore, data: bytes) -> str:
    """Write ``data`` in the pre-marker format (raw bytes named by hash); return digest."""
    digest = BlobStore.hash_bytes(data)
    target = blob_store.path_for(digest)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(data)
    return digest


def test_migrate_rewrites_legacy_blobs(blob_store: BlobStore) -> None:
    """migrate() upgrades legacy raw blobs to the marker format, then no-ops."""
    compressible = b"From: a@b\nSubject: hi\n" * 500
    d_leg = _write_legacy(blob_store, compressible)
    # A blob already in the new format must be left untouched by migration.
    d_new = blob_store.put(b"already new-format")

    rewritten = blob_store.migrate()
    assert rewritten == 1  # only the legacy blob was converted
    # The legacy blob is now marker-prefixed and compressed, and reads back exactly.
    on_disk = blob_store.path_for(d_leg).read_bytes()
    assert on_disk[0] == 1
    assert blob_store.get(d_leg) == compressible
    assert blob_store.get(d_new) == b"already new-format"
    # Idempotent: a second run finds nothing to do.
    assert blob_store.migrate() == 0


def test_migrate_flags_corrupt_legacy(blob_store: BlobStore) -> None:
    """A legacy file whose bytes do not match its name aborts the migration."""
    target = blob_store.path_for("0" * 64)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_bytes(b"not the content this digest names")
    with pytest.raises(BlobCorruptionError):
        blob_store.migrate()


def test_get_raises_on_bad_zstd_frame(blob_store: BlobStore) -> None:
    """A zstd-marked payload that is not a valid frame is flagged corrupt."""
    digest = blob_store.put(b"From: a@b\nSubject: hi\n" * 500)  # stored zstd (marker 1)
    # Corrupt the frame while keeping the zstd marker.
    blob_store.path_for(digest).write_bytes(b"\x01" + b"not a valid zstd frame")
    with pytest.raises(BlobCorruptionError):
        blob_store.get(digest)


def test_delete(blob_store: BlobStore) -> None:
    """Delete removes the blob and reports whether it existed."""
    digest = blob_store.put(b"bye")
    assert blob_store.delete(digest) is True
    assert not blob_store.exists(digest)
    # Deleting again is a no-op returning False.
    assert blob_store.delete(digest) is False
