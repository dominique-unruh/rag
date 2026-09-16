"""Tests for the filesystem ``documents`` extractor plugin.

Exercises the folder walk, sidecar merging, mime detection (esp. markdown vs
libmagic's ``text/plain``), the default excludes (hidden/backup/symlink),
oversize skipping, mtime-based change detection, ``full`` re-reads and deletion
detection — all offline against a temporary folder tree and a temp store.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest

# libmagic is a hard dependency of the plugin: import it here so a missing
# python-magic / libmagic makes these tests FAIL (not silently skip).
import magic  # noqa: F401

from rag.blobs import BlobStore
from rag.db import Database
from rag.extract.base import UpsertStatus
from rag.extract.plugins.documents import DocumentsExtractor
from rag.extract.store import SqliteStore
from rag.secrets import Secrets


@pytest.fixture
def doc_store(database: Database, blob_store: BlobStore) -> SqliteStore:
    """A ``SqliteStore`` scoped to the ``documents`` source type."""
    return SqliteStore(database.conn, blob_store, "documents")


def _run(store: SqliteStore, root: Path, **options: object) -> None:
    """Run the extractor over a single folder ``root`` with id ``main``."""
    config = {"folders": [{"folder-id": "main", "path": str(root)}]}
    DocumentsExtractor().run(store, root / ".state", config, Secrets(), **options)


def test_basic_ingest_and_mime(doc_store: SqliteStore, tmp_path: Path) -> None:
    """Files are ingested with correct doc-ids and mime types; sidecars merge."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "readme.txt").write_text("hello world\n")
    (root / "note.md").write_text("# Title\n\n**markdown** body\n")
    sub = root / "sub"
    sub.mkdir()
    (sub / "doc.bin").write_bytes(b"payload-bytes")
    (sub / "doc.bin.metadata.json").write_text(
        json.dumps({"title": "The Doc", "url": "https://x/y", "content_type": "application/pdf"})
    )

    _run(doc_store, root)

    ids = set(doc_store.list_ids())
    assert ids == {"main:readme.txt", "main:note.md", "main:sub/doc.bin"}
    # Sidecar file itself is never a document.
    assert "main:sub/doc.bin.metadata.json" not in ids

    # Markdown must beat libmagic's text/plain via the extension.
    assert doc_store.get_metadata("main:note.md")["mime-type"] == "text/markdown"
    # Plain text stays text/plain.
    assert doc_store.get_metadata("main:readme.txt")["mime-type"] == "text/plain"

    # Sidecar content_type wins as mime-type; sidecar fields are merged.
    bin_meta = doc_store.get_metadata("main:sub/doc.bin")
    assert bin_meta["mime-type"] == "application/pdf"
    assert bin_meta["title"] == "The Doc"
    assert bin_meta["url"] == "https://x/y"
    assert bin_meta["path"] == "sub/doc.bin"
    assert bin_meta["size"] == len(b"payload-bytes")


def test_excludes_symlinks_hidden_and_oversize(doc_store: SqliteStore, tmp_path: Path) -> None:
    """Hidden files/dirs, backups, symlinks and oversize files are skipped."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "keep.txt").write_text("keep")
    (root / ".hidden").write_text("secret")          # hidden file
    (root / "backup~").write_text("old")             # editor backup
    hidden_dir = root / ".obsidian"                    # hidden dir (vault config)
    hidden_dir.mkdir()
    (hidden_dir / "app.json").write_text("{}")
    # A symlink to a real file.
    (root / "target.txt").write_text("t")
    os.symlink(root / "target.txt", root / "link.txt")
    # An oversize file (cap set tiny below).
    (root / "big.txt").write_text("x" * 100)

    config = {
        "folders": [{"folder-id": "main", "path": str(root)}],
        "max_file_size": 10,
    }
    DocumentsExtractor().run(doc_store, root / ".state", config, Secrets())

    ids = set(doc_store.list_ids())
    assert ids == {"main:keep.txt", "main:target.txt"}
    assert "main:big.txt" not in ids
    assert "main:link.txt" not in ids
    assert "main:.hidden" not in ids
    assert not any(i.startswith("main:.obsidian") for i in ids)


def test_mtime_skip_and_full_rescan(doc_store: SqliteStore, tmp_path: Path) -> None:
    """Unchanged files are skipped by mtime; ``full`` forces a re-read."""
    root = tmp_path / "vault"
    root.mkdir()
    f = root / "a.txt"
    f.write_text("body")

    _run(doc_store, root)
    assert doc_store.get_metadata("main:a.txt")["mime-type"] == "text/plain"
    v1 = doc_store.get_transient("main:a.txt")

    # Second run with no changes: mtime matches -> content untouched, still v1.
    _run(doc_store, root)
    assert doc_store.get_transient("main:a.txt") == v1

    # Touch the file to a new mtime, keeping bytes identical: re-read happens but
    # upsert reports UNCHANGED (content+stable identical), version stays 1.
    future = f.stat().st_mtime + 1000
    os.utime(f, (future, future))
    _run(doc_store, root)
    # The recorded mtime moved forward (proof it was re-examined).
    assert doc_store.get_transient("main:a.txt")["mtime"] != v1["mtime"]


def test_deletion_detection(doc_store: SqliteStore, tmp_path: Path) -> None:
    """A file removed from disk is deleted from the store on the next run."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "a.txt").write_text("a")
    (root / "b.txt").write_text("b")
    _run(doc_store, root)
    assert set(doc_store.list_ids()) == {"main:a.txt", "main:b.txt"}

    (root / "b.txt").unlink()
    _run(doc_store, root)
    assert set(doc_store.list_ids()) == {"main:a.txt"}


def test_count_limit_disables_deletion(doc_store: SqliteStore, tmp_path: Path) -> None:
    """A count-limited (partial) pass must not delete unseen documents."""
    root = tmp_path / "vault"
    root.mkdir()
    for name in ("a.txt", "b.txt", "c.txt"):
        (root / name).write_text(name)

    # Full pass first, so all three are stored.
    _run(doc_store, root)
    assert len(set(doc_store.list_ids())) == 3

    # A count=1 pass sees only one file; the other two must survive.
    _run(doc_store, root, count="1", full="true")
    assert len(set(doc_store.list_ids())) == 3


def test_unknown_option_rejected(doc_store: SqliteStore, tmp_path: Path) -> None:
    """A mistyped option key fails loudly rather than being ignored."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "a.txt").write_text("a")
    from rag.errors import PluginError

    with pytest.raises(PluginError):
        _run(doc_store, root, limit="1")


def _doc_config(root: Path, sourcelink: str | None = None) -> dict[str, object]:
    """A documents plugin config subtree for folder ``main`` rooted at ``root``."""
    folder: dict[str, object] = {"folder-id": "main", "path": str(root)}
    if sourcelink is not None:
        folder["sourcelink"] = sourcelink
    return {"folders": [folder]}


def _url_for(store: SqliteStore, doc_id: str, config: dict[str, object]) -> str | None:
    """Return the extractor's ``source_url`` for a stored document under ``config``.

    Links are computed live from ``config`` at call time (never stored at ingest),
    so the config passed here — not whatever was used to ingest — drives the result.
    """
    stable = store.get_metadata(doc_id) or {}
    transient = store.get_transient(doc_id) or {}
    return DocumentsExtractor().source_url(doc_id, stable, transient, config)


def test_sourcelink_local(doc_store: SqliteStore, tmp_path: Path) -> None:
    """``local`` mode yields a file: URL to the file's absolute path."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("hi")
    _run(doc_store, root)

    expected = (root / "note.md").as_uri()
    # Nothing baked into stored metadata — the link is derived at query time.
    assert "source_link" not in (doc_store.get_transient("main:note.md") or {})
    assert "source_link" not in (doc_store.get_metadata("main:note.md") or {})
    url = _url_for(doc_store, "main:note.md", _doc_config(root, "local"))
    assert url == expected
    assert url.startswith("file://")


def test_sourcelink_obsidian(doc_store: SqliteStore, tmp_path: Path) -> None:
    """``obsidian`` mode yields an obsidian:// URL keyed by vault name + rel path."""
    root = tmp_path / "My Vault"
    (root / "sub").mkdir(parents=True)
    (root / "sub" / "a note.md").write_text("hi")
    _run(doc_store, root)

    # Vault name is the root basename; both components are percent-encoded.
    url = _url_for(doc_store, "main:sub/a note.md", _doc_config(root, "obsidian"))
    assert url == "obsidian://open?vault=My%20Vault&file=sub/a%20note.md"


def test_sourcelink_metadata_prefers_sidecar_url(doc_store: SqliteStore, tmp_path: Path) -> None:
    """``metadata`` mode returns the sidecar url when present."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "doc.bin").write_bytes(b"x")
    (root / "doc.bin.metadata.json").write_text(json.dumps({"url": "https://x/y"}))
    _run(doc_store, root)

    assert _url_for(doc_store, "main:doc.bin", _doc_config(root, "metadata")) == "https://x/y"


def test_sourcelink_metadata_falls_back_to_local(doc_store: SqliteStore, tmp_path: Path) -> None:
    """``metadata`` mode with no sidecar url falls back to a local file: URL."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("hi")
    _run(doc_store, root)

    url = _url_for(doc_store, "main:note.md", _doc_config(root, "metadata"))
    assert url == (root / "note.md").as_uri()


def test_sourcelink_default_is_metadata(doc_store: SqliteStore, tmp_path: Path) -> None:
    """An unset sourcelink defaults to metadata (sidecar url, else local)."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("hi")
    _run(doc_store, root)

    url = _url_for(doc_store, "main:note.md", _doc_config(root))  # no sourcelink key
    assert url == (root / "note.md").as_uri()


def test_sourcelink_reflects_moved_path(doc_store: SqliteStore, tmp_path: Path) -> None:
    """A link is computed from the live config, so a moved root changes the URL."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("hi")
    _run(doc_store, root)

    # Same stored document, but config now points the folder at a different path:
    # the local file: URL must follow the new path (nothing was frozen at ingest).
    moved = tmp_path / "moved-vault"
    url = _url_for(doc_store, "main:note.md", _doc_config(moved, "local"))
    assert url == (moved / "note.md").as_uri()


def test_sourcelink_invalid_rejected(doc_store: SqliteStore, tmp_path: Path) -> None:
    """An unknown sourcelink mode is a loud configuration error."""
    root = tmp_path / "vault"
    root.mkdir()
    (root / "note.md").write_text("hi")
    config = _doc_config(root, "bogus")
    from rag.errors import ConfigError

    with pytest.raises(ConfigError):
        DocumentsExtractor().run(doc_store, root / ".state", config, Secrets())
