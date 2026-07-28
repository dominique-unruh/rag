"""Tests for the ``maintenance fsck`` invariant checker."""

from __future__ import annotations

import json
from pathlib import Path

from rag.blobs import BlobStore
from rag.config import Config
from rag.db import Database
from rag.embed.models import load_models
from rag.extract.plugins.email import EmailExtractor
from rag.extract.store import SqliteStore
from rag.maintenance import fsck
from rag.translate.store import TranslationStore

# A minimal embedding model so dangling-point checks have a collection to look at.
# Models live in the top-level `models:` section; `embedding.models` lists ids.
_MODELS = {
    "test": {"type": "dense", "backend": "openai", "api_base": "http://x", "model": "m", "dim": 4},
}


def _config(storage: Path) -> Config:
    """A Config rooted at ``storage`` with the test embedding model."""
    return Config(
        {"storage": {"dir": str(storage)}, "models": _MODELS, "embedding": {"models": ["test"]}},
        storage / "config.yaml",
    )


def _ctx(storage: Path, conn, *, vectors: object | None = None) -> fsck.FsckContext:
    """Build an FsckContext over ``storage`` with an explicit connection."""
    return fsck.FsckContext(
        config=_config(storage),
        storage_dir=storage,
        conn=conn,
        blobs=BlobStore(storage / "blobs"),
        vectors=vectors,
    )


def _seed(storage: Path) -> Database:
    """Create a DB with one document that has an active two-part translation."""
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    store = SqliteStore(db.conn, blobs, "email")
    store.upsert("a", b"raw-a", {"mime-type": "message/rfc822", "title": "A"})
    TranslationStore(db.conn, blobs).write_set(
        source_type="email", doc_id="a", source_version=1,
        translator_name="dispatch", git_rev="r",
        parts=[(["body"], "hello", {}), (["sig"], "bye", {})],
    )
    return db


def test_clean_db_has_no_dangling_rows(tmp_path: Path) -> None:
    """A well-formed DB trips none of the DB-level checks."""
    storage = tmp_path / "s"
    db = _seed(storage)
    ctx = _ctx(storage, db.conn)
    try:
        assert check_all_db(ctx) == []
    finally:
        db.close()


def check_all_db(ctx: fsck.FsckContext) -> list[fsck.Problem]:
    """Run just the DB-only checks (no vector store / plugin state needed)."""
    problems: list[fsck.Problem] = []
    for check in (
        fsck.check_dangling_translations,
        fsck.check_dangling_parts,
        fsck.check_dangling_embeddings,
        fsck.check_multiple_active_translations,
        fsck.check_dangling_blob_refs,
    ):
        problems.extend(check(ctx))
    return problems


def test_dangling_translation_detected(tmp_path: Path) -> None:
    """A translation set with no backing document is reported."""
    storage = tmp_path / "s"
    db = _seed(storage)
    # Insert an orphan set (FK off so the insert is allowed) — simulates a broken
    # cascade / manual edit that fsck must catch.
    db.conn.execute("PRAGMA foreign_keys=OFF")
    db.conn.execute(
        "INSERT INTO translation_sets "
        "(source_type, doc_id, source_version, metadata, translated_at, git_rev, is_active) "
        "VALUES ('email', 'ghost', 1, x'', 't', 'r', 0)"
    )
    db.conn.commit()
    ctx = _ctx(storage, db.conn)
    try:
        problems = fsck.check_dangling_translations(ctx)
        assert len(problems) == 1
        assert "ghost" in problems[0].message
    finally:
        db.close()


def test_dangling_blob_ref_detected(tmp_path: Path) -> None:
    """A document whose blob has vanished from the store is reported."""
    storage = tmp_path / "s"
    db = _seed(storage)
    blobs = BlobStore(storage / "blobs")
    # Delete the raw-email blob under the document, leaving a dangling reference.
    digest = db.conn.execute("SELECT blob_hash FROM documents WHERE doc_id='a'").fetchone()[0]
    assert blobs.delete(digest)
    ctx = _ctx(storage, db.conn)
    try:
        problems = fsck.check_dangling_blob_refs(ctx)
        assert any(digest in p.message for p in problems)
    finally:
        db.close()


class _FakeVectors:
    """A stand-in vector store yielding fixed points for the dangling-point check."""

    def __init__(self, collection: str, points: list[tuple[object, dict]]) -> None:
        """Hold one collection and the points it should scroll."""
        self._collection = collection
        self._points = points
        self.deleted: list[object] = []  # ids passed to delete_points

    def delete_points(self, collection: str, point_ids: list[object]) -> None:
        """Record which point ids a fix asked to delete."""
        self.deleted.extend(point_ids)

    def has_collection(self, collection: str) -> bool:
        """Whether ``collection`` is the one we hold."""
        return collection == self._collection

    def list_collections(self) -> list[str]:
        """All collection names (just the one)."""
        return [self._collection]

    def count(self, collection: str) -> int:
        """Number of points held."""
        return len(self._points)

    def iter_points(self, collection: str):
        """Yield the held ``(id, payload)`` points."""
        yield from self._points


def test_dangling_point_detected(tmp_path: Path) -> None:
    """A point whose part_id has no embeddings row for its model is reported."""
    storage = tmp_path / "s"
    db = _seed(storage)
    collection = load_models(_config(storage).raw)["test"].collection
    # A point referencing part_id 999 (never embedded) is dangling.
    point = ("pid-1", {"source_type": "email", "doc_id": "a", "part_id": 999, "chunk": 0})
    vectors = _FakeVectors(collection, [point])
    ctx = _ctx(storage, db.conn, vectors=vectors)
    try:
        problems = fsck.check_dangling_points(ctx)
        assert len(problems) == 1 and "dangling" in problems[0].message
        # The fix deletes exactly the dangling point ids from the collection.
        assert problems[0].apply_fix is not None
        info = problems[0].apply_fix()
        assert vectors.deleted == ["pid-1"]
        assert "deleted 1 point" in info
    finally:
        db.close()


def test_stray_collection_warned(tmp_path: Path) -> None:
    """A collection matching no configured model is flagged as a warning."""
    storage = tmp_path / "s"
    db = _seed(storage)
    ctx = _ctx(storage, db.conn, vectors=_FakeVectors("mystery", [("p", {})]))
    try:
        problems = fsck.check_stray_collections(ctx)
        assert len(problems) == 1 and problems[0].severity == "warning"
    finally:
        db.close()


# --- Extractor.fsck: email presence-file validation ---

def _email_state(tmp_path: Path, uidmap: object) -> Path:
    """Write ``uidmap`` as the email plugin's uidmap.json and return its state dir."""
    state_dir = tmp_path / "state"
    state_dir.mkdir()
    (state_dir / "uidmap.json").write_text(json.dumps(uidmap))
    return state_dir


def test_email_fsck_clean(tmp_path: Path) -> None:
    """A well-formed presence file yields no problems."""
    state = _email_state(tmp_path, {"accounts": {"a": {"folders": {
        "INBOX": {"uidvalidity": 1, "uids": {
            "1": {"message-id": "m1", "extracted": True},
            "2": {"message-id": "m2", "extracted": False},
        }},
    }}}})
    assert EmailExtractor().fsck(state, {}, "no") == 0


def test_email_fsck_missing_file_is_healthy(tmp_path: Path) -> None:
    """No presence file yet (nothing tracked) is a healthy state."""
    state = tmp_path / "state"
    state.mkdir()
    assert EmailExtractor().fsck(state, {}, "no") == 0


def _dup_state(tmp_path: Path) -> Path:
    """Presence file with two extracted=true uids for one message-id 'dup'."""
    return _email_state(tmp_path, {"accounts": {"a": {"folders": {
        "INBOX": {"uidvalidity": 1, "uids": {"1": {"message-id": "dup", "extracted": True}}},
        "Sent": {"uidvalidity": 1, "uids": {"9": {"message-id": "dup", "extracted": True}}},
    }}}})


def test_email_fsck_two_true_reports_and_no_fix(tmp_path, capsys) -> None:
    """Invariant-2 violation is printed and counted; fix='no' leaves the file alone."""
    state = _dup_state(tmp_path)
    errors = EmailExtractor().fsck(state, {}, "no")
    out = capsys.readouterr().out
    assert errors == 1
    assert "dup" in out and "extracted=true" in out and "[ERROR]" in out
    # Untouched: both entries still extracted=true.
    data = json.loads((state / "uidmap.json").read_text())
    flags = [
        e["extracted"]
        for f in data["accounts"]["a"]["folders"].values()
        for e in f["uids"].values()
    ]
    assert flags == [True, True]


def test_email_fsck_fix_yes_clears_all(tmp_path, capsys) -> None:
    """fix='yes' clears extracted on all uids of the message-id; re-check is clean."""
    state = _dup_state(tmp_path)
    errors = EmailExtractor().fsck(state, {}, "yes")
    out = capsys.readouterr().out
    assert errors == 1                      # still reported even though fixed
    assert "Fix:" in out and "cleared extracted on 2 uid(s)" in out
    # Both flags are now False, so a re-check finds nothing.
    assert EmailExtractor().fsck(state, {}, "no") == 0


def test_email_fsck_bad_shape_and_json(tmp_path, capsys) -> None:
    """Malformed JSON and malformed entries are reported."""
    # Invalid JSON.
    state = tmp_path / "s1"
    state.mkdir()
    (state / "uidmap.json").write_text("{not json")
    assert EmailExtractor().fsck(state, {}, "no") == 1
    assert "not valid JSON" in capsys.readouterr().out

    # Entry missing 'extracted' and a bare-string (legacy) entry.
    state2 = _email_state(tmp_path, {"accounts": {"a": {"folders": {
        "INBOX": {"uidvalidity": 1, "uids": {
            "1": {"message-id": "m"},           # missing extracted
            "2": "legacy-bare-string",           # not an object
        }},
    }}}})
    assert EmailExtractor().fsck(state2, {}, "no") == 2
    out = capsys.readouterr().out
    assert "'extracted' missing" in out and "not an object" in out


# --- ask_fix helper ---

def test_ask_fix_no_never_prompts(capsys) -> None:
    """fix='no' returns False and prints nothing."""
    from rag.fsck import ask_fix

    assert ask_fix("no", "Fix: do thing") is False
    assert capsys.readouterr().out == ""


def test_ask_fix_yes_prints_and_applies(capsys) -> None:
    """fix='yes' prints the description and returns True."""
    from rag.fsck import ask_fix

    assert ask_fix("yes", "Fix: do thing") is True
    assert "Fix: do thing" in capsys.readouterr().out


def test_ask_fix_ask_reads_answer(monkeypatch) -> None:
    """fix='ask' returns True only on an explicit yes."""
    from rag.fsck import ask_fix

    monkeypatch.setattr("builtins.input", lambda prompt="": "y")
    assert ask_fix("ask", "Fix: do thing") is True
    monkeypatch.setattr("builtins.input", lambda prompt="": "n")
    assert ask_fix("ask", "Fix: do thing") is False
