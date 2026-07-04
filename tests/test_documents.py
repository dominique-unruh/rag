"""Document Store: upsert/dedup, reconcile deletion, ledgers, change planning."""

from __future__ import annotations

from rag.documents import (
    AUTO_KEYS,
    DocumentStore,
    compute_hash,
    filter_changes,
    plan_changes,
    to_document,
)


def _store(tmp_path):
    return DocumentStore(tmp_path / "documents.db")


def test_upsert_new_changed_unchanged(tmp_path):
    s = _store(tmp_path)
    assert s.upsert("notes", "/a.md", "hello", {}) is True   # new
    assert s.upsert("notes", "/a.md", "hello", {}) is False  # unchanged
    assert s.upsert("notes", "/a.md", "hello!", {}) is True  # changed
    assert s.count() == 1
    doc = s.get("notes:/a.md")
    assert doc.text == "hello!"
    assert doc.metadata["extractor"] == "notes"       # auto metadata
    assert doc.metadata["doc_id"] == "notes:/a.md"


def test_upsert_non_json_metadata(tmp_path):
    # email `date` arrives as a datetime -> must serialize (default=str).
    from datetime import datetime
    s = _store(tmp_path)
    assert s.upsert("email", "m@h", "body",
                    {"date": datetime(2026, 5, 4, 12, 0, 0)}) is True
    doc = s.get("email:m@h")
    assert "2026-05-04" in doc.metadata["date"]  # stored as string


def test_no_local_id_stored(tmp_path):
    s = _store(tmp_path)
    s.upsert("email", "a:b@h", "x", {})
    doc = s.get("email:a:b@h")
    assert not hasattr(doc, "local_id")     # local_id is extractor-internal, not stored
    assert "local_id" not in doc.metadata


def test_want_hashes_and_all(tmp_path):
    s = _store(tmp_path)
    s.upsert("notes", "a", "x", {})
    s.upsert("email", "m1@h", "y", {"subject": "hi"})
    assert set(s.want_hashes()) == {"notes:a", "email:m1@h"}
    assert {d.doc_id for d in s.all()} == {"notes:a", "email:m1@h"}


def test_reconcile_scoped_per_extractor(tmp_path):
    s = _store(tmp_path)
    s.upsert("notes", "a", "x", {})
    s.upsert("notes", "b", "y", {})
    s.upsert("email", "m", "z", {})
    # notes now only sees "a" -> "b" deleted; email untouched
    deleted = s.reconcile("notes", {"a"})
    assert deleted == ["notes:b"]
    assert set(s.want_hashes()) == {"notes:a", "email:m"}


def test_ledgers(tmp_path):
    s = _store(tmp_path)
    s.mark("embed_state", "notes:a", "h1")
    assert s.state("embed_state") == {"notes:a": "h1"}
    assert s.state("graph_state") == {}
    s.unmark("embed_state", "notes:a")
    assert s.state("embed_state") == {}


def test_compute_hash_ignores_bookkeeping_keys():
    a = compute_hash("t", {"subject": "s"})
    b = compute_hash("t", {"subject": "s", **{k: "x" for k in AUTO_KEYS}})
    assert a == b  # AUTO_KEYS excluded from the hash
    assert compute_hash("t", {"subject": "s"}) != compute_hash("t", {"subject": "z"})


def test_plan_changes():
    want = {"a": "h1", "b": "h2", "c": "h3"}
    have = {"a": "h1", "b": "OLD", "d": "h4"}
    to_process, to_delete = plan_changes(want, have)
    assert set(to_process) == {"b", "c"}  # b changed, c new
    assert to_delete == ["d"]             # d gone


def test_filter_changes_glob_and_max():
    proc = ["notes:/a.md", "notes:/b.md", "email:m@h"]
    dele = ["notes:/gone.md", "email:x@h"]
    # glob on doc_id restricts both sides
    p, d = filter_changes(proc, dele, docs_pattern="notes:*")
    assert p == ["notes:/a.md", "notes:/b.md"] and d == ["notes:/gone.md"]
    # max caps the process list only
    p2, d2 = filter_changes(proc, dele, max_items=1)
    assert p2 == ["notes:/a.md"] and d2 == dele
    # combined
    p3, _ = filter_changes(proc, dele, docs_pattern="*@h", max_items=1)
    assert p3 == ["email:m@h"]


def test_to_document_sets_id_and_excludes(tmp_path):
    s = _store(tmp_path)
    s.upsert("notes", "/a.md", "body", {"file_name": "a.md"})
    doc = to_document(s.get("notes:/a.md"))
    assert doc.id_ == "notes:/a.md"
    for key in AUTO_KEYS:
        assert key in doc.excluded_embed_metadata_keys
