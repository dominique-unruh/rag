"""Extraction: scaffolding upsert+reconcile, extractors, email id, registry."""

from __future__ import annotations

from llama_index.core import Document

from rag import config, extract
from rag.documents import DocumentStore
from rag.extract.base import run_extractor
from rag.extract.email import EmailExtractor, _local_id_and_metadata
from rag.extract.notes import NotesExtractor


class _FakeExtractor:
    name = "fake"

    def __init__(self, items):
        self._items = items  # list of (local_id, text, metadata)

    def enabled(self, cfg):
        return True

    def iter_documents(self, cfg):
        yield from self._items


def test_run_extractor_upserts_and_reconciles(tmp_path):
    store = DocumentStore(tmp_path / "d.db")
    ex = _FakeExtractor([("1", "a", {}), ("2", "b", {})])
    r = run_extractor(ex, cfg=None, store=store)
    assert r == {"total": 2, "new": 2, "deleted": 0}

    # second run drops item "2" -> reconciled away
    ex2 = _FakeExtractor([("1", "a", {})])
    r2 = run_extractor(ex2, cfg=None, store=store)
    assert r2["new"] == 0 and r2["deleted"] == 1
    assert set(store.want_hashes()) == {"fake:1"}


def test_notes_extractor_yields_path_ids(project):
    root, _ = project
    cfg = config.load(root)
    ex = NotesExtractor()
    assert ex.enabled(cfg) is True
    items = list(ex.iter_documents(cfg))
    assert len(items) == 1
    local_id, text, meta = items[0]
    assert local_id.endswith("a.md")
    assert "Alice" in text


# --- email id resolution --------------------------------------------------

class _LazyHeaders:
    def __init__(self, d):
        self._d = d

    def get(self, k):
        return self._d.get(k)


def test_email_message_id_from_lazyheaders():
    doc = Document(text="body", metadata={
        "headers": _LazyHeaders({"message-id": ("<abc@x>",)}), "uid": "9"})
    local_id, meta = _local_id_and_metadata(doc, "INBOX")
    assert local_id == "abc@x"
    assert "headers" not in meta
    assert meta["folder"] == "INBOX"


def test_email_uid_fallback():
    doc = Document(text="body", metadata={"headers": _LazyHeaders({}), "uid": "42"})
    local_id, _ = _local_id_and_metadata(doc, "Sent")
    assert local_id == "uid:Sent:42"


def test_email_hash_fallback_when_nothing():
    doc = Document(text="body", metadata={})
    local_id, _ = _local_id_and_metadata(doc, "INBOX")
    assert local_id.startswith("hash:")


def test_registry_and_disabled(project):
    root, _ = project
    cfg = config.load(root)  # notes enabled, email disabled (fixture)
    assert set(extract.EXTRACTORS) == {"notes", "email"}
    assert extract.EXTRACTORS["notes"].enabled(cfg) is True
    assert extract.EXTRACTORS["email"].enabled(cfg) is False
