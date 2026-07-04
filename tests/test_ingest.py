"""Ingest glue: email id assignment, dedup gate, notes reader, email loader."""

from __future__ import annotations

from datetime import date

import pytest
from llama_index.core import Document
from llama_index.core.storage.docstore import SimpleDocumentStore

from rag import config, ingest


# --- _assign_email_id -----------------------------------------------------

def test_assign_id_from_message_id():
    doc = Document(text="t", metadata={"headers": {"message-id": ("<abc@x>",)}, "uid": "9"})
    ingest._assign_email_id(doc, "INBOX")
    assert doc.id_ == "abc@x"
    assert "headers" not in doc.metadata  # bulky blob dropped
    assert "uid" in doc.excluded_embed_metadata_keys


def test_assign_id_string_header_and_case():
    doc = Document(text="t", metadata={"headers": {"Message-ID": "<Z@y>"}})
    ingest._assign_email_id(doc, "INBOX")
    assert doc.id_ == "Z@y"


def test_assign_id_from_lazyheaders_mapping():
    # imap-tools returns a LazyHeaders object (not a dict); must still work.
    class LazyHeaders:
        def __init__(self, d):
            self._d = d

        def get(self, k):
            return self._d.get(k)

    doc = Document(text="t", metadata={
        "headers": LazyHeaders({"message-id": ("<abc@x>",)}), "uid": "9"})
    ingest._assign_email_id(doc, "INBOX")
    assert doc.id_ == "abc@x"  # Message-ID used, not the uid fallback


def test_assign_id_uid_fallback():
    doc = Document(text="t", metadata={"headers": {}, "uid": "42"})
    ingest._assign_email_id(doc, "Sent")
    assert doc.id_ == "uid:Sent:42"


def test_assign_id_keeps_random_when_nothing_available():
    doc = Document(text="t", metadata={})
    original = doc.id_
    ingest._assign_email_id(doc, "INBOX")
    assert doc.id_ == original  # untouched, still dedups on content hash


# --- select_new_or_changed / mark_ingested --------------------------------

def test_select_new_changed_unchanged():
    ds = SimpleDocumentStore()
    d1 = Document(text="one", id_="a")
    d2 = Document(text="two", id_="b")

    first = ingest.select_new_or_changed([d1, d2], ds)
    assert {d.id_ for d in first} == {"a", "b"}  # both new; docstore untouched
    assert ds.get_document_hash("a") is None  # select does NOT mutate

    for d in first:
        ingest.mark_ingested(d, ds)

    assert ingest.select_new_or_changed([d1, d2], ds) == []  # now recorded

    changed = ingest.select_new_or_changed([Document(text="one!", id_="a"), d2], ds)
    assert {d.id_ for d in changed} == {"a"}  # content changed -> reselected


def test_mark_ingested_is_per_document_crash_safe():
    # Recording only the item actually inserted means an interrupted run leaves
    # the rest still pending next time.
    ds = SimpleDocumentStore()
    docs = [Document(text=f"d{i}", id_=str(i)) for i in range(3)]
    ingest.mark_ingested(docs[0], ds)  # only the first "succeeded"
    remaining = ingest.select_new_or_changed(docs, ds)
    assert {d.id_ for d in remaining} == {"1", "2"}


# --- load_notes -----------------------------------------------------------

def test_load_notes_reads_md_with_path_ids(project):
    root, _ = project
    cfg = config.load(root)
    docs = ingest.load_notes(cfg)
    assert len(docs) == 1
    assert docs[0].id_.endswith("a.md")


def test_load_notes_disabled_returns_empty(project):
    root, _ = project
    cfg = config.load(root)
    cfg.notes.enabled = False
    assert ingest.load_notes(cfg) == []


# --- load_emails (mocked reader, no network) ------------------------------

class _FakeReader:
    last_kwargs = None

    def __init__(self, host, username, password):
        self.creds = (host, username, password)

    def load_data(self, **kwargs):
        _FakeReader.last_kwargs = kwargs
        return [
            Document(text="e1", metadata={"headers": {"message-id": ("<m1@h>",)}, "uid": "1"}),
            Document(text="e2", metadata={"headers": {}, "uid": "2"}),
        ]


def test_load_emails_assigns_ids_and_builds_since(project, monkeypatch):
    root, _ = project
    # enable email
    text = (root / "config.yaml").read_text().replace(
        "email:\n  enabled: false",
        "email:\n  enabled: true\n  host: h\n  username: u\n"
        "  since_date: 2025-01-01\n  password: p",
    )
    (root / "config.yaml").write_text(text, encoding="utf-8")
    cfg = config.load(root)

    import llama_index.readers.imap as imap_mod

    monkeypatch.setattr(imap_mod, "ImapReader", _FakeReader)

    docs = ingest.load_emails(cfg)
    assert [d.id_ for d in docs] == ["m1@h", "uid:INBOX:2"]
    # server-side SINCE built from the configured cutoff date
    crit = _FakeReader.last_kwargs["search_criteria"]
    assert "SINCE" in str(crit) and "2025" in str(crit)


def test_load_emails_disabled_returns_empty(project):
    root, _ = project
    cfg = config.load(root)
    assert cfg.email.enabled is False
    assert ingest.load_emails(cfg) == []


# --- CLI parser -----------------------------------------------------------

def test_parser_defaults():
    args = ingest.build_parser().parse_args([])
    assert args.verbose is False
    assert args.delay == 0.0


def test_parser_verbose_and_delay():
    args = ingest.build_parser().parse_args(["--verbose", "--delay=60"])
    assert args.verbose is True
    assert args.delay == 60.0


def test_parser_delay_float():
    args = ingest.build_parser().parse_args(["--delay", "1.5"])
    assert args.delay == 1.5
    assert args.verbose is False
