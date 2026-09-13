"""Tests for module 2 (translation): translators, discovery, store, pipeline."""

from __future__ import annotations

import io
from pathlib import Path

import pytest

from rag.blobs import BlobStore
from rag.config import Config
from rag.db import Database
from rag.extract.store import SqliteStore
from rag.translate.base import Document, Translation
from rag.translate.discovery import TranslatorError, list_translators, load_translator
from rag.translate.pipeline import TranslateOptions, TranslationError, _validate, run_translation
from rag.translate.store import SourceDocument, TranslationStore
from rag.translate.translators.dispatch import translator as dispatch
from rag.translate.translators.text import translator as text_translator

# A multipart email: plain body + one text attachment, with an encoded subject.
_EMAIL = (
    "From: A <a@x>\r\n"
    "To: B <b@y>\r\n"
    "Subject: =?utf-8?q?Gr=C3=BC=C3=9Fe?=\r\n"
    "MIME-Version: 1.0\r\n"
    "Content-Type: multipart/mixed; boundary=BND\r\n\r\n"
    "--BND\r\nContent-Type: text/plain; charset=utf-8\r\n\r\nHello body.\r\n"
    "--BND\r\nContent-Type: text/plain\r\n"
    "Content-Disposition: attachment; filename=note.txt\r\n\r\nAttached text.\r\n"
    "--BND--\r\n"
).encode()


# --- translators ---------------------------------------------------------
def test_text_translator_html_and_passthrough() -> None:
    """HTML becomes Markdown; plain text and Markdown pass through."""
    html = Document("d", [], b"<h1>Hi</h1><p>yo</p>", {"mime-type": "text/html"}, {})
    assert text_translator.translate(html)[0].text == "# Hi\n\nyo"
    plain = Document("d", [], b"just text", {"mime-type": "text/plain"}, {})
    assert plain.mime_type == "text/plain"
    assert text_translator.translate(plain)[0].text == "just text"


def test_email_translator_body_and_attachment() -> None:
    """Email yields a body part (decoded headers) + attachment part via dispatch."""
    doc = Document("m1", [], _EMAIL, {"mime-type": "message/rfc822"}, {})
    parts = dispatch.translate(doc)
    by_sub = {tuple(p.sub_doc_id): p for p in parts}
    # Body: decoded subject prepended, body text present.
    body = by_sub[("body",)]
    assert "**Subject:** Grüße" in body.text
    assert "Hello body." in body.text
    # Attachment routed through dispatch -> text translator.
    assert by_sub[("attachment:1",)].text == "Attached text."
    # Every translation keeps the source doc_id.
    assert all(p.doc_id == "m1" for p in parts)


def test_dispatch_tags_debug_info() -> None:
    """Dispatch records where it routed each translation (nested trace)."""
    doc = Document("m1", [], _EMAIL, {"mime-type": "message/rfc822"}, {})
    by_sub = {tuple(p.sub_doc_id): p for p in dispatch.translate(doc)}
    # Body routed straight to Email.
    assert by_sub[("body",)].debug_info == "Dispatched to Email"
    # Attachment: inner dispatch -> Text, then outer dispatch -> Email.
    assert by_sub[("attachment:1",)].debug_info == "Dispatched to Text\nDispatched to Email"


def test_pdf_translator_returns_single_part() -> None:
    """The PDF translator returns exactly one translation for a valid PDF."""
    from pypdf import PdfWriter

    writer = PdfWriter()
    writer.add_blank_page(width=72, height=72)
    buf = io.BytesIO()
    writer.write(buf)
    from rag.translate.translators.pdf import translator as pdf_translator

    doc = Document("p", [], buf.getvalue(), {"mime-type": "application/pdf"}, {})
    result = pdf_translator.translate(doc)
    assert len(result) == 1 and result[0].doc_id == "p"


def _minimal_docx(body_xml: str) -> bytes:
    """Build a minimal valid .docx (OOXML zip) whose body is ``body_xml``."""
    import zipfile

    content_types = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Types xmlns="http://schemas.openxmlformats.org/package/2006/content-types">'
        '<Default Extension="rels" ContentType="application/vnd.openxmlformats-package.relationships+xml"/>'
        '<Override PartName="/word/document.xml" '
        'ContentType="application/vnd.openxmlformats-officedocument.wordprocessingml.document.main+xml"/>'
        "</Types>"
    )
    rels = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">'
        '<Relationship Id="rId1" '
        'Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/officeDocument" '
        'Target="word/document.xml"/></Relationships>'
    )
    document = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>'
        '<w:document xmlns:w="http://schemas.openxmlformats.org/wordprocessingml/2006/main">'
        f"<w:body>{body_xml}</w:body></w:document>"
    )
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as archive:
        archive.writestr("[Content_Types].xml", content_types)
        archive.writestr("_rels/.rels", rels)
        archive.writestr("word/document.xml", document)
    return buf.getvalue()


def test_docx_translator_converts_to_markdown() -> None:
    """The docx translator maps Word styles to Markdown (headings, bold)."""
    from rag.translate.translators.docx import translator as docx_translator

    body = (
        '<w:p><w:pPr><w:pStyle w:val="Heading1"/></w:pPr><w:r><w:t>My Title</w:t></w:r></w:p>'
        "<w:p><w:r><w:t>Hello </w:t></w:r>"
        "<w:r><w:rPr><w:b/></w:rPr><w:t>bold</w:t></w:r>"
        "<w:r><w:t> world.</w:t></w:r></w:p>"
    )
    mime = "application/vnd.openxmlformats-officedocument.wordprocessingml.document"
    doc = Document("dx", [], _minimal_docx(body), {"mime-type": mime}, {})
    result = docx_translator.translate(doc)
    assert len(result) == 1 and result[0].doc_id == "dx"
    assert result[0].text == "# My Title\n\nHello **bold** world."


def test_dispatch_routes_docx_and_dotx() -> None:
    """Dispatch routes both the docx and dotx (template) mime types to Docx."""
    body = "<w:p><w:r><w:t>plain para</w:t></w:r></w:p>"
    for mime in (
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    ):
        doc = Document("d", [], _minimal_docx(body), {"mime-type": mime}, {})
        parts = dispatch.translate(doc)
        assert len(parts) == 1
        assert parts[0].text == "plain para"
        assert parts[0].debug_info == "Dispatched to Docx"


def test_dispatch_unknown_mime_yields_nothing() -> None:
    """An unhandled mime type produces no translation."""
    doc = Document("d", [], b"\x00\x01", {"mime-type": "application/octet-stream"}, {})
    assert dispatch.translate(doc) == []


def test_dispatch_ignored_mime_yields_nothing() -> None:
    """A deliberately-ignored mime (e.g. S/MIME signature) is routed to `ignore`."""
    doc = Document("d", [], b"\x00\x01", {"mime-type": "application/x-pkcs7-signature"}, {})
    assert dispatch.translate(doc) == []
    assert load_translator("ignore").translate(doc) == []


# --- discovery -----------------------------------------------------------
def test_discovery() -> None:
    """Built-in translators are discoverable; unknown names raise."""
    assert {"dispatch", "email", "text", "pdf"} <= set(list_translators())
    assert load_translator("text") is text_translator
    with pytest.raises(TranslatorError):
        load_translator("does-not-exist")


# --- pipeline invariants -------------------------------------------------
def _source() -> SourceDocument:
    """A minimal source document for validation tests."""
    return SourceDocument("email", "m1", 1, b"", {}, {})


def test_validate_rejects_wrong_doc_id() -> None:
    """A translation with a mismatched doc_id is rejected."""
    with pytest.raises(TranslationError):
        _validate(_source(), [Translation("OTHER", ["body"], "t", {})])


def test_validate_rejects_duplicate_sub_doc_id() -> None:
    """Two translations sharing a sub_doc_id are rejected."""
    dup = [Translation("m1", ["body"], "a", {}), Translation("m1", ["body"], "b", {})]
    with pytest.raises(TranslationError):
        _validate(_source(), dup)


# --- store ---------------------------------------------------------------
def test_store_select_and_write(database: Database, blob_store: BlobStore) -> None:
    """select_documents finds untranslated/stale docs; write_set stores an active set."""
    src = SqliteStore(database.conn, blob_store, "email")
    src.upsert("m1", _EMAIL, {"mime-type": "message/rfc822", "subject": "Hi"})
    store = TranslationStore(database.conn, blob_store)

    # Untranslated -> selected.
    assert [d.doc_id for d in store.select_documents()] == ["m1"]

    set_id = store.write_set(
        source_type="email", doc_id="m1", source_version=1,
        translator_name="dispatch", git_rev="rev",
        parts=[(["body"], "A", {}), (["attachment:1"], "B", {})],
    )
    conn = database.conn
    assert conn.execute("SELECT COUNT(*) FROM translation_parts WHERE set_id=?", (set_id,)).fetchone()[0] == 2
    keys = sorted(r[0] for r in conn.execute("SELECT part_key FROM translation_parts WHERE set_id=?", (set_id,)))
    assert keys == ['["attachment:1"]', '["body"]']

    # Up to date now -> not selected by default, but --force still yields it.
    assert list(store.select_documents()) == []
    assert [d.doc_id for d in store.select_documents(force=True)] == ["m1"]

    # A newer set deactivates the previous one (at most one active).
    set_id2 = store.write_set(
        source_type="email", doc_id="m1", source_version=1,
        translator_name="dispatch", git_rev="rev", parts=[(["body"], "newer", {})],
    )
    active = conn.execute("SELECT id FROM translation_sets WHERE is_active=1").fetchall()
    assert len(active) == 1 and active[0][0] == set_id2


def test_store_selects_stale(database: Database, blob_store: BlobStore) -> None:
    """A document whose active set is from an older version is re-selected."""
    src = SqliteStore(database.conn, blob_store, "email")
    src.upsert("m1", _EMAIL, {"mime-type": "message/rfc822"})
    store = TranslationStore(database.conn, blob_store)
    store.write_set(source_type="email", doc_id="m1", source_version=1,
                    translator_name="dispatch", git_rev="rev", parts=[(["body"], "A", {})])
    assert list(store.select_documents()) == []
    # Change the source -> version bumps to 2 -> active set (v1) is now stale.
    src.upsert("m1", _EMAIL + b"more", {"mime-type": "message/rfc822"})
    assert [d.doc_id for d in store.select_documents()] == ["m1"]


# --- pipeline end-to-end (inspection mode; no git/DB write needed) -------
def test_run_translation_inspect(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """run_translation --inspect translates a seeded email and prints it."""
    storage = tmp_path / "storage"
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    SqliteStore(db.conn, blobs, "email").upsert("m1", _EMAIL, {"mime-type": "message/rfc822"})
    config = Config({"storage": {"dir": str(storage)}}, tmp_path / "config.yaml")

    processed = run_translation(config, TranslateOptions(inspect=True), db=db)

    assert processed == 1
    out = capsys.readouterr().out
    # The body (with decoded subject) and the attachment both appear.
    assert "**Subject:** Grüße" in out
    assert "Attached text." in out
    # Nothing was written to the DB in inspection mode.
    assert db.conn.execute("SELECT COUNT(*) FROM translation_sets").fetchone()[0] == 0
    db.close()
