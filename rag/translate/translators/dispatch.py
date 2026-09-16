"""The ``dispatch`` translator (PLAN §5.4) — the default.

Routes a document to the right translator based on its mime type. Sub-translators
are imported lazily inside :meth:`translate` to avoid an import cycle (the email
translator dispatches its attachments back through here).
"""

from __future__ import annotations

import sys
from dataclasses import replace

from ..base import Document, Translation, Translator

# How many leading bytes to inspect when guessing whether unknown content is text.
_TEXT_SNIFF_BYTES: int = 8192
# Minimum fraction of decoded characters that must be printable/whitespace for a
# content sniff to count the document as human-readable text.
_TEXT_PRINTABLE_RATIO: float = 0.9


def _looks_like_text(mime: str, content: bytes) -> bool:
    """Return whether a document should get an identity (text) translation.

    True when the mime type is any ``text/*`` (e.g. ``text/x-tex``, ``text/csv``
    — declared human-readable), or when the type is unrecognised but the bytes
    themselves look like text: UTF-8 decodable, no NUL bytes, and overwhelmingly
    printable/whitespace characters. Binary blobs (images, archives, …) return
    False so they are skipped rather than dumped as mojibake.
    """
    if mime.startswith("text/"):
        return True
    if not content:
        # Nothing to index and no signal that it is text — leave it to the caller.
        return False
    sample = content[:_TEXT_SNIFF_BYTES]
    # A NUL byte is a reliable binary marker.
    if b"\x00" in sample:
        return False
    try:
        decoded = sample.decode("utf-8")
    except UnicodeDecodeError:
        return False
    # Count anything that is ordinary text: common whitespace or >= space.
    printable = sum(1 for ch in decoded if ch in "\t\n\r\f\v" or ch >= " ")
    return bool(decoded) and printable / len(decoded) >= _TEXT_PRINTABLE_RATIO

# Mime types (without parameters) handled by the `text` translator.
_TEXT_TYPES: frozenset[str] = frozenset(
    {"text/plain", "text/markdown", "text/html", "application/xhtml+xml"}
)

# Word OOXML mime types handled by the `docx` translator (document + template).
_DOCX_TYPES: frozenset[str] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
        "application/vnd.openxmlformats-officedocument.wordprocessingml.template",
    }
)

# Excel OOXML mime types handled by the `xlsx` translator (workbook + template).
_XLSX_TYPES: frozenset[str] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.template",
    }
)

# PowerPoint OOXML mime types handled by the `pptx` translator: the presentation
# family — deck (.pptx), template (.potx) and slideshow (.ppsx).
_PPTX_TYPES: frozenset[str] = frozenset(
    {
        "application/vnd.openxmlformats-officedocument.presentationml.presentation",
        "application/vnd.openxmlformats-officedocument.presentationml.template",
        "application/vnd.openxmlformats-officedocument.presentationml.slideshow",
    }
)

# Container/archive mime types handled by the `archive` translator, which unpacks
# them and dispatches each contained file. Covers ZIP, the TAR family, the
# standalone gzip/bzip2/xz compression streams (all stdlib-decodable), plus 7z and
# RAR (via py7zr / rarfile). The archive translator sniffs the real format from the
# content, so the exact spelling here only has to be broad enough to route to it.
_ARCHIVE_TYPES: frozenset[str] = frozenset(
    {
        "application/zip",
        "application/x-zip-compressed",
        "application/x-tar",
        "application/gzip",
        "application/x-gzip",
        "application/x-bzip2",
        "application/x-xz",
        "application/x-7z-compressed",
        "application/vnd.rar",
        "application/x-rar-compressed",
        "application/x-rar",
    }
)

# Mime types deliberately dropped (no indexable content) — routed to `ignore`,
# so they are silently skipped rather than warned about. Cryptographic signature
# parts: S/MIME (both the `x-` and IANA-registered spellings) and PGP.
_IGNORE_TYPES: frozenset[str] = frozenset(
    {
        "application/x-pkcs7-signature",
        "application/pkcs7-signature",
        "application/pgp-signature",
    }
)


class Dispatch(Translator):
    """Routes a document to email / text / pdf translators by mime type."""

    def translate(self, document: Document) -> list[Translation]:
        """Dispatch on ``document.mime_type``; unknown types yield no translation."""
        # Normalise: lower-case, drop any ";charset=…" parameter.
        mime = (document.mime_type or "").split(";", 1)[0].strip().lower()
        # Lazy imports break the dispatch<->email cycle.
        if mime == "message/rfc822":
            from .email import translator as target
        elif mime in _TEXT_TYPES:
            from .text import translator as target
        elif mime == "application/pdf":
            from .pdf import translator as target
        elif mime in _DOCX_TYPES:
            from .docx import translator as target
        elif mime in _XLSX_TYPES:
            from .xlsx import translator as target
        elif mime in _PPTX_TYPES:
            from .pptx import translator as target
        elif mime in _ARCHIVE_TYPES:
            from .archive import translator as target
        elif mime in _IGNORE_TYPES:
            from .ignore import translator as target
        elif _looks_like_text(mime, document.content):
            # No dedicated translator, but the document is human-readable text
            # (declared text/*, e.g. .tex/.csv, or unrecognised-but-looks-like
            # text): translate it as-is via the identity `text` translator.
            from .text import translator as target
        else:
            # Unknown, non-text type: nothing indexable to translate.
            print(
                f"warning: no translator for mime {mime!r} "
                f"({document.doc_id}:{'/'.join(document.sub_doc_id) or 'root'})",
                file=sys.stderr,
            )
            return []
        results = target.translate(document)
        # Tag each translation's debug_info with where it was dispatched (nested
        # dispatches accumulate a trace). debug_info is test-only, never stored.
        label = f"Dispatched to {type(target).__name__}"
        return [
            replace(t, debug_info=f"{t.debug_info}\n{label}" if t.debug_info else label)
            for t in results
        ]


# Module-level instance selected by --translator dispatch (the default).
translator: Dispatch = Dispatch()
