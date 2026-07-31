"""The ``pdf`` translator (PLAN §5.4).

Extracts the text from a PDF using the well-maintained ``pypdf`` library and
returns it as a single Markdown translation.
"""

from __future__ import annotations

import io
import logging
import re
import sys

from pypdf import PdfReader

from ..base import Document, Translation, Translator, make_translation

# pypdf reports recoverable problems in non-compliant PDFs through the standard
# ``logging`` machinery (logger ``pypdf`` and its children). Left alone these go
# to stderr via the root logger and drown the translate output in noise, so we
# route them through :class:`_PdfLogHandler` below.
_HARMLESS_PATTERNS: list[re.Pattern[str]] = [
    # Malformed xref/object pointers that pypdf transparently recovers from;
    # e.g. "Ignoring wrong pointing object 6 0 (offset 0)". Purely cosmetic.
    re.compile(r"^Ignoring wrong pointing object "),
]
"""Regexes for pypdf log messages that are safe to drop entirely."""


class _PdfLogHandler(logging.Handler):
    """Filters pypdf's log records: drop known-harmless noise, surface the rest.

    Messages matching :data:`_HARMLESS_PATTERNS` are discarded; anything else is
    printed to stderr with a ``[PDF parser]:`` prefix so its origin is clear.
    """

    def emit(self, record: logging.LogRecord) -> None:
        """Route one pypdf log record to stderr or the bit bucket."""
        # getMessage() applies pypdf's %-style / dict formatting for us.
        message: str = record.getMessage()
        if any(pattern.search(message) for pattern in _HARMLESS_PATTERNS):
            # Known recoverable noise — say nothing.
            return
        # Unknown pypdf message: keep it, but mark where it came from.
        print(f"[PDF parser]: {message}", file=sys.stderr)


def _install_pdf_log_handler() -> None:
    """Attach :class:`_PdfLogHandler` to the ``pypdf`` logger, once.

    Propagation is disabled so records do not also reach the root logger (which
    would re-emit the unprefixed original).
    """
    logger: logging.Logger = logging.getLogger("pypdf")
    # Idempotent: don't stack duplicate handlers if this module is re-imported.
    if any(isinstance(handler, _PdfLogHandler) for handler in logger.handlers):
        return
    logger.addHandler(_PdfLogHandler())
    logger.propagate = False


# Install the handler at import time, before any PDF is parsed.
_install_pdf_log_handler()


class PDF(Translator):
    """Extracts plain text from a PDF document."""

    def translate(self, document: Document) -> list[Translation]:
        """Return the PDF's extracted text as one translation.

        Pages are separated by blank lines. Extraction is best-effort — pages that
        yield no text (e.g. scanned images) contribute nothing (OCR is future work).
        """
        reader = PdfReader(io.BytesIO(document.content))
        # extract_text() may return None for image-only pages; treat as empty.
        pages = [(page.extract_text() or "").strip() for page in reader.pages]
        text = "\n\n".join(page for page in pages if page)
        return [make_translation(document, text)]


# Module-level instance selected by --translator pdf.
translator: PDF = PDF()
