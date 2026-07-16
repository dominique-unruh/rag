"""The ``pdf`` translator (PLAN §5.4).

Extracts the text from a PDF using the well-maintained ``pypdf`` library and
returns it as a single Markdown translation.
"""

from __future__ import annotations

import io

from pypdf import PdfReader

from ..base import Document, Translation, Translator, make_translation


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
