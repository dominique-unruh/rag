"""The ``docx`` translator (PLAN §5.4).

Converts a Word OOXML document (``.docx``/``.dotx``) to Markdown using the
``mammoth`` library (which maps Word styles to semantic HTML — headings, lists,
tables, bold/italic) and the shared HTML→Markdown helper.
"""

from __future__ import annotations

import io
import sys

import mammoth

from ..base import Document, Translation, Translator, make_translation
from ..markdown import html_to_markdown


class Docx(Translator):
    """Converts a Word OOXML document to Markdown."""

    def translate(self, document: Document) -> list[Translation]:
        """Return the document's text as one Markdown translation.

        ``mammoth`` produces semantic HTML from the docx, which is then converted
        to Markdown. mammoth also reports conversion *messages*; only ``error``
        ones are surfaced (a docx routinely emits many harmless "unrecognised
        style" warnings — the text is still extracted, so they would be noise).
        """
        result = mammoth.convert_to_html(io.BytesIO(document.content))
        # Surface only genuine errors; warnings (unmapped styles) are cosmetic.
        for message in result.messages:
            if getattr(message, "type", None) == "error":
                print(f"[docx parser]: {message}", file=sys.stderr)
        text = html_to_markdown(result.value)
        return [make_translation(document, text)]


# Module-level instance selected by --translator docx.
translator: Docx = Docx()
