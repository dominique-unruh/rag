"""The ``text`` translator (PLAN §5.4).

Handles plain text, HTML and Markdown: plain text and Markdown pass through
unchanged; HTML is converted to Markdown. Nothing else.
"""

from __future__ import annotations

from ..base import Document, Translation, Translator, make_translation
from ..markdown import html_to_markdown

# Mime types treated as HTML (converted to Markdown).
_HTML_TYPES: frozenset[str] = frozenset({"text/html", "application/xhtml+xml"})


class Text(Translator):
    """Pass-through / HTML→Markdown translator for textual documents."""

    def translate(self, document: Document) -> list[Translation]:
        """Return the document as Markdown (converting only if it is HTML)."""
        # Decode the bytes leniently — charset detection is out of scope here.
        text = document.content.decode("utf-8", errors="replace")
        # Normalise the mime type (drop any ";charset=…" parameter).
        mime = (document.mime_type or "").split(";", 1)[0].strip().lower()
        if mime in _HTML_TYPES:
            # HTML → Markdown; plain text and Markdown are kept as-is.
            text = html_to_markdown(text)
        return [make_translation(document, text)]


# Module-level instance selected by --translator text.
translator: Text = Text()
