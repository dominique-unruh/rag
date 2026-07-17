"""The ``ignore`` translator (PLAN §5.4).

A no-op translator that always produces no translation. Used by ``dispatch`` for
mime types that carry no indexable content and should be dropped silently (e.g.
S/MIME signature parts), so they neither warn nor create empty parts.
"""

from __future__ import annotations

from ..base import Document, Translation, Translator


class Ignore(Translator):
    """Translator that deliberately yields nothing for any document."""

    def translate(self, document: Document) -> list[Translation]:
        """Return no translations — the document is intentionally dropped."""
        # Nothing to index for this mime type; produce zero parts.
        return []


# Module-level instance selected by --translator ignore (and by dispatch).
translator: Ignore = Ignore()
