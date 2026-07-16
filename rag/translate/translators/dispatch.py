"""The ``dispatch`` translator (PLAN §5.4) — the default.

Routes a document to the right translator based on its mime type. Sub-translators
are imported lazily inside :meth:`translate` to avoid an import cycle (the email
translator dispatches its attachments back through here).
"""

from __future__ import annotations

import sys

from ..base import Document, Translation, Translator

# Mime types (without parameters) handled by the `text` translator.
_TEXT_TYPES: frozenset[str] = frozenset(
    {"text/plain", "text/markdown", "text/html", "application/xhtml+xml"}
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
        else:
            # Unknown type: nothing to translate (TBD as types are added).
            print(
                f"warning: no translator for mime {mime!r} "
                f"({document.doc_id}:{'/'.join(document.sub_doc_id) or 'root'})",
                file=sys.stderr,
            )
            return []
        return target.translate(document)


# Module-level instance selected by --translator dispatch (the default).
translator: Dispatch = Dispatch()
