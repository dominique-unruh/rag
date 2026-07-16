"""HTML → Markdown conversion helper (PLAN §5.5).

Backed by the well-maintained ``markdownify`` library rather than a hand-rolled
converter. Shared by the ``email`` and ``text`` translators.
"""

from __future__ import annotations

from markdownify import markdownify


def html_to_markdown(html: str) -> str:
    """Convert an HTML string to Markdown, trimmed of surrounding whitespace."""
    # ATX headings read better than the default underline style for our purposes.
    return markdownify(html, heading_style="ATX").strip()
