"""Terminal helpers for chat output — OSC 8 hyperlinks.

OSC 8 lets a terminal show clickable link *text* with a hidden URL (so a citation
can read "email — Subject" yet open the original message). Unsupported terminals
would show raw escape codes, so we only emit them to an interactive TTY and fall
back to appending the plain URL otherwise.
"""

from __future__ import annotations

import os
import re
from typing import TextIO

# OSC 8 sequence: ESC ] 8 ; params ; URL ST  <text>  ESC ] 8 ; ; ST. We use BEL
# (\a) as the string terminator — widely accepted and shorter than ESC-backslash.
_OSC8_START: str = "\033]8;;"
_OSC8_END: str = "\a"

# Regex extracting citation markers from a model answer: [1], [2, 3], [4,5,6] …
_CITATION_RE: re.Pattern[str] = re.compile(r"\[(\d+(?:\s*,\s*\d+)*)\]")


def supports_hyperlinks(stream: TextIO) -> bool:
    """Whether ``stream`` is an interactive terminal we may emit OSC 8 links to.

    Conservative: requires a TTY and honours ``NO_COLOR`` (a common opt-out for
    terminal escapes). Dumb terminals are excluded. When false, callers should show
    the URL as plain text instead.
    """
    if os.environ.get("NO_COLOR"):
        return False
    if os.environ.get("TERM") == "dumb":
        return False
    try:
        return bool(stream.isatty())
    except (AttributeError, ValueError):  # pragma: no cover - unusual stream
        return False


def hyperlink(url: str, text: str, *, enabled: bool) -> str:
    """Return ``text`` as an OSC 8 hyperlink to ``url`` (or a plain fallback).

    With ``enabled`` false (non-TTY, piped, opted out), returns ``"text (url)"`` so
    the destination is still visible and copy-pasteable.
    """
    if not enabled:
        return f"{text} ({url})"
    return f"{_OSC8_START}{url}{_OSC8_END}{text}{_OSC8_START}{_OSC8_END}"


def cited_indices(answer: str, count: int) -> list[int]:
    """Return the distinct 1-based citation numbers referenced in ``answer``.

    Parses ``[n]`` and grouped ``[n, m]`` markers, keeps those within ``1..count``
    (dropping hallucinated/out-of-range numbers), and preserves first-mention order.
    """
    seen: list[int] = []
    for group in _CITATION_RE.findall(answer):
        # A group like "2, 3" holds several citations in one bracket.
        for token in group.split(","):
            number = int(token.strip())
            if 1 <= number <= count and number not in seen:
                seen.append(number)
    return seen
