"""Text chunking for embedding (PLAN §6.3).

A simple character-based sliding window with overlap. Good enough for Markdown
translations; a smarter (sentence/heading-aware) splitter can replace it later
without touching the rest of the pipeline.
"""

from __future__ import annotations


def chunk_text(text: str, size: int, overlap: int) -> list[str]:
    """Split ``text`` into chunks of ``size`` characters overlapping by ``overlap``.

    Returns an empty list for empty/whitespace-only text. The last chunk may be
    shorter. Requires ``0 <= overlap < size``.
    """
    if not text.strip():
        return []
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError(f"invalid chunking: size={size}, overlap={overlap}")
    chunks: list[str] = []
    # Advance by (size - overlap) each step so consecutive chunks share `overlap`.
    step = size - overlap
    start = 0
    length = len(text)
    while start < length:
        chunk = text[start : start + size]
        chunks.append(chunk)
        if start + size >= length:
            # Reached the end; stop (avoids an extra empty tail chunk).
            break
        start += step
    return chunks
