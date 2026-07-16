"""Local sparse (BM25-style) vectors (PLAN §6).

Produces sparse term-frequency vectors entirely locally — no model, no service.
Each text becomes a map of ``term-id -> term-frequency``; the vector store keeps
the collection with an IDF modifier so document-frequency weighting (the BM25
"IDF" part) is applied at query time from the collection itself.

Term ids are a stable 32-bit hash of the (lower-cased) token, matching Qdrant's
``uint32`` sparse indices. Hash collisions are possible but rare and harmless at
this scale.
"""

from __future__ import annotations

import hashlib
import re

# Tokeniser: Unicode word characters. Good enough for mixed DE/EN mail + notes;
# a smarter analyzer (stemming, stopwords) can replace this later.
_TOKEN_RE: re.Pattern[str] = re.compile(r"\w+", re.UNICODE)


def _term_id(term: str) -> int:
    """Return a stable 32-bit id for ``term`` (Qdrant sparse indices are uint32)."""
    # blake2b with a 4-byte digest gives a well-distributed 32-bit value.
    return int.from_bytes(hashlib.blake2b(term.encode("utf-8"), digest_size=4).digest(), "big")


def sparse_vector(text: str) -> dict[int, float]:
    """Return the term-frequency sparse vector of ``text`` as ``{term-id: tf}``.

    Empty for text with no tokens. Frequencies are raw counts; IDF weighting is
    applied by the vector store at query time.
    """
    counts: dict[int, float] = {}
    for match in _TOKEN_RE.findall(text.lower()):
        term_id = _term_id(match)
        counts[term_id] = counts.get(term_id, 0.0) + 1.0
    return counts
