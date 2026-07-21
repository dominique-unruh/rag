"""Module 4 — chat over the RAG (PLAN §7).

Only **simple mode** (a.k.a. *retrieve-then-read*) is implemented: for each user
turn we retrieve the most relevant chunks from the embedding indices, fuse them,
and let the chat LLM answer from that context. The *agentic* (complex) mode, in
which the LLM plans its own retrieval, is still in ``PLAN.md``.
"""

from __future__ import annotations
