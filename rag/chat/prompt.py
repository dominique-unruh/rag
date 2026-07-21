"""Prompt construction for simple-mode chat (PLAN §7).

Pure helpers (no I/O) that turn retrieved chunks + a question into the messages
sent to the LLM, so they can be unit-tested in isolation. Simple mode is
*retrieve-then-read*: the context is stuffed into the user turn and the model is
told to answer from it and cite sources by number.
"""

from __future__ import annotations

from collections.abc import Sequence

from .retrieve import RetrievedChunk

# The system instruction: grounds the model in the provided context and asks for
# numbered citations, so answers stay traceable to source documents.
_SYSTEM_PROMPT: str = (
    "You are a helpful assistant answering questions over the user's personal data "
    "(emails, notes, …). Answer using only the context passages provided with each "
    "question. Cite the passages you use by their number, like [1] or [2, 3]. If the "
    "answer is not in the context, say so plainly instead of guessing. Be concise."
)


def system_prompt() -> str:
    """Return the system instruction for simple-mode chat."""
    return _SYSTEM_PROMPT


def _citation_label(chunk: RetrievedChunk) -> str:
    """Return a short human label for a chunk's source (title/date/source_type)."""
    # Prefer the title; fall back to the doc id so the label is never empty.
    title = chunk.title or chunk.doc_id
    parts = [chunk.source_type, title]
    if chunk.date:
        parts.append(chunk.date)
    return " — ".join(parts)


def format_context(chunks: Sequence[RetrievedChunk]) -> str:
    """Render retrieved ``chunks`` as a numbered context block for the prompt."""
    if not chunks:
        return "(no relevant passages were found)"
    blocks: list[str] = []
    for index, chunk in enumerate(chunks, start=1):
        # Number each passage so the model can cite it as [n]; label its source.
        blocks.append(f"[{index}] ({_citation_label(chunk)})\n{chunk.text}")
    return "\n\n".join(blocks)


def build_user_message(query: str, chunks: Sequence[RetrievedChunk]) -> str:
    """Build the augmented user turn: the context block followed by the question."""
    context = format_context(chunks)
    return (
        f"Context passages:\n\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer from the context above, citing passages by number."
    )
