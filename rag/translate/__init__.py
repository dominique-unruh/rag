"""Module 2 — processing / translation (PLAN §5).

Turns extracted documents into Markdown translations. The conversion work is done
by stateless :class:`~rag.translate.base.Translator` subclasses under
``rag.translate.translators`` (one file per translator); the pipeline
(:mod:`rag.translate.pipeline`) selects documents, invokes the chosen translator,
enforces invariants, and does all DB/blob bookkeeping.
"""

from __future__ import annotations

from .base import Document, Translation, Translator, make_translation

__all__ = ["Document", "Translation", "Translator", "make_translation"]
