"""Translator API: ``Document``, ``Translation``, the ``Translator`` base class
and the ``make_translation`` helper (PLAN §5.1).

A translator is **stateless** pure conversion: given one :class:`Document` it
returns a list of :class:`Translation`s. All bookkeeping, DB work and invariant
enforcement live in the pipeline (:mod:`rag.translate.pipeline`), not here.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True)
class Document:
    """One input to a translator.

    Carries everything a translator needs. For a document taken straight from the
    extraction DB, ``sub_doc_id`` is the empty list; translators extend it as they
    descend into sub-parts (e.g. email attachments).
    """

    doc_id: str
    """The source document id (shared by every part it produces)."""

    sub_doc_id: list[str]
    """Path of this (sub-)document within its top-level document; ``[]`` at top."""

    content: bytes
    """Raw bytes to translate."""

    stable_metadata: dict[str, Any]
    """Stable metadata from the ``documents`` row (or of a synthetic sub-document)."""

    transient_metadata: dict[str, Any] = field(default_factory=dict)
    """Transient metadata (folder/read/…); available but rarely needed here."""

    @property
    def mime_type(self) -> str | None:
        """The mime type — stored in the metadata, exposed here for convenience.

        Equivalent to ``stable_metadata.get("mime-type")``.
        """
        return self.stable_metadata.get("mime-type")


@dataclass(frozen=True)
class Translation:
    """One Markdown translation produced from a document.

    Carries **stable** metadata only — translations have no transient metadata.
    ``doc_id`` must equal the source document's id (enforced by the pipeline).
    """

    doc_id: str
    """Must equal the source document's ``doc_id``."""

    sub_doc_id: list[str]
    """Identifies this part within the document (unique within a translation set)."""

    text: str
    """The Markdown result."""

    stable_metadata: dict[str, Any] = field(default_factory=dict)
    """Per-part stable metadata."""

    debug_info: str = ""
    """Optional free-text debugging info. Has no effect on the pipeline and is
    **never stored in the DB** — for testing/inspection only."""


class Translator(ABC):
    """Converts one document into a list of translations.

    Implementations **must be stateless** — an instance holds no per-document
    state and can be reused and shared freely across documents and threads.
    """

    @abstractmethod
    def translate(self, document: Document) -> list[Translation]:
        """Translate ``document`` into zero or more Markdown translations."""


def make_translation(
    document: Document,
    text: str,
    *,
    sub_doc_id: list[str] | None = None,
    extra_metadata: dict[str, Any] | None = None,
) -> Translation:
    """Build a :class:`Translation` from a :class:`Document` and its translated text.

    Convenience so translators don't repeat plumbing: copies the document's
    ``doc_id`` and ``sub_doc_id`` (unless ``sub_doc_id`` is overridden) and seeds
    the part metadata from the document's stable metadata, updated with any
    ``extra_metadata``.
    """
    # Default to the document's own sub_doc_id; translators override when they
    # emit a nested part (e.g. an email body or attachment).
    resolved_sub = list(document.sub_doc_id if sub_doc_id is None else sub_doc_id)
    # Seed metadata from the source, then layer any translator-supplied extras.
    metadata: dict[str, Any] = dict(document.stable_metadata)
    if extra_metadata:
        metadata.update(extra_metadata)
    return Translation(
        doc_id=document.doc_id,
        sub_doc_id=resolved_sub,
        text=text,
        stable_metadata=metadata,
    )
