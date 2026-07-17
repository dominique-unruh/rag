"""Exception types shared across the pipeline.

Keeping the exceptions in one small module avoids import cycles between the
core (``db``, ``blobs``) and the extraction layer that raises them.
"""

from __future__ import annotations


class RagError(Exception):
    """Base class for all errors raised by this project.

    Catching :class:`RagError` catches every domain-specific failure while
    letting genuinely unexpected exceptions (bugs) propagate.
    """


class ConfigError(RagError):
    """Raised when the YAML configuration is missing or malformed."""


class SecretNotFoundError(RagError):
    """Raised when a required secret cannot be resolved and no prompt is possible."""


class DocumentExistsError(RagError):
    """Raised by ``Store.upsert(..., exist_ok=False)`` when the document already exists.

    Signals that a plugin expected to create a fresh document but found one
    already present (likely a duplicated ``doc_id``).
    """


class DocumentMissingError(RagError):
    """Raised when an operation expected a document that does not exist.

    Produced by ``Store.upsert(..., missing_ok=False)`` and
    ``Store.delete(..., missing_ok=False)`` when the target ``doc_id`` is absent.
    """


class PluginError(RagError):
    """Raised for plugin discovery/loading problems (missing class, bad name, …)."""


class TimestampError(RagError):
    """Raised when a timestamp string cannot be parsed by :mod:`rag.timestamps`."""


class VectorStoreBusyError(RagError):
    """Raised when the embedded vector store is locked by another process.

    Embedded Qdrant allows a single writer per directory; a second opener (another
    ``rag embed`` run, or the db-browser reading it) hits this. The fix is to close
    the other user and retry.
    """
