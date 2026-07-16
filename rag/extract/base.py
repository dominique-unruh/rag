"""Extraction plugin API: the ``Extractor`` base class and ``Store`` handle (PLAN §4.1).

A plugin subclasses :class:`Extractor` and implements :meth:`Extractor.run`. The
framework derives the plugin's ``source_type`` from its subdirectory name, opens
a :class:`Store` scoped to that id, and calls ``run``. All version handling and
change detection live in the framework (in :meth:`Store.upsert`), not in plugins.
"""

from __future__ import annotations

import enum
from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ..secrets import Secrets


class UpsertStatus(enum.Enum):
    """Outcome of an :meth:`Store.upsert` call."""

    CREATED = "created"
    """A new document was inserted at version 1."""

    UPDATED = "updated"
    """An existing document changed; its version was bumped."""

    UNCHANGED = "unchanged"
    """Content and metadata were identical; nothing was written."""


@dataclass(frozen=True)
class UpsertResult:
    """What :meth:`Store.upsert` did, returned to the plugin."""

    status: UpsertStatus
    """Whether the document was created, updated or left unchanged."""

    version: int
    """The document's version after the call."""


class Store(ABC):
    """System-facing API given to a plugin, scoped to one ``source_type``.

    Every method is implicitly restricted to the plugin's own ``source_type``, so
    a plugin can neither see nor modify another plugin's documents. Concrete
    implementations (e.g. :class:`~rag.extract.store.SqliteStore`) back this with
    the database and blob store.
    """

    @abstractmethod
    def upsert(
        self,
        doc_id: str,
        content: bytes,
        metadata: Mapping[str, Any],
        *,
        transient: Mapping[str, Any] | None = None,
        exist_ok: bool = True,
        missing_ok: bool = True,
    ) -> UpsertResult:
        """Create or update a document.

        ``metadata`` is the **stable** (content-defining) metadata; ``transient``
        is optional informational state (IMAP folder, read flag, …) that must not
        affect the version.

        The framework hashes ``content``, compares content + stable ``metadata``
        against the stored row, and inserts (version 1) if new, bumps the version
        if content or stable metadata changed, or leaves the version alone if only
        the transient metadata (or nothing) changed. Transient metadata is always
        written. The returned :class:`UpsertResult` reports the outcome *with
        respect to the versioned content*: a transient-only change reports
        :attr:`UpsertStatus.UNCHANGED`.

        ``exist_ok=False`` raises :class:`~rag.errors.DocumentExistsError` if the
        document already exists; ``missing_ok=False`` raises
        :class:`~rag.errors.DocumentMissingError` if it does not yet exist. Both
        default to a permissive upsert.
        """

    @abstractmethod
    def delete(self, doc_id: str, *, missing_ok: bool = True) -> bool:
        """Delete a document, cascading to its translations and embeddings.

        Returns whether a row existed. ``missing_ok=False`` raises
        :class:`~rag.errors.DocumentMissingError` when the document is absent.
        """

    @abstractmethod
    def exists(self, doc_id: str) -> bool:
        """Return whether a document with this id currently exists."""

    @abstractmethod
    def list_ids(
        self,
        *,
        stable: Mapping[str, Any] | None = None,
        transient: Mapping[str, Any] | None = None,
    ) -> Iterable[str]:
        """Return current ``doc_id``s for this ``source_type``, optionally filtered.

        ``stable`` / ``transient`` are metadata filters against the respective
        JSONB column. Each maps a metadata key to either a scalar (equality) or a
        list/tuple/set (membership, i.e. ``IN``); all given conditions must hold
        (AND). With no filters, every id for this source type is returned.

        Plugins use the unfiltered form to detect vanished documents, and the
        filtered form to find documents by metadata (e.g. all mail whose transient
        ``folder`` is now excluded).
        """

    @abstractmethod
    def get_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Return the stored **stable** metadata for ``doc_id``, or ``None``."""

    @abstractmethod
    def get_transient(self, doc_id: str) -> dict[str, Any] | None:
        """Return the stored **transient** metadata for ``doc_id``, or ``None``."""


class Extractor(ABC):
    """A data-source plugin. One subclass per source type.

    The ``source_type`` is not declared here — the framework derives it from the
    plugin's subdirectory name (e.g. ``extract/plugins/email/`` -> ``"email"``)
    and enforces it. The :class:`Store` handed to :meth:`run` is already bound to
    that id.
    """

    @abstractmethod
    def run(
        self,
        store: Store,
        state_dir: Path,
        config: Mapping[str, Any],
        secrets: Secrets,
        **options: Any,
    ) -> None:
        """Fetch from the source and reconcile it into ``store``.

        The plugin decides what "reconcile" means: upsert everything it finds and
        delete what has vanished. Parameters:

        * ``store`` — the scoped :class:`Store` (the only path to the system).
        * ``state_dir`` — a private, persistent directory for this plugin's own
          bookkeeping across runs (opaque to the system; created by the caller).
        * ``config`` — this plugin's subtree of the YAML config.
        * ``secrets`` — the shared secret resolver (for passwords/API keys).
        * ``options`` — plugin-specific CLI arguments.
        """

    def source_url(
        self, doc_id: str, stable: Mapping[str, Any], transient: Mapping[str, Any]
    ) -> str | None:
        """Return a URL linking to the original source of a document, or ``None``.

        Given a document's id and its stable/transient metadata, produce a URL
        that opens the original item (e.g. the message in a mail client). The
        default is ``None`` (no link); plugins override when a meaningful URL
        exists. Not abstract — a plugin may legitimately have no source URL.
        """
        return None
