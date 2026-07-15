"""SQLite-backed implementation of the extraction :class:`Store` (PLAN §4).

Binds the generic :class:`~rag.extract.base.Store` API to a specific
``source_type``, the database connection and the blob store. Owns version
handling and change detection so plugins never deal with either.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterable, Mapping
from datetime import datetime, timezone
from typing import Any

from ..blobs import BlobStore
from ..errors import DocumentExistsError, DocumentMissingError
from .base import Store, UpsertResult, UpsertStatus


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string for ``updated_at``."""
    # UTC keeps timestamps comparable regardless of the machine's timezone.
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(metadata: Mapping[str, Any]) -> str:
    """Serialise ``metadata`` to a stable JSON string.

    Keys are sorted and separators are compact so equal mappings always yield the
    same text — important for deterministic storage and change detection.
    """
    return json.dumps(metadata, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


class SqliteStore(Store):
    """A :class:`Store` scoped to one ``source_type`` over one connection."""

    def __init__(self, conn: sqlite3.Connection, blobs: BlobStore, source_type: str) -> None:
        """Bind the store to a connection, blob store and ``source_type``.

        ``source_type`` is fixed here (from the plugin's directory name); every
        query below filters on it, enforcing per-plugin isolation.
        """
        # Live SQLite connection (shared with the rest of the run).
        self._conn: sqlite3.Connection = conn
        # Content-addressed blob store for document source bytes.
        self._blobs: BlobStore = blobs
        # The plugin id this store is locked to; never taken from the plugin.
        self._source_type: str = source_type

    def _fetch(self, doc_id: str) -> sqlite3.Row | None:
        """Return the current row for ``doc_id`` in this source_type, or ``None``.

        Renders both JSONB metadata columns back to text so callers can parse
        them with :func:`json.loads`.
        """
        # Scoped by source_type so plugins are isolated from each other.
        cursor = self._conn.execute(
            "SELECT version, blob_hash, "
            "json(stable_metadata) AS stable_metadata, "
            "json(transient_metadata) AS transient_metadata "
            "FROM documents WHERE source_type = ? AND doc_id = ?",
            (self._source_type, doc_id),
        )
        return cursor.fetchone()

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
        """Insert or update ``doc_id``; see :meth:`Store.upsert` for semantics."""
        # Hash first so we can compare against any stored blob without writing.
        new_hash = BlobStore.hash_bytes(content)
        # Canonicalise both metadata sets; transient defaults to empty.
        new_stable_json = _canonical_json(metadata)
        new_transient_json = _canonical_json(transient or {})
        existing = self._fetch(doc_id)

        if existing is None:
            # --- fresh insert path ---------------------------------------
            if not missing_ok:
                # Plugin asserted the document should already exist.
                raise DocumentMissingError(
                    f"{self._source_type}:{doc_id} does not exist (missing_ok=False)"
                )
            # Persist the source bytes, then create the row at version 1.
            self._blobs.put(content)
            with self._conn:  # own transaction per document (PLAN §5)
                self._conn.execute(
                    "INSERT INTO documents (source_type, doc_id, version, blob_hash, "
                    "stable_metadata, transient_metadata, updated_at) "
                    "VALUES (?, ?, 1, ?, jsonb(?), jsonb(?), ?)",
                    (self._source_type, doc_id, new_hash, new_stable_json,
                     new_transient_json, _now_iso()),
                )
            return UpsertResult(UpsertStatus.CREATED, 1)

        # --- existing document path --------------------------------------
        if not exist_ok:
            # Plugin asserted this was a first-time insert.
            raise DocumentExistsError(
                f"{self._source_type}:{doc_id} already exists (exist_ok=False)"
            )

        old_version = int(existing["version"])
        # Compare canonically: re-serialise the stored stable metadata the same
        # way so key ordering never masquerades as a change. Transient metadata is
        # deliberately excluded from this comparison.
        old_stable_json = _canonical_json(json.loads(existing["stable_metadata"]))
        old_transient_json = _canonical_json(json.loads(existing["transient_metadata"]))
        content_changed = existing["blob_hash"] != new_hash
        stable_changed = old_stable_json != new_stable_json
        transient_changed = old_transient_json != new_transient_json

        if not content_changed and not stable_changed:
            # Versioned content is identical. Still refresh transient metadata if
            # it changed (folder moved, marked read, …) — without bumping version.
            if transient_changed:
                with self._conn:
                    self._conn.execute(
                        "UPDATE documents SET transient_metadata = jsonb(?), updated_at = ? "
                        "WHERE source_type = ? AND doc_id = ?",
                        (new_transient_json, _now_iso(), self._source_type, doc_id),
                    )
            # Reported UNCHANGED w.r.t. the pipeline regardless of transient writes.
            return UpsertResult(UpsertStatus.UNCHANGED, old_version)

        # Content or stable metadata changed: store the (possibly new) blob, bump
        # the version, and write both metadata sets.
        self._blobs.put(content)
        new_version = old_version + 1
        with self._conn:
            self._conn.execute(
                "UPDATE documents SET version = ?, blob_hash = ?, "
                "stable_metadata = jsonb(?), transient_metadata = jsonb(?), updated_at = ? "
                "WHERE source_type = ? AND doc_id = ?",
                (new_version, new_hash, new_stable_json, new_transient_json,
                 _now_iso(), self._source_type, doc_id),
            )
        return UpsertResult(UpsertStatus.UPDATED, new_version)

    def delete(self, doc_id: str, *, missing_ok: bool = True) -> bool:
        """Delete ``doc_id`` and cascade; see :meth:`Store.delete`."""
        if not self.exists(doc_id):
            if not missing_ok:
                # Plugin asserted the document should be present.
                raise DocumentMissingError(
                    f"{self._source_type}:{doc_id} does not exist (missing_ok=False)"
                )
            # Deleting an absent document is a no-op.
            return False
        # Deleting the documents row cascades to translation_sets ->
        # translation_parts -> embeddings via ON DELETE CASCADE (foreign_keys=ON).
        # (Vector-store point removal is handled by the embed module, which owns
        # the vector store; no embeddings exist during extraction-only runs.)
        with self._conn:
            self._conn.execute(
                "DELETE FROM documents WHERE source_type = ? AND doc_id = ?",
                (self._source_type, doc_id),
            )
        return True

    def exists(self, doc_id: str) -> bool:
        """Return whether ``doc_id`` exists in this source_type."""
        cursor = self._conn.execute(
            "SELECT 1 FROM documents WHERE source_type = ? AND doc_id = ? LIMIT 1",
            (self._source_type, doc_id),
        )
        return cursor.fetchone() is not None

    def list_ids(self) -> Iterable[str]:
        """Return all current ``doc_id``s for this source_type as a list."""
        cursor = self._conn.execute(
            "SELECT doc_id FROM documents WHERE source_type = ? ORDER BY doc_id",
            (self._source_type,),
        )
        # Materialise to a list so callers can iterate repeatedly / take len().
        return [row["doc_id"] for row in cursor.fetchall()]

    def get_metadata(self, doc_id: str) -> dict[str, Any] | None:
        """Return stored **stable** metadata for ``doc_id`` as a dict, or ``None``."""
        row = self._fetch(doc_id)
        if row is None:
            return None
        # Stored as JSONB; json() (in _fetch) rendered it back to text for parsing.
        return json.loads(row["stable_metadata"])

    def get_transient(self, doc_id: str) -> dict[str, Any] | None:
        """Return stored **transient** metadata for ``doc_id`` as a dict, or ``None``."""
        row = self._fetch(doc_id)
        if row is None:
            return None
        return json.loads(row["transient_metadata"])
