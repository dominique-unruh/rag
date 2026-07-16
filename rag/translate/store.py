"""Database access for the translation pipeline (PLAN §5.2).

Selects source documents that need translating and writes translation sets +
parts. Kept separate from the pipeline logic so the SQL lives in one place.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any

from ..blobs import BlobStore


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


def _canonical_json(value: Any) -> str:
    """Serialise ``value`` to a stable, compact JSON string."""
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


@dataclass(frozen=True)
class SourceDocument:
    """A source document row selected for translation."""

    source_type: str
    """Owning plugin id."""

    doc_id: str
    """Document id."""

    version: int
    """Current source version (recorded on the produced translation set)."""

    content: bytes
    """Raw source bytes (from the blob store)."""

    stable_metadata: dict[str, Any]
    """Stable metadata of the document."""

    transient_metadata: dict[str, Any]
    """Transient metadata of the document."""


class TranslationStore:
    """Reads source documents and writes translation sets/parts."""

    def __init__(self, conn: sqlite3.Connection, blobs: BlobStore) -> None:
        """Bind to a DB connection and the blob store."""
        # Live SQLite connection (writer).
        self._conn: sqlite3.Connection = conn
        # Blob store for source reads and translation writes.
        self._blobs: BlobStore = blobs

    def select_documents(
        self,
        *,
        source_type: str | None = None,
        doc_id: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> Iterator[SourceDocument]:
        """Yield documents to translate.

        By default only documents that **need** translating are yielded: those
        with no active translation set, or whose active set was made from an older
        source version (stale). ``force=True`` yields all selected documents
        regardless. ``source_type`` / ``doc_id`` restrict the selection; ``limit``
        caps the count.
        """
        # Left-join the active set so we can tell what is missing or stale.
        clauses: list[str] = []
        params: list[Any] = []
        if source_type is not None:
            clauses.append("d.source_type = ?")
            params.append(source_type)
        if doc_id is not None:
            clauses.append("d.doc_id = ?")
            params.append(doc_id)
        if not force:
            # Needs translating: no active set, or active set is from an older
            # source version than the document currently has.
            clauses.append("(ts.source_version IS NULL OR ts.source_version <> d.version)")
        where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
        limit_sql = " LIMIT ?" if limit is not None else ""
        if limit is not None:
            params.append(limit)
        query = (
            "SELECT d.source_type, d.doc_id, d.version, d.blob_hash, "
            "json(d.stable_metadata) AS stable, json(d.transient_metadata) AS transient "
            "FROM documents d "
            "LEFT JOIN translation_sets ts "
            "  ON ts.source_type = d.source_type AND ts.doc_id = d.doc_id AND ts.is_active = 1"
            f"{where} ORDER BY d.source_type, d.doc_id{limit_sql}"
        )
        for row in self._conn.execute(query, params):
            yield SourceDocument(
                source_type=row["source_type"],
                doc_id=row["doc_id"],
                version=int(row["version"]),
                content=self._blobs.get(row["blob_hash"]),
                stable_metadata=json.loads(row["stable"]),
                transient_metadata=json.loads(row["transient"]),
            )

    def write_set(
        self,
        *,
        source_type: str,
        doc_id: str,
        source_version: int,
        translator_name: str,
        git_rev: str,
        parts: Sequence[tuple[list[str], str, Mapping[str, Any]]],
    ) -> int:
        """Write one active translation set with its parts, atomically.

        Deactivates any existing active set for the document, inserts the new set
        (marked active) and its parts (each part's text stored as a Markdown
        blob), all in a single transaction (PLAN §5). Returns the new set id.
        """
        set_metadata = _canonical_json({"translator": translator_name})
        now = _now_iso()
        with self._conn:  # one transaction: set + all its parts
            # Only one active set per document — clear the previous one first.
            self._conn.execute(
                "UPDATE translation_sets SET is_active = 0 "
                "WHERE source_type = ? AND doc_id = ? AND is_active = 1",
                (source_type, doc_id),
            )
            cursor = self._conn.execute(
                "INSERT INTO translation_sets "
                "(source_type, doc_id, source_version, metadata, translated_at, git_rev, is_active) "
                "VALUES (?, ?, ?, jsonb(?), ?, ?, 1)",
                (source_type, doc_id, source_version, set_metadata, now, git_rev),
            )
            set_id = int(cursor.lastrowid)
            for sub_doc_id, text, metadata in parts:
                # Store the translated text as an immutable blob.
                blob_hash = self._blobs.put(text.encode("utf-8"))
                self._conn.execute(
                    "INSERT INTO translation_parts (set_id, sub_doc_id, blob_hash, metadata) "
                    "VALUES (?, jsonb(?), ?, jsonb(?))",
                    (set_id, _canonical_json(list(sub_doc_id)), blob_hash,
                     _canonical_json(dict(metadata))),
                )
        return set_id
