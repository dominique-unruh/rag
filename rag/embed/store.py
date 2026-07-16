"""Database bookkeeping for embedding (PLAN §6.2).

Selects documents whose active translation set needs embedding for a given model
(missing or stale), and records the per-``(part, model)`` embeddings rows.
"""

from __future__ import annotations

import json
import sqlite3
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime, timezone


def _now_iso() -> str:
    """Return the current UTC time as an ISO-8601 string."""
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class PartRecord:
    """One part of an active translation set to be embedded."""

    part_id: int
    """`translation_parts.id`."""

    sub_doc_id: list[str]
    """The part's sub_doc_id."""

    blob_hash: str
    """Reference to the part's translated Markdown blob."""


@dataclass(frozen=True)
class DocRecord:
    """A document (its active set) to embed for one model."""

    source_type: str
    doc_id: str
    set_id: int
    parts: list[PartRecord]


class EmbedStore:
    """Reads what needs embedding and records what was embedded, for one model."""

    def __init__(self, conn: sqlite3.Connection, model_id: str) -> None:
        """Bind to a connection and the embedding ``model_id`` being processed."""
        self._conn: sqlite3.Connection = conn
        self._model: str = model_id

    def select_docs(
        self,
        *,
        source_type: str | None = None,
        doc_id: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> Iterator[DocRecord]:
        """Yield documents whose active set needs embedding for this model.

        A document needs embedding when any part of its active set has no
        embeddings row for this model referencing that set (missing or stale).
        ``force`` yields every selected document regardless. Filters restrict the
        selection; ``limit`` caps the count.
        """
        clauses = ["ts.is_active = 1"]
        params: list[object] = []
        if source_type is not None:
            clauses.append("ts.source_type = ?")
            params.append(source_type)
        if doc_id is not None:
            clauses.append("ts.doc_id = ?")
            params.append(doc_id)
        where = " AND ".join(clauses)
        active_sets = self._conn.execute(
            f"SELECT id, source_type, doc_id FROM translation_sets ts "
            f"WHERE {where} ORDER BY ts.source_type, ts.doc_id",
            params,
        ).fetchall()

        yielded = 0
        for srow in active_sets:
            set_id = int(srow["id"])
            part_rows = self._conn.execute(
                "SELECT tp.id AS part_id, json(tp.sub_doc_id) AS sub, tp.blob_hash, "
                "(SELECT COUNT(*) FROM embeddings e "
                " WHERE e.part_id = tp.id AND e.model = ? AND e.set_id = ?) AS embedded "
                "FROM translation_parts tp WHERE tp.set_id = ? ORDER BY tp.part_key",
                (self._model, set_id, set_id),
            ).fetchall()
            # Needs embedding if forced, or any part is not yet embedded for this
            # model against the *current* set.
            needs = force or any(int(p["embedded"]) == 0 for p in part_rows)
            if not needs:
                continue
            parts = [
                PartRecord(int(p["part_id"]), json.loads(p["sub"]), p["blob_hash"])
                for p in part_rows
            ]
            yield DocRecord(srow["source_type"], srow["doc_id"], set_id, parts)
            yielded += 1
            if limit is not None and yielded >= limit:
                return

    def record(self, doc: DocRecord) -> None:
        """Record that ``doc``'s parts are now embedded for this model.

        Replaces any prior embeddings rows for this document + model (covering the
        old set's parts) with rows for the current set's parts, in one transaction.
        """
        now = _now_iso()
        with self._conn:
            # Drop stale rows for this document+model (old set's parts included).
            self._conn.execute(
                "DELETE FROM embeddings WHERE source_type = ? AND doc_id = ? AND model = ?",
                (doc.source_type, doc.doc_id, self._model),
            )
            for part in doc.parts:
                self._conn.execute(
                    "INSERT INTO embeddings "
                    "(source_type, doc_id, set_id, part_id, model, embedded_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (doc.source_type, doc.doc_id, doc.set_id, part.part_id, self._model, now),
                )
