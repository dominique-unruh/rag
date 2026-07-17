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

from ..errors import ConfigError


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

    def ensure_fingerprint(self, fingerprint: str) -> None:
        """Guard against incompatible config changes for this model (PLAN §6).

        On first use, records the model's config ``fingerprint``. On later runs,
        raises :class:`ConfigError` if the current fingerprint differs — meaning
        the model's index would be polluted with incompatible vectors. The fix is
        to use a new model id, or wipe this model's collection + rows and re-embed.
        """
        row = self._conn.execute(
            "SELECT fingerprint FROM embedding_models WHERE model = ?", (self._model,)
        ).fetchone()
        if row is None:
            # First run for this model — remember its config.
            with self._conn:
                self._conn.execute(
                    "INSERT INTO embedding_models (model, fingerprint, created_at) "
                    "VALUES (?, ?, ?)",
                    (self._model, fingerprint, _now_iso()),
                )
            return
        if row["fingerprint"] != fingerprint:
            raise ConfigError(
                f"embedding model {self._model!r} config changed since it was first "
                f"embedded (incompatible):\n  stored:  {row['fingerprint']}\n  current: "
                f"{fingerprint}\nUse a new model id, or wipe this model's collection and "
                f"its embeddings rows to re-embed from scratch."
            )

    def _active_set_filter(
        self, source_type: str | None, doc_id: str | None
    ) -> tuple[str, list[object]]:
        """Build the WHERE clause + params selecting the active sets to consider.

        Shared by :meth:`select_docs` and :meth:`count_docs` so the two never
        drift apart on which sets they look at. Always restricts to active sets;
        optionally narrows by ``source_type`` / ``doc_id``.
        """
        clauses = ["ts.is_active = 1"]
        params: list[object] = []
        if source_type is not None:
            clauses.append("ts.source_type = ?")
            params.append(source_type)
        if doc_id is not None:
            clauses.append("ts.doc_id = ?")
            params.append(doc_id)
        return " AND ".join(clauses), params

    def count_docs(
        self,
        *,
        source_type: str | None = None,
        doc_id: str | None = None,
        limit: int | None = None,
        force: bool = False,
    ) -> int:
        """Return how many documents :meth:`select_docs` would yield for the same args.

        Runs the "needs embedding" predicate as a single SQL count (no chunking or
        blob reads) so a caller can show ``n/total`` progress before the run. Uses
        the same active-set filter as :meth:`select_docs`; ``force`` counts every
        active set, otherwise only those with a part not yet embedded for this
        model against the current set. ``limit`` caps the result to match the run.
        """
        where, params = self._active_set_filter(source_type, doc_id)
        if force:
            # Force re-embeds every active set, so the count is just those sets.
            sql = f"SELECT COUNT(*) AS n FROM translation_sets ts WHERE {where}"
            count_params: list[object] = params
        else:
            # A set needs embedding if it has a part with no embeddings row for
            # this model against that set. Mirror select_docs' per-part predicate.
            sql = (
                f"SELECT COUNT(*) AS n FROM translation_sets ts WHERE {where} "
                f"AND EXISTS (SELECT 1 FROM translation_parts tp WHERE tp.set_id = ts.id "
                f"AND NOT EXISTS (SELECT 1 FROM embeddings e "
                f"WHERE e.part_id = tp.id AND e.model = ? AND e.set_id = ts.id))"
            )
            # The model param binds the inner embeddings check.
            count_params = [*params, self._model]
        count = int(self._conn.execute(sql, count_params).fetchone()["n"])
        # The run stops at limit, so the reachable total is capped to match.
        return min(count, limit) if limit is not None else count

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
        where, params = self._active_set_filter(source_type, doc_id)
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
