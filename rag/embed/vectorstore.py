"""Vector store wrapper over embedded Qdrant (PLAN §6).

One Qdrant store (a local directory) holds a separate **collection per embedding
model**. Embedded Qdrant takes an exclusive lock on its directory, matching the
"embed holds the vector store for its run" concurrency policy (PLAN §9).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

from qdrant_client import QdrantClient, models

from ..errors import VectorStoreBusyError

# Map our config distance names to Qdrant's enum.
_DISTANCES: dict[str, models.Distance] = {
    "cosine": models.Distance.COSINE,
    "dot": models.Distance.DOT,
    "euclid": models.Distance.EUCLID,
}

# Named sparse vector used for sparse collections (Qdrant sparse vectors are
# always named). IDF weighting is applied by the collection at query time.
SPARSE_VECTOR_NAME: str = "text"


class VectorStore:
    """Thin wrapper around an embedded Qdrant client, one collection per model."""

    def __init__(self, path: Path) -> None:
        """Open (or create) the embedded Qdrant store rooted at ``path``.

        Embedded Qdrant takes an exclusive lock on ``path``; if another process
        already holds it, raise a clear :class:`VectorStoreBusyError` instead of
        leaking Qdrant's internal ``RuntimeError``/lock traceback.
        """
        # Directory is created by Qdrant; ensure the parent exists first.
        path.mkdir(parents=True, exist_ok=True)
        try:
            self._client: QdrantClient = QdrantClient(path=str(path))
        except RuntimeError as exc:
            # Qdrant's local client raises a bare RuntimeError when the directory
            # is already locked by another client — turn it into an actionable one.
            raise VectorStoreBusyError(
                f"the vector store at {path} is in use by another process "
                f"(embedded Qdrant allows only one at a time). Close the db-browser "
                f"or any other running 'rag embed'/chat and try again."
            ) from exc

    def ensure_collection(self, collection: str, *, kind: str, dim: int, distance: str) -> None:
        """Create ``collection`` for a dense or sparse model if it is absent.

        ``kind`` is ``"dense"`` or ``"sparse"``. A dense collection is checked for
        a matching vector size on re-open (mixing dims corrupts retrieval, PLAN
        §6); a sparse collection uses a named sparse vector with an IDF modifier.
        """
        if self._client.collection_exists(collection):
            if kind == "dense":
                info = self._client.get_collection(collection)
                existing = info.config.params.vectors.size  # type: ignore[union-attr]
                if existing != dim:
                    raise ValueError(
                        f"collection {collection!r} has dim {existing}, expected {dim}; "
                        "use a fresh collection for a different model/dim"
                    )
            return
        if kind == "sparse":
            # Sparse vectors: no fixed size; IDF applied at query time.
            self._client.create_collection(
                collection,
                vectors_config={},
                sparse_vectors_config={
                    SPARSE_VECTOR_NAME: models.SparseVectorParams(modifier=models.Modifier.IDF)
                },
            )
        else:
            metric = _DISTANCES.get(distance.lower(), models.Distance.COSINE)
            self._client.create_collection(
                collection,
                vectors_config=models.VectorParams(size=dim, distance=metric),
            )

    def replace_document(
        self, collection: str, source_type: str, doc_id: str, points: list[models.PointStruct]
    ) -> None:
        """Delete a document's existing points, then upsert its new ``points``.

        Deleting first means a re-embed (e.g. after the active set changed) never
        leaves stale points from the old set behind.
        """
        self._delete_document(collection, source_type, doc_id)
        if points:
            self._client.upsert(collection, points=points)

    def _delete_document(self, collection: str, source_type: str, doc_id: str) -> None:
        """Delete all points of one document from ``collection`` (by payload filter)."""
        self._client.delete(
            collection,
            points_selector=models.FilterSelector(filter=_doc_filter(source_type, doc_id)),
        )

    def delete_document(self, collection: str, source_type: str, doc_id: str) -> None:
        """Public: remove all of a document's points from ``collection`` (GC/cleanup)."""
        if self._client.collection_exists(collection):
            self._delete_document(collection, source_type, doc_id)

    def count(self, collection: str) -> int:
        """Return the number of points in ``collection`` (for tests/stats)."""
        return self._client.count(collection).count

    def close(self) -> None:
        """Release the Qdrant store (and its directory lock)."""
        self._client.close()


def _doc_filter(source_type: str, doc_id: str) -> models.Filter:
    """Build a Qdrant payload filter matching one document's points."""
    return models.Filter(
        must=[
            models.FieldCondition(key="source_type", match=models.MatchValue(value=source_type)),
            models.FieldCondition(key="doc_id", match=models.MatchValue(value=doc_id)),
        ]
    )
