"""Retrieval for chat (PLAN §7).

For a user query, search each selected embedding model's index (dense via the
backend, sparse via local BM25 term vectors), **fuse** the ranked lists with
Reciprocal Rank Fusion, and reconstruct the text of the winning chunks so the LLM
can read them.

Chunk text is *not* stored in the vector payload (only identity is), so we rebuild
it deterministically: a point's payload names its ``part_id`` and ``chunk`` index,
we fetch that part's translated Markdown blob and re-chunk it with the same
parameters the embed pass used — yielding the exact chunk that was embedded.
"""

from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from ..blobs import BlobStore
from ..config import Config
from ..embed.backends import embed_texts
from ..embed.chunk import chunk_text
from ..embed.models import ModelSpec
from ..embed.sparse import sparse_vector
from ..embed.vectorstore import SearchHit, VectorStore
from ..secrets import Secrets

# RRF damping constant. The standard value (Cormack et al. 2009); larger values
# flatten the contribution of top ranks, smaller ones sharpen it.
_RRF_K: int = 60

# How many hits to pull from each model before fusion. A few times the final
# top_k so fusion has enough overlap to work with, but bounded for a CPU box.
_PER_MODEL_MULTIPLIER: int = 4
_MIN_PER_MODEL: int = 20

# Identity of one embedded chunk across models (dense + sparse point ids differ,
# but they describe the same logical chunk — this tuple is what fusion keys on).
ChunkKey = tuple[str, str, int, int]


@dataclass(frozen=True)
class RetrievedChunk:
    """One chunk returned to the chat layer, with reconstructed text and citation."""

    source_type: str
    """The chunk's document source_type (e.g. ``email``)."""

    doc_id: str
    """The chunk's document id."""

    part_id: int
    """`translation_parts.id` of the part this chunk came from."""

    chunk: int
    """The chunk's index within its part."""

    text: str
    """The reconstructed chunk text (what the LLM reads)."""

    title: str | None
    """The document's title, for citation (may be ``None``)."""

    date: str | None
    """The document's date, for citation (may be ``None``)."""

    url: str | None
    """Link to the original source (from the extractor's ``source_url``), or ``None``."""

    score: float
    """The fused RRF score (higher = more relevant across models)."""


def _hit_key(payload: dict[str, object]) -> ChunkKey | None:
    """Return the :data:`ChunkKey` for a hit payload, or ``None`` if it is malformed."""
    try:
        return (
            str(payload["source_type"]),
            str(payload["doc_id"]),
            int(payload["part_id"]),  # type: ignore[arg-type]
            int(payload["chunk"]),  # type: ignore[arg-type]
        )
    except (KeyError, TypeError, ValueError):
        # A point missing the identity payload can't be reconstructed — skip it.
        return None


def rrf_fuse(ranked_lists: Iterable[Sequence[ChunkKey]], k: int = _RRF_K) -> list[tuple[ChunkKey, float]]:
    """Fuse several ranked lists of chunk keys via Reciprocal Rank Fusion.

    Each list is one model's hits in descending relevance. A key's fused score is
    ``sum(1 / (k + rank))`` over the lists it appears in (rank is 0-based), so a
    chunk ranked highly by several models beats one ranked highly by just one.
    Returns ``(key, score)`` pairs sorted by score descending.
    """
    scores: dict[ChunkKey, float] = {}
    for ranked in ranked_lists:
        for rank, key in enumerate(ranked):
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


def _source_url(
    config: Config,
    source_type: str,
    doc_id: str,
    stable: dict[str, object],
    transient: dict[str, object],
) -> str | None:
    """Ask the owning extractor for a link to the original document, or ``None``.

    Mirrors the db-browser: loads the plugin for ``source_type`` and calls its
    ``source_url``, passing the plugin's live config subtree so path-derived links
    stay current. Any failure (unknown plugin, plugin error) degrades to no link —
    a chat citation must never break because a plugin's URL logic raised.
    """
    # Imported lazily to keep the module import light and avoid cycles.
    from ..extract.discovery import load_plugin

    try:
        return load_plugin(source_type)().source_url(
            doc_id, stable, transient, config.plugin_config(source_type)
        )
    except Exception:
        return None


def _chunk_at(text: str, size: int, overlap: int, index: int) -> str | None:
    """Return the ``index``-th chunk of ``text`` under the given params, or ``None``.

    Mirrors the embed pass exactly (same :func:`chunk_text`), so the chunk lines up
    with the embedded point. ``None`` if the index is out of range (e.g. the part's
    content changed since embedding — the caller then skips that hit).
    """
    chunks = chunk_text(text, size, overlap)
    if 0 <= index < len(chunks):
        return chunks[index]
    return None


class Retriever:
    """Runs multi-model search + RRF fusion + text reconstruction for chat."""

    def __init__(
        self,
        vectors: VectorStore,
        conn: sqlite3.Connection,
        blobs: BlobStore,
        *,
        config: Config,
        chunk_size: int,
        overlap: int,
        secrets: Secrets,
        verbose: bool = False,
    ) -> None:
        """Bind the retriever to its stores and the embed pass's chunk parameters."""
        # Full config — needed so source links reflect the live folder paths/modes.
        self._config: Config = config
        # The open vector store (server or embedded-local) to search.
        self._vectors: VectorStore = vectors
        # Metadata connection, for part blob hashes and document titles/dates.
        self._conn: sqlite3.Connection = conn
        # Blob store holding the translated Markdown of each part.
        self._blobs: BlobStore = blobs
        # Chunking parameters — MUST match those the embed pass used.
        self._chunk_size: int = chunk_size
        self._overlap: int = overlap
        # Secret resolver for dense backends that need an API key.
        self._secrets: Secrets = secrets
        # Print per-model retrieval progress to stderr when set.
        self._verbose: bool = verbose
        # Per-query caches so repeated parts/documents are read once.
        self._part_text: dict[int, str | None] = {}
        # (title, date, source_url) per (source_type, doc_id).
        self._doc_meta: dict[tuple[str, str], tuple[str | None, str | None, str | None]] = {}

    def _log(self, message: str) -> None:
        """Print a verbose progress line to stderr (no-op unless verbose)."""
        if self._verbose:
            print(message, file=sys.stderr)

    def _search_model(self, spec: ModelSpec, query: str, limit: int) -> list[SearchHit]:
        """Search one model's collection for ``query``; empty if it has no index yet."""
        if not self._vectors.has_collection(spec.collection):
            # Model configured but never embedded — nothing to contribute.
            self._log(f"  {spec.id} ({spec.type}): no index yet, skipped")
            return []
        # Time the search (a dense query also embeds the query text — the slow bit).
        started = time.monotonic()
        if spec.type == "sparse":
            counts = sparse_vector(query)
            hits = self._vectors.search_sparse(
                spec.collection, list(counts.keys()), list(counts.values()), limit
            )
        else:
            # Dense: embed the query (with the model's query instruction) then search.
            vector = embed_texts(spec, [query], is_query=True, secrets=self._secrets)[0]
            hits = self._vectors.search_dense(spec.collection, vector, limit)
        elapsed = time.monotonic() - started
        self._log(f"  {spec.id} ({spec.type}): {len(hits)} hit(s) in {elapsed:.2f}s")
        return hits

    def _part_blob_text(self, part_id: int) -> str | None:
        """Return the decoded Markdown of part ``part_id`` (cached), or ``None`` if gone."""
        if part_id not in self._part_text:
            row = self._conn.execute(
                "SELECT blob_hash FROM translation_parts WHERE id = ?", (part_id,)
            ).fetchone()
            # The part may have been superseded/deleted since the point was written.
            if row is None:
                self._part_text[part_id] = None
            else:
                self._part_text[part_id] = self._blobs.get(row["blob_hash"]).decode(
                    "utf-8", errors="replace"
                )
        return self._part_text[part_id]

    def _document_meta(
        self, source_type: str, doc_id: str
    ) -> tuple[str | None, str | None, str | None]:
        """Return ``(title, date, url)`` for a document (cached), for citations.

        ``url`` is the extractor's ``source_url`` (a link to the original document,
        e.g. a mail-link for email), computed from the document's stable/transient
        metadata; ``None`` when the plugin offers no link.
        """
        key = (source_type, doc_id)
        if key not in self._doc_meta:
            row = self._conn.execute(
                "SELECT title, date, json(stable_metadata) AS stable, "
                "json(transient_metadata) AS transient "
                "FROM documents WHERE source_type = ? AND doc_id = ?",
                (source_type, doc_id),
            ).fetchone()
            if row is None:
                self._doc_meta[key] = (None, None, None)
            else:
                # Decode the metadata blobs and ask the plugin for a source link.
                stable = json.loads(row["stable"]) if row["stable"] else {}
                transient = json.loads(row["transient"]) if row["transient"] else {}
                url = _source_url(self._config, source_type, doc_id, stable, transient)
                self._doc_meta[key] = (row["title"], row["date"], url)
        return self._doc_meta[key]

    def retrieve(
        self, query: str, specs: Sequence[ModelSpec], top_k: int
    ) -> list[RetrievedChunk]:
        """Retrieve, fuse and reconstruct the ``top_k`` best chunks for ``query``.

        ``specs`` are the embedding models to query (a mix of dense and sparse gives
        hybrid semantic + lexical recall). Each model's hits are fused via RRF; the
        winners are reconstructed to text (skipping any whose part/chunk no longer
        exists). Fewer than ``top_k`` chunks may be returned.
        """
        # Reset per-query caches so a long session doesn't grow unbounded.
        self._part_text = {}
        self._doc_meta = {}
        per_model = max(_MIN_PER_MODEL, top_k * _PER_MODEL_MULTIPLIER)
        self._log(
            f"retrieving: querying {len(specs)} model(s) for up to {per_model} hit(s) each..."
        )

        # Collect each model's hits as an ordered list of chunk keys (for fusion),
        # remembering one representative payload per key for reconstruction.
        ranked_lists: list[list[ChunkKey]] = []
        payloads: dict[ChunkKey, dict[str, object]] = {}
        for spec in specs:
            ranked: list[ChunkKey] = []
            for hit in self._search_model(spec, query, per_model):
                key = _hit_key(hit.payload)
                if key is None:
                    continue
                ranked.append(key)
                payloads.setdefault(key, hit.payload)
            ranked_lists.append(ranked)

        results: list[RetrievedChunk] = []
        for key, score in rrf_fuse(ranked_lists):
            source_type, doc_id, part_id, chunk_index = key
            text = self._part_blob_text(part_id)
            if text is None:
                continue
            chunk = _chunk_at(text, self._chunk_size, self._overlap, chunk_index)
            if chunk is None:
                # Part content changed since embedding — its chunk index is stale.
                continue
            title, date, url = self._document_meta(source_type, doc_id)
            results.append(
                RetrievedChunk(
                    source_type=source_type,
                    doc_id=doc_id,
                    part_id=part_id,
                    chunk=chunk_index,
                    text=chunk,
                    title=title,
                    date=date,
                    url=url,
                    score=score,
                )
            )
            if len(results) >= top_k:
                break
        return results
