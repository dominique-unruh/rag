"""Embedding pipeline (PLAN §6.3).

Runs one model's embed pass: for each document whose active translation set needs
embedding, chunk its parts, embed the chunks, replace the document's points in
that model's collection, and record the bookkeeping.
"""

from __future__ import annotations

import sys
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass

from qdrant_client import models

from ..blobs import BlobStore
from ..config import Config
from ..db import Database
from ..interrupt import interrupt_requested
from ..secrets import Secrets
from .backends import embed_texts
from .chunk import chunk_text
from .models import ModelSpec, chunk_params, get_model, model_fingerprint
from .sparse import sparse_vector
from .store import DocRecord, EmbedStore
from .vectorstore import SPARSE_VECTOR_NAME, open_vector_store

# Filenames/subdirs under the storage directory.
_DB_FILENAME: str = "metadata.db"
_BLOBS_SUBDIR: str = "blobs"

# Namespace for deterministic point ids (so re-embedding is idempotent).
_POINT_NS: uuid.UUID = uuid.UUID("6f4e2b90-1c2d-4e6a-9b7f-0a1b2c3d4e5f")

# A function that maps a list of texts to a list of vectors (injectable in tests).
EmbedFn = Callable[[list[str]], list[list[float]]]


@dataclass(frozen=True)
class EmbedOptions:
    """Options controlling an embed run."""

    model: str
    """The embedding model id to run (its own index)."""

    source_type: str | None = None
    """Restrict to one source_type."""

    doc_id: str | None = None
    """Restrict to one document id."""

    limit: int | None = None
    """Max number of documents to process."""

    force: bool = False
    """Re-embed even documents already up to date for this model."""

    verbose: bool = False
    """Print one progress line per embedded document."""


def _format_duration(seconds: float) -> str:
    """Format an elapsed ``seconds`` compactly, e.g. ``45s``, ``5m30s``, ``1h2m3s``.

    Only the non-zero leading units are shown; seconds are always shown (so a
    sub-minute run reads ``42s`` and a fast one ``0s``).
    """
    total = int(round(seconds))
    hours, rem = divmod(total, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours}h{minutes}m{secs}s"
    if minutes:
        return f"{minutes}m{secs}s"
    return f"{secs}s"


def _point_id(model_id: str, part_id: int, chunk_index: int) -> str:
    """Return a deterministic point id for a (model, part, chunk).

    Deterministic so re-embedding the same content overwrites the same points.
    """
    return str(uuid.uuid5(_POINT_NS, f"{model_id}:{part_id}:{chunk_index}"))


def _chunk_vector(spec: ModelSpec, chunks: list[str], embed: EmbedFn) -> list[object]:
    """Return the per-chunk vectors for ``chunks`` in the model's representation.

    Dense: fixed-length float vectors from the backend. Sparse: locally-computed
    term-frequency sparse vectors (named for Qdrant), no external call.
    """
    if spec.type == "sparse":
        vectors: list[object] = []
        for chunk in chunks:
            counts = sparse_vector(chunk)
            vectors.append(
                {SPARSE_VECTOR_NAME: models.SparseVector(
                    indices=list(counts.keys()), values=list(counts.values())
                )}
            )
        return vectors
    # Dense: one float vector per chunk from the backend.
    return list(embed(chunks))


def _chunk_document(
    doc: DocRecord, blobs: BlobStore, chunk_size: int, overlap: int
) -> list[tuple[PartRecord, list[str]]]:
    """Return each part of ``doc`` with its (possibly empty) list of text chunks.

    Chunking is cheap and done up front so the caller knows the total chunk count
    (for progress reporting) before the expensive embed call.
    """
    result: list[tuple[PartRecord, list[str]]] = []
    for part in doc.parts:
        text = blobs.get(part.blob_hash).decode("utf-8", errors="replace")
        result.append((part, chunk_text(text, chunk_size, overlap)))
    return result


def _embed_chunked(
    doc: DocRecord, chunked: list[tuple[PartRecord, list[str]]], spec: ModelSpec, embed: EmbedFn
) -> list[models.PointStruct]:
    """Embed the pre-chunked parts of ``doc``; return the Qdrant points to store."""
    points: list[models.PointStruct] = []
    for part, chunks in chunked:
        if not chunks:
            # Empty part (e.g. an image-only PDF) — no points, but the part is
            # still recorded as embedded so it is not reselected forever.
            continue
        vectors = _chunk_vector(spec, chunks, embed)
        for index, vector in enumerate(vectors):
            points.append(
                models.PointStruct(
                    id=_point_id(spec.id, part.part_id, index),
                    vector=vector,
                    payload={
                        "source_type": doc.source_type,
                        "doc_id": doc.doc_id,
                        "set_id": doc.set_id,
                        "part_id": part.part_id,
                        "chunk": index,
                    },
                )
            )
    return points


def run_embed(
    config: Config,
    options: EmbedOptions,
    *,
    db: Database | None = None,
    embed_fn: EmbedFn | None = None,
    secrets: Secrets | None = None,
) -> int:
    """Run the embed pass for one model; return the number of documents embedded.

    Opens the model's collection in the embedded vector store, iterates the
    documents whose active set needs embedding (per ``options``), embeds their
    parts and replaces their points, recording the bookkeeping. ``embed_fn`` may
    override the backend (used by tests).
    """
    storage = config.storage_dir
    spec = get_model(config.raw, options.model)
    chunk_size, overlap = chunk_params(config.raw)
    resolver = secrets if secrets is not None else Secrets(allow_prompt=False)
    # Default embedder calls the configured backend for this model.
    embed: EmbedFn = embed_fn or (lambda texts: embed_texts(spec, texts, secrets=resolver))

    blobs = BlobStore(storage / _BLOBS_SUBDIR)
    # One-time upgrade of any pre-marker blobs to the compressed format (no-op once
    # done). A pipeline run is a natural, single-writer moment to do this.
    blobs.migrate()
    # Open the configured vector store (Qdrant server if vector_store.url is set,
    # else the embedded-local store under storage/qdrant). Reuses the same secret
    # resolver so a server API key is resolved under the run's TTY/headless policy.
    vectors = open_vector_store(config, secrets=resolver)
    owns_db = db is None
    database = db if db is not None else Database.open(storage / _DB_FILENAME)
    try:
        store = EmbedStore(database.conn, spec.id)
        # Guard first: refuse if the model's incompatibility-defining config changed.
        store.ensure_fingerprint(model_fingerprint(spec))
        vectors.ensure_collection(spec.collection, kind=spec.type, dim=spec.dim, distance=spec.distance)
        # Precount how many documents will be embedded so verbose output can show
        # n/total progress. Only when verbose — it is a wasted query otherwise.
        total = (
            store.count_docs(
                source_type=options.source_type,
                doc_id=options.doc_id,
                limit=options.limit,
                force=options.force,
            )
            if options.verbose
            else 0
        )
        processed = 0
        for doc in store.select_docs(
            source_type=options.source_type,
            doc_id=options.doc_id,
            limit=options.limit,
            force=options.force,
        ):
            # Chunk up front (cheap) so we can report the chunk count before the
            # expensive embed call, and print the start line without a newline.
            chunked = _chunk_document(doc, blobs, chunk_size, overlap)
            if options.verbose:
                n_chunks = sum(len(chunks) for _, chunks in chunked)
                sub = ", ".join("/".join(p.sub_doc_id) or "(root)" for p in doc.parts)
                # processed is the count already done; this doc is the next one.
                print(
                    f"[{processed + 1}/{total}] "
                    f"embedding {doc.source_type}:{doc.doc_id} (set {doc.set_id}) "
                    f"-> {n_chunks} chunk(s) [{sub}] into {spec.id}...",
                    end="",
                    flush=True,
                    file=sys.stderr,
                )
            # Time the actual embedding + store so the Done line can report it.
            started = time.monotonic()
            points = _embed_chunked(doc, chunked, spec, embed)
            # Replace this document's points in the model's collection, then record.
            vectors.replace_document(spec.collection, doc.source_type, doc.doc_id, points)
            store.record(doc)
            processed += 1
            if options.verbose:
                print(f" Done ({_format_duration(time.monotonic() - started)}).", file=sys.stderr)
            # Checkpoint: this document's points + bookkeeping are committed here —
            # a safe point to honour a graceful interrupt.
            if interrupt_requested():
                print(f"interrupted after {processed} document(s).", file=sys.stderr)
                break
        return processed
    finally:
        vectors.close()
        if owns_db:
            database.close()
