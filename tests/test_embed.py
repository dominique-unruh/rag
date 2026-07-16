"""Tests for module 3 (embedding): chunking, model registry, pipeline."""

from __future__ import annotations

from pathlib import Path

import pytest

from rag.blobs import BlobStore
from rag.config import Config
from rag.db import Database
from rag.embed.chunk import chunk_text
from rag.embed.models import chunk_params, get_model, load_models
from rag.embed.pipeline import EmbedOptions, run_embed
from rag.embed.vectorstore import VectorStore
from rag.errors import ConfigError
from rag.extract.store import SqliteStore
from rag.translate.store import TranslationStore

# Embedding config used by the pipeline tests (backend unused — embed_fn injected).
_EMB_CFG = {
    "chunk_size": 20,
    "chunk_overlap": 5,
    "models": [
        {"id": "test", "type": "dense", "backend": "openai",
         "api_base": "http://x", "model": "m", "dim": 4},
    ],
}
# A fake embedder: one fixed 4-d vector per text (dim matches the model).
_FAKE = lambda texts: [[1.0, 0.0, 0.0, 0.0] for _ in texts]  # noqa: E731


# --- chunking ------------------------------------------------------------
def test_chunk_text_basic() -> None:
    """Chunks cover the text with the requested size/overlap."""
    assert chunk_text("", 10, 2) == []
    assert chunk_text("abcdef", 10, 2) == ["abcdef"]
    # size 4, overlap 1 -> step 3 over 'abcdefg': 'abcd'(0-3), 'defg'(3-6) covers 'g'.
    assert chunk_text("abcdefg", 4, 1) == ["abcd", "defg"]


def test_chunk_text_validates() -> None:
    """Bad size/overlap combinations raise."""
    with pytest.raises(ValueError):
        chunk_text("x", 5, 5)


# --- model registry ------------------------------------------------------
def test_model_registry() -> None:
    """Models parse from config; unknown ids and bad configs raise."""
    raw = {"embedding": _EMB_CFG}
    assert set(load_models(raw)) == {"test"}
    assert chunk_params(raw) == (20, 5)
    spec = get_model(raw, "test")
    assert spec.dim == 4 and spec.backend == "openai" and spec.collection == "emb_test"
    with pytest.raises(ConfigError):
        get_model(raw, "nope")
    with pytest.raises(ConfigError):
        # dim is required for a dense model.
        load_models({"embedding": {"models": [{"id": "x", "backend": "openai"}]}})


# --- pipeline (embedded Qdrant + fake embedder) --------------------------
def _seed(storage: Path, text: str = "Hello world, this is the body text.") -> Database:
    """Create a storage dir with one document + active translation set."""
    db = Database.open(storage / "metadata.db")
    blobs = BlobStore(storage / "blobs")
    SqliteStore(db.conn, blobs, "email").upsert(
        "m1", b"raw", {"mime-type": "message/rfc822", "title": "Hi"}
    )
    TranslationStore(db.conn, blobs).write_set(
        source_type="email", doc_id="m1", source_version=1,
        translator_name="dispatch", git_rev="r", parts=[(["body"], text, {})],
    )
    return db


def _config(storage: Path) -> Config:
    """A Config pointing at ``storage`` with the test embedding model."""
    return Config({"storage": {"dir": str(storage)}, "embedding": _EMB_CFG}, storage / "config.yaml")


def _vector_count(storage: Path) -> int:
    """Open the vector store and count points in the test collection."""
    vs = VectorStore(storage / "qdrant")
    try:
        return vs.count("emb_test")
    finally:
        vs.close()


def test_embed_pipeline_end_to_end(tmp_path: Path) -> None:
    """Embed once, verify points + bookkeeping, and that it is idempotent."""
    storage = tmp_path / "s"
    db = _seed(storage)
    config = _config(storage)
    opts = EmbedOptions(model="test")

    processed = run_embed(config, opts, db=db, embed_fn=_FAKE)
    assert processed == 1
    # The body (35 chars, size 20/overlap 5 -> 2 chunks) produced points.
    assert _vector_count(storage) == 2
    # One embeddings row (one part) for this model.
    assert db.conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model='test'").fetchone()[0] == 1

    # Re-run: already up to date -> nothing processed.
    assert run_embed(config, opts, db=db, embed_fn=_FAKE) == 0
    # Force: re-embeds.
    assert run_embed(config, EmbedOptions(model="test", force=True), db=db, embed_fn=_FAKE) == 1
    assert _vector_count(storage) == 2
    db.close()


def test_embed_reembeds_on_new_active_set(tmp_path: Path) -> None:
    """A new active translation set makes the document stale -> re-embedded."""
    storage = tmp_path / "s"
    db = _seed(storage)
    config = _config(storage)
    run_embed(config, EmbedOptions(model="test"), db=db, embed_fn=_FAKE)

    # New active set (shorter body -> fewer chunks), deactivating the old one.
    TranslationStore(db.conn, BlobStore(storage / "blobs")).write_set(
        source_type="email", doc_id="m1", source_version=1,
        translator_name="dispatch", git_rev="r", parts=[(["body"], "short", {})],
    )
    assert run_embed(config, EmbedOptions(model="test"), db=db, embed_fn=_FAKE) == 1
    # "short" is one chunk -> the old 2 points were replaced by 1.
    assert _vector_count(storage) == 1
    db.close()


def test_embed_empty_part_records_but_no_points(tmp_path: Path) -> None:
    """An empty translation still gets an embeddings row but no vector points."""
    storage = tmp_path / "s"
    db = _seed(storage, text="")
    config = _config(storage)
    assert run_embed(config, EmbedOptions(model="test"), db=db, embed_fn=_FAKE) == 1
    assert _vector_count(storage) == 0
    assert db.conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model='test'").fetchone()[0] == 1
    db.close()
