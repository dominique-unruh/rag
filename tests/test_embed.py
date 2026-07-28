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
from rag.embed import vectorstore as vectorstore_mod
from rag.embed.vectorstore import open_qdrant_client, open_vector_store
from rag.errors import ConfigError, VectorStoreBusyError
from rag.extract.store import SqliteStore
from rag.translate.store import TranslationStore

# One dense model block (backend unused — the pipeline tests inject embed_fn). The
# id is the map key in the top-level `models:` section, so it is not in the block.
_MODEL = {"type": "dense", "backend": "openai", "api_base": "http://x", "model": "m", "dim": 4}


def _raw(*, models: dict | None = None, enabled: list | None = None,
         chunk_size: int = 20, chunk_overlap: int = 5) -> dict:
    """Build a raw config: top-level `models:` keyed by id + an `embedding` section.

    ``models`` maps id -> block (defaults to a single ``test`` dense model);
    ``enabled`` is the id list ``embedding.models`` references (defaults to all).
    """
    models = models if models is not None else {"test": _MODEL}
    enabled = enabled if enabled is not None else list(models)
    return {
        "models": models,
        "embedding": {"chunk_size": chunk_size, "chunk_overlap": chunk_overlap, "models": enabled},
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
    raw = _raw()
    assert set(load_models(raw)) == {"test"}
    assert chunk_params(raw) == (20, 5)
    spec = get_model(raw, "test")
    assert spec.dim == 4 and spec.backend == "openai" and spec.collection == "emb_test"
    with pytest.raises(ConfigError):
        get_model(raw, "nope")
    with pytest.raises(ConfigError):
        # dim is required for a dense model.
        load_models(_raw(models={"x": {"backend": "openai"}}))


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
    return Config({"storage": {"dir": str(storage)}, **_raw()}, storage / "config.yaml")


def _vector_count(storage: Path, collection: str = "emb_test") -> int:
    """Open the vector store and count points in ``collection``."""
    vs = open_vector_store(_config(storage))
    try:
        return vs.count(collection)
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


# Sparse (BM25) embedding config — no backend service needed.
_SPARSE_RAW = _raw(models={"sp": {"type": "sparse", "backend": "bm25"}})


def test_sparse_model_registry() -> None:
    """A sparse model parses; dim is 0/unused; bad sparse backend raises."""
    spec = get_model(_SPARSE_RAW, "sp")
    assert spec.type == "sparse" and spec.backend == "bm25" and spec.dim == 0
    with pytest.raises(ConfigError):
        load_models(_raw(models={"x": {"type": "sparse", "backend": "ollama"}}))


def test_embed_sparse_pipeline(tmp_path: Path) -> None:
    """Sparse embedding stores BM25 term-frequency vectors locally (no service)."""
    storage = tmp_path / "s"
    db = _seed(storage)
    config = Config({"storage": {"dir": str(storage)}, **_SPARSE_RAW},
                    storage / "config.yaml")
    # No embed_fn needed — sparse vectors are computed locally.
    assert run_embed(config, EmbedOptions(model="sp"), db=db) == 1
    assert _vector_count(storage, "emb_sp") == 2
    assert db.conn.execute(
        "SELECT COUNT(*) FROM embeddings WHERE model='sp'").fetchone()[0] == 1
    # Idempotent.
    assert run_embed(config, EmbedOptions(model="sp"), db=db) == 0
    db.close()


def test_fingerprint_guard(tmp_path: Path) -> None:
    """Changing a model's incompatibility-defining config is refused; safe knobs pass."""
    storage = tmp_path / "s"
    db = _seed(storage)
    run_embed(_config(storage), EmbedOptions(model="test"), db=db, embed_fn=_FAKE)

    # Change the underlying model (same id) -> fingerprint mismatch -> refused.
    changed = _raw(models={"test": dict(_MODEL, model="other")})
    bad = Config({"storage": {"dir": str(storage)}, **changed}, storage / "c.yaml")
    with pytest.raises(ConfigError):
        run_embed(bad, EmbedOptions(model="test"), db=db, embed_fn=_FAKE)

    # Changing only the query instruction is safe (query-time only) -> no error.
    safe = _raw(models={"test": dict(_MODEL, query_instruction="Query: ")})
    ok = Config({"storage": {"dir": str(storage)}, **safe}, storage / "c.yaml")
    assert run_embed(ok, EmbedOptions(model="test", force=True), db=db, embed_fn=_FAKE) == 1
    db.close()


def test_vector_store_busy_error(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A locked embedded Qdrant directory surfaces a clean VectorStoreBusyError."""
    def _boom(*_a: object, **_k: object) -> object:
        # Mimic embedded Qdrant's bare RuntimeError when the dir is already locked.
        raise RuntimeError("Storage folder is already accessed by another instance")

    monkeypatch.setattr(vectorstore_mod, "QdrantClient", _boom)
    # No vector_store.url -> embedded-local branch, which maps the lock error.
    config = Config({"storage": {"dir": str(tmp_path)}}, tmp_path / "config.yaml")
    with pytest.raises(VectorStoreBusyError):
        open_qdrant_client(config)


def test_vector_store_rejects_server_only_options_without_url(tmp_path: Path) -> None:
    """on_disk / quantization without a server url are refused, not silently ignored."""
    base = {"storage": {"dir": str(tmp_path)}}
    # on_disk without url.
    cfg = Config({**base, "vector_store": {"on_disk": True}}, tmp_path / "c.yaml")
    with pytest.raises(ConfigError):
        open_vector_store(cfg)
    # quantization without url.
    cfg = Config(
        {**base, "vector_store": {"quantization": {"type": "scalar"}}}, tmp_path / "c.yaml"
    )
    with pytest.raises(ConfigError):
        open_vector_store(cfg)
    # on_disk: false without url is fine (matches the in-RAM reality) -> no raise.
    cfg = Config({**base, "vector_store": {"on_disk": False}}, tmp_path / "c.yaml")
    open_vector_store(cfg).close()


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
