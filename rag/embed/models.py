"""Embedding model registry (PLAN §6.1).

Parses the enabled embedding models into :class:`ModelSpec`s. The models
themselves are described once in the top-level ``models:`` section (see
:mod:`rag.models`); ``embedding.models`` is just the list of ids to embed with.
Each model has a stable id (treated like a version), a type, a backend (``openai``
/ ``ollama``), and the parameters that backend and its index need.
"""

from __future__ import annotations

import json
import re
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError
from ..models import model_block

# Default chunking parameters if the config omits them.
_DEFAULT_CHUNK_SIZE: int = 1024
_DEFAULT_CHUNK_OVERLAP: int = 128

# Default per-request batching / timeout for dense backends if the config omits them.
_DEFAULT_BATCH_SIZE: int = 16
_DEFAULT_TIMEOUT: float = 300.0


@dataclass(frozen=True)
class ModelSpec:
    """One configured embedding model."""

    id: str
    """Stable model id (also the ``--model`` value and the `embeddings.model` value)."""

    type: str
    """``dense`` or ``sparse`` (only ``dense`` is implemented so far)."""

    backend: str
    """``openai`` (OpenAI-compatible endpoint) or ``ollama`` (native Ollama)."""

    model: str
    """The backend's model name (e.g. ``qwen3-embedding:8b``)."""

    dim: int
    """Vector dimension (must match the model's output; fixes the collection)."""

    collection: str
    """Vector-store collection name for this model's index."""

    distance: str = "cosine"
    """Similarity metric (``cosine`` / ``dot`` / ``euclid``)."""

    api_base: str | None = None
    """OpenAI-compatible base URL (e.g. ``http://localhost:11434/v1``)."""

    host: str | None = None
    """Native Ollama host (e.g. ``http://localhost:11434``)."""

    api_key_name: str | None = None
    """Secret name for the API key (resolved via the secrets mechanism), if any."""

    query_instruction: str | None = None
    """Optional instruction prefix prepended to *query* texts (not documents)."""

    batch_size: int = _DEFAULT_BATCH_SIZE
    """Max chunks sent per backend request (dense only). Smaller batches keep each
    request under ``timeout`` and bound memory; larger batches are more efficient.
    Raise it once the backend runs on a GPU. Unused for sparse (local) models."""

    timeout: float = _DEFAULT_TIMEOUT
    """HTTP timeout (seconds) for one backend request. Generous for slow/CPU
    embedders. Applies per ``batch_size`` request, not per document."""


def _collection_name(model_id: str) -> str:
    """Return a safe default vector-store collection name for ``model_id``."""
    # Keep it filesystem/collection-safe; models ids are usually already clean.
    return "emb_" + re.sub(r"[^A-Za-z0-9_]+", "_", model_id)


def _embedding_section(config_raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``embedding`` config section (empty mapping if absent)."""
    section = config_raw.get("embedding", {})
    if not isinstance(section, Mapping):
        raise ConfigError("top-level 'embedding' must be a mapping")
    return section


def chunk_params(config_raw: Mapping[str, Any]) -> tuple[int, int]:
    """Return ``(chunk_size, chunk_overlap)`` from config, with defaults."""
    section = _embedding_section(config_raw)
    size = int(section.get("chunk_size", _DEFAULT_CHUNK_SIZE))
    overlap = int(section.get("chunk_overlap", _DEFAULT_CHUNK_OVERLAP))
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ConfigError(f"invalid chunking: size={size}, overlap={overlap}")
    return size, overlap


# Allowed backends per model type.
_DENSE_BACKENDS: frozenset[str] = frozenset({"openai", "ollama"})
_SPARSE_BACKENDS: frozenset[str] = frozenset({"bm25"})


def _parse_model(entry: Mapping[str, Any]) -> ModelSpec:
    """Build a :class:`ModelSpec` from one config entry, validating required keys."""
    model_id = entry.get("id")
    if not model_id:
        raise ConfigError("each embedding model needs an 'id'")
    model_type = str(entry.get("type", "dense"))
    if model_type not in ("dense", "sparse"):
        raise ConfigError(f"model {model_id!r}: type must be 'dense' or 'sparse'")

    if model_type == "dense":
        backend = str(entry.get("backend", "openai"))
        if backend not in _DENSE_BACKENDS:
            raise ConfigError(f"model {model_id!r}: dense backend must be one of {sorted(_DENSE_BACKENDS)}")
        if "dim" not in entry:
            raise ConfigError(f"model {model_id!r}: 'dim' is required for a dense model")
        dim = int(entry["dim"])
    else:  # sparse
        backend = str(entry.get("backend", "bm25"))
        if backend not in _SPARSE_BACKENDS:
            raise ConfigError(f"model {model_id!r}: sparse backend must be one of {sorted(_SPARSE_BACKENDS)}")
        # Sparse vectors have no fixed dimension; dim is unused.
        dim = 0

    # Per-request batching / timeout (dense backends). Validated defensively so a
    # bad config fails loudly rather than sending a degenerate request.
    batch_size = int(entry.get("batch_size", _DEFAULT_BATCH_SIZE))
    if batch_size < 1:
        raise ConfigError(f"model {model_id!r}: 'batch_size' must be >= 1")
    timeout = float(entry.get("timeout", _DEFAULT_TIMEOUT))
    if timeout <= 0:
        raise ConfigError(f"model {model_id!r}: 'timeout' must be > 0")

    return ModelSpec(
        id=str(model_id),
        type=model_type,
        backend=backend,
        model=str(entry.get("model", model_id)),
        dim=dim,
        collection=str(entry.get("collection") or _collection_name(str(model_id))),
        distance=str(entry.get("distance", "cosine")),
        api_base=entry.get("api_base"),
        host=entry.get("host"),
        api_key_name=entry.get("api_key_name"),
        query_instruction=entry.get("query_instruction"),
        batch_size=batch_size,
        timeout=timeout,
    )


def load_models(config_raw: Mapping[str, Any]) -> dict[str, ModelSpec]:
    """Return the enabled embedding models keyed by id.

    Reads ``embedding.models`` — a list of model ids — and resolves each against the
    top-level ``models:`` section (:func:`rag.models.model_block`). Returns an empty
    dict when ``embedding.models`` is absent.
    """
    section = _embedding_section(config_raw)
    ids = section.get("models", [])
    if not isinstance(ids, list):
        raise ConfigError("embedding.models must be a list of model ids")
    models: dict[str, ModelSpec] = {}
    for model_id in ids:
        if not isinstance(model_id, str):
            raise ConfigError(f"embedding.models entries must be model ids (strings), got {model_id!r}")
        # Pull the shared block from the top-level registry, then parse as embedding.
        spec = _parse_model(model_block(config_raw, model_id))
        models[spec.id] = spec
    return models


def model_fingerprint(spec: ModelSpec) -> str:
    """Return a canonical fingerprint of a model's *incompatibility-defining* config.

    Includes only fields whose change makes existing stored vectors incompatible
    or is baked immutably into the collection: the embedder ``model`` identity,
    ``type`` (dense/sparse), ``dim`` and ``distance`` metric. Deliberately
    excludes freely-changeable knobs: the query instruction (applied at query
    time only, never to stored vectors), the transport (backend/host/api_base)
    and the collection name.
    """
    return json.dumps(
        {"model": spec.model, "type": spec.type, "dim": spec.dim, "distance": spec.distance},
        sort_keys=True,
        separators=(",", ":"),
    )


def get_model(config_raw: Mapping[str, Any], model_id: str) -> ModelSpec:
    """Return the :class:`ModelSpec` for ``model_id``, or raise :class:`ConfigError`."""
    models = load_models(config_raw)
    if model_id not in models:
        available = ", ".join(sorted(models)) or "(none configured)"
        raise ConfigError(f"unknown embedding model {model_id!r}; available: {available}")
    return models[model_id]
