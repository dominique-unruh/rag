"""Chat LLM registry (PLAN §7).

Parses a model block from the top-level ``models:`` section (see :mod:`rag.models`)
into a :class:`ChatModelSpec`; ``chat.default_model`` names which id to use by
default. A chat model uses the same transport options as an embedding model (see
:mod:`rag.embed.models`): an **OpenAI-compatible** endpoint or a **native Ollama**
server. Unlike an embedder, a chat model additionally carries generation knobs
(context window, temperature, token cap).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError
from ..models import model_block, models_section

# Allowed chat backends (same two transports as the dense embedding backends).
_CHAT_BACKENDS: frozenset[str] = frozenset({"openai", "ollama"})

# Default per-request HTTP timeout (seconds). Generous: a CPU-only chat model can
# take a long time to produce a full answer, and streaming holds the connection open.
_DEFAULT_TIMEOUT: float = 600.0

# Default number of retrieved chunks fed to the model when config omits it.
_DEFAULT_TOP_K: int = 8


@dataclass(frozen=True)
class ChatModelSpec:
    """One configured chat LLM."""

    id: str
    """Stable model id (the ``--model`` value and the ``chat.default_model`` value)."""

    backend: str
    """``openai`` (OpenAI-compatible endpoint) or ``ollama`` (native Ollama)."""

    model: str
    """The backend's model name (e.g. ``qwen2.5:7b-instruct``)."""

    host: str | None = None
    """Native Ollama host (e.g. ``http://localhost:11434``)."""

    api_base: str | None = None
    """OpenAI-compatible base URL (e.g. ``http://localhost:11434/v1``)."""

    api_key_name: str | None = None
    """Secret name for the API key (resolved via the secrets mechanism), if any."""

    timeout: float = _DEFAULT_TIMEOUT
    """HTTP timeout (seconds) for one generation request."""

    num_ctx: int | None = None
    """Ollama context-window override (tokens); ``None`` uses the model default."""

    temperature: float | None = None
    """Sampling temperature; ``None`` uses the backend default."""

    max_tokens: int | None = None
    """Cap on generated tokens (OpenAI ``max_tokens`` / Ollama ``num_predict``)."""


def _chat_section(config_raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the ``chat`` config section (empty mapping if absent)."""
    section = config_raw.get("chat", {})
    if not isinstance(section, Mapping):
        raise ConfigError("top-level 'chat' must be a mapping")
    return section


def _parse_model(entry: Mapping[str, Any]) -> ChatModelSpec:
    """Build a :class:`ChatModelSpec` from one config entry, validating required keys."""
    model_id = entry.get("id")
    if not model_id:
        raise ConfigError("each chat model needs an 'id'")
    backend = str(entry.get("backend", "ollama"))
    if backend not in _CHAT_BACKENDS:
        raise ConfigError(
            f"chat model {model_id!r}: backend must be one of {sorted(_CHAT_BACKENDS)}"
        )
    timeout = float(entry.get("timeout", _DEFAULT_TIMEOUT))
    if timeout <= 0:
        raise ConfigError(f"chat model {model_id!r}: 'timeout' must be > 0")

    # Optional generation knobs; kept as-is (None => backend default).
    num_ctx = entry.get("num_ctx")
    temperature = entry.get("temperature")
    max_tokens = entry.get("max_tokens")

    return ChatModelSpec(
        id=str(model_id),
        backend=backend,
        model=str(entry.get("model", model_id)),
        host=entry.get("host"),
        api_base=entry.get("api_base"),
        api_key_name=entry.get("api_key_name"),
        timeout=timeout,
        num_ctx=int(num_ctx) if num_ctx is not None else None,
        temperature=float(temperature) if temperature is not None else None,
        max_tokens=int(max_tokens) if max_tokens is not None else None,
    )


def get_chat_model(config_raw: Mapping[str, Any], model_id: str | None) -> ChatModelSpec:
    """Return the :class:`ChatModelSpec` for ``model_id`` (or the configured default).

    ``model_id`` is a model id from the top-level ``models:`` section. With ``None``
    it falls back to ``chat.default_model``; if that is also unset and exactly one
    model is defined, that one is used. The referenced block is parsed as a chat
    model — a non-chat backend (e.g. a ``bm25`` embedder) is rejected here. Raises
    :class:`ConfigError` on an unknown/ambiguous/absent selection.
    """
    if model_id is None:
        # Prefer the explicit default; else the sole model when unambiguous.
        default = _chat_section(config_raw).get("default_model")
        if default is not None:
            model_id = str(default)
        else:
            models = models_section(config_raw)
            if len(models) == 1:
                model_id = next(iter(models))
            elif not models:
                raise ConfigError("no models configured; add a 'models' section to config.yaml")
            else:
                raise ConfigError(
                    "no chat.default_model set; pass --model "
                    f"(available: {', '.join(sorted(models))})"
                )
    # Resolve the shared block (raises with the available ids if unknown), parse as chat.
    return _parse_model(model_block(config_raw, model_id))


def retrieval_top_k(config_raw: Mapping[str, Any]) -> int:
    """Return the default number of chunks to retrieve (``chat.retrieval.top_k``)."""
    retrieval = _chat_section(config_raw).get("retrieval", {})
    if not isinstance(retrieval, Mapping):
        raise ConfigError("chat.retrieval must be a mapping")
    top_k = int(retrieval.get("top_k", _DEFAULT_TOP_K))
    if top_k < 1:
        raise ConfigError("chat.retrieval.top_k must be >= 1")
    return top_k
