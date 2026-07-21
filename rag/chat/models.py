"""Chat LLM registry (PLAN §7).

Parses the ``chat`` config section into :class:`ChatModelSpec`s. A chat model uses
the same transport options as an embedding model (see :mod:`rag.embed.models`): an
**OpenAI-compatible** endpoint or a **native Ollama** server. Unlike an embedder,
a chat model additionally carries generation knobs (context window, temperature,
token cap).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

from ..errors import ConfigError

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


def load_chat_models(config_raw: Mapping[str, Any]) -> dict[str, ChatModelSpec]:
    """Return all configured chat models keyed by id (empty if the section is absent)."""
    section = _chat_section(config_raw)
    entries = section.get("models", [])
    if not isinstance(entries, list):
        raise ConfigError("chat.models must be a list")
    models: dict[str, ChatModelSpec] = {}
    for entry in entries:
        if not isinstance(entry, Mapping):
            raise ConfigError("each chat model must be a mapping")
        spec = _parse_model(entry)
        models[spec.id] = spec
    return models


def get_chat_model(config_raw: Mapping[str, Any], model_id: str | None) -> ChatModelSpec:
    """Return the :class:`ChatModelSpec` for ``model_id`` (or the configured default).

    With ``model_id`` ``None`` uses ``chat.default_model``; if that is also unset
    and exactly one model is configured, that one is used. Raises
    :class:`ConfigError` on an unknown/ambiguous/absent selection.
    """
    models = load_chat_models(config_raw)
    if not models:
        raise ConfigError(
            "no chat models configured; add a 'chat.models' entry to config.yaml"
        )
    if model_id is None:
        # Fall back to the explicit default, then to the sole model if unambiguous.
        default = _chat_section(config_raw).get("default_model")
        if default is not None:
            model_id = str(default)
        elif len(models) == 1:
            return next(iter(models.values()))
        else:
            raise ConfigError(
                "several chat models configured but no chat.default_model set; "
                f"pass --model (available: {', '.join(sorted(models))})"
            )
    if model_id not in models:
        available = ", ".join(sorted(models)) or "(none configured)"
        raise ConfigError(f"unknown chat model {model_id!r}; available: {available}")
    return models[model_id]


def retrieval_top_k(config_raw: Mapping[str, Any]) -> int:
    """Return the default number of chunks to retrieve (``chat.retrieval.top_k``)."""
    retrieval = _chat_section(config_raw).get("retrieval", {})
    if not isinstance(retrieval, Mapping):
        raise ConfigError("chat.retrieval must be a mapping")
    top_k = int(retrieval.get("top_k", _DEFAULT_TOP_K))
    if top_k < 1:
        raise ConfigError("chat.retrieval.top_k must be >= 1")
    return top_k
