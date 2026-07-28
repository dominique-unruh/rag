"""Top-level ``models:`` config section — the shared model registry (PLAN §10).

Every model (embedding and chat alike) is described once under the top-level
``models:`` mapping, keyed by id. Consuming sections reference a model by that id:
``embedding.models`` lists the ids to embed with, ``chat.default_model`` names the
default chat model. The block here is *untyped* — there is no embedding/chat
discriminator — so a model is whatever the referencing section interprets it as:
each consumer parses the referenced block with its own spec parser (see
:mod:`rag.embed.models` and :mod:`rag.chat.models`). A backend the consumer does
not accept is rejected by that consumer's parser.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from .errors import ConfigError


def models_section(config_raw: Mapping[str, Any]) -> Mapping[str, Any]:
    """Return the top-level ``models:`` mapping (id -> block); empty if absent."""
    section = config_raw.get("models", {})
    if not isinstance(section, Mapping):
        raise ConfigError("top-level 'models' must be a mapping of id -> model config")
    return section


def model_block(config_raw: Mapping[str, Any], model_id: str) -> Mapping[str, Any]:
    """Return the config block for ``model_id`` from ``models:``, with ``id`` injected.

    Raises :class:`ConfigError` (naming the available ids) when ``model_id`` is not
    defined. The returned mapping carries an ``id`` key so the per-consumer parsers —
    which read ``entry["id"]`` — keep working against the keyed layout unchanged.
    """
    section = models_section(config_raw)
    block = section.get(model_id)
    if block is None:
        available = ", ".join(sorted(section)) or "(none configured)"
        raise ConfigError(f"unknown model {model_id!r}; available: {available}")
    if not isinstance(block, Mapping):
        raise ConfigError(f"model {model_id!r} must be a mapping")
    # Inject the id (the map key) so downstream parsers see it as a normal field.
    return {"id": model_id, **block}
