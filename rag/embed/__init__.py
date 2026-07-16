"""Module 3 — embedding / RAG index (PLAN §6).

Chunks the parts of each active translation set and embeds them into a vector
index, with **one index per embedding model**. Several models can be populated
side by side (dense + sparse for hybrid, or old/new during a migration); each is
embedded by its own ``embed`` pass and tracked per ``(part, model)`` in the
``embeddings`` table.
"""

from __future__ import annotations

from .models import ModelSpec, get_model, load_models

__all__ = ["ModelSpec", "get_model", "load_models"]
