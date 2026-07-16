"""Embedding backends (PLAN §2, §6.1).

Turns texts into vectors via either an OpenAI-compatible endpoint or a native
Ollama server. Uses the standard library (no extra HTTP dependency).
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from ..errors import RagError
from ..secrets import Secrets
from .models import ModelSpec

# HTTP timeout (seconds) for an embedding request. Embedding can be slow for big
# local models, so this is generous.
_HTTP_TIMEOUT: float = 300.0


class EmbedBackendError(RagError):
    """Raised when an embedding backend request fails."""


def _post_json(url: str, payload: dict[str, object], headers: dict[str, str]) -> dict[str, object]:
    """POST ``payload`` as JSON to ``url`` and return the parsed JSON response."""
    data = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.URLError as exc:
        raise EmbedBackendError(f"embedding request to {url} failed: {exc}") from exc


def _apply_instruction(spec: ModelSpec, texts: list[str], is_query: bool) -> list[str]:
    """Prepend the query instruction to ``texts`` for queries (asymmetric models)."""
    if is_query and spec.query_instruction:
        # Instruction goes on queries only, never on stored documents.
        return [spec.query_instruction + t for t in texts]
    return texts


def _embed_openai(spec: ModelSpec, texts: list[str], secrets: Secrets) -> list[list[float]]:
    """Embed via an OpenAI-compatible ``/embeddings`` endpoint."""
    if not spec.api_base:
        raise EmbedBackendError(f"model {spec.id!r}: 'api_base' required for backend 'openai'")
    url = spec.api_base.rstrip("/") + "/embeddings"
    headers: dict[str, str] = {}
    if spec.api_key_name:
        # Resolve the API key via the shared secrets mechanism (never the YAML).
        headers["Authorization"] = f"Bearer {secrets.get(spec.api_key_name)}"
    body = _post_json(url, {"model": spec.model, "input": texts}, headers)
    items = body.get("data")
    if not isinstance(items, list):
        raise EmbedBackendError(f"unexpected embeddings response from {url}")
    # Responses carry an index; sort by it so vectors line up with inputs.
    ordered = sorted(items, key=lambda d: d.get("index", 0))
    return [list(d["embedding"]) for d in ordered]


def _embed_ollama(spec: ModelSpec, texts: list[str]) -> list[list[float]]:
    """Embed via a native Ollama ``/api/embed`` endpoint (batch input)."""
    if not spec.host:
        raise EmbedBackendError(f"model {spec.id!r}: 'host' required for backend 'ollama'")
    url = spec.host.rstrip("/") + "/api/embed"
    body = _post_json(url, {"model": spec.model, "input": texts}, {})
    vectors = body.get("embeddings")
    if not isinstance(vectors, list):
        raise EmbedBackendError(f"unexpected embeddings response from {url}")
    return [list(v) for v in vectors]


def embed_texts(
    spec: ModelSpec, texts: list[str], *, is_query: bool = False, secrets: Secrets | None = None
) -> list[list[float]]:
    """Embed ``texts`` with the model ``spec``, returning one vector per text.

    ``is_query`` applies the model's query instruction (for asymmetric models).
    ``secrets`` supplies API keys when the backend needs one.
    """
    if not texts:
        return []
    prepared = _apply_instruction(spec, texts, is_query)
    resolver = secrets if secrets is not None else Secrets(allow_prompt=False)
    if spec.backend == "openai":
        return _embed_openai(spec, prepared, resolver)
    if spec.backend == "ollama":
        return _embed_ollama(spec, prepared)
    raise EmbedBackendError(f"model {spec.id!r}: unknown backend {spec.backend!r}")
