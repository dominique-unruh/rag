"""Chat LLM backends (PLAN §7).

Streams a chat completion from either a native Ollama server (``/api/chat``) or an
OpenAI-compatible endpoint (``/chat/completions``), using only the standard library
(no extra HTTP dependency) — mirroring :mod:`rag.embed.backends`.

Streaming is deliberate: on a CPU-only box a full answer can take a long time, so
tokens are yielded as they arrive to keep the REPL responsive. The body builders
and delta parsers are split out as pure functions so they can be unit-tested
without a live server.
"""

from __future__ import annotations

import http.client
import json
import urllib.error
import urllib.request
from collections.abc import Iterator

from ..errors import RagError
from ..secrets import Secrets
from .models import ChatModelSpec

# A single chat message: ``{"role": "system"|"user"|"assistant", "content": str}``.
Message = dict[str, str]


class ChatBackendError(RagError):
    """Raised when a chat backend request fails."""


def _ollama_options(spec: ChatModelSpec) -> dict[str, object]:
    """Return the Ollama ``options`` block for ``spec`` (omitting unset knobs)."""
    options: dict[str, object] = {}
    # Only send knobs the user set, so the model's own defaults apply otherwise.
    if spec.num_ctx is not None:
        options["num_ctx"] = spec.num_ctx
    if spec.temperature is not None:
        options["temperature"] = spec.temperature
    if spec.max_tokens is not None:
        # Ollama calls the generation cap num_predict.
        options["num_predict"] = spec.max_tokens
    return options


def _ollama_body(spec: ChatModelSpec, messages: list[Message]) -> dict[str, object]:
    """Build the JSON body for a streaming Ollama ``/api/chat`` request."""
    body: dict[str, object] = {"model": spec.model, "messages": messages, "stream": True}
    options = _ollama_options(spec)
    if options:
        body["options"] = options
    return body


def _openai_body(spec: ChatModelSpec, messages: list[Message]) -> dict[str, object]:
    """Build the JSON body for a streaming OpenAI ``/chat/completions`` request."""
    body: dict[str, object] = {"model": spec.model, "messages": messages, "stream": True}
    # OpenAI-style knobs are top-level (not nested), unlike Ollama's options block.
    if spec.temperature is not None:
        body["temperature"] = spec.temperature
    if spec.max_tokens is not None:
        body["max_tokens"] = spec.max_tokens
    return body


def _ollama_delta(obj: dict[str, object]) -> str:
    """Extract the text delta from one Ollama stream chunk (``message.content``)."""
    message = obj.get("message")
    if isinstance(message, dict):
        content = message.get("content")
        if isinstance(content, str):
            return content
    return ""


def _openai_delta(obj: dict[str, object]) -> str:
    """Extract the text delta from one OpenAI stream chunk (``choices[0].delta.content``)."""
    choices = obj.get("choices")
    if isinstance(choices, list) and choices:
        delta = choices[0].get("delta") if isinstance(choices[0], dict) else None
        if isinstance(delta, dict):
            content = delta.get("content")
            if isinstance(content, str):
                return content
    return ""


def _open_stream(
    url: str, body: dict[str, object], headers: dict[str, str], timeout: float
) -> urllib.request.addinfourl:
    """POST ``body`` as JSON and return the open response for line-by-line reading."""
    data = json.dumps(body).encode("utf-8")
    request = urllib.request.Request(url, data=data, method="POST")
    request.add_header("Content-Type", "application/json")
    for name, value in headers.items():
        request.add_header(name, value)
    try:
        # urlopen returns a file-like response that yields bytes incrementally as
        # the server streams them — exactly what we iterate over below.
        return urllib.request.urlopen(request, timeout=timeout)
    except urllib.error.HTTPError as exc:
        # The server answered with an error status; its body usually says why (e.g.
        # Ollama's 404 "model ... not found"). Surface it so the message is actionable.
        detail = ""
        try:
            detail = exc.read().decode("utf-8", errors="replace").strip()
        except Exception:  # pragma: no cover - body may be unreadable
            pass
        suffix = f": {detail}" if detail else ""
        raise ChatBackendError(
            f"chat request to {url} failed: HTTP {exc.code}{suffix}"
        ) from exc
    except (urllib.error.URLError, OSError, http.client.HTTPException) as exc:
        raise ChatBackendError(f"chat request to {url} failed: {exc}") from exc


def _stream_ollama(spec: ChatModelSpec, messages: list[Message]) -> Iterator[str]:
    """Yield text deltas from a native Ollama ``/api/chat`` stream (newline-delimited JSON)."""
    if not spec.host:
        raise ChatBackendError(f"chat model {spec.id!r}: 'host' required for backend 'ollama'")
    url = spec.host.rstrip("/") + "/api/chat"
    response = _open_stream(url, _ollama_body(spec, messages), {}, spec.timeout)
    try:
        for raw in response:
            line = raw.decode("utf-8").strip()
            if not line:
                continue
            obj = json.loads(line)
            yield _ollama_delta(obj)
            if obj.get("done"):
                break
    finally:
        response.close()


def _stream_openai(spec: ChatModelSpec, messages: list[Message], secrets: Secrets) -> Iterator[str]:
    """Yield text deltas from an OpenAI-compatible ``/chat/completions`` SSE stream."""
    if not spec.api_base:
        raise ChatBackendError(f"chat model {spec.id!r}: 'api_base' required for backend 'openai'")
    url = spec.api_base.rstrip("/") + "/chat/completions"
    headers: dict[str, str] = {}
    if spec.api_key_name:
        # Resolve the API key via the shared secrets mechanism (never the YAML).
        headers["Authorization"] = f"Bearer {secrets.get(spec.api_key_name)}"
    response = _open_stream(url, _openai_body(spec, messages), headers, spec.timeout)
    try:
        for raw in response:
            line = raw.decode("utf-8").strip()
            # SSE frames are ``data: {json}`` lines, terminated by ``data: [DONE]``.
            if not line.startswith("data:"):
                continue
            payload = line[len("data:"):].strip()
            if payload == "[DONE]":
                break
            yield _openai_delta(json.loads(payload))
    finally:
        response.close()


def stream_chat(
    spec: ChatModelSpec, messages: list[Message], *, secrets: Secrets | None = None
) -> Iterator[str]:
    """Stream a chat completion for ``messages``, yielding text deltas in order.

    Dispatches on ``spec.backend``. ``secrets`` supplies an API key when the
    OpenAI-compatible backend needs one; a non-prompting resolver is used by
    default so a batch context never blocks on input.
    """
    resolver = secrets if secrets is not None else Secrets(allow_prompt=False)
    if spec.backend == "ollama":
        yield from _stream_ollama(spec, messages)
    elif spec.backend == "openai":
        yield from _stream_openai(spec, messages, resolver)
    else:  # pragma: no cover - registry already validates the backend
        raise ChatBackendError(f"chat model {spec.id!r}: unknown backend {spec.backend!r}")
