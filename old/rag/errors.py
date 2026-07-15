"""User-facing error type and helpers.

`UserError` marks a failure that is the user's to fix (bad config, wrong model
name, unreachable/auth-failing endpoint) — the CLI prints it as one clean line.
Anything that is NOT a UserError is treated as an internal bug and keeps its full
traceback. So: catch provider/library exceptions *at the point they happen* and
re-raise them as UserError; never let the CLI blanket-catch broad exception types.
"""

from __future__ import annotations

from contextlib import contextmanager

try:  # openai ships as a transitive dependency
    from openai import OpenAIError
except Exception:  # pragma: no cover
    class OpenAIError(Exception):
        pass


class UserError(Exception):
    """A failure with a message meant for the user; no traceback shown."""


def format_openai(exc: "OpenAIError") -> str:
    """Extract the human message (and HTTP status) from an openai error."""
    message = None
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        message = body.get("message")
        if not message and isinstance(body.get("error"), dict):
            message = body["error"].get("message")
    status = getattr(exc, "status_code", None)
    text = message or str(exc)
    return f"[{status}] {text}" if status else text


@contextmanager
def endpoint_errors(hint: str = ""):
    """Convert any openai/endpoint exception raised inside into a UserError."""
    try:
        yield
    except OpenAIError as exc:
        where = f" {hint}" if hint else ""
        raise UserError(f"endpoint error{where}: {format_openai(exc)}") from exc
