"""errors: openai formatting + endpoint_errors -> UserError conversion."""

from __future__ import annotations

import pytest
from openai import OpenAIError

from rag import config, errors, runtime


def test_config_and_dim_errors_are_user_errors():
    assert issubclass(config.ConfigError, errors.UserError)
    assert issubclass(runtime.EmbeddingDimError, errors.UserError)


def test_format_openai_nested_message_and_status():
    class E(OpenAIError):
        status_code = 400
        body = {"error": {"message": "does not support matryoshka"}}

    out = errors.format_openai(E())
    assert "[400]" in out and "does not support matryoshka" in out


def test_format_openai_flat_message():
    class E(OpenAIError):
        status_code = 401
        body = {"message": "Invalid API key format"}

    assert errors.format_openai(E()) == "[401] Invalid API key format"


def test_format_openai_fallback_to_str():
    assert "boom" in errors.format_openai(OpenAIError("boom"))


def test_endpoint_errors_converts_to_user_error():
    with pytest.raises(errors.UserError) as ei:
        with errors.endpoint_errors("(http://x/v1)"):
            raise OpenAIError("kaput")
    msg = str(ei.value)
    assert "endpoint error (http://x/v1)" in msg and "kaput" in msg


def test_endpoint_errors_passes_through_non_openai():
    # Non-provider exceptions are internal bugs; must NOT be converted.
    with pytest.raises(ValueError):
        with errors.endpoint_errors():
            raise ValueError("bug")
