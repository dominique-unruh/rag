"""CLI wrapper: expected failures become clean messages, no traceback."""

from __future__ import annotations

import pytest

from rag import cli, config, runtime


def _boom(exc):
    def main(argv):
        raise exc

    return main


def test_run_success_passes_through():
    assert cli.run(lambda argv: 0) == 0


def test_run_config_error_clean(capsys):
    rc = cli.run(_boom(config.ConfigError("bad config.yaml")))
    assert rc == 1
    err = capsys.readouterr().err
    assert "error: bad config.yaml" in err
    assert "Traceback" not in err


def test_run_embedding_dim_error_clean(capsys):
    rc = cli.run(_boom(runtime.EmbeddingDimError("dim 4096 != 768")))
    assert rc == 1
    assert "dim 4096 != 768" in capsys.readouterr().err


def test_run_keyboard_interrupt():
    assert cli.run(_boom(KeyboardInterrupt())) == 130


def test_run_debug_env_reraises(monkeypatch):
    monkeypatch.setenv("RAGPROTO_DEBUG", "1")
    with pytest.raises(config.ConfigError):
        cli.run(_boom(config.ConfigError("x")))


def test_run_does_not_catch_raw_openai_error():
    # cli.run catches only UserError; a raw provider error is an internal path
    # that must be converted at its call site, not blanket-caught here.
    from openai import OpenAIError

    with pytest.raises(OpenAIError):
        cli.run(_boom(OpenAIError("raw")))


def test_unexpected_error_propagates():
    # Not a UserError -> real traceback (bug surfacing).
    with pytest.raises(ValueError):
        cli.run(_boom(ValueError("unexpected")))
