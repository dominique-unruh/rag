"""Config loading, validation, secret resolution, and cutoff logic."""

from __future__ import annotations

import os
from datetime import date, timedelta
from pathlib import Path

import pytest

from rag import config


def _enable_email(root: Path, extra: str) -> None:
    """Flip the fixture's disabled email block to an enabled one plus `extra`."""
    text = (root / "config.yaml").read_text().replace(
        "email:\n  enabled: false",
        "email:\n  enabled: true\n  host: h\n  username: u\n" + extra,
    )
    (root / "config.yaml").write_text(text, encoding="utf-8")


def test_load_valid(project):
    root, _ = project
    cfg = config.load(root)
    assert cfg.models.embed_dim == 8
    assert cfg.models.api_key == "none"  # inline default
    assert cfg.notes.enabled and os.path.isabs(cfg.notes.directory)
    assert cfg.qdrant.path and cfg.qdrant.url is None
    assert cfg.graph.llm_extraction is False


def test_missing_config_file(tmp_path: Path):
    with pytest.raises(config.ConfigError, match="config.yaml not found"):
        config.load(tmp_path)


def test_missing_required_key(project):
    root, _ = project
    (root / "config.yaml").write_text(
        "models:\n  api_base: x\n  chat_model: m\n  embed_model: e\n"
        "notes:\n  enabled: true\n  directory: /tmp\n"
        "qdrant:\n  path: /tmp/q\n  collection: c\n"
        "storage:\n  dir: /tmp/s\n",
        encoding="utf-8",
    )
    with pytest.raises(config.ConfigError, match="models.embed_dim"):
        config.load(root)


def test_qdrant_requires_exactly_one_target(project):
    root, _ = project
    text = (root / "config.yaml").read_text().replace(
        "  collection: personal_rag",
        "  url: http://localhost:6333\n  collection: personal_rag",
    )
    (root / "config.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(config.ConfigError, match="exactly one"):
        config.load(root)


def test_no_source_enabled(project):
    root, _ = project
    text = (root / "config.yaml").read_text().replace(
        "notes:\n  enabled: true", "notes:\n  enabled: false"
    )
    (root / "config.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(config.ConfigError, match="At least one source"):
        config.load(root)


# --- api key resolution ---------------------------------------------------

def test_api_key_from_file(project, secret_file):
    root, wc = project
    key_path = secret_file("openai.key", "sk-from-file\n")
    wc(api_key_line=f"  api_key_file: {key_path}")
    cfg = config.load(root)
    assert cfg.models.api_key == "sk-from-file"  # trailing newline stripped


def test_api_key_inline_none_used_verbatim(project):
    root, wc = project
    wc(api_key_line="  api_key: none")  # inline wins, no keyring touched
    cfg = config.load(root)
    assert cfg.models.api_key == "none"


def test_api_key_from_keyring(project, monkeypatch):
    root, wc = project
    wc(api_key_line="")  # no inline, no file -> keyring
    calls = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, account):
            calls["get"] = (service, account)
            return "sk-from-keyring"

        @staticmethod
        def set_password(service, account, value):
            calls["set"] = True

    monkeypatch.setattr(config, "_keyring", lambda: FakeKeyring)
    cfg = config.load(root)
    assert cfg.models.api_key == "sk-from-keyring"
    assert calls["get"] == (config.KEYRING_SERVICE, "api_key@http://127.0.0.1:9/v1")
    assert "set" not in calls  # already present -> not re-stored


def test_api_key_prompt_then_stored_in_keyring(project, monkeypatch):
    root, wc = project
    wc(api_key_line="")
    stored = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, account):
            return None  # miss

        @staticmethod
        def set_password(service, account, value):
            stored[(service, account)] = value

    monkeypatch.setattr(config, "_keyring", lambda: FakeKeyring)
    monkeypatch.setattr(config, "_prompt_api_key", lambda api_base: "sk-typed")
    cfg = config.load(root)
    assert cfg.models.api_key == "sk-typed"
    assert stored[(config.KEYRING_SERVICE, "api_key@http://127.0.0.1:9/v1")] == "sk-typed"


def test_api_key_file_missing_fails(project):
    root, wc = project
    wc(api_key_line="  api_key_file: /no/such/key")
    with pytest.raises(config.ConfigError, match="api_key_file points to a missing file"):
        config.load(root)


# --- since cutoff ---------------------------------------------------------

def test_since_days_resolves_relative(project):
    root, _ = project
    _enable_email(root, "  since_days: 30\n  password: p")
    cfg = config.load(root)
    assert cfg.email.since == date.today() - timedelta(days=30)


def test_since_date_overrides_since_days(project):
    root, _ = project
    _enable_email(root, "  since_days: 30\n  since_date: 2025-01-01\n  password: p")
    cfg = config.load(root)
    assert cfg.email.since == date(2025, 1, 1)


def test_email_enabled_requires_host(project):
    root, _ = project
    text = (root / "config.yaml").read_text().replace(
        "email:\n  enabled: false",
        "email:\n  enabled: true\n  username: u\n  since_days: 30\n  password: p",
    )
    (root / "config.yaml").write_text(text, encoding="utf-8")
    with pytest.raises(config.ConfigError, match="email.host"):
        config.load(root)


# --- password resolution: inline / file / keyring / prompt ----------------

def test_password_inline(project):
    root, _ = project
    _enable_email(root, "  since_days: 30\n  password: inline-secret")
    cfg = config.load(root)
    assert cfg.email.password == "inline-secret"


def test_password_from_file(project, secret_file):
    root, _ = project
    pw_path = secret_file("imap.pass", "file-secret\n")
    _enable_email(root, f"  since_days: 30\n  password_file: {pw_path}")
    cfg = config.load(root)
    assert cfg.email.password == "file-secret"


def test_password_from_keyring(project, monkeypatch):
    root, _ = project
    _enable_email(root, "  since_days: 30")  # no inline/file -> keyring
    calls = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, user):
            calls["get"] = (service, user)
            return "keyring-secret"

        @staticmethod
        def set_password(service, user, value):
            calls["set"] = (service, user, value)

    monkeypatch.setattr(config, "_keyring", lambda: FakeKeyring)
    cfg = config.load(root)
    assert cfg.email.password == "keyring-secret"
    assert calls["get"] == (config.KEYRING_SERVICE, "u")
    assert "set" not in calls  # already present -> not re-stored


def test_password_prompt_then_stored_in_keyring(project, monkeypatch):
    root, _ = project
    _enable_email(root, "  since_days: 30")
    stored = {}

    class FakeKeyring:
        @staticmethod
        def get_password(service, user):
            return None  # miss

        @staticmethod
        def set_password(service, user, value):
            stored[(service, user)] = value

    monkeypatch.setattr(config, "_keyring", lambda: FakeKeyring)
    monkeypatch.setattr(config, "_prompt_password", lambda user: "typed-secret")
    cfg = config.load(root)
    assert cfg.email.password == "typed-secret"
    assert stored[(config.KEYRING_SERVICE, "u")] == "typed-secret"


def test_password_no_keyring_non_interactive_fails(project, monkeypatch):
    root, _ = project
    _enable_email(root, "  since_days: 30")
    monkeypatch.setattr(config, "_keyring", lambda: None)  # no backend
    # simulate non-interactive: real _prompt_password checks isatty
    import sys

    monkeypatch.setattr(sys.stdin, "isatty", lambda: False)
    with pytest.raises(config.ConfigError, match="no terminal is available"):
        config.load(root)


def test_loose_secret_file_permissions_warn(project, secret_file, capsys):
    root, _ = project
    pw_path = secret_file("imap.pass", "s", mode=0o644)
    _enable_email(root, f"  since_days: 30\n  password_file: {pw_path}")
    config.load(root)
    assert "chmod 600" in capsys.readouterr().out


# --- _keyring backend selection ------------------------------------------

def test_keyring_returns_module_when_default_ok(monkeypatch):
    import keyring

    monkeypatch.setattr(keyring, "get_keyring", lambda: object())
    assert config._keyring() is keyring


def test_keyring_falls_back_to_secretservice_when_default_broken(monkeypatch):
    import keyring
    from keyring.backends import SecretService

    def boom():
        raise RuntimeError("libsecret required")

    class FakeBackend:
        priority = 5

    captured = {}
    monkeypatch.setattr(keyring, "get_keyring", boom)
    monkeypatch.setattr(SecretService, "Keyring", lambda: FakeBackend())
    monkeypatch.setattr(keyring, "set_keyring", lambda b: captured.setdefault("backend", b))

    kr = config._keyring()
    assert kr is keyring
    assert isinstance(captured["backend"], FakeBackend)


def test_keyring_none_when_no_backend_usable(monkeypatch):
    import keyring
    from keyring.backends import SecretService

    def boom(*a, **k):
        raise RuntimeError("libsecret required")

    monkeypatch.setattr(keyring, "get_keyring", boom)
    monkeypatch.setattr(SecretService, "Keyring", boom)  # fallback also unavailable
    assert config._keyring() is None
