"""Tests for the global secret resolver, focusing on ``--reauth`` behaviour."""

from __future__ import annotations

from typing import Any

import pytest

from rag import secrets as secrets_mod
from rag.secrets import Secrets


@pytest.fixture()
def fake_keyring(monkeypatch: pytest.MonkeyPatch) -> dict[tuple[str, str], str]:
    """Replace the keyring with an in-memory ``(service, key) -> value`` store."""
    store: dict[tuple[str, str], str] = {}

    def get_password(service: str, key: str) -> str | None:
        """Return the stored secret or ``None``."""
        return store.get((service, key))

    def set_password(service: str, key: str, value: str) -> None:
        """Persist a secret in the fake store."""
        store[(service, key)] = value

    monkeypatch.setattr(secrets_mod.keyring, "get_password", get_password)
    monkeypatch.setattr(secrets_mod.keyring, "set_password", set_password)
    return store


def test_reauth_keep_uses_stored(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    """Answering the reauth prompt with 'keep' (Enter) returns the stored secret."""
    fake_keyring[("rag", "k")] = "old"
    # Enter (empty) == keep; getpass must never be called in this path.
    monkeypatch.setattr("builtins.input", lambda _prompt: "")
    monkeypatch.setattr(secrets_mod, "getpass", lambda *_a, **_k: pytest.fail("prompted"))

    resolver = Secrets(allow_prompt=True, reauth=True)
    assert resolver.get("k") == "old"
    # Store is unchanged.
    assert fake_keyring[("rag", "k")] == "old"


def test_reauth_replace_overwrites(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    """Answering 'n' prompts for a new secret and writes it back to the keyring."""
    fake_keyring[("rag", "k")] = "old"
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr(secrets_mod, "getpass", lambda *_a, **_k: "new")

    resolver = Secrets(allow_prompt=True, reauth=True)
    assert resolver.get("k") == "new"
    # The replacement was persisted for later runs.
    assert fake_keyring[("rag", "k")] == "new"


def test_reauth_blank_keeps_stored(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    """Choosing to replace but entering nothing keeps the stored secret."""
    fake_keyring[("rag", "k")] = "old"
    monkeypatch.setattr("builtins.input", lambda _prompt: "n")
    monkeypatch.setattr(secrets_mod, "getpass", lambda *_a, **_k: "")

    resolver = Secrets(allow_prompt=True, reauth=True)
    assert resolver.get("k") == "old"
    assert fake_keyring[("rag", "k")] == "old"


def test_no_reauth_does_not_prompt(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    """Without reauth, a stored secret is returned with no prompting at all."""
    fake_keyring[("rag", "k")] = "old"
    monkeypatch.setattr("builtins.input", lambda *_a: pytest.fail("prompted"))
    monkeypatch.setattr(secrets_mod, "getpass", lambda *_a, **_k: pytest.fail("prompted"))

    assert Secrets(allow_prompt=True, reauth=False).get("k") == "old"


def test_reauth_ignored_when_not_prompting(
    monkeypatch: pytest.MonkeyPatch, fake_keyring: dict[tuple[str, str], str]
) -> None:
    """Reauth is inert on a non-TTY run (allow_prompt=False): stored value stands."""
    fake_keyring[("rag", "k")] = "old"
    monkeypatch.setattr("builtins.input", lambda *_a: pytest.fail("prompted"))

    assert Secrets(allow_prompt=False, reauth=True).get("k") == "old"
