"""Tests for the email plugin's OAuth2 helpers (no network / no browser needed)."""

from __future__ import annotations

import base64
import hashlib

import pytest

from rag.errors import PluginError
from rag.extract.plugins.email import oauth2


def test_pkce_pair_challenge_is_s256_of_verifier() -> None:
    """The challenge must be the base64url SHA-256 of the verifier, unpadded."""
    verifier, challenge = oauth2._pkce_pair()
    # Recompute the S256 transform and compare against the returned challenge.
    expected = (
        base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
        .rstrip(b"=")
        .decode("ascii")
    )
    assert challenge == expected
    # Verifier length must sit within RFC 7636's 43..128-char window.
    assert 43 <= len(verifier) <= 128
    # base64url output carries no padding characters.
    assert "=" not in challenge


def test_pkce_pair_is_random_each_call() -> None:
    """Two pairs must differ so a replayed verifier can never be reused."""
    assert oauth2._pkce_pair()[0] != oauth2._pkce_pair()[0]


def test_auth_code_login_rejects_state_mismatch(monkeypatch: pytest.MonkeyPatch) -> None:
    """A redirect whose ``state`` does not match must abort before token exchange."""

    class _FakeServer:
        """Stand-in for the loopback server that yields a forged-state callback."""

        server_address = ("127.0.0.1", 5555)
        timeout = 1.0
        # A redirect carrying a code but the WRONG state (not the generated one).
        result = {"code": ["abc"], "state": ["not-the-real-state"]}

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Accept the (address, handler) signature and ignore it."""

        def server_close(self) -> None:
            """No-op: nothing real to release."""

    # Swap in the fake server and neuter the browser open; _post_form must NOT run.
    monkeypatch.setattr(oauth2, "_CallbackServer", _FakeServer)
    monkeypatch.setattr(oauth2.webbrowser, "open", lambda url: True)
    monkeypatch.setattr(
        oauth2, "_post_form", lambda *a, **k: pytest.fail("token exchange must not run")
    )

    with pytest.raises(PluginError, match="state mismatch"):
        oauth2._auth_code_login("common", "cid", None, ["scope"])


def test_auth_code_login_exchanges_code_for_tokens(monkeypatch: pytest.MonkeyPatch) -> None:
    """A well-formed redirect must exchange the code and return both tokens."""
    captured: dict[str, str] = {}

    class _FakeServer:
        """Loopback stand-in that echoes back the state chosen by the caller."""

        server_address = ("127.0.0.1", 5555)
        timeout = 1.0
        result: dict[str, list[str]] | None = None

        def __init__(self, *args: object, **kwargs: object) -> None:
            """Ignore (address, handler)."""

        def server_close(self) -> None:
            """No-op."""

    def _fake_open(url: str) -> bool:
        """Pull the generated ``state`` out of the authorize URL and echo it back."""
        # The poll loop reads server.result on the next handle_request; set it now.
        import urllib.parse

        state = urllib.parse.parse_qs(urllib.parse.urlparse(url).query)["state"][0]
        _FakeServer.result = {"code": ["the-code"], "state": [state]}
        return True

    def _fake_post(url: str, fields: dict[str, str]) -> dict[str, str]:
        """Assert the PKCE + auth-code fields are present, then return tokens."""
        captured.update(fields)
        return {"access_token": "AT", "refresh_token": "RT"}

    monkeypatch.setattr(oauth2, "_CallbackServer", _FakeServer)
    monkeypatch.setattr(oauth2.webbrowser, "open", _fake_open)
    monkeypatch.setattr(oauth2, "_post_form", _fake_post)

    access, refresh = oauth2._auth_code_login("common", "cid", None, ["scope"])
    assert (access, refresh) == ("AT", "RT")
    # The exchange must carry the PKCE verifier and the authorization_code grant.
    assert captured["grant_type"] == oauth2._AUTH_GRANT
    assert captured["code"] == "the-code"
    assert "code_verifier" in captured
    assert captured["redirect_uri"].startswith("http://localhost:")
