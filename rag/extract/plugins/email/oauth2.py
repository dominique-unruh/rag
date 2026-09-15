"""OAuth2 (XOAUTH2) credential handling for IMAP accounts — Microsoft 365.

Some IMAP servers (notably ``outlook.office365.com``) no longer accept a plain
password over ``LOGIN``/``PLAIN`` and require OAuth2 bearer tokens via the SASL
``XOAUTH2`` mechanism. This module turns an account's OAuth2 config into a live
*access token* usable for that mechanism.

Token model (see the discussion in PLAN §10):

* The **access token** is short-lived (~1h) and never persisted — it is fetched
  fresh on every run and kept only in memory.
* The **refresh token** is long-lived and *is* persisted, as one JSON file per
  account under the plugin's ``secrets/`` subdirectory. Microsoft rotates the
  refresh token on most refreshes, so the freshly returned one is written back
  every time; losing that write eventually strands the account on a stale token.

The very first login is interactive. The **authorization-code flow with a
loopback redirect and PKCE** (RFC 8252) is preferred: the CLI opens the system
browser, the user signs in, and the browser redirects to a short-lived
``http://localhost:<port>`` listener the CLI runs to capture the auth code. This
is what desktop mail clients (e.g. Thunderbird) actually use, and — crucially —
many locked-down M365 tenants block the *device-code* flow via Conditional Access
("authentication flow restricted by your admin") while still permitting this
interactive browser sign-in. The **device-code flow** (RFC 8628) remains as a
fallback for headless hosts with no launchable browser. Either way the resulting
refresh token seeds the ``secrets/`` store and every later run refreshes silently.
"""

from __future__ import annotations

import base64
import hashlib
import http.server
import json
import secrets
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import webbrowser
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from ....errors import PluginError
from ....interrupt import interrupt_requested

# Well-known *public* client id shipped by Mozilla Thunderbird. It is not a
# secret (it identifies the app, not the user), has public-client flows enabled
# (so device-code works without a client_secret), and is accepted by many M365
# tenants without the user registering their own Azure application. Overridable
# per account via ``oauth.client_id`` when a tenant blocks it.
_DEFAULT_CLIENT_ID: str = "9e5f94bc-e8a4-4e73-b8be-63364c29d753"

# Default Azure AD tenant. ``common`` lets both work/school and personal
# accounts authenticate; a single-tenant org can pin its own tenant id here.
_DEFAULT_TENANT: str = "common"

# Default OAuth2 scopes. ``IMAP.AccessAsUser.All`` grants IMAP access as the
# signed-in user; ``offline_access`` is what makes the server return a refresh
# token at all (without it there is nothing long-lived to persist). NOTE the
# resource host is ``outlook.office.com`` (the OAuth resource), NOT the IMAP
# server host ``outlook.office365.com`` — using the latter yields AADSTS70011.
_DEFAULT_SCOPES: tuple[str, ...] = (
    "https://outlook.office.com/IMAP.AccessAsUser.All",
    "offline_access",
)

# Microsoft identity-platform endpoint templates (v2.0). ``{tenant}`` is filled
# from config; the authorize, device-code and token exchanges hit this same host.
_AUTHORIZE_URL: str = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/authorize"
_DEVICECODE_URL: str = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/devicecode"
_TOKEN_URL: str = "https://login.microsoftonline.com/{tenant}/oauth2/v2.0/token"

# Grant type identifier for the RFC 8628 device-code token poll.
_DEVICE_GRANT: str = "urn:ietf:params:oauth:grant-type:device_code"

# Grant type identifier for the RFC 6749 authorization-code exchange.
_AUTH_GRANT: str = "authorization_code"

# Network timeout (seconds) for the token/device HTTP calls. Deliberately short:
# these are small requests to a highly-available endpoint.
_HTTP_TIMEOUT: float = 30.0

# How long (seconds) to wait for the user to complete the interactive browser
# sign-in and for the redirect to reach the loopback listener before giving up.
_REDIRECT_TIMEOUT: float = 300.0


def _post_form(url: str, fields: Mapping[str, str]) -> dict[str, Any]:
    """POST ``fields`` as ``application/x-www-form-urlencoded`` and parse JSON.

    Returns the decoded JSON object. OAuth2 error responses come back as HTTP
    400 with a JSON body carrying an ``error`` code, so an :class:`HTTPError`
    whose body is JSON is *returned* (not raised) — the caller inspects the
    ``error`` field. Any non-JSON failure is turned into a :class:`PluginError`.
    """
    # Encode the form body; the identity endpoint expects urlencoded pairs.
    data = urllib.parse.urlencode(fields).encode("utf-8")
    request = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}
    )
    try:
        with urllib.request.urlopen(request, timeout=_HTTP_TIMEOUT) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        # OAuth2 errors are 4xx with a JSON body ({"error": ..., ...}); surface
        # that structured body to the caller rather than treating it as fatal.
        body = exc.read().decode("utf-8", errors="replace")
        try:
            return json.loads(body)
        except json.JSONDecodeError:
            # A non-JSON HTTP error is a genuine transport/server fault.
            raise PluginError(
                f"OAuth2 request to {url} failed: HTTP {exc.code} {exc.reason}: {body}"
            ) from exc
    except urllib.error.URLError as exc:
        # DNS/connection-level failure — nothing to parse.
        raise PluginError(f"OAuth2 request to {url} failed: {exc.reason}") from exc


def _oauth_config(account: Mapping[str, Any]) -> tuple[str, str, list[str], str | None]:
    """Return ``(tenant, client_id, scopes, client_secret)`` for ``account``.

    Reads the account's optional ``oauth`` subtree, falling back to the built-in
    Thunderbird public client / ``common`` tenant / IMAP scopes. ``client_secret``
    is ``None`` for a public client (the normal case); it is only set for a
    confidential app registration that requires one.
    """
    oauth = account.get("oauth") or {}
    if not isinstance(oauth, Mapping):
        raise PluginError("account 'oauth' config must be a mapping")
    tenant = str(oauth.get("tenant", _DEFAULT_TENANT))
    client_id = str(oauth.get("client_id", _DEFAULT_CLIENT_ID))
    # Scopes may be given as a YAML list; default to the IMAP + offline_access set.
    configured_scopes = oauth.get("scopes")
    if configured_scopes is None:
        scopes = list(_DEFAULT_SCOPES)
    elif isinstance(configured_scopes, Sequence) and not isinstance(configured_scopes, str):
        scopes = [str(s) for s in configured_scopes]
    else:
        raise PluginError("account 'oauth.scopes' must be a list of scope strings")
    # Optional confidential-client secret; absent for the public-client default.
    secret = oauth.get("client_secret")
    client_secret = str(secret) if secret else None
    return tenant, client_id, scopes, client_secret


def _token_path(secrets_dir: Path, account_id: str) -> Path:
    """Return the refresh-token JSON file path for ``account_id``.

    Mirrors the sync-state file naming: the id is sanitised to a safe filename so
    an arbitrary ``account-id`` can never escape the ``secrets/`` directory.
    """
    import re  # local: only needed for the one-off filename sanitisation.

    # Replace anything outside a safe filename charset with an underscore.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id)
    return secrets_dir / f"{safe}.json"


def _load_refresh_token(path: Path) -> str | None:
    """Return the stored refresh token at ``path``, or ``None`` if absent."""
    if not path.exists():
        # No prior login for this account.
        return None
    data = json.loads(path.read_text())
    token = data.get("refresh_token")
    return str(token) if token else None


def _save_refresh_token(path: Path, refresh_token: str) -> None:
    """Persist ``refresh_token`` to ``path`` atomically with 0600 permissions.

    Written to a temp file then renamed so a crash mid-write cannot corrupt or
    truncate the stored token. The file is chmod'd to owner-only *before* the
    rename so the secret is never briefly world-readable.
    """
    # Ensure the containing secrets/ dir exists and is owner-only.
    path.parent.mkdir(parents=True, exist_ok=True)
    path.parent.chmod(0o700)
    tmp = path.with_suffix(".json.tmp")
    # Record when it was obtained for debuggability; only the token is load-bearing.
    tmp.write_text(json.dumps({"refresh_token": refresh_token}, indent=2))
    # Tighten permissions before publishing the file into place.
    tmp.chmod(0o600)
    tmp.replace(path)


def _refresh_access_token(
    tenant: str,
    client_id: str,
    client_secret: str | None,
    scopes: Sequence[str],
    refresh_token: str,
) -> tuple[str, str] | None:
    """Exchange a refresh token for a fresh ``(access_token, refresh_token)``.

    Returns the new access token together with the refresh token to persist (the
    rotated one if the server returned it, else the one we sent). Returns ``None``
    when the refresh token is no longer valid (``invalid_grant``) so the caller
    can fall back to an interactive device-code login.
    """
    fields: dict[str, str] = {
        "client_id": client_id,
        "grant_type": "refresh_token",
        "refresh_token": refresh_token,
        # Scope is echoed so the returned access token targets IMAP.
        "scope": " ".join(scopes),
    }
    # Confidential clients must present their secret; public clients omit it.
    if client_secret:
        fields["client_secret"] = client_secret
    result = _post_form(_TOKEN_URL.format(tenant=tenant), fields)
    if "access_token" in result:
        # Persist the rotated refresh token when present; otherwise keep the old.
        new_refresh = str(result.get("refresh_token") or refresh_token)
        return str(result["access_token"]), new_refresh
    # A stale/revoked refresh token reports invalid_grant -> re-login needed.
    if result.get("error") == "invalid_grant":
        return None
    # Any other error is unexpected (misconfig, tenant policy) -> surface it.
    raise PluginError(
        f"OAuth2 refresh failed for tenant {tenant!r}: "
        f"{result.get('error')}: {result.get('error_description', '')}"
    )


def _pkce_pair() -> tuple[str, str]:
    """Return a fresh ``(code_verifier, code_challenge)`` PKCE pair (S256).

    The verifier is a high-entropy URL-safe random string; the challenge is the
    base64url-encoded SHA-256 of the verifier with padding stripped, per RFC 7636.
    PKCE lets a *public* client (no secret) exchange the auth code securely.
    """
    # 64 random bytes -> ~86 url-safe chars, well within the 43..128 RFC range.
    verifier = secrets.token_urlsafe(64)
    digest = hashlib.sha256(verifier.encode("ascii")).digest()
    challenge = base64.urlsafe_b64encode(digest).rstrip(b"=").decode("ascii")
    return verifier, challenge


class _CallbackServer(http.server.HTTPServer):
    """Loopback HTTP server that captures a single OAuth2 redirect callback.

    The redirect lands as a browser ``GET http://localhost:<port>/?code=...``;
    the handler stashes the parsed query params here for the caller to read.
    """

    # Parsed query params of the captured redirect, or ``None`` until one arrives.
    result: dict[str, list[str]] | None

    def __init__(self, address: tuple[str, int], handler: type[http.server.BaseHTTPRequestHandler]) -> None:
        """Bind the loopback server and initialise the empty result slot."""
        super().__init__(address, handler)
        self.result = None


class _CallbackHandler(http.server.BaseHTTPRequestHandler):
    """One-shot handler: record the auth-code redirect and show a closing page."""

    def do_GET(self) -> None:  # noqa: N802 (name mandated by BaseHTTPRequestHandler)
        """Capture the ``code``/``error`` redirect; ignore incidental requests."""
        params = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        # Browsers also fetch /favicon.ico etc.; only a request carrying the
        # auth-code (or an error) is the redirect we are waiting for.
        if "code" not in params and "error" not in params:
            self.send_response(404)
            self.end_headers()
            return
        # Hand the params up to the server instance for the polling loop to read.
        server: _CallbackServer = self.server  # type: ignore[assignment]
        server.result = params
        body = (
            b"<html><body><h3>Sign-in complete.</h3>"
            b"<p>You may close this tab and return to the terminal.</p>"
            b"</body></html>"
        )
        self.send_response(200 if "code" in params else 400)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002 (base API name)
        """Silence the default stderr access log; it would clutter CLI output."""


def _browser_available() -> bool:
    """Return whether a launchable web browser exists for the auth-code flow.

    Falls back to the device-code flow when this is ``False`` (e.g. a headless
    server with no ``$DISPLAY`` and no registered browser).
    """
    try:
        webbrowser.get()
        return True
    except webbrowser.Error:
        return False


def _auth_code_login(
    tenant: str,
    client_id: str,
    client_secret: str | None,
    scopes: Sequence[str],
) -> tuple[str, str]:
    """Run the RFC 8252 auth-code + PKCE loopback flow; return ``(access, refresh)``.

    Binds an ephemeral ``http://localhost:<port>`` listener, opens the system
    browser at the authorize endpoint, waits for the redirect carrying the auth
    code, then exchanges that code (with the PKCE verifier) for tokens. Preferred
    over device-code because Conditional Access often blocks device-code while
    allowing interactive browser sign-in. Honours a graceful interrupt / timeout
    while waiting for the user.
    """
    verifier, challenge = _pkce_pair()
    # Opaque value echoed back in the redirect; guards against a stray/forged callback.
    state = secrets.token_urlsafe(16)

    # 1. Bind the loopback listener on an OS-assigned port BEFORE building the URL,
    #    since the redirect_uri must carry that exact port. 127.0.0.1 is reached via
    #    the registered ``http://localhost`` redirect (Microsoft allows any port).
    server = _CallbackServer(("127.0.0.1", 0), _CallbackHandler)
    # handle_request() returns after this long without a request, so the poll loop
    # can re-check the interrupt flag and deadline between browser round-trips.
    server.timeout = 1.0
    try:
        port = server.server_address[1]
        redirect_uri = f"http://localhost:{port}"

        # 2. Build the authorize URL and send the user there.
        query = urllib.parse.urlencode(
            {
                "client_id": client_id,
                "response_type": "code",
                "redirect_uri": redirect_uri,
                "scope": " ".join(scopes),
                "state": state,
                "code_challenge": challenge,
                "code_challenge_method": "S256",
            }
        )
        authorize_url = f"{_AUTHORIZE_URL.format(tenant=tenant)}?{query}"
        print(
            "Opening your browser to sign in. If it does not open, visit this URL "
            f"manually:\n{authorize_url}",
            file=sys.stderr,
        )
        # Best-effort auto-open; the printed URL covers the case where it fails.
        webbrowser.open(authorize_url)

        # 3. Serve requests until the redirect arrives, the user interrupts, or we
        #    time out waiting for them to finish in the browser.
        deadline = time.monotonic() + _REDIRECT_TIMEOUT
        while server.result is None:
            if interrupt_requested():
                raise PluginError("OAuth2 browser login interrupted before completion")
            if time.monotonic() > deadline:
                raise PluginError("OAuth2 browser login timed out waiting for sign-in")
            server.handle_request()
        params = server.result
    finally:
        # Release the port promptly regardless of outcome.
        server.server_close()

    # 4. Validate the callback: an OAuth2 error, a mismatched state, or a missing
    #    code all mean we must not proceed to the token exchange.
    if "error" in params:
        raise PluginError(
            f"OAuth2 browser login failed: {params['error'][0]}: "
            f"{params.get('error_description', [''])[0]}"
        )
    if params.get("state", [None])[0] != state:
        raise PluginError("OAuth2 browser login failed: redirect state mismatch")
    code = params["code"][0]

    # 5. Exchange the auth code (+ PKCE verifier) for tokens.
    fields: dict[str, str] = {
        "client_id": client_id,
        "grant_type": _AUTH_GRANT,
        "code": code,
        "redirect_uri": redirect_uri,
        "code_verifier": verifier,
        "scope": " ".join(scopes),
    }
    if client_secret:
        fields["client_secret"] = client_secret
    result = _post_form(_TOKEN_URL.format(tenant=tenant), fields)
    if "access_token" not in result:
        raise PluginError(
            f"OAuth2 code exchange failed: {result.get('error')}: "
            f"{result.get('error_description', '')}"
        )
    return str(result["access_token"]), str(result.get("refresh_token", ""))


def _device_code_login(
    tenant: str,
    client_id: str,
    client_secret: str | None,
    scopes: Sequence[str],
) -> tuple[str, str]:
    """Run the RFC 8628 device-code flow; return ``(access_token, refresh_token)``.

    Requests a device/user code, prints the verification URL + code to stderr,
    then polls the token endpoint until the user approves (or the code expires).
    Honours a graceful interrupt between polls.
    """
    # 1. Ask for a device code + user code for these scopes.
    start = _post_form(
        _DEVICECODE_URL.format(tenant=tenant),
        {"client_id": client_id, "scope": " ".join(scopes)},
    )
    if "device_code" not in start:
        raise PluginError(
            f"OAuth2 device-code request failed for tenant {tenant!r}: "
            f"{start.get('error')}: {start.get('error_description', '')}"
        )
    device_code = str(start["device_code"])
    # The server provides a ready-made human instruction message; show it verbatim.
    print(start.get("message", "Follow the URL and enter the code to sign in."), file=sys.stderr)
    # Poll cadence and overall deadline come from the server response.
    interval = int(start.get("interval", 5))
    deadline = time.monotonic() + int(start.get("expires_in", 900))

    poll_fields: dict[str, str] = {
        "client_id": client_id,
        "grant_type": _DEVICE_GRANT,
        "device_code": device_code,
    }
    if client_secret:
        poll_fields["client_secret"] = client_secret

    # 2. Poll until the user approves, the code expires, or we are interrupted.
    while True:
        if interrupt_requested():
            raise PluginError("OAuth2 device-code login interrupted before approval")
        if time.monotonic() > deadline:
            raise PluginError("OAuth2 device-code login timed out waiting for approval")
        # Wait the server-mandated interval before each poll (avoids being throttled).
        time.sleep(interval)
        result = _post_form(_TOKEN_URL.format(tenant=tenant), poll_fields)
        if "access_token" in result:
            # Approved: hand back both tokens for use + persistence.
            return str(result["access_token"]), str(result.get("refresh_token", ""))
        error = result.get("error")
        if error == "authorization_pending":
            # User has not finished yet; keep polling at the same cadence.
            continue
        if error == "slow_down":
            # Server asks us to back off; RFC 8628 says add 5s to the interval.
            interval += 5
            continue
        # authorization_declined / expired_token / bad_verification_code / other.
        raise PluginError(
            f"OAuth2 device-code login failed: {error}: {result.get('error_description', '')}"
        )


def _interactive_login(
    tenant: str,
    client_id: str,
    client_secret: str | None,
    scopes: Sequence[str],
) -> tuple[str, str]:
    """Run whichever interactive login the environment supports.

    Prefers the auth-code loopback flow (works even on Conditional-Access tenants
    that block device-code); falls back to the device-code flow only when no
    browser can be launched, e.g. a headless host.
    """
    if _browser_available():
        return _auth_code_login(tenant, client_id, client_secret, scopes)
    # No launchable browser: the loopback redirect could never arrive, so use the
    # copy-a-code device flow instead (note: some tenants block this one).
    print(
        "No usable web browser found; falling back to the device-code flow.",
        file=sys.stderr,
    )
    return _device_code_login(tenant, client_id, client_secret, scopes)


def access_token(
    account: Mapping[str, Any],
    secrets_dir: Path,
    *,
    allow_prompt: bool,
) -> str:
    """Return a live XOAUTH2 access token for ``account``.

    Resolution order:

    1. If a refresh token is stored under ``secrets_dir``, silently exchange it
       for a fresh access token (persisting the rotated refresh token).
    2. Otherwise — or if the stored token is no longer valid — run the interactive
       login (auth-code loopback, or device-code when no browser is available),
       but only when ``allow_prompt`` (a TTY is available); headless runs raise
       instead of blocking on a login they cannot complete.

    The returned access token is ephemeral and kept only by the caller for the
    duration of the IMAP connection.
    """
    account_id = str(account["account-id"])
    tenant, client_id, scopes, client_secret = _oauth_config(account)
    token_path = _token_path(secrets_dir, account_id)

    # 1. Try a silent refresh from the persisted refresh token.
    stored = _load_refresh_token(token_path)
    if stored is not None:
        refreshed = _refresh_access_token(tenant, client_id, client_secret, scopes, stored)
        if refreshed is not None:
            access, new_refresh = refreshed
            # Write back the (possibly rotated) refresh token so the chain survives.
            _save_refresh_token(token_path, new_refresh)
            return access
        # Stored token rejected (expired/revoked) -> fall through to a fresh login.

    # 2. No usable refresh token: interactive login (TTY only).
    if not allow_prompt:
        raise PluginError(
            f"account {account_id!r} needs an interactive OAuth2 login but none is "
            f"possible (no TTY). Run `rag extract email account={account_id}` once "
            f"in a terminal to authorise it, then re-run headless."
        )
    access, new_refresh = _interactive_login(tenant, client_id, client_secret, scopes)
    if not new_refresh:
        # Without offline_access the server returns no refresh token; every run
        # would then require a fresh interactive login — treat as misconfig.
        raise PluginError(
            f"OAuth2 login for {account_id!r} returned no refresh token; ensure "
            f"'offline_access' is among the configured 'oauth.scopes'."
        )
    _save_refresh_token(token_path, new_refresh)
    return access
