"""Secret resolution (PLAN §10).

Secrets never live in the YAML config or the repo. They are resolved, in order,
from the desktop keyring, then an optional file, and finally — only when running
interactively — an input prompt whose answer is cached back into the keyring.
Both the core and extraction plugins use this same mechanism.
"""

from __future__ import annotations

import sys
from getpass import getpass
from pathlib import Path

import keyring

from .errors import SecretNotFoundError

# Keyring "service" namespace under which all of this project's secrets are
# stored, so they are grouped and do not collide with other applications.
_SERVICE: str = "rag"


class Secrets:
    """Resolver for passwords / API keys shared by the core and plugins."""

    def __init__(
        self,
        service: str = _SERVICE,
        *,
        allow_prompt: bool | None = None,
        reauth: bool = False,
    ) -> None:
        """Create a resolver.

        ``service`` is the keyring namespace. ``allow_prompt`` controls whether an
        interactive prompt is used as a last resort; when left ``None`` it is
        auto-detected from whether stdin is a TTY (so batch/headless runs never
        block on input). ``reauth`` re-verifies each stored secret: before using a
        keyring value it asks whether to keep it, letting the user replace a stale
        one (e.g. an outdated password) without deleting anything by hand.
        """
        # Keyring service namespace for stored secrets.
        self._service: str = service
        # Whether we may fall back to an interactive prompt. Auto-detect from the
        # terminal unless the caller forced a value.
        self._allow_prompt: bool = sys.stdin.isatty() if allow_prompt is None else allow_prompt
        # Re-verify stored secrets interactively (only meaningful when prompting).
        self._reauth: bool = reauth

    def _reauth_prompt(self, key: str, prompt_text: str | None) -> str | None:
        """Ask whether to keep the stored secret for ``key``; replace it if not.

        Returns the new secret (already written back to the keyring) if the user
        chose to re-enter it, or ``None`` to keep using the stored value. An empty
        replacement is treated as "keep" so an accidental blank never wipes it.
        """
        # Default to keeping the stored value (Enter == yes) so the common case is
        # a single keystroke; only an explicit "n" triggers re-entry.
        answer = input(f"Use stored secret for {key!r}? [Y/n] ").strip().lower()
        if answer not in ("n", "no"):
            return None
        value = getpass(prompt_text or f"Enter new secret for {key!r}: ")
        if not value:
            # Blank input: don't overwrite with nothing — keep the stored secret.
            return None
        # Persist the replacement so later runs (and the rest of this run) use it.
        keyring.set_password(self._service, key, value)
        return value

    def get(
        self,
        key: str,
        *,
        file: Path | None = None,
        prompt_text: str | None = None,
    ) -> str:
        """Resolve the secret named ``key``.

        Resolution order:
        1. the keyring entry ``(service, key)``;
        2. the first line of ``file`` (if given and present);
        3. an interactive prompt (only when prompting is allowed), whose value is
           written back to the keyring for next time.

        Raises :class:`SecretNotFoundError` when every source is exhausted.
        """
        # 1. Desktop keyring — the normal persistent store.
        from_keyring = keyring.get_password(self._service, key)
        if from_keyring:
            # In reauth mode, give the user a chance to replace a stale stored
            # secret (e.g. a changed password) without deleting it by hand. Only
            # possible when we can prompt; otherwise the stored value stands.
            if self._reauth and self._allow_prompt:
                replaced = self._reauth_prompt(key, prompt_text)
                if replaced is not None:
                    return replaced
            return from_keyring

        # 2. Plain file fallback (e.g. a 0600 secrets file on a server).
        if file is not None:
            path = file.expanduser()
            if path.exists():
                # Only the first line is the secret; trailing newlines are stripped.
                return path.read_text().splitlines()[0]

        # 3. Interactive prompt, caching the answer into the keyring.
        if self._allow_prompt:
            value = getpass(prompt_text or f"Enter secret for {key!r}: ")
            if value:
                # Persist so the user is only ever prompted once.
                keyring.set_password(self._service, key, value)
                return value

        raise SecretNotFoundError(
            f"could not resolve secret {key!r} (tried keyring/file; prompting "
            f"{'disabled' if not self._allow_prompt else 'produced no value'})"
        )
