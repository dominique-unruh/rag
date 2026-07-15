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

    def __init__(self, service: str = _SERVICE, *, allow_prompt: bool | None = None) -> None:
        """Create a resolver.

        ``service`` is the keyring namespace. ``allow_prompt`` controls whether an
        interactive prompt is used as a last resort; when left ``None`` it is
        auto-detected from whether stdin is a TTY (so batch/headless runs never
        block on input).
        """
        # Keyring service namespace for stored secrets.
        self._service: str = service
        # Whether we may fall back to an interactive prompt. Auto-detect from the
        # terminal unless the caller forced a value.
        self._allow_prompt: bool = sys.stdin.isatty() if allow_prompt is None else allow_prompt

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
