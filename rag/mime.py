"""Shared content-based mime detection (libmagic + extension fallback).

Used by the documents extractor (module 1) and the archive translator (module 2)
so both name file types identically. Detection **content-sniffs** the bytes with
libmagic and only falls back to the filename extension when magic is
uninformative — content sniffing is a hard requirement, never a silent
extension-only downgrade. Callers layer their own overrides on top (the
documents extractor trusts a sidecar ``content_type`` first, for instance).
"""

from __future__ import annotations

import mimetypes
from typing import Any

from .errors import PluginError

# Returned when neither libmagic nor the extension can name a type.
FALLBACK_MIME: str = "application/octet-stream"

# Extension -> mime overrides applied before ``mimetypes.guess_type`` so the
# result does not depend on the host's ``mime.types``. libmagic content-sniffs
# markdown as plain ``text/plain`` (it has no markdown rule), so the extension is
# the only reliable signal for it.
_EXT_MIME: dict[str, str] = {
    ".md": "text/markdown",
    ".markdown": "text/markdown",
}

# Mime types libmagic emits when it can only tell "this is text" / "this is
# bytes". When magic returns one of these but the extension names something more
# specific, prefer the extension (that is subtype correctness, not a fallback).
_GENERIC_MIMES: frozenset[str] = frozenset({"text/plain", "application/octet-stream"})


def _require_magic() -> Any:
    """Import and return the ``magic`` module, or fail loudly with an install hint.

    The import is deliberately lazy (here, not at module top) so a missing
    libmagic breaks only the features that actually sniff content, not unrelated
    imports (plugin discovery, translator discovery, the other extractors).
    """
    try:
        import magic  # type: ignore[import-untyped]
    except ImportError as exc:  # module missing, or libmagic shared lib absent
        raise PluginError(
            "content-based mime detection needs libmagic. Install the Python "
            "binding ('pip install python-magic') and the system library "
            "(e.g. 'apt install libmagic1' / 'brew install libmagic')."
        ) from exc
    return magic


def _ext_suffix(name: str) -> str:
    """Return the lower-cased final extension of a file name (``""`` if none).

    Splits on the last dot of the basename only, so directory components in a
    path (``docs/note.md``) never contribute a spurious extension.
    """
    base = name.rsplit("/", 1)[-1]
    dot = base.rfind(".")
    return base[dot:].lower() if dot > 0 else ""


def _ext_mime(name: str) -> str | None:
    """Return an extension-based mime type for a file name, or ``None`` if unknown."""
    # Our built-in overrides win over the system mime.types database.
    override = _EXT_MIME.get(_ext_suffix(name))
    if override is not None:
        return override
    guessed, _encoding = mimetypes.guess_type(name)
    return guessed


class MimeDetector:
    """Names a file's mime type from its bytes and its name (libmagic + extension).

    Construct once and reuse: constructing it resolves libmagic eagerly (failing
    loudly if it is unavailable) so callers hit the missing-dependency error up
    front rather than on the first file.
    """

    def __init__(self) -> None:
        """Resolve libmagic now, so an absent library fails before any work starts."""
        self._magic: Any = _require_magic()

    def detect(self, name: str, data: bytes) -> str:
        """Return the mime type for ``data`` (named ``name``), combining magic + ext.

        libmagic content-sniffs the bytes; but when magic can only say
        ``text/plain`` / ``application/octet-stream`` and the extension names
        something more specific (e.g. markdown), the extension is used.
        :data:`FALLBACK_MIME` is the last resort.
        """
        # Content sniff (required) plus the extension guess.
        magic_mime = self._magic.from_buffer(data, mime=True) if data else None
        ext_mime = _ext_mime(name)
        # Prefer the extension only when magic is uninformative but the extension
        # is specific; otherwise magic wins, then extension, then the fallback.
        if magic_mime in _GENERIC_MIMES and ext_mime is not None and ext_mime not in _GENERIC_MIMES:
            return ext_mime
        return magic_mime or ext_mime or FALLBACK_MIME
