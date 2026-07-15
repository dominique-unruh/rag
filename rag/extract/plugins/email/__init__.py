"""Email extraction plugin (``source_type`` = ``email``) — PLAN §4.4.

Extracts messages from one or more configured IMAP accounts. Each message becomes
a document with ``doc_id = "<account-id>:<message-id>"`` and a blob holding the
full raw RFC 5322 message. Stable metadata (date/sender/recipients/subject/
account) defines the content; transient metadata (folder, read flag) tracks
volatile IMAP state without bumping the version.

The heavy lifting (IMAP I/O, sync-state bookkeeping) lives in
:mod:`rag.extract.plugins.email.imap`; this module holds the pure parsing
helpers and the :class:`EmailExtractor` entry point so the parsing is unit
testable without a live server.
"""

from __future__ import annotations

from collections.abc import Mapping
from email.message import Message
from email.utils import getaddresses, parsedate_to_datetime, parseaddr
from email.header import decode_header
from pathlib import Path
from typing import Any

from ....errors import ConfigError
from ....secrets import Secrets
from ...base import Extractor, Store
from .imap import extract_account

# MIME type recorded for the raw email blobs.
_EMAIL_MIME: str = "message/rfc822"


def _decode_mime_words(raw: str | None) -> str:
    """Decode RFC 2047 encoded-word header text into a plain Unicode string.

    Handles headers like ``=?utf-8?q?...?=``; unknown charsets fall back to
    latin-1 so we never crash on odd senders. Returns ``""`` for ``None``.
    """
    if raw is None:
        return ""
    parts: list[str] = []
    # decode_header splits the header into (bytes-or-str, charset) chunks.
    for chunk, charset in decode_header(raw):
        if isinstance(chunk, bytes):
            # Decode with the declared charset, tolerating errors/garbage.
            parts.append(chunk.decode(charset or "utf-8", errors="replace"))
        else:
            parts.append(chunk)
    # Join and collapse internal whitespace/newlines from folded headers.
    return " ".join("".join(parts).split())


def _clean_address(raw: str) -> str:
    """Return a single address as ``NAME <EMAIL>`` (or just ``EMAIL`` if no name)."""
    # parseaddr splits "Display <addr>" into (name, addr).
    name, addr = parseaddr(raw)
    name = _decode_mime_words(name)
    if name:
        return f"{name} <{addr}>"
    return addr


def _clean_address_list(raw: str) -> str:
    """Return a comma-separated cleaned list of addresses from a header value."""
    # getaddresses handles multiple comma-separated addresses robustly.
    cleaned: list[str] = []
    for name, addr in getaddresses([raw or ""]):
        if not addr:
            # Skip empty/garbage entries produced by malformed headers.
            continue
        decoded_name = _decode_mime_words(name)
        cleaned.append(f"{decoded_name} <{addr}>" if decoded_name else addr)
    return ", ".join(cleaned)


def _header_date_iso(msg: Message) -> str:
    """Return the message ``Date`` header as an ISO-8601 string, or ``""``."""
    raw = msg.get("Date")
    if not raw:
        return ""
    try:
        # parsedate_to_datetime yields an aware datetime for well-formed headers.
        return parsedate_to_datetime(raw).isoformat()
    except (TypeError, ValueError):
        # Malformed Date header: better to record nothing than a wrong value.
        return ""


def build_stable_metadata(msg: Message, account_id: str) -> dict[str, Any]:
    """Build the content-defining metadata for a parsed email ``msg``.

    Keys: ``date``, ``sender``, ``recipients`` (To + Cc), ``subject`` and the
    owning ``account``. All are stable — a change means the message content
    genuinely changed.
    """
    # Combine To and Cc into one recipients list for a simple, searchable field.
    to = msg.get("To", "")
    cc = msg.get("Cc", "")
    combined = ", ".join(part for part in (to, cc) if part)
    return {
        "date": _header_date_iso(msg),
        "sender": _clean_address(msg.get("From", "")),
        "recipients": _clean_address_list(combined),
        "subject": _decode_mime_words(msg.get("Subject")),
        "account": account_id,
        "mime-type": _EMAIL_MIME,
    }


def build_transient_metadata(folder: str, *, read: bool) -> dict[str, Any]:
    """Build the volatile IMAP state for a message.

    ``folder`` is where the message currently lives; ``read`` reflects the
    ``\\Seen`` flag. Changing either updates the row without bumping the version.
    """
    return {"folder": folder, "read": read}


def _accounts_from_config(config: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    """Return the list of account mappings from the plugin config.

    Validates that ``accounts`` is present and is a list of mappings, each with a
    non-empty ``account-id``.
    """
    accounts = config.get("accounts")
    if not isinstance(accounts, list) or not accounts:
        raise ConfigError("email plugin requires a non-empty 'accounts' list in its config")
    for account in accounts:
        if not isinstance(account, Mapping) or not account.get("account-id"):
            raise ConfigError("each email account needs a mapping with an 'account-id'")
    return accounts


class EmailExtractor(Extractor):
    """Extractor that syncs one or more IMAP accounts into the store."""

    def run(
        self,
        store: Store,
        state_dir: Path,
        config: Mapping[str, Any],
        secrets: Secrets,
        **options: Any,
    ) -> None:
        """Reconcile configured IMAP accounts into ``store``.

        Honours the ``account`` option to restrict to a single account; all other
        options (``since``/``until``/``count``/``folder``) are passed through to
        the per-account sync. Delegates the IMAP work to
        :func:`~rag.extract.plugins.email.imap.extract_account`.
        """
        accounts = _accounts_from_config(config)
        # Optional restriction to a single configured account.
        only_account = options.get("account")
        for account in accounts:
            account_id = str(account["account-id"])
            # Skip accounts the caller filtered out.
            if only_account is not None and account_id != only_account:
                continue
            # Delegate a single account's extraction (connect, sync, upsert/delete).
            extract_account(
                account=account,
                store=store,
                state_dir=state_dir,
                secrets=secrets,
                options=options,
            )
