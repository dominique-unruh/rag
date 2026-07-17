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

import re
from collections.abc import Mapping
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

from ....errors import ConfigError
from ....interrupt import interrupt_requested
from ....secrets import Secrets
from ....timestamps import format_timestamp
from ...base import Extractor, Store
from .imap import extract_account

# MIME type recorded for the raw email blobs.
_EMAIL_MIME: str = "message/rfc822"

# Base URL for mail-link source links (the part before the '#'; the mail-link
# format leaves it unspecified). See
# https://github.com/dominique-unruh/mail-link/wiki/Mail-link-format
_MAIL_LINK_BASE: str = "https://qis.rwth-aachen.de/people/unruh/tools/mail-link/"

# Mapping of mail-link fragment field name -> stable-metadata key that fills it.
# The mail-link "subject" field is sourced from the global `title` key.
_MAIL_LINK_FIELDS: tuple[tuple[str, str], ...] = (
    ("subject", "title"),
    ("date", "date"),
    ("from", "sender"),
    ("to", "recipients"),
)

def _format_address(addr: Address) -> str:
    """Render one parsed :class:`Address` as ``NAME <EMAIL>`` (or just ``EMAIL``)."""
    # headerregistry already RFC 2047-decodes the display name for us.
    name = addr.display_name
    return f"{name} <{addr.addr_spec}>" if name else addr.addr_spec


def _address_field(msg: EmailMessage, *fields: str) -> str:
    """Return a cleaned, comma-joined address list from the given header ``fields``.

    Relies on the stdlib ``policy.default`` header parser (``AddressHeader``),
    which handles encoded words, groups and malformed input robustly.
    """
    out: list[str] = []
    for field in fields:
        header = msg[field]
        if header is None:
            continue
        # ``.addresses`` is provided by the AddressHeader value type.
        for addr in header.addresses:
            if addr.addr_spec:
                out.append(_format_address(addr))
    return ", ".join(out)


def _received_datetime(msg: EmailMessage) -> "datetime | None":
    """Approximate the message date from the ``Received`` trace headers.

    Each ``Received`` header ends with ``; <date>``. Headers are tried in order
    (the first/topmost — added last in transit, closest to delivery — wins), and
    the first parseable date is returned. ``None`` if none parse.
    """
    for received in msg.get_all("Received", []):
        text = str(received)
        if ";" not in text:
            continue
        # The timestamp is the part after the final ';'.
        candidate = text.rsplit(";", 1)[1].strip()
        try:
            moment = parsedate_to_datetime(candidate)
        except (TypeError, ValueError):
            continue
        if moment is not None:
            return moment
    return None


def _message_ids(header: object) -> list[str]:
    """Extract bare message-ids from a References / In-Reply-To header value.

    Returns the ids without angle brackets (matching the ``doc_id`` form), in
    order. Falls back to whitespace-splitting if the header has no ``<...>``
    tokens. Empty list when the header is absent/empty.
    """
    if header is None:
        return []
    text = str(header)
    # Message-ids are normally wrapped in angle brackets; grab those first.
    ids = re.findall(r"<([^>]+)>", text)
    if not ids:
        # Malformed header without brackets — best-effort whitespace split.
        ids = text.split()
    return [i.strip() for i in ids if i.strip()]


def _header_datetime(msg: EmailMessage) -> "datetime | None":
    """Return the message's timezone-aware datetime, or ``None``.

    Prefers the ``Date`` header (``DateHeader.datetime``); when that is missing or
    unparseable, falls back to an approximation from the ``Received`` trace
    headers. ``None`` only if neither yields a usable date.
    """
    header = msg["Date"]
    moment = getattr(header, "datetime", None) if header is not None else None
    if moment is not None:
        return moment
    # Date header absent/invalid — approximate from the Received trace.
    return _received_datetime(msg)


def build_stable_metadata(msg: EmailMessage) -> dict[str, Any]:
    """Build the content-defining metadata for a parsed email ``msg``.

    ``msg`` must be parsed with ``policy=email.policy.default`` so headers are
    decoded and addresses/dates come back structured. Keys: ``sender``,
    ``recipients`` (To + Cc), the global ``title`` (= the decoded Subject),
    ``mime-type``, ``message-id`` (the message's own bare id, when present) and —
    only when the Date header parses — ``date`` in the shared canonical format
    (:func:`rag.timestamps.format_timestamp`). The owning account is deliberately
    *not* here (it is transient).
    """
    subject = msg["Subject"]
    meta: dict[str, Any] = {
        "sender": _address_field(msg, "From"),
        "recipients": _address_field(msg, "To", "Cc"),
        # The email's title is its Subject (global `title` key). str() of a decoded
        # header yields the Unicode subject ("" when absent).
        "title": str(subject) if subject is not None else "",
        "mime-type": _EMAIL_MIME,
    }
    # The message's own id (bare, no angle brackets — same form as the doc_id and
    # the threading headers), when the header is present.
    own_ids = _message_ids(msg["Message-ID"])
    if own_ids:
        meta["message-id"] = own_ids[0]
    # Store the date only if it parsed; normalise to the canonical format so it
    # matches every other extractor. Omit it entirely otherwise.
    moment = _header_datetime(msg)
    if moment is not None:
        meta["date"] = format_timestamp(moment)
    # Threading headers as lists of bare message-ids (present ones only). These
    # match the doc_id form, so they reference other messages.
    references = _message_ids(msg["References"])
    if references:
        meta["references"] = references
    in_reply_to = _message_ids(msg["In-Reply-To"])
    if in_reply_to:
        meta["in-reply-to"] = in_reply_to
    return meta


def build_transient_metadata(folder: str, account_id: str, *, read: bool) -> dict[str, Any]:
    """Build the volatile IMAP state for a message.

    ``account`` is where the message currently resides, ``folder`` the mailbox
    within it, and ``read`` reflects the ``\\Seen`` flag. All are transient:
    moving the message between accounts/folders or reading it updates the row
    without bumping the version.
    """
    # TODO: add IMAP UID
    # TODO: add other IMAP flags (starred? important?), not sure which ones there are.
    return {"account": account_id, "folder": folder, "read": read}


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

    def source_url(
        self, doc_id: str, stable: Mapping[str, Any], transient: Mapping[str, Any]
    ) -> str | None:
        """Return a mail-link URL for the message.

        Format: ``<base>#<message-id>(&field=value)*`` with each component
        percent-encoded UTF-8 (spaces as ``+``). The ``doc_id`` is the Message-ID;
        the fragment also carries subject/date/from/to when available. See
        https://github.com/dominique-unruh/mail-link/wiki/Mail-link-format
        """
        # The message-id heads the fragment (fully percent-encoded).
        parts = [quote(doc_id, safe="")]
        # Append the descriptive fields that are present in stable metadata.
        for field, key in _MAIL_LINK_FIELDS:
            value = stable.get(key)
            if value:
                parts.append(f"{field}={quote_plus(str(value))}")
        return _MAIL_LINK_BASE + "#" + "&".join(parts)

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
        # The count budget is shared across ALL accounts (count=1 means one item
        # total, not one per account). None = unlimited.
        remaining: int | None = int(options["count"]) if options.get("count") is not None else None
        for account in accounts:
            account_id = str(account["account-id"])
            # Skip accounts the caller filtered out.
            if only_account is not None and account_id != only_account:
                continue
            # Stop once the shared budget is exhausted.
            if remaining is not None and remaining <= 0:
                break
            # Delegate one account's extraction; thread the leftover budget onward.
            remaining = extract_account(
                account=account,
                store=store,
                state_dir=state_dir,
                secrets=secrets,
                options=options,
                remaining=remaining,
            )
            # A graceful interrupt stops the whole run, not just one account.
            if interrupt_requested():
                break
