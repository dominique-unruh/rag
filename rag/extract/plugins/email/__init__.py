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

import json
import re
from collections.abc import Mapping
from datetime import datetime
from email.headerregistry import Address
from email.message import EmailMessage
from email.utils import parsedate_to_datetime
from pathlib import Path
from typing import Any
from urllib.parse import quote, quote_plus

from ....errors import ConfigError, PluginError
from ....fsck import FixMode, ask_fix, print_problem
from ....interrupt import interrupt_requested
from ....secrets import Secrets
from ....timestamps import format_timestamp
from ...base import Extractor, Store
from .imap import (
    _SECRETS_SUBDIR,
    _UIDMAP_FILENAME,
    _entry_message_id,
    _load_uidmap,
    _save_uidmap,
    _truthy,
    collect_extracted_mids,
    extract_account,
    run_deletions,
)

# MIME type recorded for the raw email blobs.
_EMAIL_MIME: str = "message/rfc822"

# Every option key the email extractor accepts. A key=value token outside this
# set is rejected (a typo like ``limit=1`` instead of ``count=1`` would otherwise
# be silently ignored, changing behaviour with no warning). ``verbose`` is the
# reserved flag the CLI injects from ``-v``; the rest are user-facing.
_USER_OPTIONS: tuple[str, ...] = (
    "since", "until", "count", "folder", "account", "full", "delete",
)
_KNOWN_OPTIONS: frozenset[str] = frozenset(_USER_OPTIONS) | {"verbose"}

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


def _clear_extracted(data: Mapping[str, Any], message_id: str) -> int:
    """Set ``extracted=false`` on every uid entry with ``message_id`` in ``data``.

    Mutates the loaded presence-file mapping in place and returns how many entries
    were changed. Used by :meth:`EmailExtractor.fsck` to repair an invariant-2
    violation (more than one uid flagged extracted for one message-id).
    """
    cleared = 0
    for account_entry in data.get("accounts", {}).values():
        folders = account_entry.get("folders", {}) if isinstance(account_entry, dict) else {}
        for folder_entry in folders.values():
            uids = folder_entry.get("uids", {}) if isinstance(folder_entry, dict) else {}
            for entry in uids.values():
                # Only object-form entries carry the flag; clear the true ones.
                if isinstance(entry, dict) and entry.get("message-id") == message_id and entry.get("extracted"):
                    entry["extracted"] = False
                    cleared += 1
    return cleared


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
        :func:`~rag.extract.plugins.email.imap.extract_account`. An unrecognised
        option key raises :class:`PluginError` rather than being ignored.

        With ``delete=true`` this instead runs the opt-in cross-account deletion
        pass (:func:`~rag.extract.plugins.email.imap.run_deletions`): it scans
        **every** account to refresh the joint UID map and removes documents whose
        message-id survives nowhere. That pass is separate because it is expensive
        (it connects to all accounts) and inherently global, so it ignores the
        ``account``/``folder``/``count`` filters.
        """
        # Reject unrecognised options up front so a mistyped key (e.g. ``limit=1``
        # for ``count=1``) fails loudly instead of being silently ignored.
        unknown = sorted(set(options) - _KNOWN_OPTIONS)
        if unknown:
            raise PluginError(
                f"unknown option(s): {', '.join(unknown)}. "
                f"Recognised options: {', '.join(_USER_OPTIONS)}."
            )
        accounts = _accounts_from_config(config)
        # Opt-in deletion pass: expensive + global, handled entirely separately
        # from the normal additive sync.
        if _truthy(options.get("delete")):
            run_deletions(
                accounts=accounts, store=store, state_dir=state_dir,
                secrets=secrets, options=options,
            )
            return
        # Optional restriction to a single configured account.
        only_account = options.get("account")
        # The count budget is shared across ALL accounts (count=1 means one item
        # total, not one per account). None = unlimited.
        remaining: int | None = int(options["count"]) if options.get("count") is not None else None
        # Load the joint UID map ONCE and share it (plus the extracted-set) across
        # every account. Invariant 2 is global — a message-id present in several
        # accounts must be stored, and flagged extracted, exactly once — so dedup
        # state cannot be per-account. ``full`` starts from an empty set so every
        # in-scope message is re-fetched (dedup still applies within the run). Only
        # loaded when there is work: no accounts => nothing to read from disk.
        if accounts:
            uidmap = _load_uidmap(state_dir)
            extracted_mids: set[str] = (
                set() if _truthy(options.get("full")) else collect_extracted_mids(uidmap)
            )
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
                uidmap=uidmap,
                extracted_mids=extracted_mids,
            )
            # A graceful interrupt stops the whole run, not just one account.
            if interrupt_requested():
                break

    def statistics(self, state_dir: Path, config: Mapping[str, Any]) -> dict[str, Any]:
        """Report what the email plugin is tracking, from its state dir only.

        Reads the joint UID map (and the ``secrets/`` dir) — no IMAP, no network.
        Per account: number of folders and tracked UIDs; plus the count of
        distinct message-ids across all accounts (the dedup'd message total) and
        how many accounts have a stored OAuth2 refresh token.
        """
        # Load the joint uid->message-id map (empty skeleton if never synced).
        uidmap = _load_uidmap(state_dir)
        accounts_stats: dict[str, Any] = {}
        # Collect distinct message-ids across everything for a dedup'd total.
        all_message_ids: set[str] = set()
        total_uids = 0
        for account_id, account_entry in sorted(uidmap.get("accounts", {}).items()):
            folders = account_entry.get("folders", {})
            # Sum UID entries and gather message-ids for this account.
            account_uids = 0
            for folder_entry in folders.values():
                uids = folder_entry.get("uids", {})
                account_uids += len(uids)
                # Entries are {"message-id", "extracted"} objects (or legacy bare
                # doc_id strings); _entry_message_id reads both shapes.
                all_message_ids.update(_entry_message_id(e) for e in uids.values())
            total_uids += account_uids
            accounts_stats[account_id] = {"folders": len(folders), "tracked_uids": account_uids}
        # Count accounts with a persisted OAuth2 refresh token (one file each).
        secrets_dir = state_dir / _SECRETS_SUBDIR
        oauth_tokens = len(list(secrets_dir.glob("*.json"))) if secrets_dir.is_dir() else 0
        return {
            "accounts": accounts_stats,
            "tracked_uids_total": total_uids,
            "distinct_message_ids": len(all_message_ids),
            "oauth_tokens_stored": oauth_tokens,
        }

    def fsck(self, state_dir: Path, config: Mapping[str, Any], fix: FixMode) -> int:
        """Validate the presence file (``uidmap.json``) shape and its invariants.

        Checks, without touching the network: the file is valid JSON with the
        expected ``accounts -> folders -> uids`` nesting; every uid entry is a
        ``{"message-id": str, "extracted": bool}`` object; and invariant 2 — no two
        uids may carry ``extracted: true`` for the same message-id (only one stored
        copy exists per message-id). Prints each problem as found and returns the
        error count.

        Fixable: invariant-2 violations. Since we cannot know which uid was the
        stored copy, the fix sets ``extracted=false`` for **all** uids of that
        message-id (a normal extract then re-stores exactly one).
        """
        errors = 0
        path = state_dir / _UIDMAP_FILENAME
        if not path.exists():
            # Nothing tracked yet is a valid, healthy state.
            return 0
        # Read the raw file so a malformed JSON is itself reported (rather than
        # silently swallowed by the tolerant loader).
        try:
            data = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            print_problem(f"[email] {_UIDMAP_FILENAME} is unreadable or not valid JSON: {exc}")
            return 1
        accounts = data.get("accounts") if isinstance(data, dict) else None
        if not isinstance(accounts, dict):
            print_problem(f"[email] {_UIDMAP_FILENAME}: top-level 'accounts' is missing or not an object")
            return 1
        # message-id -> number of uids flagged extracted=true (for invariant 2).
        true_holders: dict[str, int] = {}
        for account_id, account_entry in accounts.items():
            folders = account_entry.get("folders") if isinstance(account_entry, dict) else None
            if not isinstance(folders, dict):
                print_problem(f"[email] account {account_id!r}: 'folders' is missing or not an object")
                errors += 1
                continue
            for folder, folder_entry in folders.items():
                uids = folder_entry.get("uids") if isinstance(folder_entry, dict) else None
                if not isinstance(uids, dict):
                    print_problem(f"[email] {account_id}/{folder}: 'uids' is missing or not an object")
                    errors += 1
                    continue
                for uid, entry in uids.items():
                    # Each entry must be the object form; message-id a non-empty
                    # string and extracted a real boolean.
                    if not isinstance(entry, dict):
                        print_problem(
                            f"[email] {account_id}/{folder} uid {uid}: entry is not an object "
                            f"(got {type(entry).__name__})"
                        )
                        errors += 1
                        continue
                    mid = entry.get("message-id")
                    if not isinstance(mid, str) or not mid:
                        print_problem(
                            f"[email] {account_id}/{folder} uid {uid}: 'message-id' missing or not a non-empty string"
                        )
                        errors += 1
                        continue
                    extracted = entry.get("extracted")
                    if not isinstance(extracted, bool):
                        print_problem(
                            f"[email] {account_id}/{folder} uid {uid}: 'extracted' missing or not a boolean"
                        )
                        errors += 1
                        continue
                    if extracted:
                        true_holders[mid] = true_holders.get(mid, 0) + 1
        # Invariant 2: at most one extracted=true per message-id. Fixable.
        for mid, count in sorted(true_holders.items()):
            if count <= 1:
                continue
            print_problem(
                f"[email] message-id {mid!r} has {count} uids with extracted=true (must be <= 1)"
            )
            errors += 1
            if ask_fix(
                fix,
                f"Fix: set extracted=false for all {count} uids with message-id {mid!r} "
                f"(a normal extract then re-stores exactly one)",
            ):
                cleared = _clear_extracted(data, mid)
                _save_uidmap(state_dir, data)
                print(f"  cleared extracted on {cleared} uid(s) for {mid!r}")
        return errors
