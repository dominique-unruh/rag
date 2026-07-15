"""IMAP connection, sync-state bookkeeping and message reconciliation (PLAN §4.4).

Separated from the plugin entry point so the pure parsing helpers in the parent
package stay import-light and testable. The sync state (per-folder
``UIDVALIDITY`` + a map of seen ``UID`` -> ``doc_id``) lives as one JSON file per
account under the plugin's private ``state_dir``, letting each run cheaply detect
new and deleted messages.
"""

from __future__ import annotations

import email
import imaplib
import json
import re
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from ....secrets import Secrets
from ....timestamps import parse_timestamp
from ...base import Store

# Default IMAP ports for the SSL and plaintext transports.
_DEFAULT_SSL_PORT: int = 993
_DEFAULT_PLAIN_PORT: int = 143

# Regex pulling the folder name out of an IMAP LIST response line. The name is
# the final token, optionally double-quoted.
_LIST_NAME_RE: re.Pattern[bytes] = re.compile(rb'"?([^"]*)"?\s*$')

# Abbreviated month names for building IMAP ``DD-Mon-YYYY`` search dates, which
# must be in English regardless of locale.
_IMAP_MONTHS: tuple[str, ...] = (
    "Jan", "Feb", "Mar", "Apr", "May", "Jun",
    "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
)


def _password(secrets: Secrets, account: Mapping[str, Any]) -> str:
    """Resolve the IMAP password for ``account`` via the shared secret resolver.

    The keyring key is ``imap:<host>:<username>`` — the actual credential
    identity. Keying by host + username (rather than username alone) disambiguates
    accounts that share a username across different servers, and stays stable if
    the ``account-id`` is later renamed. Falls back to an interactive prompt
    cached to the keyring.
    """
    account_id = str(account["account-id"])
    host = str(account["host"])
    username = str(account.get("username", account_id))
    # Composite, server-scoped key so same-username/different-server accounts
    # never share a password entry.
    key = f"imap:{host}:{username}"
    return secrets.get(
        key,
        prompt_text=f"IMAP password for {username} at {host} ({account_id}): ",
    )


def _connect(account: Mapping[str, Any], password: str) -> imaplib.IMAP4:
    """Open and authenticate an IMAP connection for ``account``.

    Uses SSL unless the account sets ``ssl: false``; picks a sensible default port
    for the chosen transport when none is configured.
    """
    host = str(account["host"])
    use_ssl = bool(account.get("ssl", True))
    # Default the port based on transport unless explicitly configured.
    port = int(account.get("port", _DEFAULT_SSL_PORT if use_ssl else _DEFAULT_PLAIN_PORT))
    username = str(account.get("username", account["account-id"]))
    # Choose the SSL or plaintext client accordingly.
    client: imaplib.IMAP4 = (
        imaplib.IMAP4_SSL(host, port) if use_ssl else imaplib.IMAP4(host, port)
    )
    client.login(username, password)
    return client


def _state_path(state_dir: Path, account_id: str) -> Path:
    """Return the JSON sync-state file path for ``account_id``."""
    # Sanitise the id so it is always a safe filename.
    safe = re.sub(r"[^A-Za-z0-9_.-]", "_", account_id)
    return state_dir / f"{safe}.json"


def _load_state(path: Path) -> dict[str, Any]:
    """Load the sync state from ``path``, returning an empty skeleton if absent."""
    if not path.exists():
        # Fresh account: no folders seen yet.
        return {"folders": {}}
    return json.loads(path.read_text())


def _save_state(path: Path, state: Mapping[str, Any]) -> None:
    """Persist the sync ``state`` to ``path`` atomically."""
    # Write to a temp file then rename so a crash cannot corrupt the state file.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(state, indent=2, sort_keys=True))
    tmp.replace(path)


def _parse_folder_name(line: bytes) -> str:
    """Extract the mailbox name from a single IMAP LIST response ``line``."""
    match = _LIST_NAME_RE.search(line.strip())
    # Fall back to the raw decoded line if the regex somehow misses.
    name = match.group(1) if match else line.strip()
    return name.decode("utf-8", errors="replace")


def _list_folders(
    client: imaplib.IMAP4, account: Mapping[str, Any], folder_option: str | None
) -> list[str]:
    """Return the folders to process for ``account``.

    Precedence: an explicit ``folder`` CLI option, else the account's configured
    ``folders`` list, else every folder reported by the server.
    """
    if folder_option:
        # Caller pinned a single folder for this run.
        return [folder_option]
    configured = account.get("folders")
    if configured:
        # Account restricts itself to a fixed set of folders.
        return [str(f) for f in configured]
    # Otherwise enumerate all folders on the server.
    typ, data = client.list()
    if typ != "OK" or data is None:
        return []
    return [_parse_folder_name(line) for line in data if line]


def _uidvalidity(client: imaplib.IMAP4, folder: str) -> int:
    """Return the ``UIDVALIDITY`` of ``folder`` (identifies UID-space epochs)."""
    typ, data = client.status(_quote(folder), "(UIDVALIDITY)")
    if typ != "OK" or not data:
        # Unknown validity: use 0 so any stored non-zero value forces a resync.
        return 0
    # Response looks like: b'"INBOX" (UIDVALIDITY 12345)'.
    match = re.search(rb"UIDVALIDITY\s+(\d+)", data[0])
    return int(match.group(1)) if match else 0


def _quote(folder: str) -> str:
    """Return ``folder`` quoted for use in IMAP commands (handles spaces)."""
    # Double-quote and escape embedded quotes/backslashes per IMAP syntax.
    escaped = folder.replace("\\", "\\\\").replace('"', '\\"')
    return f'"{escaped}"'


def _imap_date(moment: datetime) -> str:
    """Format ``moment`` as an IMAP ``DD-Mon-YYYY`` search date (English month)."""
    return f"{moment.day:02d}-{_IMAP_MONTHS[moment.month - 1]}-{moment.year}"


def _search_uids(
    client: imaplib.IMAP4,
    *,
    since: datetime | None = None,
    until: datetime | None = None,
) -> list[int]:
    """Return matching UIDs in the selected folder, optionally date-filtered.

    With no ``since``/``until`` this returns every UID (used for deletion
    detection); with them it applies IMAP ``SINCE``/``BEFORE`` server-side.
    """
    criteria: list[str] = []
    if since is not None:
        # SINCE is inclusive of the given date (server internal date).
        criteria.extend(["SINCE", _imap_date(since)])
    if until is not None:
        # BEFORE is exclusive of the given date.
        criteria.extend(["BEFORE", _imap_date(until)])
    if not criteria:
        # No filter -> match everything.
        criteria = ["ALL"]
    typ, data = client.uid("SEARCH", None, *criteria)
    if typ != "OK" or not data or not data[0]:
        return []
    # data[0] is space-separated ASCII UIDs.
    return [int(tok) for tok in data[0].split()]


def _fetch_message(client: imaplib.IMAP4, uid: int) -> tuple[bytes, bool] | None:
    """Fetch the raw message and read-flag for ``uid``.

    Returns ``(raw_rfc822_bytes, is_read)`` or ``None`` if the UID could not be
    fetched (e.g. it was deleted between search and fetch).
    """
    typ, data = client.uid("FETCH", str(uid), "(RFC822 FLAGS)")
    if typ != "OK" or not data or not isinstance(data[0], tuple):
        return None
    # data[0] = (header_line_with_FLAGS, raw_body_bytes).
    header_line, raw = data[0][0], data[0][1]
    if not isinstance(raw, (bytes, bytearray)):
        return None
    # \Seen appearing in the FETCH header means the message is marked read.
    is_read = b"\\Seen" in header_line
    return bytes(raw), is_read


def _message_id(raw: bytes, account_id: str, uidvalidity: int, uid: int) -> tuple[str, Any]:
    """Parse ``raw`` and return ``(doc_id, parsed_message)``.

    ``doc_id`` is ``"<account-id>:<message-id>"``; if the message lacks a
    Message-ID header we synthesise a stable id from the UIDVALIDITY and UID.
    """
    # Parse the full message once; the caller reuses it for metadata.
    parsed = email.message_from_bytes(raw)
    header = parsed.get("Message-ID")
    if header:
        # Strip the surrounding angle brackets and any stray whitespace.
        msgid = header.strip().strip("<>").strip()
    else:
        # No Message-ID: fall back to a UID-based identifier (stable per epoch).
        msgid = f"uidvalidity{uidvalidity}-uid{uid}"
    return f"{account_id}:{msgid}", parsed


def _parse_options(options: Mapping[str, Any]) -> tuple[datetime | None, datetime | None, int | None]:
    """Parse the ``since``/``until``/``count`` run options into typed values."""
    # Timestamps go through the shared parser (accepts "1w ago", ISO, …).
    since = parse_timestamp(str(options["since"])) if options.get("since") else None
    until = parse_timestamp(str(options["until"])) if options.get("until") else None
    # count bounds how many messages are fetched this run; None = unbounded.
    count = int(options["count"]) if options.get("count") is not None else None
    return since, until, count


def extract_account(
    *,
    account: Mapping[str, Any],
    store: Store,
    state_dir: Path,
    secrets: Secrets,
    options: Mapping[str, Any],
) -> None:
    """Connect to one IMAP account and reconcile its folders into ``store``.

    For each folder: detects deletions (known UIDs no longer on the server),
    then fetches messages (new ones first, then already-known ones so their
    transient flags stay current), bounded by the ``count`` option. Sync state is
    persisted so subsequent runs stay incremental.
    """
    account_id = str(account["account-id"])
    since, until, count = _parse_options(options)
    folder_option = str(options["folder"]) if options.get("folder") else None
    # Verbose: print a one-liner per extracted/deleted message.
    verbose = bool(options.get("verbose"))

    # Resolve credentials and connect.
    password = _password(secrets, account)
    client = _connect(account, password)
    try:
        state_path = _state_path(state_dir, account_id)
        state = _load_state(state_path)
        # Remaining fetch budget for this run (None = unlimited).
        remaining = count
        for folder in _list_folders(client, account, folder_option):
            # Read-only select: we never modify the mailbox.
            typ, _ = client.select(_quote(folder), readonly=True)
            if typ != "OK":
                # Folder vanished or is inaccessible; skip it this run.
                continue
            remaining = _sync_folder(
                client=client,
                store=store,
                state=state,
                account_id=account_id,
                folder=folder,
                since=since,
                until=until,
                remaining=remaining,
                verbose=verbose,
            )
            # Persist after each folder so partial progress survives a crash.
            _save_state(state_path, state)
            if remaining is not None and remaining <= 0:
                # Exhausted the fetch budget for this run.
                break
    finally:
        # Always log out cleanly, even on error.
        try:
            client.logout()
        except OSError:
            # Connection already dropped; nothing more to do.
            pass


def _sync_folder(
    *,
    client: imaplib.IMAP4,
    store: Store,
    state: dict[str, Any],
    account_id: str,
    folder: str,
    since: datetime | None,
    until: datetime | None,
    remaining: int | None,
    verbose: bool = False,
) -> int | None:
    """Reconcile a single ``folder``; return the updated fetch ``remaining`` budget.

    Handles UIDVALIDITY changes (resync), deletion detection and bounded fetching
    of new-then-known messages, updating ``state`` in place. When ``verbose``,
    prints a one-liner per deleted and per extracted message.
    """
    from . import build_stable_metadata, build_transient_metadata  # avoid import cycle

    validity = _uidvalidity(client, folder)
    folders_state: dict[str, Any] = state.setdefault("folders", {})
    prior = folders_state.get(folder)
    # known maps str(uid) -> doc_id for messages we have already extracted.
    if prior is not None and int(prior.get("uidvalidity", 0)) == validity:
        known: dict[str, str] = dict(prior.get("uids", {}))
        validity_matched = True
    else:
        # New folder or the UID space was reset: discard the stale UID map and
        # re-fetch. Do not delete docs — the message-id-based doc_ids stay valid.
        known = {}
        validity_matched = False

    # All current UIDs, used for deletion detection.
    all_uids = set(_search_uids(client))
    if validity_matched:
        # Any previously-known UID missing from the server was deleted upstream.
        for uid_str, doc_id in list(known.items()):
            if int(uid_str) not in all_uids:
                store.delete(doc_id)
                del known[uid_str]
                if verbose:
                    # Say what we just did — deleted a vanished message.
                    print(f"deleted   {doc_id}")

    # Candidate UIDs to (re)fetch, filtered by the date window.
    candidates = _search_uids(client, since=since, until=until)
    # Process unknown UIDs first (real new mail), then known ones to refresh their
    # transient flags — all subject to the shared budget.
    unknown = [u for u in candidates if str(u) not in known]
    already = [u for u in candidates if str(u) in known]
    for uid in unknown + already:
        if remaining is not None and remaining <= 0:
            # Budget exhausted; stop fetching but keep the state gathered so far.
            break
        fetched = _fetch_message(client, uid)
        if fetched is None:
            # Message disappeared between search and fetch; ignore it.
            continue
        raw, is_read = fetched
        doc_id, parsed = _message_id(raw, account_id, validity, uid)
        # Build both metadata sets and upsert. Duplicate message-ids are fine
        # (exist_ok default True); an unchanged body just refreshes transient.
        stable = build_stable_metadata(parsed, account_id)
        transient = build_transient_metadata(folder, read=is_read)
        result = store.upsert(doc_id, raw, stable, transient=transient)
        known[str(uid)] = doc_id
        if verbose:
            # Format the message date as DD-Mon-YYYY (empty/unparseable -> "?").
            date_iso = stable["date"]
            try:
                date_str = _imap_date(datetime.fromisoformat(date_iso)) if date_iso else "?"
            except ValueError:
                date_str = "?"
            # One line per item: what happened + date + id + subject.
            print(f"{result.status.value:9} {date_str:>11} {doc_id}  {stable['subject']}")
        if remaining is not None:
            remaining -= 1

    # Record the up-to-date sync state for this folder.
    folders_state[folder] = {"uidvalidity": validity, "uids": known}
    return remaining
