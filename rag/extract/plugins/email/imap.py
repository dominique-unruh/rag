"""IMAP connection, sync-state bookkeeping and message reconciliation (PLAN §4.4).

Separated from the plugin entry point so the pure parsing helpers in the parent
package stay import-light and testable. The sync state (per-folder
``UIDVALIDITY`` + a map of seen ``UID`` -> ``doc_id``) lives as one JSON file per
account under the plugin's private ``state_dir``, letting each run cheaply detect
new and deleted messages.
"""

from __future__ import annotations

import email
import email.policy
import imaplib
import json
import re
import sys
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

# Socket timeout (seconds) for IMAP operations, so a wedged/unresponsive server
# raises instead of hanging the run forever. Overridable per account via
# ``timeout`` in the config.
_DEFAULT_TIMEOUT: float = 30.0

# Regex parsing an IMAP LIST response line: `(flags) "delim" name`. The name is
# the trailing field — a quoted string (may contain spaces) or a bare atom. The
# delimiter is a quoted char or NIL.
_LIST_NAME_RE: re.Pattern[bytes] = re.compile(
    rb'^\([^)]*\)\s+(?:"[^"]*"|NIL)\s+(.+?)\s*$', re.IGNORECASE
)

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
    # Socket timeout so a stalled server raises instead of hanging forever.
    timeout = float(account.get("timeout", _DEFAULT_TIMEOUT))
    # Choose the SSL or plaintext client accordingly.
    client: imaplib.IMAP4 = (
        imaplib.IMAP4_SSL(host, port, timeout=timeout)
        if use_ssl
        else imaplib.IMAP4(host, port, timeout=timeout)
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
    """Extract the mailbox name from a single IMAP LIST response ``line``.

    The name is kept in the server's wire form (modified UTF-7, e.g.
    ``Entw&APw-rfe``) so it can be sent back verbatim in SELECT; only surrounding
    quotes are stripped. Falls back to the whole line if parsing fails.
    """
    match = _LIST_NAME_RE.match(line.strip())
    name = match.group(1) if match else line.strip()
    # Strip a surrounding pair of double quotes, if present.
    if len(name) >= 2 and name[:1] == b'"' and name[-1:] == b'"':
        name = name[1:-1]
    return name.decode("ascii", errors="replace")


def _list_folders(
    client: imaplib.IMAP4, account: Mapping[str, Any], folder_option: str | None
) -> list[str]:
    """Return the folders to process for ``account``.

    Precedence: an explicit ``folder`` CLI option (used as-is), else the account's
    configured ``folders`` list, else every folder reported by the server. In the
    latter two cases the account's ``exclude_folders`` list is then removed.
    """
    if folder_option:
        # Caller pinned a single folder for this run — honoured verbatim.
        return [folder_option]
    configured = account.get("folders")
    if configured:
        # Account restricts itself to a fixed set of folders.
        base = [str(f) for f in configured]
    else:
        # Otherwise enumerate all folders on the server.
        typ, data = client.list()
        if typ != "OK" or data is None:
            return []
        base = [_parse_folder_name(line) for line in data if line]
    # Drop any explicitly excluded folders (e.g. Spam, Trash).
    excluded = {str(f) for f in (account.get("exclude_folders") or [])}
    return [f for f in base if f not in excluded]


def _uidvalidity(client: imaplib.IMAP4) -> int:
    """Return the ``UIDVALIDITY`` of the currently-selected folder.

    Read from the untagged response left by the preceding SELECT/EXAMINE. We must
    NOT issue ``STATUS`` on the selected mailbox — RFC 3501 forbids it and some
    servers (e.g. Dovecot) desync the connection, which then breaks the following
    ``UID SEARCH``. Returns 0 when the server did not report it (forces resync).
    """
    # imaplib stashes untagged UIDVALIDITY from the last SELECT/EXAMINE here.
    data = client.untagged_responses.get("UIDVALIDITY")
    if not data:
        return 0
    match = re.search(rb"\d+", data[0] if isinstance(data[0], bytes) else str(data[0]).encode())
    return int(match.group(0)) if match else 0


def _exists_count(client: imaplib.IMAP4) -> int:
    """Return the message count (EXISTS) of the currently-selected folder.

    Read from the SELECT/EXAMINE untagged response. Used as a safety check: if the
    server says the folder has messages but a SEARCH returns none, we treat that
    as a glitch and skip deletion detection rather than wrongly purging documents.
    """
    data = client.untagged_responses.get("EXISTS")
    if not data:
        return 0
    match = re.search(rb"\d+", data[0] if isinstance(data[0], bytes) else str(data[0]).encode())
    return int(match.group(0)) if match else 0


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


def _looks_like_email(msg: Any) -> bool:
    """Return whether ``msg`` looks like a real RFC822 email.

    Garbage bytes (e.g. from a desynced IMAP connection) parse into a header-less
    :class:`email.message.Message`; a genuine email always carries at least a
    ``Message-ID`` or ``From`` header. Used to reject non-messages before storing.
    """
    return bool(msg.get("Message-ID") or msg.get("From"))


def _message_id(raw: bytes, uidvalidity: int, uid: int) -> tuple[str, Any]:
    """Parse ``raw`` and return ``(doc_id, parsed_message)``.

    ``doc_id`` is the bare ``<message-id>`` — deliberately account-independent, so
    the same message moved between accounts keeps one identity. If the message
    lacks a Message-ID header we synthesise an id from the UIDVALIDITY and UID.
    """
    # Parse once with the modern policy so headers/addresses are decoded; the
    # caller reuses the parsed message for metadata.
    parsed = email.message_from_bytes(raw, policy=email.policy.default)
    header = parsed["Message-ID"]
    if header is not None:
        # str() renders the (decoded) header; strip brackets/whitespace.
        msgid = str(header).strip().strip("<>").strip()
    else:
        # No Message-ID: fall back to a UID-based identifier (stable per epoch).
        msgid = f"uidvalidity{uidvalidity}-uid{uid}"
    return msgid, parsed


def _truthy(value: Any) -> bool:
    """Interpret a CLI option value as a boolean.

    Accepts actual booleans and the usual string forms (``1/true/yes/on``); any
    other/absent value is False. Used for flag-like ``key=value`` options.
    """
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _parse_options(options: Mapping[str, Any]) -> tuple[datetime | None, datetime | None, int | None]:
    """Parse the ``since``/``until``/``count`` run options into typed values."""
    # Timestamps go through the shared parser (accepts "1w ago", ISO, …).
    since = parse_timestamp(str(options["since"])) if options.get("since") else None
    until = parse_timestamp(str(options["until"])) if options.get("until") else None
    # count bounds how many messages are fetched this run; None = unbounded.
    count = int(options["count"]) if options.get("count") is not None else None
    return since, until, count


def _purge_excluded(
    store: Store, account: Mapping[str, Any], folder_option: str | None, verbose: bool
) -> bool:
    """Delete docs stranded in now-excluded folders; return whether any were purged.

    Finds documents whose transient (account, folder) still points at a folder in
    the account's ``exclude_folders`` list and deletes them. Skipped when a single
    ``folder`` is pinned on the CLI (that is not a full-account reconciliation).
    Returns True if anything was deleted, signalling the caller to run a full pass.
    """
    # A pinned single-folder run is not the place to reconcile the whole account.
    if folder_option:
        return False
    excluded = sorted({str(f) for f in (account.get("exclude_folders") or [])})
    if not excluded:
        return False
    account_id = str(account["account-id"])
    # Metadata query: docs of this account currently marked as in an excluded folder.
    stranded = list(
        store.list_ids(transient={"account": account_id, "folder": excluded})
    )
    for doc_id in stranded:
        store.delete(doc_id)
        if verbose:
            print(f"deleted   {'':>11} [excluded] {doc_id}", file=sys.stderr)
    return bool(stranded)


def extract_account(
    *,
    account: Mapping[str, Any],
    store: Store,
    state_dir: Path,
    secrets: Secrets,
    options: Mapping[str, Any],
    remaining: int | None,
) -> int | None:
    """Connect to one IMAP account and reconcile its folders into ``store``.

    For each folder: detects deletions (known UIDs no longer on the server), then
    fetches only new (previously-unseen) messages, bounded by the shared
    ``remaining`` fetch budget (``None`` = unlimited). Returns the leftover budget
    so it can be threaded across accounts. Sync state is persisted so subsequent
    runs stay incremental and never re-process an unchanged message.

    Before syncing, documents that still carry a now-excluded folder in their
    transient metadata are purged (via a metadata query, not IMAP); if any were
    found, a **full pass** is run — every folder resynced from scratch, ignoring
    the stored UID memory — so messages that also live in an included folder are
    re-added.
    """
    account_id = str(account["account-id"])
    # count is applied globally by the caller; only since/until come from here.
    since, until, _ = _parse_options(options)
    folder_option = str(options["folder"]) if options.get("folder") else None
    # Verbose: print a one-liner per extracted/deleted message.
    verbose = bool(options.get("verbose"))

    # Resolve credentials and connect.
    password = _password(secrets, account)
    client = _connect(account, password)
    try:
        state_path = _state_path(state_dir, account_id)
        state = _load_state(state_path)
        # Purge docs stranded in now-excluded folders. If any were purged, force a
        # full resync so any that also live in an included folder come back. The
        # `full` run option forces a full resync unconditionally.
        full_pass = _purge_excluded(store, account, folder_option, verbose)
        if _truthy(options.get("full")):
            full_pass = True
        for folder in _list_folders(client, account, folder_option):
            # Read-only select: we never modify the mailbox.
            typ, dat = client.select(_quote(folder), readonly=True)
            if typ != "OK":
                # A folder we listed but cannot select is a real error — always
                # surface it (not just in verbose mode), then skip it this run.
                detail = dat[0].decode(errors="replace") if dat and dat[0] else typ
                print(f"warning: cannot select folder {folder!r}: {detail}", file=sys.stderr)
                continue
            # Read UIDVALIDITY + message count from the SELECT response (never STATUS).
            validity = _uidvalidity(client)
            exists = _exists_count(client)
            remaining = _sync_folder(
                client=client,
                store=store,
                state=state,
                account_id=account_id,
                folder=folder,
                validity=validity,
                exists=exists,
                since=since,
                until=until,
                remaining=remaining,
                full_pass=full_pass,
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
    # Hand the leftover budget back so the caller can thread it across accounts.
    return remaining


def _sync_folder(
    *,
    client: imaplib.IMAP4,
    store: Store,
    state: dict[str, Any],
    account_id: str,
    folder: str,
    validity: int,
    exists: int,
    since: datetime | None,
    until: datetime | None,
    remaining: int | None,
    full_pass: bool = False,
    verbose: bool = False,
) -> int | None:
    """Reconcile a single ``folder``; return the updated fetch ``remaining`` budget.

    Handles UIDVALIDITY changes (resync), deletion detection and bounded fetching
    of new (previously-unseen) messages only, updating ``state`` in place. When
    ``full_pass``, the stored UID memory is ignored and every message is
    re-fetched (used after an excluded-folder purge). When ``verbose``, prints a
    one-liner per deleted and per extracted message.
    """
    from . import build_stable_metadata, build_transient_metadata  # avoid import cycle

    folders_state: dict[str, Any] = state.setdefault("folders", {})
    prior = folders_state.get(folder)
    # known maps str(uid) -> doc_id for messages we have already extracted.
    # A full pass deliberately ignores the stored UID memory and re-fetches all.
    if not full_pass and prior is not None and int(prior.get("uidvalidity", 0)) == validity:
        known: dict[str, str] = dict(prior.get("uids", {}))
        validity_matched = True
    else:
        # Full pass, new folder, or reset UID space: discard the stale UID map and
        # re-fetch. Do not delete docs — the message-id-based doc_ids stay valid.
        known = {}
        validity_matched = False

    # All current UIDs, used for deletion detection.
    all_uids = set(_search_uids(client))
    # Safety: if the folder is non-empty but the search returned nothing, treat it
    # as a glitch and skip deletion (deletions cascade — never purge on a bad read).
    search_trustworthy = bool(all_uids) or exists == 0
    if validity_matched and search_trustworthy:
        # Any previously-known UID missing from the server was deleted upstream.
        # Each deletion consumes one unit of the shared budget, like an upsert.
        for uid_str, doc_id in list(known.items()):
            if remaining is not None and remaining <= 0:
                # Budget exhausted; remaining deletions handled on the next run.
                break
            if int(uid_str) not in all_uids:
                store.delete(doc_id)
                del known[uid_str]
                if verbose:
                    # Say what we just did — deleted a vanished message.
                    print(f"deleted   {'':>11} [{folder}] {doc_id}")
                if remaining is not None:
                    remaining -= 1

    # Candidate UIDs, filtered by the date window. Only fetch UIDs we have not
    # seen before: an already-known message is not re-processed when nothing
    # changed IMAP-side. (New mail = new UID; a move/delete is handled via the
    # deletion pass + a new UID in the destination folder.)
    candidates = _search_uids(client, since=since, until=until)
    unknown = [u for u in candidates if str(u) not in known]
    if verbose:
        # Per-folder summary so an empty run is explainable.
        print(
            f"[{folder}] {len(all_uids)} on server, {len(known)} known, "
            f"{len(unknown)} new",
            file=sys.stderr,
        )
    for uid in unknown:
        if remaining is not None and remaining <= 0:
            # Budget exhausted; stop fetching but keep the state gathered so far.
            break
        fetched = _fetch_message(client, uid)
        if fetched is None:
            # Message disappeared between search and fetch; ignore it.
            continue
        raw, is_read = fetched
        doc_id, parsed = _message_id(raw, validity, uid)
        # Guard: refuse to persist bytes that are not a real email (e.g. garbage
        # from a desynced IMAP connection). Mark the UID seen so it is not
        # re-fetched, and skip without consuming the fetch budget.
        if not _looks_like_email(parsed):
            print(
                f"warning: UID {uid} in [{folder}] is not a parseable email; skipping",
                file=sys.stderr,
            )
            known[str(uid)] = doc_id
            continue
        # Build both metadata sets and upsert. Duplicate message-ids are fine
        # (exist_ok default True); an unchanged body just refreshes transient.
        stable = build_stable_metadata(parsed)
        transient = build_transient_metadata(folder, account_id, read=is_read)
        result = store.upsert(doc_id, raw, stable, transient=transient)
        known[str(uid)] = doc_id
        if verbose:
            # Format the message date as DD-Mon-YYYY (missing/unparseable -> "?").
            date_iso = stable.get("date", "")
            try:
                date_str = _imap_date(datetime.fromisoformat(date_iso)) if date_iso else "?"
            except ValueError:
                date_str = "?"
            # One line per item: what happened + date + folder + id + title (subject).
            print(f"{result.status.value:9} {date_str:>11} [{folder}] {doc_id}  {stable.get('title', '')}")
        if remaining is not None:
            remaining -= 1

    # Record the up-to-date sync state for this folder.
    folders_state[folder] = {"uidvalidity": validity, "uids": known}
    return remaining
