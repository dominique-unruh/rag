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

from ....errors import PluginError
from ....interrupt import interrupt_requested
from ....secrets import Secrets
from ....timestamps import parse_timestamp
from ...base import Store
from . import oauth2

# Subdirectory (under the plugin's state_dir) holding persistent OAuth2 refresh
# tokens, one JSON file per account. Owner-only; never committed.
_SECRETS_SUBDIR: str = "secrets"

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


def _authenticate(client: imaplib.IMAP4, mechanism: str, username: str, credential: str) -> None:
    """Authenticate ``client`` using the configured ``mechanism``.

    ``login`` uses the IMAP ``LOGIN`` command (the imaplib default). ``plain`` uses
    SASL ``AUTHENTICATE PLAIN`` — needed by servers (e.g. RWTH) that reject the
    ``LOGIN`` command outright but accept the PLAIN mechanism over TLS. ``xoauth2``
    uses SASL ``XOAUTH2`` with an OAuth2 bearer token as the ``credential`` —
    required by servers (e.g. Microsoft 365) that no longer accept passwords.
    """
    if mechanism == "login":
        client.login(username, credential)
    elif mechanism == "plain":
        # SASL PLAIN response is "\0<authcid>\0<passwd>"; imaplib base64-encodes
        # whatever the callback returns. authzid is left empty (same as authcid).
        credentials = f"\0{username}\0{credential}".encode("utf-8")
        client.authenticate("PLAIN", lambda _challenge: credentials)
    elif mechanism == "xoauth2":
        # SASL XOAUTH2 response is "user=<user>^Aauth=Bearer <token>^A^A" (^A =
        # 0x01); imaplib base64-encodes whatever the callback returns. Here the
        # ``credential`` is a short-lived OAuth2 access token, not a password.
        response = f"user={username}\x01auth=Bearer {credential}\x01\x01".encode("utf-8")
        client.authenticate("XOAUTH2", lambda _challenge: response)
    else:
        raise PluginError(
            f"unknown IMAP auth mechanism {mechanism!r}; use 'login', 'plain' or 'xoauth2'"
        )


def _connect(account: Mapping[str, Any], credential: str) -> imaplib.IMAP4:
    """Open and authenticate an IMAP connection for ``account``.

    Uses SSL unless the account sets ``ssl: false``; picks a sensible default port
    for the chosen transport when none is configured. The ``auth`` option selects
    the authentication mechanism (``login`` default, ``plain`` for SASL PLAIN, or
    ``xoauth2`` for an OAuth2 bearer token). ``credential`` is the password for
    ``login``/``plain`` and the OAuth2 access token for ``xoauth2``.
    """
    host = str(account["host"])
    use_ssl = bool(account.get("ssl", True))
    # Default the port based on transport unless explicitly configured.
    port = int(account.get("port", _DEFAULT_SSL_PORT if use_ssl else _DEFAULT_PLAIN_PORT))
    username = str(account.get("username", account["account-id"]))
    # Authentication mechanism; normalised to lower case for a forgiving match.
    mechanism = str(account.get("auth", "login")).lower()
    # Socket timeout so a stalled server raises instead of hanging forever.
    timeout = float(account.get("timeout", _DEFAULT_TIMEOUT))
    # Choose the SSL or plaintext client accordingly.
    client: imaplib.IMAP4 = (
        imaplib.IMAP4_SSL(host, port, timeout=timeout)
        if use_ssl
        else imaplib.IMAP4(host, port, timeout=timeout)
    )
    account_id = str(account["account-id"])
    try:
        _authenticate(client, mechanism, username, credential)
    except imaplib.IMAP4.error as exc:
        # Turn imaplib's opaque auth failure into a clear, actionable message.
        # A "BAD" reply means the server rejected the auth command itself (some
        # servers reject the LOGIN command but accept 'auth: plain'), as opposed
        # to "NO" for bad credentials — either way, name the account/host/user so
        # it's obvious which one failed and how to fix it.
        raise PluginError(
            f"IMAP authentication failed for account {account_id!r} "
            f"({username} at {host}:{port}, auth={mechanism}): {exc}. "
            f"Check the username/password, or try a different 'auth' mechanism "
            f"('login', 'plain' or 'xoauth2') — some servers reject 'login' but "
            f"accept 'plain', and M365 requires 'xoauth2'."
        ) from exc
    return client


# Filename of the joint UID map: one file shared by ALL accounts (not one per
# account). Deletion detection needs to know whether a message-id survives
# *anywhere* across every account/folder, so the maps are kept together.
_UIDMAP_FILENAME: str = "uidmap.json"


def _uidmap_path(state_dir: Path) -> Path:
    """Return the path of the joint UID map file under ``state_dir``."""
    return state_dir / _UIDMAP_FILENAME


def _load_uidmap(state_dir: Path) -> dict[str, Any]:
    """Load the joint UID→message-id map for all accounts.

    Shape: ``{"accounts": {<account_id>: {"folders": {<folder>: {"uidvalidity":
    int, "uids": {<uid>: <doc_id>}}}}}}``. Returns an empty skeleton when the file
    is absent, first migrating any legacy per-account state files (the previous
    ``<account_id>.json`` format) so a switch to the joint map does not force a
    full re-download.
    """
    path = _uidmap_path(state_dir)
    if path.exists():
        return json.loads(path.read_text())
    # Migration: fold any legacy per-account files (same per-folder shape) into
    # the joint structure. Best-effort — an unreadable file is simply skipped.
    data: dict[str, Any] = {"accounts": {}}
    for legacy in sorted(state_dir.glob("*.json")):
        if legacy.name == _UIDMAP_FILENAME:
            continue
        try:
            old = json.loads(legacy.read_text())
        except (json.JSONDecodeError, OSError):
            continue
        if isinstance(old, Mapping) and "folders" in old:
            # The legacy filename stem is the (sanitised) account id.
            data["accounts"][legacy.stem] = {"folders": old["folders"]}
    return data


def _save_uidmap(state_dir: Path, data: Mapping[str, Any]) -> None:
    """Persist the joint UID map atomically (temp file + rename)."""
    path = _uidmap_path(state_dir)
    # Write to a temp file then rename so a crash cannot corrupt the map.
    tmp = path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(data, indent=2, sort_keys=True))
    tmp.replace(path)


def _account_folders(uidmap: dict[str, Any], account_id: str) -> dict[str, Any]:
    """Return (creating) the per-folder sub-map for ``account_id`` in ``uidmap``."""
    accounts: dict[str, Any] = uidmap.setdefault("accounts", {})
    account: dict[str, Any] = accounts.setdefault(account_id, {})
    return account.setdefault("folders", {})


def _open_client(
    account: Mapping[str, Any], state_dir: Path, secrets: Secrets
) -> imaplib.IMAP4:
    """Resolve credentials for ``account`` and return an authenticated client.

    XOAUTH2 accounts (e.g. M365) mint a short-lived OAuth2 access token from the
    persisted refresh token under the plugin's ``secrets/`` dir; every other
    account uses a keyring password. Shared by the additions and deletion passes.
    """
    mechanism = str(account.get("auth", "login")).lower()
    if mechanism == "xoauth2":
        credential = oauth2.access_token(
            account, state_dir / _SECRETS_SUBDIR, allow_prompt=secrets.allow_prompt
        )
    else:
        credential = _password(secrets, account)
    return _connect(account, credential)


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


def _fetch_msgid(client: imaplib.IMAP4, uid: int, validity: int) -> str | None:
    """Return the doc_id for ``uid`` by fetching only its ``Message-ID`` header.

    Used by the deletion scan to learn a message's identity cheaply (no body
    download). Returns the bare Message-ID, or a synthesised
    ``uidvalidity{v}-uid{u}`` id when the header is absent (matching
    :func:`_message_id`), or ``None`` if the fetch failed (message vanished
    mid-scan) so the caller can skip it.
    """
    # BODY.PEEK avoids setting \Seen; fetch just the Message-ID header field.
    typ, data = client.uid("FETCH", str(uid), "(BODY.PEEK[HEADER.FIELDS (MESSAGE-ID)])")
    if typ != "OK" or not data or not isinstance(data[0], tuple):
        return None
    raw = data[0][1]
    if not isinstance(raw, (bytes, bytearray)):
        return None
    # Parse the tiny header block; reuse the same id derivation as full messages.
    parsed = email.message_from_bytes(bytes(raw), policy=email.policy.default)
    header = parsed["Message-ID"]
    if header is not None:
        return str(header).strip().strip("<>").strip()
    # No Message-ID: same UID-based fallback as _message_id (stable per epoch).
    return f"uidvalidity{validity}-uid{uid}"


def extract_account(
    *,
    account: Mapping[str, Any],
    store: Store,
    state_dir: Path,
    secrets: Secrets,
    options: Mapping[str, Any],
    remaining: int | None,
) -> int | None:
    """Connect to one IMAP account and add new messages into ``store``.

    **Additions only.** For each folder, fetches previously-unseen messages
    (bounded by the shared ``remaining`` fetch budget; ``None`` = unlimited),
    upserts them, and refreshes the account's entries in the joint UID map.
    Returns the leftover budget so it can be threaded across accounts.

    Deletions are deliberately *not* handled here: a message-id may live in
    several folders/accounts, so deciding it is truly gone requires scanning
    every account at once. That is the opt-in :func:`run_deletions` pass.
    """
    account_id = str(account["account-id"])
    # count is applied globally by the caller; only since/until come from here.
    since, until, _ = _parse_options(options)
    folder_option = str(options["folder"]) if options.get("folder") else None
    # Verbose: print a one-liner per extracted message.
    verbose = bool(options.get("verbose"))
    # full: ignore the stored UID memory and re-fetch every message in scope.
    full = _truthy(options.get("full"))

    client = _open_client(account, state_dir, secrets)
    try:
        # The joint UID map holds every account; grab this account's sub-map.
        uidmap = _load_uidmap(state_dir)
        folders_state = _account_folders(uidmap, account_id)
        for folder in _list_folders(client, account, folder_option):
            # Read-only select: we never modify the mailbox.
            typ, dat = client.select(_quote(folder), readonly=True)
            if typ != "OK":
                # A folder we listed but cannot select is a real error — always
                # surface it (not just in verbose mode), then skip it this run.
                detail = dat[0].decode(errors="replace") if dat and dat[0] else typ
                print(f"warning: cannot select folder {folder!r}: {detail}", file=sys.stderr)
                continue
            # Read UIDVALIDITY from the SELECT response (never STATUS).
            validity = _uidvalidity(client)
            remaining = _sync_folder(
                client=client,
                store=store,
                folders_state=folders_state,
                account_id=account_id,
                folder=folder,
                validity=validity,
                since=since,
                until=until,
                remaining=remaining,
                full=full,
                verbose=verbose,
            )
            # Persist after each folder so partial progress survives a crash.
            _save_uidmap(state_dir, uidmap)
            if remaining is not None and remaining <= 0:
                # Exhausted the fetch budget for this run.
                break
            # Graceful interrupt requested mid-run: stop after the current folder
            # (its progress is already persisted above).
            if interrupt_requested():
                print(f"interrupted; stopping account {account_id!r}.", file=sys.stderr)
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
    folders_state: dict[str, Any],
    account_id: str,
    folder: str,
    validity: int,
    since: datetime | None,
    until: datetime | None,
    remaining: int | None,
    full: bool = False,
    verbose: bool = False,
) -> int | None:
    """Add previously-unseen messages in ``folder``; return the updated budget.

    Fetches only UIDs not already in the joint map (or every UID when ``full``),
    upserts each and records its ``uid -> doc_id`` in ``folders_state`` in place.
    **Never deletes** — a vanished UID may just be a move, so removal is left to
    the cross-account :func:`run_deletions` pass. When ``verbose``, prints a
    one-liner per extracted message.
    """
    from . import build_stable_metadata, build_transient_metadata  # avoid import cycle

    prior = folders_state.get(folder)
    # known maps str(uid) -> doc_id for messages we have already extracted.
    # ``full`` deliberately ignores the stored UID memory and re-fetches all.
    if not full and prior is not None and int(prior.get("uidvalidity", 0)) == validity:
        known: dict[str, str] = dict(prior.get("uids", {}))
    else:
        # Full pass, new folder, or reset UID space: discard the stale UID map.
        known = {}

    # Candidate UIDs, filtered by the date window. Only fetch UIDs we have not
    # seen before: an already-known message is not re-processed when nothing
    # changed IMAP-side. (New mail = new UID; deletions are handled separately.)
    candidates = _search_uids(client, since=since, until=until)
    unknown = [u for u in candidates if str(u) not in known]
    if verbose:
        # Per-folder summary so an empty run is explainable.
        print(
            f"[{folder}] {len(candidates)} in window, {len(known)} known, "
            f"{len(unknown)} new",
            file=sys.stderr,
        )
    for uid in unknown:
        if remaining is not None and remaining <= 0:
            # Budget exhausted; stop fetching but keep the state gathered so far.
            break
        # Checkpoint: honour a graceful interrupt between messages. State gathered
        # so far is persisted by the caller after this folder returns.
        if interrupt_requested():
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


def _scan_account_uids(
    client: imaplib.IMAP4,
    account: Mapping[str, Any],
    folders_state: dict[str, Any],
    verbose: bool,
) -> bool:
    """Refresh ``folders_state`` from the server for one account's included folders.

    For every included folder, rebuilds ``uid -> doc_id`` from the server's live
    UID set, reusing message-ids already in the map and fetching just the
    ``Message-ID`` header for unknown UIDs. Returns ``False`` if a graceful
    interrupt was requested (the caller must then abort without deleting, since
    the scan is incomplete); ``True`` when the account was fully scanned.

    Safety: a folder that reports messages (EXISTS > 0) but returns an empty
    search is treated as a glitch — its prior entries are kept rather than being
    wiped, so a bad read can never make a message look deleted.
    """
    account_id = str(account["account-id"])
    # None => all included folders (honours configured `folders`/`exclude_folders`);
    # the deletion scan is global and ignores any single-folder CLI pin.
    for folder in _list_folders(client, account, None):
        typ, dat = client.select(_quote(folder), readonly=True)
        if typ != "OK":
            detail = dat[0].decode(errors="replace") if dat and dat[0] else typ
            print(f"warning: cannot select folder {folder!r}: {detail}", file=sys.stderr)
            continue
        validity = _uidvalidity(client)
        exists = _exists_count(client)
        all_uids = _search_uids(client)
        # Glitch guard: non-empty folder yet empty search -> keep prior entries.
        if not all_uids and exists != 0:
            print(
                f"warning: empty search on non-empty folder {folder!r} in "
                f"{account_id!r}; keeping its prior UID map",
                file=sys.stderr,
            )
            continue
        # Reuse message-ids we already know for this epoch; only unknowns cost a fetch.
        prior = folders_state.get(folder)
        prior_uids: dict[str, str] = (
            dict(prior.get("uids", {}))
            if prior is not None and int(prior.get("uidvalidity", 0)) == validity
            else {}
        )
        fresh: dict[str, str] = {}
        for uid in all_uids:
            # Between folders is a safe checkpoint; abort mid-folder without
            # committing partial results (prior entry stays intact).
            if interrupt_requested():
                return False
            key = str(uid)
            doc_id = prior_uids.get(key)
            if doc_id is None:
                # Unknown UID: resolve its identity via a cheap header-only fetch.
                doc_id = _fetch_msgid(client, uid, validity)
                if doc_id is None:
                    # Fetch failed (vanished mid-scan) — leave it out of presence.
                    continue
            fresh[key] = doc_id
        folders_state[folder] = {"uidvalidity": validity, "uids": fresh}
        if verbose:
            print(f"[{account_id}/{folder}] scanned {len(fresh)} message(s)", file=sys.stderr)
    return True


def run_deletions(
    *,
    accounts: list[Mapping[str, Any]],
    store: Store,
    state_dir: Path,
    secrets: Secrets,
    options: Mapping[str, Any],
) -> None:
    """Delete documents whose message-id no longer lives in **any** account.

    The opt-in, expensive counterpart to :func:`extract_account` (triggered by
    ``delete=true``). It refreshes the joint UID map by scanning every account's
    included folders — ignoring any ``account``/``folder`` filter, since presence
    is a global question — then deletes every stored document whose id is not a
    live message-id anywhere. This subsumes the old excluded-folder purge: a
    message only in an excluded folder is not in the map, so it is deleted.

    Conservative on failure: an unreachable account keeps its prior map entries
    (its documents stay "present"), and an interrupted scan deletes nothing.
    """
    verbose = bool(options.get("verbose"))
    uidmap = _load_uidmap(state_dir)
    for account in accounts:
        account_id = str(account["account-id"])
        try:
            client = _open_client(account, state_dir, secrets)
        except (PluginError, OSError, imaplib.IMAP4.error) as exc:
            # Cannot reach this account: keep its stored entries (its documents
            # stay present and are not deleted this run) and move on.
            print(
                f"warning: skipping account {account_id!r} in deletion scan "
                f"(cannot connect): {exc}",
                file=sys.stderr,
            )
            continue
        try:
            folders_state = _account_folders(uidmap, account_id)
            completed = _scan_account_uids(client, account, folders_state, verbose)
            # Persist the refreshed map after each account (progress survives a crash).
            _save_uidmap(state_dir, uidmap)
        finally:
            try:
                client.logout()
            except OSError:
                pass
        if not completed or interrupt_requested():
            # Incomplete scan -> presence is unreliable -> never delete on it.
            print("interrupted; deletion scan incomplete, deleting nothing.", file=sys.stderr)
            return

    # Presence = every message-id currently mapped across all accounts/folders.
    present: set[str] = set()
    for account_entry in uidmap.get("accounts", {}).values():
        for folder_entry in account_entry.get("folders", {}).values():
            present.update(folder_entry.get("uids", {}).values())

    # Delete any stored document whose id survives in no folder of any account.
    deleted = 0
    for doc_id in list(store.list_ids()):
        if doc_id not in present:
            store.delete(doc_id)
            deleted += 1
            if verbose:
                print(f"deleted   [gone] {doc_id}", file=sys.stderr)
    print(f"deletion scan complete: removed {deleted} document(s).", file=sys.stderr)
