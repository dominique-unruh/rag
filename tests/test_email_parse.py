"""Tests for the email plugin's pure parsing helpers (no IMAP server needed)."""

from __future__ import annotations

import email.policy
from email import message_from_string

import pytest

from rag.errors import PluginError
from rag.extract.plugins.email import build_stable_metadata, build_transient_metadata
from rag.extract.plugins.email.imap import (
    _authenticate,
    _demote_siblings,
    _entry_extracted,
    _entry_message_id,
    _list_folders,
    _looks_like_email,
    _truthy,
    _uid_already_stored,
    collect_extracted_mids,
)


class _FakeIMAP:
    """Records which auth path imaplib would take, without a real server."""

    def __init__(self) -> None:
        """Start with no recorded auth call."""
        self.login_args: tuple[str, str] | None = None
        self.authenticate_args: tuple[str, bytes] | None = None

    def login(self, user: str, password: str) -> None:
        """Record a LOGIN-command auth."""
        self.login_args = (user, password)

    def authenticate(self, mechanism: str, authobject: object) -> None:
        """Record a SASL AUTHENTICATE, capturing the callback's response bytes."""
        self.authenticate_args = (mechanism, authobject(b""))  # type: ignore[operator]


def test_authenticate_login() -> None:
    """The 'login' mechanism uses the IMAP LOGIN command."""
    client = _FakeIMAP()
    _authenticate(client, "login", "user", "pw")  # type: ignore[arg-type]
    assert client.login_args == ("user", "pw")
    assert client.authenticate_args is None


def test_authenticate_plain() -> None:
    """The 'plain' mechanism sends SASL PLAIN as \\0user\\0pw."""
    client = _FakeIMAP()
    _authenticate(client, "plain", "user", "pw")  # type: ignore[arg-type]
    assert client.login_args is None
    assert client.authenticate_args == ("PLAIN", b"\x00user\x00pw")


def test_authenticate_unknown_mechanism() -> None:
    """An unknown mechanism fails clearly rather than silently doing nothing."""
    with pytest.raises(PluginError):
        _authenticate(_FakeIMAP(), "oauth", "u", "p")  # type: ignore[arg-type]


def _parse(raw: str) -> "email.message.EmailMessage":
    """Parse a raw email with the modern policy (as the plugin does)."""
    return message_from_string(raw, policy=email.policy.default)


def test_looks_like_email() -> None:
    """Real emails are accepted; header-less garbage is rejected."""
    good = message_from_string("From: a@x\r\nSubject: Hi\r\n\r\nbody")
    with_msgid = message_from_string("Message-ID: <m@x>\r\n\r\nbody")
    garbage = message_from_string("\x01\x02 random binary-ish payload no headers")
    assert _looks_like_email(good)
    assert _looks_like_email(with_msgid)
    assert not _looks_like_email(garbage)


def test_truthy() -> None:
    """CLI flag values parse to booleans across the usual forms."""
    assert all(_truthy(v) for v in [True, "1", "true", "YES", "on"])
    assert not any(_truthy(v) for v in [False, None, "0", "no", "", "off"])

# A representative raw email covering display names, multiple recipients and an
# RFC 2047 encoded subject.
_RAW = (
    "From: Alice Example <alice@example.org>\r\n"
    "To: Bob <bob@example.com>, carol@example.net\r\n"
    "Cc: Dave <dave@example.org>\r\n"
    "Subject: =?utf-8?q?Caf=C3=A9_meeting?=\r\n"
    "Date: Wed, 15 Jul 2026 09:30:00 +0000\r\n"
    "Message-ID: <abc123@example.org>\r\n"
    "\r\n"
    "Body text.\r\n"
)


def test_build_stable_metadata() -> None:
    """Stable metadata pulls the expected fields from a parsed message.

    Exercises the stdlib policy.default parser: RFC 2047 subject decoding,
    display-name addresses and To+Cc recipient merging.
    """
    meta = build_stable_metadata(_parse(_RAW))
    assert meta["sender"] == "Alice Example <alice@example.org>"
    # The Subject is stored under the global `title` key.
    assert meta["title"] == "Café meeting"
    assert "subject" not in meta
    # account is transient, not stable.
    assert "account" not in meta
    # Recipients combine To and Cc.
    assert "bob@example.com" in meta["recipients"]
    assert "dave@example.org" in meta["recipients"]
    # Date normalised to canonical UTC ISO 8601 with the Z designator.
    assert meta["date"] == "2026-07-15T09:30:00Z"
    # The message's own id, bare (no angle brackets), like the doc_id form.
    assert meta["message-id"] == "abc123@example.org"


def test_stable_metadata_message_id_absent() -> None:
    """No Message-ID header -> the 'message-id' key is omitted."""
    assert "message-id" not in build_stable_metadata(_parse("From: a@x\r\n\r\nbody"))


def test_stable_metadata_date_normalised_to_utc() -> None:
    """A non-UTC Date header is converted to the canonical UTC (Z) format."""
    raw = "From: a@x\r\nDate: Wed, 15 Jul 2026 09:30:00 +0200\r\n\r\nbody"
    assert build_stable_metadata(_parse(raw))["date"] == "2026-07-15T07:30:00Z"


def test_stable_metadata_threading_headers() -> None:
    """References / In-Reply-To become lists of bare message-ids."""
    raw = (
        "From: a@x\r\n"
        "In-Reply-To: <parent@x>\r\n"
        "References: <root@x> <parent@x>\r\n\r\nbody"
    )
    meta = build_stable_metadata(_parse(raw))
    assert meta["in-reply-to"] == ["parent@x"]
    assert meta["references"] == ["root@x", "parent@x"]
    # Absent threading headers -> keys omitted.
    plain = build_stable_metadata(_parse("From: a@x\r\n\r\nbody"))
    assert "references" not in plain and "in-reply-to" not in plain


def test_stable_metadata_omits_missing_or_bad_date() -> None:
    """No Date header, or an unparseable one (and no Received), omits the date key."""
    no_date = build_stable_metadata(_parse("From: a@x\r\nSubject: Hi\r\n\r\nbody"))
    assert "date" not in no_date
    bad_date = build_stable_metadata(_parse("From: a@x\r\nDate: not a date\r\n\r\nbody"))
    assert "date" not in bad_date


def test_stable_metadata_date_falls_back_to_received() -> None:
    """An unparseable Date falls back to the first parseable Received timestamp."""
    raw = (
        "From: a@x\r\n"
        "Received: from m2 by n2; Wed, 15 Jul 2026 11:00:00 +0000\r\n"
        "Received: from m1 by n1; Wed, 15 Jul 2026 09:30:00 +0200\r\n"
        "Date: garbage\r\n\r\nbody"
    )
    # First Received header wins (11:00 UTC).
    assert build_stable_metadata(_parse(raw))["date"] == "2026-07-15T11:00:00Z"


def test_list_folders_excludes() -> None:
    """Configured folders are returned minus the exclude list (client unused here)."""
    account = {"folders": ["INBOX", "Spam", "Sent"], "exclude_folders": ["Spam"]}
    assert _list_folders(None, account, None) == ["INBOX", "Sent"]


def test_source_url() -> None:
    """The email extractor builds a mail-link URL from message-id + fields."""
    from rag.extract.plugins.email import EmailExtractor

    url = EmailExtractor().source_url(
        "abc@example.org", {"title": "Hi there", "sender": "A <a@x>"}, {}
    )
    # Base + fragment, message-id percent-encoded, fields appended.
    assert url.startswith("https://qis.rwth-aachen.de/people/unruh/tools/mail-link/#")
    assert "abc%40example.org" in url
    assert "subject=Hi+there" in url
    assert "from=A+%3Ca%40x%3E" in url
    # Empty metadata -> just the encoded message-id in the fragment.
    bare = EmailExtractor().source_url("m@x", {}, {})
    assert bare.endswith("#m%40x")


def test_build_transient_metadata() -> None:
    """Transient metadata records account, folder and read state."""
    assert build_transient_metadata("INBOX", "main", read=True) == {
        "account": "main",
        "folder": "INBOX",
        "read": True,
    }


def test_entry_message_id_reads_both_shapes() -> None:
    """message-id is read from the new object shape and the legacy bare string."""
    assert _entry_message_id({"message-id": "m1", "extracted": True}) == "m1"
    assert _entry_message_id("m2") == "m2"  # legacy bare doc_id


def test_entry_extracted_defaults_false_for_legacy() -> None:
    """extracted reads the flag; legacy strings / missing flag count as False."""
    assert _entry_extracted({"message-id": "m", "extracted": True}) is True
    assert _entry_extracted({"message-id": "m", "extracted": False}) is False
    assert _entry_extracted({"message-id": "m"}) is False  # flag absent
    assert _entry_extracted("m") is False  # legacy bare string


def test_uid_already_stored() -> None:
    """A uid is 'stored' if its own flag is set or a sibling holds its message-id."""
    mids: set[str] = {"sib"}
    assert _uid_already_stored(None, mids) is False  # unknown uid must be fetched
    assert _uid_already_stored({"message-id": "x", "extracted": True}, mids) is True
    assert _uid_already_stored({"message-id": "sib", "extracted": False}, mids) is True
    assert _uid_already_stored({"message-id": "y", "extracted": False}, mids) is False


def test_run_rejects_unknown_option() -> None:
    """A mistyped option key (e.g. limit=1 for count=1) fails loudly, not silently."""
    from rag.extract.plugins.email import EmailExtractor

    # Validation happens before any store/config use, so the args can be dummies.
    with pytest.raises(PluginError, match="unknown option"):
        EmailExtractor().run(None, None, {}, None, limit="1")  # type: ignore[arg-type]


def test_run_accepts_known_options(monkeypatch: pytest.MonkeyPatch) -> None:
    """Recognised option keys pass the up-front validation."""
    from rag.extract.plugins.email import EmailExtractor

    # Stub out the account loading so we stop right after validation succeeds.
    import rag.extract.plugins.email as email_pkg

    monkeypatch.setattr(email_pkg, "_accounts_from_config", lambda config: [])
    # No PluginError => the known keys were accepted (empty account list = no work).
    EmailExtractor().run(None, None, {}, None, count="1", folder="INBOX")  # type: ignore[arg-type]


def test_demote_siblings_clears_other_holders_across_folders_and_accounts() -> None:
    """Promoting one uid demotes every other copy of its message-id, any account."""
    working = {"7": {"message-id": "m", "extracted": True}}  # just-promoted, acct A/Sent
    accounts_state = {
        "A": {"folders": {
            "Sent": {"uidvalidity": 1, "uids": working},
            "Archive": {"uidvalidity": 1, "uids": {
                "3": {"message-id": "m", "extracted": True},  # stale sibling -> demote
                "4": {"message-id": "other", "extracted": True},  # unrelated -> keep
            }},
        }},
        "B": {"folders": {
            # Same message-id in a *different* account: must also be demoted.
            "INBOX": {"uidvalidity": 9, "uids": {
                "7": {"message-id": "m", "extracted": True},  # same uid#, other acct
            }},
        }},
    }
    _demote_siblings(accounts_state, working, "A", "Sent", "7", "m")
    assert working["7"]["extracted"] is True  # kept (the promoted copy)
    assert accounts_state["A"]["folders"]["Archive"]["uids"]["3"]["extracted"] is False
    assert accounts_state["A"]["folders"]["Archive"]["uids"]["4"]["extracted"] is True
    assert accounts_state["B"]["folders"]["INBOX"]["uids"]["7"]["extracted"] is False


def test_collect_extracted_mids_spans_all_accounts() -> None:
    """The run-wide extracted set gathers message-ids flagged in every account."""
    uidmap = {"accounts": {
        "A": {"folders": {"INBOX": {"uids": {
            "1": {"message-id": "m1", "extracted": True},
            "2": {"message-id": "m2", "extracted": False},  # not stored -> excluded
        }}}},
        "B": {"folders": {"Sent": {"uids": {
            "5": {"message-id": "m3", "extracted": True},
        }}}},
    }}
    assert collect_extracted_mids(uidmap) == {"m1", "m3"}
