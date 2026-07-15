"""Tests for the email plugin's pure parsing helpers (no IMAP server needed)."""

from __future__ import annotations

from email import message_from_string

from rag.extract.plugins.email import (
    build_stable_metadata,
    build_transient_metadata,
    _clean_address,
    _clean_address_list,
    _decode_mime_words,
)

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


def test_clean_address_with_name() -> None:
    """A display name is rendered as ``NAME <EMAIL>``."""
    assert _clean_address("Alice Example <alice@example.org>") == "Alice Example <alice@example.org>"


def test_clean_address_without_name() -> None:
    """A bare address is returned unchanged."""
    assert _clean_address("bob@example.com") == "bob@example.com"


def test_clean_address_list_multiple() -> None:
    """Multiple recipients are cleaned and comma-joined."""
    result = _clean_address_list("Bob <bob@example.com>, carol@example.net")
    assert result == "Bob <bob@example.com>, carol@example.net"


def test_decode_mime_words() -> None:
    """RFC 2047 encoded words are decoded to Unicode."""
    assert _decode_mime_words("=?utf-8?q?Caf=C3=A9?=") == "Café"


def test_build_stable_metadata() -> None:
    """Stable metadata pulls the expected fields from a parsed message."""
    msg = message_from_string(_RAW)
    meta = build_stable_metadata(msg, "main")
    assert meta["sender"] == "Alice Example <alice@example.org>"
    assert meta["subject"] == "Café meeting"
    assert meta["account"] == "main"
    # Recipients combine To and Cc.
    assert "bob@example.com" in meta["recipients"]
    assert "dave@example.org" in meta["recipients"]
    assert meta["date"].startswith("2026-07-15T09:30:00")


def test_build_transient_metadata() -> None:
    """Transient metadata records folder and read state."""
    assert build_transient_metadata("INBOX", read=True) == {"folder": "INBOX", "read": True}
