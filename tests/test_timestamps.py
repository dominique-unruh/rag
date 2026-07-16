"""Tests for the shared timestamp parser."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

import pytest

from rag.errors import TimestampError
from rag.timestamps import format_timestamp, parse_timestamp, to_utc

# Fixed reference instant so relative parsing is deterministic.
_REF = datetime(2026, 7, 15, 12, 0, 0, tzinfo=timezone.utc)


def test_now_returns_reference() -> None:
    """``now`` resolves to the supplied reference time."""
    assert parse_timestamp("now", now=_REF) == _REF


def test_today_is_local_midnight() -> None:
    """``today`` truncates the reference to midnight."""
    result = parse_timestamp("today", now=_REF)
    assert (result.hour, result.minute, result.second) == (0, 0, 0)


def test_yesterday_is_one_day_before_today() -> None:
    """``yesterday`` is exactly one day before ``today``."""
    today = parse_timestamp("today", now=_REF)
    assert parse_timestamp("yesterday", now=_REF) == today - timedelta(days=1)


@pytest.mark.parametrize(
    "text,delta",
    [
        ("30s ago", timedelta(seconds=30)),
        ("15m ago", timedelta(minutes=15)),
        ("2h ago", timedelta(hours=2)),
        ("1d ago", timedelta(days=1)),
        ("1w ago", timedelta(weeks=1)),
        ("3d", timedelta(days=3)),  # "ago" is optional
    ],
)
def test_relative_offsets(text: str, delta: timedelta) -> None:
    """Relative forms subtract the right duration from the reference."""
    assert parse_timestamp(text, now=_REF) == _REF - delta


def test_absolute_iso() -> None:
    """An absolute ISO datetime is parsed and preserved."""
    assert parse_timestamp("2025-01-02T03:04:05+00:00") == datetime(
        2025, 1, 2, 3, 4, 5, tzinfo=timezone.utc
    )


def test_invalid_raises() -> None:
    """Unparseable input raises :class:`TimestampError`."""
    with pytest.raises(TimestampError):
        parse_timestamp("not a time")


def test_format_timestamp_uses_utc_z() -> None:
    """Canonical formatting normalises to UTC and uses the Z designator."""
    aware = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    assert format_timestamp(aware) == "2026-01-01T10:00:00Z"


def test_to_utc_converts() -> None:
    """``to_utc`` normalises an aware datetime to UTC."""
    aware = datetime(2026, 1, 1, 12, tzinfo=timezone(timedelta(hours=2)))
    assert to_utc(aware) == datetime(2026, 1, 1, 10, tzinfo=timezone.utc)
