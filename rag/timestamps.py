"""Shared timestamp parsing (PLAN §10.1).

Lives in the core so every module and plugin parses times identically. Accepts
absolute ISO-8601 forms and convenience relative forms such as ``1d ago`` or
keywords like ``now`` / ``today`` / ``yesterday``.
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta, timezone

from .errors import TimestampError

# Map single-letter relative-unit suffixes to the timedelta keyword they feed.
# Weeks are handled as 7 days because timedelta has no ``weeks``-only shortcut
# that reads better here.
_UNIT_SECONDS: dict[str, int] = {
    "s": 1,            # seconds
    "m": 60,           # minutes
    "h": 60 * 60,      # hours
    "d": 60 * 60 * 24,       # days
    "w": 60 * 60 * 24 * 7,   # weeks
}

# Pattern for relative offsets like "1d ago", "30m ago", "2 h ago". The amount
# is an integer, the unit one of the keys above; a trailing "ago" is optional
# but conventional.
_RELATIVE_RE: re.Pattern[str] = re.compile(
    r"^\s*(?P<amount>\d+)\s*(?P<unit>[smhdw])\s*(?:ago)?\s*$",
    re.IGNORECASE,
)


def _local_now(now: datetime | None) -> datetime:
    """Return the reference "now" as a timezone-aware datetime.

    ``now`` is injectable so tests are deterministic; when omitted we use the
    current local time made timezone-aware.
    """
    if now is not None:
        # Caller supplied a reference time; ensure it carries a timezone.
        return now if now.tzinfo is not None else now.astimezone()
    # Default: current instant in the machine's local timezone.
    return datetime.now().astimezone()


def parse_timestamp(text: str, *, now: datetime | None = None) -> datetime:
    """Parse ``text`` into a timezone-aware :class:`datetime`.

    Supported forms:
    * keywords: ``now``, ``today`` (local midnight), ``yesterday``;
    * relative: ``<n><unit> [ago]`` with unit in ``s/m/h/d/w`` (e.g. ``1w ago``);
    * absolute: anything :meth:`datetime.fromisoformat` accepts (e.g.
      ``2025-01-01`` or ``2025-01-01T09:30:00+02:00``).

    Naive absolute inputs are assumed to be in local time. Raises
    :class:`TimestampError` on anything unrecognised.
    """
    raw = text.strip()
    if not raw:
        # An empty string is never a valid timestamp; fail loudly.
        raise TimestampError("empty timestamp string")

    reference = _local_now(now)
    lowered = raw.lower()

    # --- keyword forms -----------------------------------------------------
    if lowered == "now":
        return reference
    if lowered == "today":
        # Local midnight at the start of the reference day.
        return reference.replace(hour=0, minute=0, second=0, microsecond=0)
    if lowered == "yesterday":
        # Midnight one day before "today".
        midnight = reference.replace(hour=0, minute=0, second=0, microsecond=0)
        return midnight - timedelta(days=1)

    # --- relative form -----------------------------------------------------
    relative = _RELATIVE_RE.match(lowered)
    if relative is not None:
        # Convert the amount+unit into a duration and subtract from "now".
        amount = int(relative.group("amount"))
        unit = relative.group("unit").lower()
        seconds = amount * _UNIT_SECONDS[unit]
        return reference - timedelta(seconds=seconds)

    # --- absolute ISO-8601 form -------------------------------------------
    try:
        parsed = datetime.fromisoformat(raw)
    except ValueError as exc:
        # Not a keyword, not relative, not ISO: unrecognised.
        raise TimestampError(f"cannot parse timestamp: {text!r}") from exc
    # Treat a naive ISO datetime as local time so comparisons stay meaningful.
    return parsed if parsed.tzinfo is not None else parsed.astimezone()


def to_utc(moment: datetime) -> datetime:
    """Return ``moment`` converted to UTC (assuming local time if naive)."""
    # Naive inputs are interpreted as local time before converting.
    aware = moment if moment.tzinfo is not None else moment.astimezone()
    return aware.astimezone(timezone.utc)
