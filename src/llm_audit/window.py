"""Time windows.

Two different clocks are involved and they are not interchangeable:

* a **file's mtime** is cheap and safe for elimination - a record cannot be
  newer than the file that holds it, so a file older than the cutoff cannot
  contain an in-window record, and can be skipped without parsing it;
* a **record's ``timestamp``** is what actually scopes the numbers. Sessions
  span days (one measured run went 17.5 h), so an mtime filter alone would drag
  a long session's entire history into a 24 h window.
"""

from __future__ import annotations

import re
from datetime import UTC, datetime, timedelta

_RELATIVE = re.compile(r"^(\d+(?:\.\d+)?)\s*([smhdw])$")
_UNITS = {"s": 1, "m": 60, "h": 3600, "d": 86400, "w": 604800}


def now() -> datetime:
    """The current time, timezone-aware UTC."""
    return datetime.now(UTC)


def parse_timestamp(raw: object) -> datetime | None:
    """Parse a transcript timestamp into an aware UTC datetime."""
    if not isinstance(raw, str) or not raw:
        return None
    text = raw.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=UTC)
    return parsed.astimezone(UTC)


def parse_since(spec: str | None, reference: datetime | None = None) -> datetime | None:
    """Turn ``24h`` / ``7d`` / an ISO date into an aware cutoff.

    Accepted forms:
      ``24h``, ``90m``, ``7d``, ``2w``, ``30s``  - relative to now
      ``2026-10-26``                              - midnight UTC that day
      ``2026-10-26T12:00:00Z``                    - an exact instant
      empty / ``all``                             - no window at all
    """
    if spec is None:
        return None
    text = spec.strip().lower()
    if not text or text in ("all", "none", "0"):
        return None

    relative = _RELATIVE.match(text)
    if relative:
        amount = float(relative.group(1))
        seconds = amount * _UNITS[relative.group(2)]
        return (reference or now()) - timedelta(seconds=seconds)

    if text.isdigit():
        # a bare number is read as hours, the common case
        return (reference or now()) - timedelta(hours=int(text))

    parsed = parse_timestamp(spec)
    if parsed is not None:
        return parsed
    raise ValueError(
        f"unrecognised --since value {spec!r}; use 24h, 90m, 7d, a bare hour "
        f"count, an ISO date (2026-10-26) or an ISO instant"
    )


def in_window(timestamp: object, since: datetime | None) -> bool:
    """True when a record belongs to the window.

    With no window everything belongs. With a window, a record that cannot be
    dated is excluded rather than assumed in - the numbers must not quietly
    include something that cannot be placed.
    """
    if since is None:
        return True
    parsed = parse_timestamp(timestamp)
    return parsed is not None and parsed >= since


def describe(since: datetime | None) -> str:
    if since is None:
        return "all time"
    return f"since {since.strftime('%Y-%m-%dT%H:%M:%SZ')}"


__all__ = ["describe", "in_window", "now", "parse_since", "parse_timestamp"]
