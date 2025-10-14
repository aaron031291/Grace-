"""Small timezone helper utilities used across the Grace codebase."""

from datetime import datetime, timezone
from typing import Optional


def now_utc() -> datetime:
    """Return a timezone-aware datetime in UTC."""
    return datetime.now(timezone.utc)


def iso_now_utc() -> str:
    """Return an ISO 8601 UTC timestamp string."""
    return now_utc().isoformat()


def to_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert a datetime to UTC timezone-aware datetime if not None.

    If dt is naive, assume system local timezone and attach UTC (best-effort).
    """
    if dt is None:
        return None
    if dt.tzinfo is None:
        # assume UTC for naive datetimes to avoid accidental local offsets
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
