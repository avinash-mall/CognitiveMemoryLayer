"""Shared storage utilities (e.g. datetime normalization for PostgreSQL)."""

from datetime import datetime, timezone
from typing import Optional


def naive_utc(dt: Optional[datetime]) -> Optional[datetime]:
    """Convert to naive UTC for PostgreSQL TIMESTAMP WITHOUT TIME ZONE."""
    if dt is None:
        return None
    if dt.tzinfo:
        return dt.astimezone(timezone.utc).replace(tzinfo=None)
    return dt
