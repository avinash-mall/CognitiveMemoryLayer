"""Shared storage utilities (e.g. datetime normalization for PostgreSQL)."""

from datetime import UTC, datetime


def naive_utc(dt: datetime | None) -> datetime | None:
    """Convert to naive UTC for PostgreSQL TIMESTAMP WITHOUT TIME ZONE."""
    if dt is None:
        return None
    if dt.tzinfo:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt
