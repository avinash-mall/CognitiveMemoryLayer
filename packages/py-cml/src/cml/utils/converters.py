"""Shared converters for mapping API/dashboard payloads to SDK models."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any

from cml.models.responses import MemoryItem


def dashboard_item_to_memory_item(raw: dict[str, Any]) -> MemoryItem:
    """Map dashboard memory list item to MemoryItem."""
    ts = raw.get("timestamp") or raw.get("written_at")
    if isinstance(ts, str):
        ts = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    return MemoryItem(
        id=raw["id"],
        text=raw.get("text", ""),
        type=raw.get("type", "memory"),
        confidence=float(raw.get("confidence", 0.5)),
        relevance=float(raw.get("importance", raw.get("relevance", 0.5))),
        timestamp=ts or datetime.now(UTC),
        metadata={
            k: v
            for k, v in raw.items()
            if k
            not in (
                "id",
                "text",
                "type",
                "confidence",
                "relevance",
                "importance",
                "timestamp",
                "written_at",
            )
        },
    )
