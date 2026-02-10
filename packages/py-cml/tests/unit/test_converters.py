"""Tests for cml.utils.converters."""

from __future__ import annotations

from datetime import datetime, timezone
from uuid import UUID

import pytest

from cml.models.responses import MemoryItem
from cml.utils.converters import dashboard_item_to_memory_item


def test_dashboard_item_to_memory_item_full() -> None:
    """Maps dashboard item with all fields to MemoryItem."""
    raw = {
        "id": "00000000-0000-0000-0000-000000000001",
        "text": "User prefers dark mode",
        "type": "preference",
        "confidence": 0.9,
        "relevance": 0.85,
        "timestamp": "2025-01-15T12:00:00Z",
    }
    item = dashboard_item_to_memory_item(raw)
    assert isinstance(item, MemoryItem)
    assert item.id == UUID("00000000-0000-0000-0000-000000000001")
    assert item.text == "User prefers dark mode"
    assert item.type == "preference"
    assert item.confidence == 0.9
    assert item.relevance == 0.85
    assert item.timestamp == datetime(2025, 1, 15, 12, 0, 0, tzinfo=timezone.utc)
    assert item.metadata == {}


def test_dashboard_item_uses_written_at_when_no_timestamp() -> None:
    """Uses written_at when timestamp is missing."""
    raw = {
        "id": "00000000-0000-0000-0000-000000000002",
        "text": "A fact",
        "written_at": "2025-02-01T00:00:00+00:00",
    }
    item = dashboard_item_to_memory_item(raw)
    assert item.timestamp == datetime(2025, 2, 1, 0, 0, 0, tzinfo=timezone.utc)
    assert item.text == "A fact"
    assert item.type == "memory"
    assert item.confidence == 0.5
    assert item.relevance == 0.5


def test_dashboard_item_uses_importance_fallback_for_relevance() -> None:
    """Uses importance when relevance is missing."""
    raw = {
        "id": "00000000-0000-0000-0000-000000000003",
        "text": "Important",
        "importance": 0.7,
    }
    item = dashboard_item_to_memory_item(raw)
    assert item.relevance == 0.7


def test_dashboard_item_fallback_timestamp_timezone_aware() -> None:
    """When no timestamp/written_at, fallback is timezone-aware (UTC)."""
    raw = {"id": "00000000-0000-0000-0000-000000000004", "text": "No time"}
    item = dashboard_item_to_memory_item(raw)
    assert item.timestamp.tzinfo is not None
    assert item.timestamp.tzinfo == timezone.utc


def test_dashboard_item_metadata_passthrough() -> None:
    """Extra keys become metadata (excluding known fields)."""
    raw = {
        "id": "00000000-0000-0000-0000-000000000005",
        "text": "x",
        "timestamp": "2025-01-01T00:00:00Z",
        "source": "import",
        "namespace": "ns1",
    }
    item = dashboard_item_to_memory_item(raw)
    assert item.metadata.get("source") == "import"
    assert item.metadata.get("namespace") == "ns1"


def test_dashboard_item_empty_text() -> None:
    """Handles missing text (defaults to empty string)."""
    raw = {"id": "00000000-0000-0000-0000-000000000006"}
    item = dashboard_item_to_memory_item(raw)
    assert item.text == ""
