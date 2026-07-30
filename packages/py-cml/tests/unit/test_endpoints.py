from __future__ import annotations

from datetime import UTC, datetime

from cml._endpoints import (
    build_read_body,
    build_write_body,
    eval_mode_headers,
)
from cml.models.enums import MemoryType


def test_build_write_body_serializes_optional_fields() -> None:
    timestamp = datetime(2025, 1, 2, 3, 4, 5, tzinfo=UTC)

    body = build_write_body(
        "remember this",
        context_tags=["prefs"],
        session_id="sess-1",
        memory_type=MemoryType.PREFERENCE,
        namespace="ns-1",
        metadata={"source": "test"},
        turn_id="turn-1",
        agent_id="agent-1",
        timestamp=timestamp,
    )

    assert body["content"] == "remember this"
    assert body["memory_type"] == "preference"
    assert body["timestamp"] == "2025-01-02T03:04:05Z"
    assert body["namespace"] == "ns-1"


def test_build_read_body_serializes_filters_and_format() -> None:
    since = datetime(2025, 1, 1, tzinfo=UTC)
    until = datetime(2025, 1, 3, tzinfo=UTC)

    body = build_read_body(
        "tea",
        max_results=5,
        context_filter=["food"],
        memory_types=[MemoryType.PREFERENCE],
        since=since,
        until=until,
        response_format="llm_context",
        user_timezone="America/New_York",
    )

    assert body["query"] == "tea"
    assert body["max_results"] == 5
    assert body["memory_types"] == ["preference"]
    assert body["format"] == "llm_context"
    assert body["since"] == "2025-01-01T00:00:00Z"
    assert body["until"] == "2025-01-03T00:00:00Z"
    assert body["user_timezone"] == "America/New_York"


def test_eval_mode_headers_returns_header_only_when_enabled() -> None:
    assert eval_mode_headers(True) == {"X-Eval-Mode": "true"}
    assert eval_mode_headers(False) is None
