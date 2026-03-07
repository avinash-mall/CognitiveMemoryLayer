from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from cml._endpoints import (
    build_create_session_body,
    build_forget_body,
    build_read_body,
    build_turn_body,
    build_update_body,
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


def test_build_turn_body_serializes_timestamp() -> None:
    timestamp = datetime(2025, 1, 5, tzinfo=UTC)

    body = build_turn_body(
        "hello",
        assistant_response="hi",
        session_id="sess-1",
        max_context_tokens=2048,
        timestamp=timestamp,
        user_timezone="UTC",
    )

    assert body == {
        "user_message": "hello",
        "assistant_response": "hi",
        "session_id": "sess-1",
        "max_context_tokens": 2048,
        "timestamp": "2025-01-05T00:00:00Z",
        "user_timezone": "UTC",
    }


def test_build_update_body_returns_path_and_payload() -> None:
    memory_id = uuid4()

    path, body = build_update_body(
        memory_id,
        text="updated",
        confidence=0.9,
        importance=0.8,
        metadata={"k": "v"},
        feedback="correct",
    )

    assert path == f"/memory/{memory_id}"
    assert body["memory_id"] == str(memory_id)
    assert body["text"] == "updated"
    assert body["feedback"] == "correct"


def test_build_forget_body_includes_optional_filters() -> None:
    before = datetime(2025, 1, 6, tzinfo=UTC)

    body = build_forget_body(
        before=before,
        memory_type=MemoryType.CONSTRAINT,
        namespace="ns-1",
        context_tags=["guardrail"],
        dry_run=False,
    )

    assert body == {
        "dry_run": False,
        "before": before,
        "memory_type": MemoryType.CONSTRAINT,
        "namespace": "ns-1",
        "context_tags": ["guardrail"],
    }


def test_build_create_session_body_supports_optional_fields() -> None:
    body = build_create_session_body(
        session_id="sess-9",
        metadata={"source": "test"},
        context_tags=["chat"],
    )

    assert body == {
        "metadata": {"source": "test"},
        "session_id": "sess-9",
        "context_tags": ["chat"],
    }


def test_eval_mode_headers_returns_header_only_when_enabled() -> None:
    assert eval_mode_headers(True) == {"X-Eval-Mode": "true"}
    assert eval_mode_headers(False) is None
