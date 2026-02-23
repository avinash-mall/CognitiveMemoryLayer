"""Tests for Pydantic request/response model serialization."""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

import pytest
from pydantic import ValidationError as PydanticValidationError

from cml.models.enums import MemoryType
from cml.models.requests import (
    CreateSessionRequest,
    ForgetRequest,
    ReadRequest,
    TurnRequest,
    UpdateRequest,
    WriteRequest,
)
from cml.models.responses import (
    ForgetResponse,
    HealthResponse,
    MemoryItem,
    ReadResponse,
    SessionContextResponse,
    SessionResponse,
    UpdateResponse,
    WriteResponse,
)


def test_memory_item_parse_from_dict() -> None:
    """MemoryItem parses from API-style dict (id, text, type, confidence, relevance, timestamp, metadata)."""
    data = {
        "id": "00000000-0000-0000-0000-000000000001",
        "text": "User likes coffee",
        "type": "preference",
        "confidence": 0.9,
        "relevance": 0.85,
        "timestamp": "2025-01-15T12:00:00Z",
        "metadata": {"source": "chat"},
    }
    item = MemoryItem.model_validate(data)
    assert item.id == UUID("00000000-0000-0000-0000-000000000001")
    assert item.text == "User likes coffee"
    assert item.type == "preference"
    assert item.confidence == 0.9
    assert item.relevance == 0.85
    assert isinstance(item.timestamp, datetime)
    assert item.metadata == {"source": "chat"}


def test_memory_item_serialize_to_dict() -> None:
    """MemoryItem serializes to dict via model_dump."""
    item = MemoryItem(
        id=UUID("00000000-0000-0000-0000-000000000001"),
        text="Test memory",
        type="fact",
        confidence=0.8,
        relevance=0.7,
        timestamp=datetime(2025, 1, 1, 0, 0, 0),
        metadata={},
    )
    d = item.model_dump(mode="json")
    assert d["id"] == "00000000-0000-0000-0000-000000000001"
    assert d["text"] == "Test memory"
    assert d["type"] == "fact"
    assert d["confidence"] == 0.8
    assert d["relevance"] == 0.7
    assert "timestamp" in d
    assert d["metadata"] == {}


def test_write_request_content_empty_raises() -> None:
    """WriteRequest rejects empty content."""
    with pytest.raises(PydanticValidationError) as exc_info:
        WriteRequest(content="")
    assert "content" in str(exc_info.value).lower()


def test_write_request_content_max_length_raises() -> None:
    """WriteRequest rejects content exceeding 100,000 characters."""
    with pytest.raises(PydanticValidationError) as exc_info:
        WriteRequest(content="x" * 100_001)
    err_str = str(exc_info.value).lower()
    assert "content" in err_str or "100000" in err_str or "100" in err_str


def test_write_request_content_at_limit_accepted() -> None:
    """WriteRequest accepts content at exactly 100,000 characters."""
    req = WriteRequest(content="x" * 100_000)
    assert len(req.content) == 100_000


def test_write_request_model_dump_exclude_none() -> None:
    """WriteRequest model_dump(exclude_none=True) omits optional fields when None."""
    req = WriteRequest(content="hello")
    d = req.model_dump(exclude_none=True)
    assert d["content"] == "hello"
    assert "context_tags" not in d
    assert "session_id" not in d
    assert "memory_type" not in d
    assert "namespace" not in d
    assert "turn_id" not in d
    assert "agent_id" not in d
    assert "metadata" in d  # default_factory=dict gives {} so may be present
    req_full = WriteRequest(
        content="hi",
        session_id="s1",
        memory_type=MemoryType.PREFERENCE,
    )
    d_full = req_full.model_dump(exclude_none=True)
    assert d_full["session_id"] == "s1"
    assert d_full.get("memory_type") == "preference"


def test_read_request_model_dump_exclude_none() -> None:
    """ReadRequest model_dump(exclude_none=True) omits optional fields when None."""
    req = ReadRequest(query="test")
    d = req.model_dump(exclude_none=True)
    assert d["query"] == "test"
    assert d.get("max_results", 10) == 10
    assert "context_filter" not in d or d["context_filter"] is None
    assert "memory_types" not in d or d["memory_types"] is None
    assert "since" not in d or d["since"] is None
    assert "until" not in d or d["until"] is None
    assert "user_timezone" not in d or d["user_timezone"] is None


def test_read_request_user_timezone_serialized() -> None:
    """ReadRequest accepts user_timezone and includes it in payload when set."""
    req = ReadRequest(query="today?", user_timezone="America/New_York")
    d = req.model_dump(exclude_none=True, by_alias=True, mode="json")
    assert d["query"] == "today?"
    assert d["user_timezone"] == "America/New_York"
    req_none = ReadRequest(query="test")
    d_none = req_none.model_dump(exclude_none=True, by_alias=True, mode="json")
    assert "user_timezone" not in d_none


def test_turn_request_user_timezone_serialized() -> None:
    """TurnRequest accepts user_timezone and includes it in payload when set."""
    req = TurnRequest(user_message="Hi", user_timezone="Europe/London")
    d = req.model_dump(exclude_none=True, mode="json")
    assert d["user_message"] == "Hi"
    assert d["user_timezone"] == "Europe/London"
    req_none = TurnRequest(user_message="Hi")
    d_none = req_none.model_dump(exclude_none=True, mode="json")
    assert "user_timezone" not in d_none


def test_write_response_parse() -> None:
    """WriteResponse parses from API dict."""
    d = {
        "success": True,
        "memory_id": "00000000-0000-0000-0000-000000000001",
        "chunks_created": 2,
        "message": "Stored",
    }
    r = WriteResponse.model_validate(d)
    assert r.success is True
    assert r.memory_id == UUID("00000000-0000-0000-0000-000000000001")
    assert r.chunks_created == 2
    assert r.message == "Stored"
    assert r.eval_outcome is None
    assert r.eval_reason is None


def test_write_response_eval_mode() -> None:
    """WriteResponse parses eval_outcome and eval_reason when present (X-Eval-Mode)."""
    d = {
        "success": True,
        "memory_id": None,
        "chunks_created": 0,
        "message": "No significant information to store",
        "eval_outcome": "skipped",
        "eval_reason": "Below novelty threshold: 0.10 < 0.20",
    }
    r = WriteResponse.model_validate(d)
    assert r.eval_outcome == "skipped"
    assert r.eval_reason == "Below novelty threshold: 0.10 < 0.20"


def test_read_response_parse() -> None:
    """ReadResponse parses from API dict with memories."""
    d = {
        "query": "food",
        "memories": [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "text": "Likes pasta",
                "type": "preference",
                "confidence": 0.9,
                "relevance": 0.9,
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {},
            }
        ],
        "facts": [],
        "preferences": [],
        "episodes": [],
        "llm_context": "## Preferences\n- Likes pasta",
        "total_count": 1,
        "elapsed_ms": 50.0,
    }
    r = ReadResponse.model_validate(d)
    assert r.query == "food"
    assert len(r.memories) == 1
    assert r.memories[0].text == "Likes pasta"
    assert r.total_count == 1
    assert r.elapsed_ms == 50.0
    assert r.context == "## Preferences\n- Likes pasta"
    assert r.retrieval_meta is None


def test_read_response_parses_retrieval_meta() -> None:
    """ReadResponse parses optional retrieval_meta from server."""
    d = {
        "query": "test",
        "memories": [],
        "total_count": 0,
        "elapsed_ms": 10.0,
        "retrieval_meta": {"sources_completed": 2, "sources_timed_out": 0, "total_elapsed_ms": 8.5},
    }
    r = ReadResponse.model_validate(d)
    assert r.retrieval_meta is not None
    assert r.retrieval_meta["sources_completed"] == 2
    assert r.retrieval_meta["total_elapsed_ms"] == 8.5


def test_read_request_max_results_validation() -> None:
    """ReadRequest rejects max_results < 1 or > 50."""
    ReadRequest(query="q", max_results=1)
    ReadRequest(query="q", max_results=50)
    with pytest.raises(PydanticValidationError):
        ReadRequest(query="q", max_results=0)
    with pytest.raises(PydanticValidationError):
        ReadRequest(query="q", max_results=51)
    with pytest.raises(PydanticValidationError):
        ReadRequest(query="q", max_results=-1)


def test_read_request_format_alias() -> None:
    """ReadRequest accepts 'format' as alias for response_format in API payload."""
    req = ReadRequest(query="q", response_format="llm_context")  # type: ignore[call-arg]
    d = req.model_dump(by_alias=True, exclude_none=True)
    assert d.get("format") == "llm_context"
    assert "response_format" not in d or d.get("response_format") != "llm_context"


def test_turn_request_defaults() -> None:
    """TurnRequest has sensible defaults for max_context_tokens."""
    req = TurnRequest(user_message="Hello")
    assert req.assistant_response is None
    assert req.session_id is None
    assert req.max_context_tokens == 1500


def test_update_request_optional_fields() -> None:
    """UpdateRequest allows partial updates (text, confidence, etc.)."""
    mid = UUID("00000000-0000-0000-0000-000000000001")
    req = UpdateRequest(memory_id=mid, text="Updated text")
    d = req.model_dump(exclude_none=True, mode="json")
    assert d["memory_id"] == str(mid)
    assert d["text"] == "Updated text"
    assert "confidence" not in d


def test_forget_request_action_values() -> None:
    """ForgetRequest accepts action delete, archive, silence."""
    req = ForgetRequest(query="old", action="archive")
    assert req.action == "archive"
    d = req.model_dump(exclude_none=True, mode="json")
    assert d["query"] == "old"
    assert d["action"] == "archive"


def test_create_session_request_defaults() -> None:
    """CreateSessionRequest has ttl_hours and metadata defaults."""
    req = CreateSessionRequest(name="test")
    assert req.ttl_hours == 24
    assert req.metadata == {}
    assert req.name == "test"


def test_update_response_parse() -> None:
    """UpdateResponse parses from API dict."""
    d = {
        "success": True,
        "memory_id": "00000000-0000-0000-0000-000000000001",
        "version": 2,
        "message": "Updated",
    }
    r = UpdateResponse.model_validate(d)
    assert r.success is True
    assert r.memory_id == UUID("00000000-0000-0000-0000-000000000001")
    assert r.version == 2
    assert r.message == "Updated"


def test_forget_response_parse() -> None:
    """ForgetResponse parses from API dict."""
    d = {"success": True, "affected_count": 5, "message": "Forgotten"}
    r = ForgetResponse.model_validate(d)
    assert r.success is True
    assert r.affected_count == 5
    assert r.message == "Forgotten"


def test_health_response_parse() -> None:
    """HealthResponse parses from API dict."""
    d = {"status": "healthy", "version": "1.0.0", "components": {"db": "ok"}}
    r = HealthResponse.model_validate(d)
    assert r.status == "healthy"
    assert r.version == "1.0.0"
    assert r.components == {"db": "ok"}


def test_session_response_parse() -> None:
    """SessionResponse parses from API dict."""
    d = {
        "session_id": "sess-abc",
        "created_at": "2025-01-01T12:00:00Z",
        "expires_at": "2025-01-02T12:00:00Z",
    }
    r = SessionResponse.model_validate(d)
    assert r.session_id == "sess-abc"
    assert r.created_at.year == 2025 and r.created_at.month == 1 and r.created_at.day == 1
    assert r.expires_at is not None
    assert r.expires_at.day == 2


def test_session_context_response_parse() -> None:
    """SessionContextResponse parses with optional messages and context_string."""
    d = {
        "session_id": "s1",
        "messages": [],
        "tool_results": [],
        "scratch_pad": [],
        "context_string": "## Summary\nNone",
    }
    r = SessionContextResponse.model_validate(d)
    assert r.session_id == "s1"
    assert r.messages == []
    assert r.context_string == "## Summary\nNone"
