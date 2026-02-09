"""Tests for Pydantic request/response model serialization."""

from datetime import datetime
from uuid import UUID

import pytest

from cml.models.requests import ReadRequest, WriteRequest
from cml.models.responses import MemoryItem, ReadResponse, WriteResponse


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
        memory_type="preference",
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
