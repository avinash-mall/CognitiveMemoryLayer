"""Unit tests for API route helpers (_to_memory_item, _safe_500_detail)."""

from datetime import datetime, timezone
from uuid import uuid4

from src.api.routes import _safe_500_detail, _to_memory_item
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory


def _make_retrieved_memory(
    text: str = "test memory",
    relevance_score: float = 0.85,
    memory_type: MemoryType = MemoryType.SEMANTIC_FACT,
    confidence: float = 0.9,
    metadata: dict | None = None,
) -> RetrievedMemory:
    record = MemoryRecord(
        id=uuid4(),
        tenant_id="t1",
        context_tags=[],
        type=memory_type,
        text=text,
        confidence=confidence,
        timestamp=datetime.now(timezone.utc),
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )
    if metadata is not None:
        record.metadata = metadata
    return RetrievedMemory(
        record=record,
        relevance_score=relevance_score,
        retrieval_source="vector",
    )


class TestToMemoryItem:
    """Tests for _to_memory_item."""

    def test_maps_retrieved_memory_to_memory_item(self):
        mem = _make_retrieved_memory(text="User loves Python", relevance_score=0.9)
        item = _to_memory_item(mem)
        assert item.id == mem.record.id
        assert item.text == "User loves Python"
        assert item.type == "semantic_fact"
        assert item.confidence == 0.9
        assert item.relevance == 0.9
        assert item.timestamp == mem.record.timestamp
        assert item.metadata == {}

    def test_memory_type_value_used(self):
        mem = _make_retrieved_memory(memory_type=MemoryType.EPISODIC_EVENT)
        item = _to_memory_item(mem)
        assert item.type == "episodic_event"

    def test_metadata_defaults_to_empty_dict(self):
        mem = _make_retrieved_memory()
        item = _to_memory_item(mem)
        assert item.metadata == {}

    def test_metadata_passthrough(self):
        mem = _make_retrieved_memory(metadata={"source": "chat"})
        item = _to_memory_item(mem)
        assert item.metadata == {"source": "chat"}


class TestSafe500Detail:
    """Tests for _safe_500_detail."""

    def test_returns_internal_server_error_when_not_debug(self, monkeypatch):
        monkeypatch.setenv("DEBUG", "false")
        from src.core.config import get_settings

        get_settings.cache_clear()
        try:
            detail = _safe_500_detail(ValueError("secret info"))
            assert detail == "Internal server error"
        finally:
            get_settings.cache_clear()

    def test_returns_exception_message_when_debug(self, monkeypatch):
        monkeypatch.setenv("DEBUG", "true")
        from src.core.config import get_settings

        get_settings.cache_clear()
        try:
            detail = _safe_500_detail(ValueError("secret info"))
            assert "secret info" in detail
        finally:
            get_settings.cache_clear()
