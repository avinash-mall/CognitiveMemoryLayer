"""Unit tests for Phase 1: core enums, schemas, and config."""
import pytest

from src.core.config import get_settings
from src.core.enums import MemorySource, MemoryStatus, MemoryType, OperationType
from src.core.schemas import (
    EntityMention,
    EventLog,
    MemoryPacket,
    MemoryRecord,
    MemoryRecordCreate,
    Provenance,
    Relation,
    RetrievedMemory,
)


class TestEnums:
    """Core enums have expected values."""

    def test_memory_type_values(self):
        assert MemoryType.EPISODIC_EVENT.value == "episodic_event"
        assert MemoryType.SEMANTIC_FACT.value == "semantic_fact"
        assert MemoryType.PREFERENCE.value == "preference"

    def test_memory_status_values(self):
        assert MemoryStatus.ACTIVE.value == "active"
        assert MemoryStatus.DELETED.value == "deleted"

    def test_memory_source_values(self):
        assert MemorySource.USER_EXPLICIT.value == "user_explicit"
        assert MemorySource.AGENT_INFERRED.value == "agent_inferred"

    def test_operation_type_values(self):
        assert OperationType.ADD.value == "add"
        assert OperationType.UPDATE.value == "update"


class TestProvenance:
    """Provenance schema."""

    def test_provenance_minimal(self):
        p = Provenance(source=MemorySource.USER_EXPLICIT)
        assert p.source == MemorySource.USER_EXPLICIT
        assert p.evidence_refs == []
        assert p.tool_refs == []

    def test_provenance_with_refs(self):
        p = Provenance(
            source=MemorySource.AGENT_INFERRED,
            evidence_refs=["evt-1"],
            tool_refs=["tool-1"],
        )
        assert p.evidence_refs == ["evt-1"]
        assert p.tool_refs == ["tool-1"]


class TestMemoryRecordSchema:
    """MemoryRecord and MemoryRecordCreate."""

    def test_memory_record_create_roundtrip(self):
        prov = Provenance(source=MemorySource.USER_EXPLICIT)
        create = MemoryRecordCreate(
            tenant_id="t1",
            user_id="u1",
            type=MemoryType.SEMANTIC_FACT,
            text="User lives in Paris",
            key="user:location",
            confidence=0.9,
            importance=0.8,
            provenance=prov,
        )
        assert create.tenant_id == "t1"
        assert create.type == MemoryType.SEMANTIC_FACT
        assert create.key == "user:location"

    def test_memory_record_has_defaults(self):
        prov = Provenance(source=MemorySource.USER_EXPLICIT)
        record = MemoryRecord(
            tenant_id="t1",
            user_id="u1",
            type=MemoryType.PREFERENCE,
            text="Prefers dark mode",
            provenance=prov,
        )
        assert record.id is not None
        assert record.confidence == 0.5
        assert record.importance == 0.5
        assert record.status == MemoryStatus.ACTIVE
        assert record.version == 1
        assert record.access_count == 0


class TestEventLogSchema:
    """EventLog schema."""

    def test_event_log_minimal(self):
        e = EventLog(
            tenant_id="t1",
            user_id="u1",
            event_type="turn",
            payload={"message": "hello"},
        )
        assert e.id is not None
        assert e.operation is None
        assert e.memory_ids == []
        assert e.created_at is not None

    def test_event_log_with_operation(self):
        e = EventLog(
            tenant_id="t1",
            user_id="u1",
            event_type="memory_op",
            operation=OperationType.ADD,
            payload={"op": "add"},
        )
        assert e.operation == OperationType.ADD


class TestMemoryPacket:
    """MemoryPacket and to_context_string."""

    def test_memory_packet_all_memories(self):
        prov = Provenance(source=MemorySource.USER_EXPLICIT)
        record = MemoryRecord(
            tenant_id="t1",
            user_id="u1",
            type=MemoryType.SEMANTIC_FACT,
            text="A fact",
            provenance=prov,
        )
        rm = RetrievedMemory(record=record, relevance_score=0.9, retrieval_source="vector")
        packet = MemoryPacket(query="test", facts=[rm])
        assert len(packet.all_memories) == 1
        assert packet.facts[0].relevance_score == 0.9

    def test_memory_packet_to_context_string(self):
        prov = Provenance(source=MemorySource.USER_EXPLICIT)
        record = MemoryRecord(
            tenant_id="t1",
            user_id="u1",
            type=MemoryType.SEMANTIC_FACT,
            text="User likes coffee",
            provenance=prov,
        )
        rm = RetrievedMemory(record=record, relevance_score=0.9, retrieval_source="vector")
        packet = MemoryPacket(query="preferences", preferences=[rm])
        s = packet.to_context_string(max_chars=500)
        assert "## Preferences" in s
        assert "User likes coffee" in s
        assert "confidence:" in s
        assert len(s) <= 500


class TestConfig:
    """Configuration loading."""

    def test_get_settings_returns_settings(self):
        s = get_settings()
        assert s.app_name == "CognitiveMemoryLayer"
        assert s.database is not None
        assert s.database.postgres_url is not None
        assert "asyncpg" in s.database.postgres_url or "postgresql" in s.database.postgres_url

    def test_get_settings_cached(self):
        a = get_settings()
        b = get_settings()
        assert a is b

    def test_nested_settings_defaults(self):
        s = get_settings()
        assert s.embedding.provider in ("openai", "local")
        assert s.memory.working_memory_max_chunks == 10
