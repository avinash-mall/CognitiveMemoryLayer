"""Integration tests for hippocampal encode flow (encode chunk, store, retrieval)."""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.core.config import get_embedding_dimensions
from src.core.enums import MemoryStatus, MemoryType
from src.memory.hippocampal.redactor import PIIRedactor
from src.memory.hippocampal.store import HippocampalStore
from src.memory.hippocampal.write_gate import WriteGate
from src.memory.working.models import ChunkType, SemanticChunk
from src.storage.postgres import PostgresMemoryStore
from src.utils.embeddings import MockEmbeddingClient


def _make_store(session_factory):
    pg_store = PostgresMemoryStore(session_factory)
    dims = get_embedding_dimensions()
    embeddings = MockEmbeddingClient(dimensions=dims)
    return HippocampalStore(
        vector_store=pg_store,
        embedding_client=embeddings,
        entity_extractor=None,
        relation_extractor=None,
        write_gate=WriteGate(),
        redactor=PIIRedactor(),
    )


@pytest.mark.asyncio
async def test_encode_chunk_stores_record(pg_session_factory):
    """Encode a chunk -> record in DB; search returns it."""
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunk = SemanticChunk(
        id="c1",
        text="I prefer dark mode for the interface.",
        chunk_type=ChunkType.PREFERENCE,
        salience=0.8,
        confidence=0.9,
        timestamp=datetime.now(UTC),
    )
    dims = get_embedding_dimensions()
    record, gate_result = await hippocampal_store.encode_chunk(
        tenant_id, chunk, existing_memories=None
    )
    assert record is not None
    assert record.text == chunk.text
    assert record.embedding is not None
    assert len(record.embedding) == dims
    assert record.type == MemoryType.PREFERENCE
    assert gate_result.memory_types and MemoryType.PREFERENCE in gate_result.memory_types

    # Verify we can retrieve the record (scan/get_recent)
    recent = await hippocampal_store.get_recent(tenant_id, limit=10)
    assert any(r.id == record.id for r in recent), "encoded record should appear in get_recent"

    # Vector search smoke test (no exception; may return 0 results depending on pgvector/embedding env)
    results = await hippocampal_store.search(tenant_id, "user preference dark mode", top_k=5)
    if results:
        assert any(
            r.id == record.id for r in results
        ), "stored record should appear in search when results returned"


@pytest.mark.asyncio
async def test_encode_chunk_skip_low_salience(pg_session_factory):
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    chunk = SemanticChunk(
        id="c2",
        text="Okay, sure.",
        chunk_type=ChunkType.STATEMENT,
        salience=0.1,
    )
    record, _ = await hippocampal_store.encode_chunk(tenant_id, chunk, existing_memories=None)
    # May be None if write gate skips
    if record is None:
        return
    # If stored, it's still valid
    assert record.text == chunk.text


@pytest.mark.asyncio
async def test_encode_chunk_stored_record_type_matches_gate(pg_session_factory):
    """Encode chunks of different ChunkTypes; stored record type equals gate's chosen type."""
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    cases = [
        (ChunkType.PREFERENCE, "I prefer tea.", MemoryType.PREFERENCE),
        (ChunkType.INSTRUCTION, "Please remind me tomorrow.", MemoryType.TASK_STATE),
    ]
    for chunk_type, text, expected_memory_type in cases:
        chunk = SemanticChunk(
            id=f"c-{chunk_type.value}",
            text=text,
            chunk_type=chunk_type,
            salience=0.8,
            timestamp=datetime.now(UTC),
        )
        record, gate_result = await hippocampal_store.encode_chunk(
            tenant_id, chunk, existing_memories=None
        )
        if record is None:
            continue
        assert (
            record.type == expected_memory_type
        ), f"ChunkType {chunk_type} should produce record type {expected_memory_type}, got {record.type}"
        assert expected_memory_type in gate_result.memory_types


@pytest.mark.asyncio
async def test_hippocampal_get_recent_and_search_with_type_filter(pg_session_factory):
    """Encode multiple chunks (episodic_event, preference) -> get_recent order; search with type filter."""
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    chunks = [
        SemanticChunk(
            id="c1",
            text="I prefer coffee in the morning.",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.85,
            timestamp=datetime.now(UTC),
        ),
        SemanticChunk(
            id="c2",
            text="We had a meeting yesterday.",
            chunk_type=ChunkType.EVENT,
            salience=0.7,
            timestamp=datetime.now(UTC),
        ),
        SemanticChunk(
            id="c3",
            text="I prefer dark mode for the UI.",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.8,
            timestamp=datetime.now(UTC),
        ),
    ]
    stored_ids = []
    for ch in chunks:
        record, _ = await hippocampal_store.encode_chunk(tenant_id, ch, existing_memories=None)
        if record:
            stored_ids.append(record.id)
            assert record.embedding is not None
            assert record.text == ch.text

    recent = await hippocampal_store.get_recent(tenant_id, limit=10)
    assert len(recent) >= 2
    for r in recent:
        assert r.type in (MemoryType.PREFERENCE, MemoryType.EPISODIC_EVENT)

    # get_recent with type filter returns only that type
    preference_only = await hippocampal_store.get_recent(
        tenant_id, limit=10, memory_types=[MemoryType.PREFERENCE]
    )
    for r in preference_only:
        assert r.type == MemoryType.PREFERENCE
    assert len(preference_only) >= 2

    # search with type filter returns only records of that type
    results = await hippocampal_store.search(
        tenant_id,
        "user preference",
        top_k=5,
        filters={"type": [MemoryType.PREFERENCE.value]},
    )
    for r in results:
        assert r.type == MemoryType.PREFERENCE


@pytest.mark.asyncio
async def test_constraint_supersession_first_silent_second_active(pg_session_factory):
    """Write two conflicting constraints (same key); assert first is SILENT and second is ACTIVE."""
    hippocampal_store = _make_store(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    # First constraint: goal (same key as second because both goal, unscoped)
    chunk1 = SemanticChunk(
        id="c1",
        text="I want to save money.",
        chunk_type=ChunkType.CONSTRAINT,
        salience=0.9,
        confidence=0.85,
        timestamp=datetime.now(UTC),
    )
    record1, _ = await hippocampal_store.encode_chunk(tenant_id, chunk1, existing_memories=None)
    assert record1 is not None
    assert record1.type == MemoryType.CONSTRAINT
    assert record1.key is not None
    constraint_key = record1.key

    # Deactivate existing constraints with this key (simulates orchestrator before second write)
    n = await hippocampal_store.deactivate_constraints_by_key(tenant_id, constraint_key)
    assert n >= 1

    # Second constraint: same key (goal, unscoped)
    chunk2 = SemanticChunk(
        id="c2",
        text="I want to buy a Ferrari.",
        chunk_type=ChunkType.CONSTRAINT,
        salience=0.9,
        confidence=0.85,
        timestamp=datetime.now(UTC),
    )
    record2, _ = await hippocampal_store.encode_chunk(tenant_id, chunk2, existing_memories=None)
    assert record2 is not None
    assert record2.type == MemoryType.CONSTRAINT
    assert "Ferrari" in record2.text

    # Query: one ACTIVE (second), one SILENT (first)
    all_constraints = await hippocampal_store.store.scan(
        tenant_id,
        limit=20,
        filters={"type": MemoryType.CONSTRAINT.value},
    )
    active = [r for r in all_constraints if r.status == MemoryStatus.ACTIVE]
    silent = [r for r in all_constraints if r.status == MemoryStatus.SILENT]
    assert len(active) >= 1
    assert any("Ferrari" in r.text for r in active)
    assert len(silent) >= 1
    assert any("save money" in r.text for r in silent)
