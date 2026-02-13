"""Integration test: full reconsolidation flow (labile -> extract -> conflict -> revise)."""

from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.reconsolidation.service import ReconsolidationService
from src.storage.postgres import PostgresMemoryStore


@pytest.mark.asyncio
async def test_reconsolidation_process_turn_no_memories(pg_session_factory):
    """With no retrieved memories, process_turn returns quickly with zero operations."""
    store = PostgresMemoryStore(pg_session_factory)
    svc = ReconsolidationService(memory_store=store, llm_client=None)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"
    result = await svc.process_turn(
        tenant_id=tenant_id,
        scope_id=user_id,
        turn_id="turn1",
        user_message="I like pizza.",
        assistant_response="Great! I'll remember that.",
        retrieved_memories=[],
    )
    assert result.turn_id == "turn1"
    assert result.memories_processed == 0
    assert result.conflicts_found == 0
    assert result.elapsed_ms >= 0


@pytest.mark.asyncio
async def test_reconsolidation_correction_flow(pg_session_factory):
    """Store a preference, retrieve it, then correct it; reconsolidation should apply revision."""
    store = PostgresMemoryStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    created = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.PREFERENCE,
            text="I prefer coffee.",
            key="user:preference:drink",
            confidence=0.8,
            importance=0.5,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )
    created.metadata["_similarity"] = 0.9

    svc = ReconsolidationService(memory_store=store, llm_client=None)
    result = await svc.process_turn(
        tenant_id=tenant_id,
        scope_id=user_id,
        turn_id="turn2",
        user_message="Actually, I prefer tea now.",
        assistant_response="Got it, I'll remember you prefer tea.",
        retrieved_memories=[created],
    )
    assert result.turn_id == "turn2"
    assert result.memories_processed == 1
    assert result.conflicts_found >= 1
    assert len(result.operations_applied) >= 1
    success_ops = [o for o in result.operations_applied if o.get("success")]
    assert len(success_ops) >= 1
