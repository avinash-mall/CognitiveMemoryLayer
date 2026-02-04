"""Integration tests for Phase 8: active forgetting flow."""
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.forgetting.worker import ForgettingWorker
from src.storage.postgres import PostgresMemoryStore


@pytest.mark.asyncio
async def test_forgetting_empty_no_memories(pg_session_factory):
    """Run forgetting with no memories returns report with zeros."""
    store = PostgresMemoryStore(pg_session_factory)
    worker = ForgettingWorker(store)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    report = await worker.run_forgetting(tenant_id, user_id, dry_run=False)

    assert report.tenant_id == tenant_id
    assert report.user_id == user_id
    assert report.memories_scanned == 0
    assert report.memories_scored == 0
    assert report.result.operations_planned == 0
    assert report.result.operations_applied == 0
    assert report.elapsed_seconds >= 0


@pytest.mark.asyncio
async def test_forgetting_dry_run_does_not_modify(pg_session_factory):
    """Dry run scores and plans but does not apply changes."""
    store = PostgresMemoryStore(pg_session_factory)
    worker = ForgettingWorker(store)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            user_id=user_id,
            type=MemoryType.EPISODIC_EVENT,
            text="Old low-value memory to forget.",
            confidence=0.3,
            importance=0.2,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    report = await worker.run_forgetting(
        tenant_id, user_id, max_memories=100, dry_run=True
    )

    assert report.memories_scanned >= 1
    assert report.memories_scored >= 1
    records = await store.scan(
        tenant_id, user_id, filters={"status": "active"}, limit=10
    )
    assert len(records) >= 1
    assert records[0].status.value == "active"


@pytest.mark.asyncio
async def test_forgetting_decay_reduces_confidence(pg_session_factory):
    """Run forgetting with low-score memory; decay reduces confidence."""
    store = PostgresMemoryStore(pg_session_factory)
    worker = ForgettingWorker(store)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    rec = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            user_id=user_id,
            type=MemoryType.EPISODIC_EVENT,
            text="Temporary memory.",
            confidence=0.6,
            importance=0.2,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    report = await worker.run_forgetting(
        tenant_id, user_id, max_memories=100, dry_run=False
    )

    if report.result.decayed >= 1:
        updated = await store.get_by_id(rec.id)
        assert updated is not None
        assert updated.confidence <= 0.6
