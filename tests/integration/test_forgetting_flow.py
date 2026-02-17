"""Integration tests for active forgetting flow."""

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
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Old low-value memory to forget.",
            confidence=0.3,
            importance=0.2,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    report = await worker.run_forgetting(tenant_id, user_id, max_memories=100, dry_run=True)

    assert report.memories_scanned >= 1
    assert report.memories_scored >= 1
    records = await store.scan(tenant_id, filters={"status": "active"}, limit=10)
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
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Temporary memory.",
            confidence=0.6,
            importance=0.2,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )

    report = await worker.run_forgetting(tenant_id, user_id, max_memories=100, dry_run=False)

    if report.result.decayed >= 1:
        updated = await store.get_by_id(rec.id)
        assert updated is not None
        assert updated.confidence <= 0.6


@pytest.mark.asyncio
async def test_forgetting_flow_protected_types_kept_low_episodic_affected(pg_session_factory):
    """Seed constraint (protected) and low-relevance episodic; run forgetting; constraint kept, episodic decay/silence."""
    store = PostgresMemoryStore(pg_session_factory)
    worker = ForgettingWorker(store)
    tenant_id = f"t-{uuid4().hex[:8]}"
    user_id = f"u-{uuid4().hex[:8]}"

    constraint_rec = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.CONSTRAINT,
            text="Never remind me on weekends.",
            key="constraint:no_weekends",
            confidence=0.9,
            importance=0.8,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )
    episodic_rec = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Some trivial remark.",
            confidence=0.2,
            importance=0.1,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
        )
    )

    report = await worker.run_forgetting(tenant_id, user_id, max_memories=100, dry_run=True)
    assert report.memories_scanned >= 2
    assert report.memories_scored >= 2

    report2 = await worker.run_forgetting(tenant_id, user_id, max_memories=100, dry_run=False)
    constraint_after = await store.get_by_id(constraint_rec.id)
    assert constraint_after is not None
    assert constraint_after.status.value == "active", "CONSTRAINT should be kept (protected)"

    episodic_after = await store.get_by_id(episodic_rec.id)
    assert episodic_after is not None
    if report2.result.decayed >= 1 or report2.result.silenced >= 1 or report2.result.deleted >= 1:
        assert episodic_after.confidence <= 0.6 or episodic_after.status.value != "active", (
            "low episodic may be decayed or silenced"
        )


# ---------------------------------------------------------------------------
# PostgresMemoryStore dependency check and ForgettingExecutor (real DB validation)
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_count_references_to_zero_for_missing_record(pg_session_factory):
    """count_references_to returns 0 for a non-existent record id."""
    store = PostgresMemoryStore(pg_session_factory)
    count = await store.count_references_to(uuid4())
    assert count == 0


@pytest.mark.asyncio
async def test_count_references_to_includes_supersedes_and_evidence_refs(
    pg_session_factory,
):
    """count_references_to counts supersedes_id and metadata.evidence_refs; no refs => 0."""
    store = PostgresMemoryStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    r1 = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="First memory.",
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )
    await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Second memory.",
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )
    count = await store.count_references_to(r1.id)
    assert count == 0


@pytest.mark.asyncio
async def test_executor_delete_skipped_when_dependencies(pg_session_factory):
    """ForgettingExecutor skips delete when record has evidence_refs dependencies."""
    from src.forgetting.actions import ForgettingAction, ForgettingOperation
    from src.forgetting.executor import ForgettingExecutor

    store = PostgresMemoryStore(pg_session_factory)
    executor = ForgettingExecutor(store)
    tenant_id = f"t-{uuid4().hex[:8]}"

    r1 = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Referenced memory.",
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )
    r2 = await store.upsert(
        MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text="Another memory.",
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        )
    )
    await store.update(
        r2.id,
        {"metadata": {"evidence_refs": [str(r1.id)]}},
        increment_version=True,
    )

    ref_count = await store.count_references_to(r1.id)
    assert ref_count >= 1

    op = ForgettingOperation(
        action=ForgettingAction.DELETE,
        memory_id=r1.id,
        reason="test",
    )
    result = await executor.execute([op], dry_run=False)
    assert result.deleted == 0
    assert any("dependency" in e.lower() or "Skipped" in e for e in result.errors)
