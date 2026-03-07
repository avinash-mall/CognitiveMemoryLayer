from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

try:
    from cml.storage.sqlite_store import SQLiteMemoryStore
    from src.core.enums import MemorySource, MemoryStatus, MemoryType
    from src.core.schemas import MemoryRecordCreate, Provenance
except ImportError as e:
    pytest.skip(
        f"CML engine (src) and redis required for sqlite_store tests: {e}",
        allow_module_level=True,
    )


def _record(
    text: str,
    *,
    memory_type: MemoryType = MemoryType.EPISODIC_EVENT,
    tenant_id: str = "tenant-a",
    key: str | None = None,
    namespace: str | None = None,
    source_session_id: str | None = None,
    embedding: list[float] | None = None,
    timestamp: datetime | None = None,
) -> MemoryRecordCreate:
    return MemoryRecordCreate(
        tenant_id=tenant_id,
        type=memory_type,
        text=text,
        key=key,
        namespace=namespace,
        source_session_id=source_session_id,
        embedding=embedding,
        timestamp=timestamp or datetime.now(UTC),
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )


@pytest.mark.asyncio
async def test_upsert_updates_existing_record_and_get_by_key() -> None:
    store = SQLiteMemoryStore()
    await store.initialize()

    first = await store.upsert(
        _record(
            "same text",
            memory_type=MemoryType.PREFERENCE,
            key="pref:key",
            embedding=[1.0, 0.0, 0.0],
        )
    )
    second = await store.upsert(
        _record(
            "same text",
            memory_type=MemoryType.PREFERENCE,
            key="pref:key",
            embedding=[1.0, 0.0, 0.0],
        )
    )
    fetched = await store.get_by_key("tenant-a", "pref:key")

    assert second.id == first.id
    assert second.version == first.version + 1
    assert fetched is not None
    assert fetched.id == first.id

    await store.close()


@pytest.mark.asyncio
async def test_scan_count_and_delete_by_filter_support_list_values() -> None:
    store = SQLiteMemoryStore()
    await store.initialize()
    now = datetime.now(UTC)
    active_constraint = await store.upsert(
        _record(
            "active constraint",
            memory_type=MemoryType.CONSTRAINT,
            namespace="ns-1",
            source_session_id="sess-a",
            embedding=[0.9, 0.1, 0.0],
            timestamp=now,
        )
    )
    expired_constraint = await store.upsert(
        _record(
            "expired constraint",
            memory_type=MemoryType.CONSTRAINT,
            namespace="ns-1",
            source_session_id="sess-b",
            embedding=[0.8, 0.2, 0.0],
            timestamp=now - timedelta(days=2),
        )
    )
    preference = await store.upsert(
        _record(
            "user likes tea",
            memory_type=MemoryType.PREFERENCE,
            namespace="ns-1",
            source_session_id="sess-a",
            embedding=[0.2, 0.8, 0.0],
            timestamp=now - timedelta(hours=1),
        )
    )

    await store.update(expired_constraint.id, {"valid_to": now - timedelta(days=1)})
    await store.delete(preference.id)

    active_rows = await store.scan(
        "tenant-a",
        filters={
            "type": [MemoryType.CONSTRAINT.value, MemoryType.PREFERENCE.value],
            "status": [MemoryStatus.ACTIVE.value],
            "source_session_id": ["sess-a", "sess-b"],
            "namespace": ["ns-1"],
            "exclude_expired": True,
        },
    )
    deleted_count = await store.count(
        "tenant-a",
        filters={"status": [MemoryStatus.DELETED.value], "type": [MemoryType.PREFERENCE.value]},
    )
    removed = await store.delete_by_filter(
        "tenant-a",
        filters={"type": [MemoryType.CONSTRAINT.value], "namespace": ["ns-1"]},
    )

    assert [row.text for row in active_rows] == [active_constraint.text]
    assert deleted_count == 1
    assert removed == 2

    await store.close()


@pytest.mark.asyncio
async def test_vector_search_ranks_memories_and_respects_filters() -> None:
    store = SQLiteMemoryStore()
    await store.initialize()
    await store.upsert(
        _record(
            "constraint match",
            memory_type=MemoryType.CONSTRAINT,
            embedding=[1.0, 0.0, 0.0],
        )
    )
    await store.upsert(
        _record(
            "preference match",
            memory_type=MemoryType.PREFERENCE,
            embedding=[0.0, 1.0, 0.0],
        )
    )
    await store.upsert(
        _record(
            "no embedding",
            memory_type=MemoryType.CONSTRAINT,
            embedding=None,
        )
    )

    results = await store.vector_search(
        "tenant-a",
        [0.98, 0.02, 0.0],
        top_k=2,
        filters={"type": [MemoryType.CONSTRAINT.value]},
        min_similarity=0.5,
    )

    assert [record.text for record in results] == ["constraint match"]
    await store.close()


@pytest.mark.asyncio
async def test_update_meta_alias_and_hard_delete() -> None:
    store = SQLiteMemoryStore()
    await store.initialize()
    created = await store.upsert(_record("to update", embedding=[0.1, 0.2, 0.3]))
    valid_to = datetime.now(UTC) + timedelta(days=30)

    updated = await store.update(
        created.id,
        {"meta": {"flag": True}, "text": "updated text", "valid_to": valid_to},
    )
    hard_deleted = await store.delete(created.id, hard=True)
    fetched = await store.get_by_id(created.id)

    assert updated is not None
    assert updated.metadata == {"flag": True}
    assert updated.text == "updated text"
    assert updated.valid_to == valid_to
    assert hard_deleted is True
    assert fetched is None

    await store.close()
