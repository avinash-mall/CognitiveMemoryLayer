"""Integration tests for Phase 1: EventLogRepository against PostgreSQL."""
from uuid import uuid4

import pytest

from src.core.enums import OperationType
from src.core.schemas import EventLog
from src.storage.event_log import EventLogRepository


@pytest.mark.asyncio
async def test_event_log_append_and_get_by_id(event_log_repo: EventLogRepository):
    """Append an event and fetch it by ID."""
    event = EventLog(
        tenant_id="tenant-1",
        user_id="user-1",
        event_type="turn",
        operation=OperationType.ADD,
        payload={"role": "user", "content": "Hello"},
        memory_ids=[],
    )
    appended = await event_log_repo.append(event)
    assert appended.id == event.id

    found = await event_log_repo.get_by_id(event.id)
    assert found is not None
    assert found.tenant_id == "tenant-1"
    assert found.user_id == "user-1"
    assert found.event_type == "turn"
    assert found.operation == OperationType.ADD
    assert found.payload == {"role": "user", "content": "Hello"}


@pytest.mark.asyncio
async def test_event_log_get_user_events(event_log_repo: EventLogRepository):
    """List events for a user; order by created_at desc."""
    base = EventLog(
        tenant_id="tenant-2",
        user_id="user-2",
        event_type="memory_op",
        operation=OperationType.ADD,
        payload={"op": "add"},
    )
    await event_log_repo.append(base)
    await event_log_repo.append(
        EventLog(
            tenant_id="tenant-2",
            user_id="user-2",
            event_type="turn",
            payload={"msg": "second"},
        )
    )

    events = await event_log_repo.get_user_events(
        tenant_id="tenant-2",
        user_id="user-2",
        limit=10,
    )
    assert len(events) >= 2
    # Most recent first
    assert events[0].created_at >= events[1].created_at


@pytest.mark.asyncio
async def test_event_log_replay_events(event_log_repo: EventLogRepository):
    """Replay events in order (asc). Use unique tenant/user so DB state is isolated."""
    import uuid
    unique = uuid.uuid4().hex[:8]
    tenant, user = f"tenant-replay-{unique}", f"user-replay-{unique}"
    for i in range(3):
        e = EventLog(
            tenant_id=tenant,
            user_id=user,
            event_type="turn",
            payload={"seq": i},
        )
        await event_log_repo.append(e)

    replayed = [
        e async for e in event_log_repo.replay_events(tenant_id=tenant, user_id=user)
    ]
    assert len(replayed) == 3
    assert replayed[0].payload["seq"] == 0
    assert replayed[1].payload["seq"] == 1
    assert replayed[2].payload["seq"] == 2


@pytest.mark.asyncio
async def test_event_log_get_by_id_missing(event_log_repo: EventLogRepository):
    """get_by_id returns None for non-existent ID."""
    found = await event_log_repo.get_by_id(uuid4())
    assert found is None
