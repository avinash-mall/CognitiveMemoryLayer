from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api.dashboard import events_routes, fact_routes, memory_routes
from src.core.enums import MemoryStatus, MemoryType

from .dashboard_support import (
    ADMIN_AUTH,
    NeoResultStub,
    NeoSessionStub,
    ResultStub,
    SessionStub,
    make_db,
)


def _request_with_db(db: object, orchestrator: object | None = None) -> SimpleNamespace:
    state = SimpleNamespace(db=db)
    if orchestrator is not None:
        state.orchestrator = orchestrator
    return SimpleNamespace(app=SimpleNamespace(state=state))


@pytest.mark.asyncio
async def test_dashboard_memories_returns_paginated_items_and_escaped_search() -> None:
    memory_id = uuid4()
    record = SimpleNamespace(
        id=memory_id,
        tenant_id="tenant-a",
        agent_id="agent-1",
        type="semantic_fact",
        status="active",
        text="100%_done memory",
        key="pref:food",
        namespace="ns-1",
        context_tags=["conversation"],
        confidence=0.8,
        importance=0.6,
        access_count=3,
        decay_rate=0.02,
        labile=True,
        version=2,
        timestamp=datetime.now(UTC),
        written_at=datetime.now(UTC),
    )
    db, session = make_db(
        pg_results=[ResultStub(scalar=3), ResultStub(scalar_rows=[record])],
    )

    result = await memory_routes.dashboard_memories(
        page=2,
        per_page=2,
        type="semantic_fact",
        status="active",
        search="100%_done",
        tenant_id="tenant-a",
        source_session_id="sess-1",
        sort_by="written_at",
        order="asc",
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.total == 3
    assert result.page == 2
    assert result.total_pages == 2
    assert result.items[0].id == memory_id
    assert "ESCAPE" in str(session.execute_calls[1][0][0])


@pytest.mark.asyncio
async def test_dashboard_memories_returns_http_500_on_error() -> None:
    db, _ = make_db(session=SessionStub([RuntimeError("memory boom")]))

    with pytest.raises(HTTPException, match="memory boom"):
        await memory_routes.dashboard_memories(
            page=1,
            per_page=25,
            type=None,
            status=None,
            search=None,
            tenant_id=None,
            source_session_id=None,
            sort_by="timestamp",
            order="desc",
            auth=ADMIN_AUTH,
            db=db,
        )


@pytest.mark.asyncio
async def test_dashboard_memory_detail_maps_meta_and_related_events() -> None:
    memory_id = uuid4()
    event_id = uuid4()
    record = SimpleNamespace(
        id=memory_id,
        tenant_id="tenant-a",
        agent_id="agent-1",
        type="semantic_fact",
        status="active",
        text="Detailed memory",
        key="pref:food",
        namespace="ns-1",
        context_tags=["conversation"],
        source_session_id="sess-1",
        entities=[{"entity": "Paris"}],
        relations=[{"predicate": "lives_in"}],
        meta={"source": "write"},
        confidence=0.9,
        importance=0.8,
        access_count=4,
        last_accessed_at=datetime.now(UTC),
        decay_rate=0.01,
        labile=False,
        provenance={"origin": "api"},
        version=3,
        supersedes_id=None,
        content_hash="hash",
        timestamp=datetime.now(UTC),
        written_at=datetime.now(UTC),
        valid_from=datetime.now(UTC),
        valid_to=None,
    )
    event = SimpleNamespace(
        id=event_id,
        event_type="memory_op",
        operation="write",
        created_at=datetime.now(UTC),
        payload={"source": "api"},
    )
    db, _ = make_db(
        pg_results=[ResultStub(one_or_none=record), ResultStub(scalar_rows=[event])],
    )

    result = await memory_routes.dashboard_memory_detail(
        memory_id=memory_id,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.id == memory_id
    assert result.metadata == {"source": "write"}
    assert result.related_events[0]["id"] == str(event_id)


@pytest.mark.asyncio
async def test_dashboard_memory_detail_404_when_missing() -> None:
    db, _ = make_db(pg_results=[ResultStub(one_or_none=None)])

    with pytest.raises(HTTPException, match="Memory not found"):
        await memory_routes.dashboard_memory_detail(
            memory_id=uuid4(),
            auth=ADMIN_AUTH,
            db=db,
        )


@pytest.mark.asyncio
async def test_dashboard_memory_lineage_maps_versions_entities_and_jobs() -> None:
    memory_id = uuid4()
    child_id = uuid4()
    event_id = uuid4()
    fact_id = uuid4()
    job_id = uuid4()
    now = datetime.now(UTC)
    record = SimpleNamespace(
        id=memory_id,
        tenant_id="tenant-a",
        agent_id="agent-1",
        type="semantic_fact",
        status="active",
        text="Detailed memory",
        key="pref:food",
        namespace="ns-1",
        context_tags=["conversation"],
        source_session_id="sess-1",
        entities=[{"normalized": "Paris"}],
        relations=[{"predicate": "lives_in"}],
        meta={"source": "write", "consolidated": True},
        confidence=0.9,
        importance=0.8,
        access_count=4,
        last_accessed_at=now,
        decay_rate=0.01,
        labile=True,
        provenance={"origin": "api"},
        version=3,
        supersedes_id=None,
        content_hash="hash",
        timestamp=now,
        written_at=now,
        valid_from=now,
        valid_to=now,
    )
    child = SimpleNamespace(
        id=child_id,
        text="Updated memory",
        type="semantic_fact",
        status="active",
        version=4,
        key="pref:food",
        confidence=0.95,
        importance=0.7,
        timestamp=now,
        written_at=now,
        supersedes_id=memory_id,
    )
    fact = SimpleNamespace(
        id=fact_id,
        tenant_id="tenant-a",
        category="identity",
        key="pref:food",
        value="pizza",
        confidence=0.88,
        evidence_count=1,
        is_current=True,
        version=2,
        created_at=now,
        updated_at=now,
    )
    job = SimpleNamespace(
        id=job_id,
        job_type="forget",
        status="completed",
        started_at=now,
        completed_at=now,
        result={"memory_id": str(memory_id), "key": "pref:food"},
    )
    event = SimpleNamespace(
        id=event_id,
        event_type="memory_op",
        operation="write",
        created_at=now,
        payload={"source": "api"},
    )
    db, _ = make_db(
        pg_results=[
            ResultStub(one_or_none=record),
            ResultStub(scalar_rows=[event]),
            ResultStub(one_or_none=record),
            ResultStub(one_or_none=None),
            ResultStub(scalar_rows=[record, child]),
            ResultStub(scalar_rows=[fact]),
            ResultStub(scalar_rows=[job]),
        ],
        neo_session=NeoSessionStub(
            [
                NeoResultStub(
                    items=[
                        {
                            "entity": "Paris",
                            "entity_type": "city",
                            "tid": "tenant-a",
                            "sid": "scope-1",
                        }
                    ]
                )
            ]
        ),
    )

    result = await memory_routes.dashboard_memory_lineage(
        memory_id=memory_id,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.memory.id == memory_id
    assert [item.id for item in result.same_key_versions] == [memory_id, child_id]
    assert result.evidence_facts[0].id == str(fact_id)
    assert result.related_entities[0].entity == "Paris"
    assert result.related_jobs[0].id == job_id
    assert "active" in result.lifecycle_flags
    assert "labile" in result.lifecycle_flags
    assert "consolidated" in result.lifecycle_flags
    assert "temporal_window_closed" in result.lifecycle_flags


class _RowCountResult:
    def __init__(self, rowcount: int) -> None:
        self.rowcount = rowcount


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("action", "expected_status"),
    [
        ("delete", "deleted"),
        ("archive", "archived"),
        ("silence", "silent"),
    ],
)
async def test_dashboard_bulk_action_maps_statuses(
    action: str,
    expected_status: str,
) -> None:
    session = SessionStub([_RowCountResult(2)])
    db, _ = make_db(session=session)
    memory_ids = [uuid4(), uuid4()]

    result = await memory_routes.dashboard_bulk_action(
        body=SimpleNamespace(memory_ids=memory_ids, action=action),
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result == {"success": True, "affected": 2, "action": action}
    assert session.commits == 1
    stmt = session.execute_calls[0][0][0]
    assert stmt.compile().params["status"] == expected_status


@pytest.mark.asyncio
async def test_dashboard_bulk_action_rejects_unknown_actions() -> None:
    db, _ = make_db()

    with pytest.raises(HTTPException, match="Unknown action: merge"):
        await memory_routes.dashboard_bulk_action(
            body=SimpleNamespace(memory_ids=[uuid4()], action="merge"),
            auth=ADMIN_AUTH,
            db=db,
        )


@pytest.mark.asyncio
async def test_dashboard_bulk_action_wraps_database_errors() -> None:
    db, _ = make_db(session=SessionStub([RuntimeError("bulk boom")]))

    with pytest.raises(HTTPException, match="bulk boom"):
        await memory_routes.dashboard_bulk_action(
            body=SimpleNamespace(memory_ids=[uuid4()], action="delete"),
            auth=ADMIN_AUTH,
            db=db,
        )


@pytest.mark.asyncio
async def test_dashboard_export_memories_returns_json_attachment() -> None:
    memory_id = uuid4()
    record = SimpleNamespace(
        id=memory_id,
        tenant_id="tenant-a",
        type="semantic_fact",
        status="active",
        text="Exported memory",
        key="pref:food",
        confidence=0.8,
        importance=0.6,
        access_count=5,
        timestamp=datetime.now(UTC),
    )
    db, _ = make_db(pg_results=[ResultStub(scalar_rows=[record])])

    response = await memory_routes.dashboard_export_memories(
        tenant_id="tenant-a",
        auth=ADMIN_AUTH,
        db=db,
    )

    assert response.headers["Content-Disposition"] == "attachment; filename=memories_export.json"
    payload = json.loads(response.body)
    assert payload[0]["id"] == str(memory_id)
    assert payload[0]["tenant_id"] == "tenant-a"


@pytest.mark.asyncio
async def test_dashboard_facts_uses_auth_tenant_fallback_and_filters() -> None:
    fact = SimpleNamespace(
        id=uuid4(),
        tenant_id="tenant-a",
        category="identity",
        key="user:name",
        value="Alice",
        confidence=0.9,
        evidence_count=3,
        is_current=True,
        version=1,
        created_at=datetime.now(UTC),
        updated_at=datetime.now(UTC),
    )
    db, _ = make_db(pg_results=[ResultStub(scalar=1), ResultStub(scalar_rows=[fact])])

    result = await fact_routes.dashboard_facts(
        request=_request_with_db(db),
        auth=ADMIN_AUTH,
        tenant_id=None,
        category="identity",
        current_only=True,
        limit=10,
        offset=5,
    )

    assert result.total == 1
    assert result.items[0].tenant_id == "tenant-a"
    assert result.items[0].category == "identity"


@pytest.mark.asyncio
async def test_dashboard_facts_wraps_errors() -> None:
    db, _ = make_db(session=SessionStub([RuntimeError("facts boom")]))

    with pytest.raises(HTTPException, match="Failed to list facts: facts boom"):
        await fact_routes.dashboard_facts(
            request=_request_with_db(db),
            auth=ADMIN_AUTH,
        )


@pytest.mark.asyncio
async def test_dashboard_fact_detail_returns_lineage_and_supersession() -> None:
    now = datetime.now(UTC)
    fact = SimpleNamespace(
        id="fact-1",
        tenant_id="tenant-a",
        category="identity",
        key="user:name",
        subject="user",
        predicate="name",
        value="Alice",
        value_type="string",
        context_tags=["profile"],
        confidence=0.95,
        evidence_count=2,
        evidence_ids=["mem-1", "mem-2"],
        valid_from=now,
        valid_to=None,
        is_current=True,
        created_at=now,
        updated_at=now,
        version=2,
        supersedes_id="fact-0",
    )
    db, _ = make_db(pg_results=[ResultStub(one_or_none=fact)])
    fact_store = SimpleNamespace(
        get_fact_lineage=AsyncMock(return_value=[{"id": "fact-0"}]),
        get_superseded_chain=AsyncMock(return_value=[{"id": "fact-2"}]),
    )
    orchestrator = SimpleNamespace(neocortical=SimpleNamespace(facts=fact_store))

    result = await fact_routes.dashboard_fact_detail(
        request=_request_with_db(db, orchestrator=orchestrator),
        fact_id="fact-1",
        auth=ADMIN_AUTH,
    )

    assert result.id == "fact-1"
    assert result.lineage == [{"id": "fact-0"}]
    assert result.superseded_by == [{"id": "fact-2"}]
    fact_store.get_fact_lineage.assert_awaited_once_with("tenant-a", fact_id="fact-1")
    fact_store.get_superseded_chain.assert_awaited_once_with("tenant-a", "fact-1")


@pytest.mark.asyncio
async def test_dashboard_fact_evidence_returns_records_and_missing_ids() -> None:
    memory_id = uuid4()
    missing_memory_id = uuid4()
    now = datetime.now(UTC)
    fact = SimpleNamespace(
        id="fact-1",
        evidence_ids=[str(memory_id), "not-a-uuid", str(missing_memory_id)],
    )
    evidence_record = SimpleNamespace(
        id=memory_id,
        text="Supporting memory",
        type=MemoryType.SEMANTIC_FACT,
        status=MemoryStatus.ACTIVE,
        confidence=0.88,
        importance=0.67,
        source_session_id="sess-1",
        timestamp=now,
        written_at=now,
        supersedes_id=None,
        metadata={"origin": "api"},
    )
    db, _ = make_db(pg_results=[ResultStub(one_or_none=fact)])
    get_by_ids_batch = AsyncMock(return_value=[evidence_record])
    orchestrator = SimpleNamespace(
        hippocampal=SimpleNamespace(store=SimpleNamespace(get_by_ids_batch=get_by_ids_batch))
    )

    result = await fact_routes.dashboard_fact_evidence(
        request=_request_with_db(db, orchestrator=orchestrator),
        fact_id="fact-1",
        auth=ADMIN_AUTH,
    )

    assert result.fact_id == "fact-1"
    assert result.evidence[0].id == memory_id
    assert result.evidence[0].type == MemoryType.SEMANTIC_FACT.value
    assert set(result.missing_evidence_ids) == {"not-a-uuid", str(missing_memory_id)}
    fetched_ids = get_by_ids_batch.await_args.args[0]
    assert set(fetched_ids) == {memory_id, missing_memory_id}


@pytest.mark.asyncio
async def test_dashboard_invalidate_fact_handles_success_not_found_and_errors() -> None:
    success_db, success_session = make_db(session=SessionStub([_RowCountResult(1)]))
    ok = await fact_routes.dashboard_invalidate_fact(
        request=_request_with_db(success_db),
        fact_id="fact-1",
        auth=ADMIN_AUTH,
    )
    assert ok == {"success": True, "message": "Fact fact-1 invalidated"}
    assert success_session.commits == 1

    missing_db, _ = make_db(session=SessionStub([_RowCountResult(0)]))
    with pytest.raises(HTTPException, match="Fact fact-2 not found"):
        await fact_routes.dashboard_invalidate_fact(
            request=_request_with_db(missing_db),
            fact_id="fact-2",
            auth=ADMIN_AUTH,
        )

    error_db, _ = make_db(session=SessionStub([RuntimeError("invalidate boom")]))
    with pytest.raises(HTTPException, match="Failed to invalidate fact: invalidate boom"):
        await fact_routes.dashboard_invalidate_fact(
            request=_request_with_db(error_db),
            fact_id="fact-3",
            auth=ADMIN_AUTH,
        )


@pytest.mark.asyncio
async def test_dashboard_events_returns_paginated_history() -> None:
    event_id = uuid4()
    memory_id = uuid4()
    event = SimpleNamespace(
        id=event_id,
        tenant_id="tenant-a",
        scope_id="scope-1",
        agent_id="agent-1",
        event_type="memory_op",
        operation="write",
        payload={"source": "api"},
        memory_ids=[memory_id],
        parent_event_id=None,
        created_at=datetime.now(UTC),
    )
    db, session = make_db(pg_results=[ResultStub(scalar=3), ResultStub(scalar_rows=[event])])

    result = await events_routes.dashboard_events(
        page=2,
        per_page=2,
        event_type="memory_op",
        operation="write",
        tenant_id="tenant-a",
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.total == 3
    assert result.total_pages == 2
    assert result.items[0].id == event_id
    assert "ORDER BY" in str(session.execute_calls[1][0][0])


@pytest.mark.asyncio
async def test_dashboard_events_wraps_errors() -> None:
    db, _ = make_db(session=SessionStub([RuntimeError("events boom")]))

    with pytest.raises(HTTPException, match="events boom"):
        await events_routes.dashboard_events(auth=ADMIN_AUTH, db=db)
