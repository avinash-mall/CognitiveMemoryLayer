from __future__ import annotations

import json
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api.dashboard import events_routes, fact_routes, memory_routes

from .dashboard_support import ADMIN_AUTH, ResultStub, SessionStub, make_db


def _request_with_db(db: object) -> SimpleNamespace:
    return SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(db=db)))


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
