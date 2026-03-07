from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from src.api.dashboard import overview_routes

from .dashboard_support import (
    ADMIN_AUTH,
    NeoResultStub,
    NeoSessionStub,
    RedisStub,
    ResultStub,
    make_db,
)


@pytest.mark.asyncio
async def test_dashboard_overview_aggregates_counts() -> None:
    db, _ = make_db(
        pg_results=[
            ResultStub(scalar=10),
            ResultStub(all_rows=[("active", 7), ("deleted", 3)]),
            ResultStub(all_rows=[("episodic_event", 6), ("constraint", 4)]),
            ResultStub(one_or_none=(0.8, 0.6, 2.5, 0.01)),
            ResultStub(scalar=2),
            ResultStub(one_or_none=(datetime(2025, 1, 1), datetime(2025, 2, 1))),
            ResultStub(scalar=100),
            ResultStub(scalar=5),
            ResultStub(scalar=4),
            ResultStub(all_rows=[("identity", 5)]),
            ResultStub(one_or_none=(0.9, 2.0)),
            ResultStub(scalar=8),
            ResultStub(all_rows=[("memory.write", 5)]),
            ResultStub(all_rows=[("add", 5)]),
        ]
    )

    result = await overview_routes.dashboard_overview(
        tenant_id="tenant-a",
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.total_memories == 10
    assert result.active_memories == 7
    assert result.deleted_memories == 3
    assert result.by_type == {"episodic_event": 6, "constraint": 4}
    assert result.total_semantic_facts == 5
    assert result.current_semantic_facts == 4
    assert result.total_events == 8
    assert result.events_by_operation == {"add": 5}


@pytest.mark.asyncio
async def test_dashboard_overview_raises_http_500_on_error() -> None:
    db, _ = make_db(pg_results=[RuntimeError("pg failed")])

    with pytest.raises(HTTPException, match="pg failed"):
        await overview_routes.dashboard_overview(tenant_id=None, auth=ADMIN_AUTH, db=db)


@pytest.mark.asyncio
async def test_dashboard_timeline_returns_points_and_total() -> None:
    db, _ = make_db(
        pg_results=[
            ResultStub(
                all_rows=[
                    (datetime(2025, 1, 1), 2),
                    (datetime(2025, 1, 2), 5),
                ]
            )
        ]
    )

    result = await overview_routes.dashboard_timeline(
        days=7,
        tenant_id=None,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert [point.date for point in result.points] == ["2025-01-01", "2025-01-02"]
    assert result.total == 7


@pytest.mark.asyncio
async def test_dashboard_components_reports_ok_unknown_and_error() -> None:
    neo_session = NeoSessionStub(
        [
            NeoResultStub(single={"cnt": 4}),
            NeoResultStub(single={"cnt": 5}),
        ]
    )
    redis = RedisStub(info_value={"used_memory": 2 * 1024 * 1024}, db_size=3)
    db, _ = make_db(
        pg_results=[
            ResultStub(scalar=1),
            ResultStub(scalar=9),
            ResultStub(scalar=4),
            ResultStub(scalar=7),
        ],
        neo_session=neo_session,
        redis=redis,
    )

    result = await overview_routes.dashboard_components(auth=ADMIN_AUTH, db=db)

    statuses = {component.name: component.status for component in result.components}
    assert statuses == {"PostgreSQL": "ok", "Neo4j": "ok", "Redis": "ok"}
    postgres = next(component for component in result.components if component.name == "PostgreSQL")
    assert postgres.details["memory_records"] == 9

    unknown_db = SimpleNamespace(
        pg_session=db.pg_session,
        neo4j_driver=None,
        redis=None,
    )
    unknown_result = await overview_routes.dashboard_components(auth=ADMIN_AUTH, db=unknown_db)
    statuses = {component.name: component.status for component in unknown_result.components}
    assert statuses["Neo4j"] == "unknown"
    assert statuses["Redis"] == "unknown"

    error_db = SimpleNamespace(
        pg_session=lambda: (_ for _ in ()).throw(RuntimeError("pg boom")),
        neo4j_driver=object(),
        neo4j_session=lambda: (_ for _ in ()).throw(RuntimeError("neo boom")),
        redis=SimpleNamespace(ping=lambda: (_ for _ in ()).throw(RuntimeError("redis boom"))),
    )
    error_result = await overview_routes.dashboard_components(auth=ADMIN_AUTH, db=error_db)
    statuses = {component.name: component.status for component in error_result.components}
    assert statuses == {"PostgreSQL": "error", "Neo4j": "error", "Redis": "error"}


@pytest.mark.asyncio
async def test_dashboard_tenants_merges_sets_from_memories_facts_and_events() -> None:
    db, _ = make_db(
        pg_results=[
            ResultStub(all_rows=[("tenant-a", 5, 4, datetime(2025, 1, 1))]),
            ResultStub(all_rows=[("tenant-b", 3)]),
            ResultStub(all_rows=[("tenant-c", 7, datetime(2025, 1, 2))]),
        ]
    )

    result = await overview_routes.dashboard_tenants(auth=ADMIN_AUTH, db=db)

    tenant_ids = [tenant.tenant_id for tenant in result.tenants]
    assert tenant_ids == ["tenant-a", "tenant-b", "tenant-c"]


@pytest.mark.asyncio
async def test_dashboard_sessions_parses_redis_and_db_counts() -> None:
    created = datetime.now(UTC).replace(microsecond=0)
    expires = created + timedelta(hours=1)
    redis = RedisStub(
        scan_results=[(0, ["session:sess-1"])],
        values={
            "session:sess-1": json.dumps(
                {
                    "tenant_id": "tenant-a",
                    "created_at": created.isoformat(),
                    "expires_at": expires.isoformat(),
                    "name": "chat",
                }
            )
        },
        ttl_values={"session:sess-1": 120},
    )
    db, _ = make_db(
        pg_results=[ResultStub(all_rows=[("sess-1", 2), ("sess-2", 1)])],
        redis=redis,
    )

    result = await overview_routes.dashboard_sessions(
        tenant_id="tenant-a",
        auth=ADMIN_AUTH,
        db=db,
    )

    assert [session.session_id for session in result.sessions] == ["sess-1", "sess-2"]
    assert result.sessions[0].ttl_seconds == 120
    assert result.sessions[0].memory_count == 2
    assert result.total_active == 1
    assert result.total_memories_with_session == 3


@pytest.mark.asyncio
async def test_dashboard_ratelimits_parses_apikey_ip_and_other_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    redis = RedisStub(
        scan_results=[(0, ["ratelimit:apikey:abcdef123456", "ratelimit:ip:1.2.3.4", "ratelimit:misc"])],
        values={
            "ratelimit:apikey:abcdef123456": "30",
            "ratelimit:ip:1.2.3.4": "5",
            "ratelimit:misc": "1",
        },
        ttl_values={
            "ratelimit:apikey:abcdef123456": 10,
            "ratelimit:ip:1.2.3.4": 20,
            "ratelimit:misc": 30,
        },
    )
    db, _ = make_db(redis=redis)
    monkeypatch.setattr(
        overview_routes,
        "get_settings",
        lambda: SimpleNamespace(auth=SimpleNamespace(rate_limit_requests_per_minute=60)),
    )

    result = await overview_routes.dashboard_ratelimits(auth=ADMIN_AUTH, db=db)

    assert result.configured_rpm == 60
    assert [entry.key_type for entry in result.entries] == ["apikey", "ip", "other"]
    assert result.entries[0].identifier == "abcdef12..."
    assert result.entries[0].utilization_pct == 50.0


@pytest.mark.asyncio
async def test_dashboard_request_stats_reads_hourly_counters() -> None:
    now = datetime.now(UTC)
    hour_a = (now - timedelta(hours=1)).strftime("%Y-%m-%d-%H")
    hour_b = now.strftime("%Y-%m-%d-%H")
    redis = RedisStub(
        values={
            f"{overview_routes._REQUEST_COUNT_PREFIX}{hour_a}": "2",
            f"{overview_routes._REQUEST_COUNT_PREFIX}{hour_b}": "5",
        }
    )
    db, _ = make_db(redis=redis)

    result = await overview_routes.dashboard_request_stats(
        hours=2,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert len(result.points) == 2
    assert result.total_last_24h == 7
