from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest
from fastapi import HTTPException

from src.api.dashboard import jobs_routes

from .dashboard_support import ADMIN_AUTH, RedisStub, ResultStub, SessionStub, make_db


def _retrieved_memory(
    *, memory_id: str | None = None, text: str = "memory text"
) -> SimpleNamespace:
    record = SimpleNamespace(
        id=memory_id or str(uuid4()),
        text=text,
        type="semantic_fact",
        confidence=0.9,
        timestamp=datetime.now(UTC),
        metadata={"source": "test"},
        supersedes_id=None,
    )
    return SimpleNamespace(record=record, relevance_score=0.8, retrieval_source="vector")


def _request(*, orchestrator: object, db: object | None = None) -> SimpleNamespace:
    return SimpleNamespace(
        app=SimpleNamespace(state=SimpleNamespace(orchestrator=orchestrator, db=db))
    )


@pytest.mark.asyncio
async def test_dashboard_labile_merges_database_and_redis_counts() -> None:
    redis = RedisStub(
        scan_results=[(0, ["labile:scope:tenant-a:scope-1", "labile:scope:tenant-c:scope-2"])],
        lrange_values={
            "labile:scope:tenant-a:scope-1": ["sess-1", "sess-2"],
            "labile:scope:tenant-c:scope-2": ["sess-3"],
        },
        values={
            "labile:session:sess-1": json.dumps({"memories": {"m1": {}, "m2": {}}}),
            "labile:session:sess-2": "not-json",
            "labile:session:sess-3": json.dumps({"memories": {"m3": {}}}),
        },
    )
    db, _ = make_db(
        pg_results=[ResultStub(all_rows=[("tenant-a", 2), ("tenant-b", 1)])],
        redis=redis,
    )

    result = await jobs_routes.dashboard_labile(tenant_id=None, auth=ADMIN_AUTH, db=db)

    by_tenant = {item.tenant_id: item for item in result.tenants}
    assert [item.tenant_id for item in result.tenants] == ["tenant-a", "tenant-b", "tenant-c"]
    assert by_tenant["tenant-a"].db_labile_count == 2
    assert by_tenant["tenant-a"].redis_scope_count == 1
    assert by_tenant["tenant-a"].redis_session_count == 2
    assert by_tenant["tenant-a"].redis_memory_count == 2
    assert by_tenant["tenant-c"].redis_memory_count == 1
    assert result.total_db_labile == 3
    assert result.total_redis_scopes == 2
    assert result.total_redis_sessions == 3
    assert result.total_redis_memories == 3


@pytest.mark.asyncio
async def test_dashboard_labile_returns_http_500_on_error() -> None:
    db, _ = make_db(session=SessionStub([RuntimeError("labile boom")]))

    with pytest.raises(HTTPException, match="labile boom"):
        await jobs_routes.dashboard_labile(tenant_id=None, auth=ADMIN_AUTH, db=db)


@pytest.mark.asyncio
async def test_dashboard_retrieval_formats_results_and_llm_context(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = SimpleNamespace(all_memories=[_retrieved_memory()])
    orchestrator = SimpleNamespace(read=AsyncMock(return_value=packet))

    class _Builder:
        def to_llm_context(self, packet: object, max_tokens: int) -> str:
            assert max_tokens == 2000
            assert packet is not None
            return "LLM context"

    monkeypatch.setattr("src.retrieval.packet_builder.MemoryPacketBuilder", _Builder)
    body = jobs_routes.DashboardRetrievalRequest(
        tenant_id="tenant-a",
        query="pizza",
        max_results=5,
        format="llm_context",
    )

    result = await jobs_routes.dashboard_retrieval(
        body=body,
        request=_request(orchestrator=orchestrator),
        auth=ADMIN_AUTH,
    )

    assert result.query == "pizza"
    assert result.total_count == 1
    assert result.llm_context == "LLM context"
    assert str(result.results[0].id)
    assert result.results[0].retrieval_source == "vector"


@pytest.mark.asyncio
async def test_dashboard_retrieval_ignores_llm_context_builder_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    packet = SimpleNamespace(all_memories=[_retrieved_memory()])
    orchestrator = SimpleNamespace(read=AsyncMock(return_value=packet))

    class _Builder:
        def to_llm_context(self, packet: object, max_tokens: int) -> str:
            raise RuntimeError("builder failed")

    monkeypatch.setattr("src.retrieval.packet_builder.MemoryPacketBuilder", _Builder)
    body = jobs_routes.DashboardRetrievalRequest(
        tenant_id="tenant-a",
        query="query",
        max_results=3,
        format="llm_context",
    )

    result = await jobs_routes.dashboard_retrieval(
        body=body,
        request=_request(orchestrator=orchestrator),
        auth=ADMIN_AUTH,
    )

    assert result.total_count == 1
    assert result.llm_context is None


@pytest.mark.asyncio
async def test_dashboard_jobs_returns_items_filters_and_duration() -> None:
    started = datetime(2025, 1, 1, 10, 0, 0)
    completed = started + timedelta(seconds=12.34)
    job = SimpleNamespace(
        id=uuid4(),
        job_type="consolidate",
        tenant_id="tenant-a",
        user_id="user-1",
        dry_run=False,
        status="completed",
        result={"status": "completed"},
        error=None,
        started_at=started,
        completed_at=completed,
    )
    db, session = make_db(
        pg_results=[ResultStub(scalar=1), ResultStub(scalar_rows=[job])],
    )

    result = await jobs_routes.dashboard_jobs(
        tenant_id="tenant-a",
        job_type="consolidate",
        limit=10,
        auth=ADMIN_AUTH,
        db=db,
    )

    assert result.total == 1
    assert len(result.items) == 1
    assert result.items[0].duration_seconds == 12.34
    count_query = str(session.execute_calls[0][0][0])
    rows_query = str(session.execute_calls[1][0][0])
    assert "tenant_id" in count_query
    assert "job_type" in count_query
    assert "ORDER BY" in rows_query


class _FailFirstCommitSession(SessionStub):
    async def commit(self) -> None:
        self.commits += 1
        if self.commits == 1:
            raise RuntimeError("tracking failed")


@pytest.mark.asyncio
async def test_dashboard_consolidate_updates_tracking_and_returns_success() -> None:
    report = SimpleNamespace(
        episodes_sampled=5,
        clusters_formed=2,
        gists_extracted=3,
        elapsed_seconds=1.5,
    )
    orchestrator = SimpleNamespace(
        consolidation=SimpleNamespace(consolidate=AsyncMock(return_value=report))
    )
    db, session = make_db(session=SessionStub([ResultStub()]))

    result = await jobs_routes.dashboard_consolidate(
        body=jobs_routes.DashboardConsolidateRequest(tenant_id="tenant-a", user_id="user-1"),
        request=_request(orchestrator=orchestrator, db=db),
        auth=ADMIN_AUTH,
    )

    assert result["status"] == "completed"
    assert result["episodes_sampled"] == 5
    assert len(session.added) == 1
    assert session.added[0].job_type == "consolidate"
    assert session.commits == 2
    assert len(session.execute_calls) == 1


@pytest.mark.asyncio
async def test_dashboard_consolidate_marks_failed_jobs() -> None:
    orchestrator = SimpleNamespace(
        consolidation=SimpleNamespace(consolidate=AsyncMock(side_effect=RuntimeError("nope")))
    )
    db, session = make_db(session=SessionStub([ResultStub()]))

    with pytest.raises(HTTPException, match="Consolidation failed: nope"):
        await jobs_routes.dashboard_consolidate(
            body=jobs_routes.DashboardConsolidateRequest(tenant_id="tenant-a", user_id="user-1"),
            request=_request(orchestrator=orchestrator, db=db),
            auth=ADMIN_AUTH,
        )

    assert len(session.execute_calls) == 1


@pytest.mark.asyncio
async def test_dashboard_forget_ignores_initial_tracking_failure_and_returns_success() -> None:
    report = SimpleNamespace(
        memories_scanned=10,
        memories_scored=8,
        duplicates_found=1,
        duplicates_resolved=1,
        elapsed_seconds=2.5,
        result=SimpleNamespace(
            operations_planned=4,
            operations_applied=3,
            deleted=1,
            decayed=1,
            silenced=1,
            compressed=0,
            errors=["warn"],
        ),
    )
    orchestrator = SimpleNamespace(
        forgetting=SimpleNamespace(run_forgetting=AsyncMock(return_value=report))
    )
    db, session = make_db(session=_FailFirstCommitSession([ResultStub()]))

    result = await jobs_routes.dashboard_forget(
        body=jobs_routes.DashboardForgetRequest(
            tenant_id="tenant-a",
            user_id="user-1",
            dry_run=True,
            max_memories=25,
        ),
        request=_request(orchestrator=orchestrator, db=db),
        auth=ADMIN_AUTH,
    )

    assert result["status"] == "completed"
    assert result["dry_run"] is True
    assert result["operations_applied"] == 3
    assert session.commits == 2
    assert len(session.execute_calls) == 1


@pytest.mark.asyncio
async def test_dashboard_forget_marks_failed_jobs() -> None:
    orchestrator = SimpleNamespace(
        forgetting=SimpleNamespace(
            run_forgetting=AsyncMock(side_effect=RuntimeError("forget boom"))
        )
    )
    db, session = make_db(session=SessionStub([ResultStub()]))

    with pytest.raises(HTTPException, match="Forgetting failed: forget boom"):
        await jobs_routes.dashboard_forget(
            body=jobs_routes.DashboardForgetRequest(tenant_id="tenant-a", dry_run=False),
            request=_request(orchestrator=orchestrator, db=db),
            auth=ADMIN_AUTH,
        )

    assert len(session.execute_calls) == 1


@pytest.mark.asyncio
async def test_dashboard_reconsolidate_returns_sessions_released() -> None:
    tracker = SimpleNamespace(release_all_for_tenant=AsyncMock(return_value=4))
    orchestrator = SimpleNamespace(reconsolidation=SimpleNamespace(labile_tracker=tracker))
    db, session = make_db(session=SessionStub([ResultStub()]))

    result = await jobs_routes.dashboard_reconsolidate(
        body=jobs_routes.DashboardReconsolidateRequest(tenant_id="tenant-a", user_id="user-1"),
        request=_request(orchestrator=orchestrator, db=db),
        auth=ADMIN_AUTH,
    )

    assert result == {"status": "completed", "tenant_id": "tenant-a", "sessions_released": 4}
    assert len(session.execute_calls) == 1


@pytest.mark.asyncio
async def test_dashboard_reconsolidate_marks_failed_jobs() -> None:
    tracker = SimpleNamespace(
        release_all_for_tenant=AsyncMock(side_effect=RuntimeError("release boom"))
    )
    orchestrator = SimpleNamespace(reconsolidation=SimpleNamespace(labile_tracker=tracker))
    db, session = make_db(session=SessionStub([ResultStub()]))

    with pytest.raises(HTTPException, match="Reconsolidation failed: release boom"):
        await jobs_routes.dashboard_reconsolidate(
            body=jobs_routes.DashboardReconsolidateRequest(tenant_id="tenant-a"),
            request=_request(orchestrator=orchestrator, db=db),
            auth=ADMIN_AUTH,
        )

    assert len(session.execute_calls) == 1


@pytest.mark.asyncio
async def test_dashboard_database_reset_requires_alembic_ini(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(jobs_routes, "_project_root", lambda: tmp_path)

    with pytest.raises(HTTPException, match="alembic.ini not found"):
        await jobs_routes.dashboard_database_reset(auth=ADMIN_AUTH)


@pytest.mark.asyncio
async def test_dashboard_database_reset_runs_downgrade_and_upgrade(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "alembic.ini").write_text("[alembic]\n", encoding="utf-8")
    calls: list[tuple[str, object]] = []

    class _Config:
        def __init__(self, path: str) -> None:
            self.path = path

    def _downgrade(config: object, target: str) -> None:
        calls.append((target, config))

    def _upgrade(config: object, target: str) -> None:
        calls.append((target, config))

    monkeypatch.setattr(jobs_routes, "_project_root", lambda: tmp_path)
    monkeypatch.setattr("alembic.config.Config", _Config)
    monkeypatch.setattr("alembic.command.downgrade", _downgrade)
    monkeypatch.setattr("alembic.command.upgrade", _upgrade)

    result = await jobs_routes.dashboard_database_reset(auth=ADMIN_AUTH)

    assert result["success"] is True
    assert [target for target, _ in calls] == ["base", "head"]
    assert str(tmp_path) in jobs_routes.sys.path


@pytest.mark.asyncio
async def test_dashboard_database_reset_wraps_exceptions(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    (tmp_path / "alembic.ini").write_text("[alembic]\n", encoding="utf-8")
    monkeypatch.setattr(jobs_routes, "_project_root", lambda: tmp_path)
    monkeypatch.setattr(
        "alembic.command.downgrade",
        lambda config, target: (_ for _ in ()).throw(RuntimeError("reset failed")),
    )

    with pytest.raises(HTTPException, match="Database reset failed."):
        await jobs_routes.dashboard_database_reset(auth=ADMIN_AUTH)
