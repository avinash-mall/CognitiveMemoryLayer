"""Dashboard job routes: labile, retrieval test, jobs, consolidate, forget, reconsolidate, database reset."""

import json
import sys
import uuid as uuid_mod
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, update

from ...storage.connection import DatabaseManager
from ...storage.models import DashboardJobModel, MemoryRecordModel
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    DashboardConsolidateRequest,
    DashboardForgetRequest,
    DashboardJobItem,
    DashboardJobsResponse,
    DashboardLabileResponse,
    DashboardReconsolidateRequest,
    DashboardRetrievalRequest,
    DashboardRetrievalResponse,
    RetrievalResultItem,
    TenantLabileInfo,
    _get_db,
    logger,
)

router = APIRouter()


def _project_root() -> Path:
    """Project root (directory containing alembic.ini and migrations/)."""
    return Path(__file__).resolve().parent.parent.parent


@router.get("/labile", response_model=DashboardLabileResponse)
async def dashboard_labile(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Labile (short-term) memory overview: DB labile counts, Redis scope and session counts. Optionally filter by tenant."""
    try:
        async with db.pg_session() as session:
            q = (
                select(MemoryRecordModel.tenant_id, func.count().label("cnt"))
                .where(MemoryRecordModel.labile.is_(True))
                .group_by(MemoryRecordModel.tenant_id)
            )
            if tenant_id:
                q = q.where(MemoryRecordModel.tenant_id == tenant_id)
            rows = (await session.execute(q)).all()
            db_map = {r[0]: r[1] for r in rows}

        redis_scope_map: dict[str, dict[str, int]] = {}
        if db.redis:
            cursor = 0
            while True:
                match_pattern = f"labile:scope:{tenant_id}:*" if tenant_id else "labile:scope:*"
                cursor, keys = await db.redis.scan(cursor, match=match_pattern, count=200)
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    parts = key_str.removeprefix("labile:scope:").split(":", 1)
                    tid = parts[0] if parts else "unknown"
                    if tid not in redis_scope_map:
                        redis_scope_map[tid] = {"scopes": 0, "sessions": 0, "memories": 0}
                    redis_scope_map[tid]["scopes"] += 1

                    scope_sessions = await db.redis.lrange(key_str, 0, -1)
                    session_count = len(scope_sessions)
                    redis_scope_map[tid]["sessions"] += session_count

                    for sess_key in scope_sessions:
                        sess_key_str = (
                            sess_key.decode() if isinstance(sess_key, bytes) else sess_key
                        )
                        rk = f"labile:session:{sess_key_str}"
                        data = await db.redis.get(rk)
                        if data:
                            try:
                                doc = json.loads(data.decode() if isinstance(data, bytes) else data)
                                redis_scope_map[tid]["memories"] += len(doc.get("memories", {}))
                            except (json.JSONDecodeError, KeyError):
                                pass
                if cursor == 0:
                    break

        all_tenants = sorted(set(db_map.keys()) | set(redis_scope_map.keys()))
        tenants = [
            TenantLabileInfo(
                tenant_id=tid,
                db_labile_count=db_map.get(tid, 0),
                redis_scope_count=redis_scope_map.get(tid, {}).get("scopes", 0),
                redis_session_count=redis_scope_map.get(tid, {}).get("sessions", 0),
                redis_memory_count=redis_scope_map.get(tid, {}).get("memories", 0),
            )
            for tid in all_tenants
        ]

        return DashboardLabileResponse(
            tenants=tenants,
            total_db_labile=sum(db_map.values()),
            total_redis_scopes=sum(v.get("scopes", 0) for v in redis_scope_map.values()),
            total_redis_sessions=sum(v.get("sessions", 0) for v in redis_scope_map.values()),
            total_redis_memories=sum(v.get("memories", 0) for v in redis_scope_map.values()),
        )
    except Exception as e:
        logger.error("dashboard_labile_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/retrieval", response_model=DashboardRetrievalResponse)
async def dashboard_retrieval(
    body: DashboardRetrievalRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Test memory retrieval: same as POST /memory/read but uses dashboard admin auth. For debugging and validation."""
    try:
        orchestrator = request.app.state.orchestrator
        start = datetime.now(UTC)
        packet = await orchestrator.read(
            tenant_id=body.tenant_id,
            query=body.query,
            max_results=body.max_results,
            context_filter=body.context_filter,
            memory_types=body.memory_types,
        )
        elapsed_ms = (datetime.now(UTC) - start).total_seconds() * 1000

        results: list[RetrievalResultItem] = []
        for mem in packet.all_memories:
            results.append(
                RetrievalResultItem(
                    id=mem.record.id,
                    text=mem.record.text,
                    type=mem.record.type,
                    confidence=mem.record.confidence,
                    relevance_score=mem.relevance_score,
                    retrieval_source=mem.retrieval_source,
                    timestamp=mem.record.timestamp,
                    metadata=mem.record.metadata if hasattr(mem.record, "metadata") else {},
                )
            )

        llm_context = None
        if body.format == "llm_context":
            try:
                from ...retrieval.packet_builder import MemoryPacketBuilder

                builder = MemoryPacketBuilder()
                llm_context = builder.to_llm_context(packet, max_tokens=2000)
            except Exception as e:
                logger.warning("dashboard_retrieval_llm_context_failed", error=str(e))

        return DashboardRetrievalResponse(
            query=body.query,
            results=results,
            llm_context=llm_context,
            total_count=len(results),
            elapsed_ms=round(elapsed_ms, 2),
        )
    except Exception as e:
        logger.error("dashboard_retrieval_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/jobs", response_model=DashboardJobsResponse)
async def dashboard_jobs(
    tenant_id: str | None = Query(None),
    job_type: str | None = Query(None),
    limit: int = Query(50, ge=1, le=200),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """List recent consolidation and forgetting job history. Filter by tenant or job type."""
    try:
        async with db.pg_session() as session:
            filters: list = []
            if tenant_id:
                filters.append(DashboardJobModel.tenant_id == tenant_id)
            if job_type:
                filters.append(DashboardJobModel.job_type == job_type)

            count_q = select(func.count()).select_from(DashboardJobModel)
            for f in filters:
                count_q = count_q.where(f)
            total = (await session.execute(count_q)).scalar() or 0

            q = select(DashboardJobModel).order_by(DashboardJobModel.started_at.desc()).limit(limit)
            for f in filters:
                q = q.where(f)
            rows = (await session.execute(q)).scalars().all()

            items = []
            for j in rows:
                duration = None
                if j.started_at and j.completed_at:
                    duration = round((j.completed_at - j.started_at).total_seconds(), 2)
                items.append(
                    DashboardJobItem(
                        id=cast("UUID", j.id),
                        job_type=cast("str", j.job_type),
                        tenant_id=cast("str", j.tenant_id),
                        user_id=cast("str | None", j.user_id),
                        dry_run=cast("bool", j.dry_run or False),
                        status=cast("str", j.status),
                        result=cast("dict[str, Any] | None", j.result),
                        error=cast("str | None", j.error),
                        started_at=cast("datetime | None", j.started_at),
                        completed_at=cast("datetime | None", j.completed_at),
                        duration_seconds=duration,
                    )
                )
            return DashboardJobsResponse(items=items, total=total)
    except Exception as e:
        logger.error("dashboard_jobs_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/consolidate")
async def dashboard_consolidate(
    body: DashboardConsolidateRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Trigger memory consolidation for a tenant. Creates a tracked job. Admin-only."""
    db: DatabaseManager = request.app.state.db
    user_id = body.user_id or body.tenant_id
    job_id = uuid_mod.uuid4()
    now = datetime.now(UTC).replace(tzinfo=None)

    try:
        async with db.pg_session() as session:
            session.add(
                DashboardJobModel(
                    id=job_id,
                    job_type="consolidate",
                    tenant_id=body.tenant_id,
                    user_id=user_id,
                    dry_run=False,
                    status="running",
                    started_at=now,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.debug("dashboard_job_tracking_failed", error=str(exc))

    try:
        orchestrator = request.app.state.orchestrator
        report = await orchestrator.consolidation.consolidate(
            tenant_id=body.tenant_id,
            user_id=user_id,
        )
        result_data = {
            "status": "completed",
            "tenant_id": body.tenant_id,
            "user_id": user_id,
            "episodes_sampled": report.episodes_sampled,
            "clusters_formed": report.clusters_formed,
            "gists_extracted": report.gists_extracted,
            "elapsed_seconds": getattr(report, "elapsed_seconds", None),
        }
        try:
            async with db.pg_session() as session:
                await session.execute(
                    update(DashboardJobModel)
                    .where(DashboardJobModel.id == job_id)
                    .values(
                        status="completed",
                        result=result_data,
                        completed_at=datetime.now(UTC).replace(tzinfo=None),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("dashboard_job_tracking_failed", error=str(exc))
        return result_data
    except Exception as e:
        try:
            async with db.pg_session() as session:
                await session.execute(
                    update(DashboardJobModel)
                    .where(DashboardJobModel.id == job_id)
                    .values(
                        status="failed",
                        error=str(e),
                        completed_at=datetime.now(UTC).replace(tzinfo=None),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("dashboard_job_tracking_failed", error=str(exc))
        logger.error("dashboard_consolidate_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


@router.post("/forget")
async def dashboard_forget(
    body: DashboardForgetRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Trigger active forgetting for a tenant. Creates a tracked job. Admin-only."""
    db: DatabaseManager = request.app.state.db
    user_id = body.user_id or body.tenant_id
    job_id = uuid_mod.uuid4()
    now = datetime.now(UTC).replace(tzinfo=None)

    try:
        async with db.pg_session() as session:
            session.add(
                DashboardJobModel(
                    id=job_id,
                    job_type="forget",
                    tenant_id=body.tenant_id,
                    user_id=user_id,
                    dry_run=body.dry_run,
                    status="running",
                    started_at=now,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.debug("dashboard_job_tracking_failed", error=str(exc))

    try:
        orchestrator = request.app.state.orchestrator
        report = await orchestrator.forgetting.run_forgetting(
            tenant_id=body.tenant_id,
            user_id=user_id,
            max_memories=body.max_memories,
            dry_run=body.dry_run,
        )
        result_data = {
            "status": "completed",
            "tenant_id": body.tenant_id,
            "user_id": user_id,
            "dry_run": body.dry_run,
            "memories_scanned": report.memories_scanned,
            "memories_scored": report.memories_scored,
            "operations_planned": report.result.operations_planned,
            "operations_applied": report.result.operations_applied,
            "deleted": report.result.deleted,
            "decayed": report.result.decayed,
            "silenced": report.result.silenced,
            "compressed": report.result.compressed,
            "duplicates_found": report.duplicates_found,
            "duplicates_resolved": report.duplicates_resolved,
            "elapsed_seconds": report.elapsed_seconds,
            "errors": report.result.errors,
        }
        try:
            async with db.pg_session() as session:
                await session.execute(
                    update(DashboardJobModel)
                    .where(DashboardJobModel.id == job_id)
                    .values(
                        status="completed",
                        result=result_data,
                        completed_at=datetime.now(UTC).replace(tzinfo=None),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("dashboard_job_tracking_failed", error=str(exc))
        return result_data
    except Exception as e:
        try:
            async with db.pg_session() as session:
                await session.execute(
                    update(DashboardJobModel)
                    .where(DashboardJobModel.id == job_id)
                    .values(
                        status="failed",
                        error=str(e),
                        completed_at=datetime.now(UTC).replace(tzinfo=None),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("dashboard_job_tracking_failed", error=str(exc))
        logger.error("dashboard_forget_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Forgetting failed: {e}")


@router.post("/reconsolidate")
async def dashboard_reconsolidate(
    body: DashboardReconsolidateRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Release all labile state for a tenant. No belief revision; clears labile sessions. Admin-only."""
    db: DatabaseManager = request.app.state.db
    job_id = uuid_mod.uuid4()
    now = datetime.now(UTC).replace(tzinfo=None)

    try:
        async with db.pg_session() as session:
            session.add(
                DashboardJobModel(
                    id=job_id,
                    job_type="reconsolidate",
                    tenant_id=body.tenant_id,
                    user_id=body.user_id or body.tenant_id,
                    dry_run=False,
                    status="running",
                    started_at=now,
                )
            )
            await session.commit()
    except Exception as exc:
        logger.debug("dashboard_job_tracking_failed", error=str(exc))

    try:
        orchestrator = request.app.state.orchestrator
        sessions_released = (
            await orchestrator.reconsolidation.labile_tracker.release_all_for_tenant(body.tenant_id)
        )
        result_data = {
            "status": "completed",
            "tenant_id": body.tenant_id,
            "sessions_released": sessions_released,
        }
        try:
            async with db.pg_session() as session:
                await session.execute(
                    update(DashboardJobModel)
                    .where(DashboardJobModel.id == job_id)
                    .values(
                        status="completed",
                        result=result_data,
                        completed_at=datetime.now(UTC).replace(tzinfo=None),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("dashboard_job_tracking_failed", error=str(exc))
        return result_data
    except Exception as e:
        try:
            async with db.pg_session() as session:
                await session.execute(
                    update(DashboardJobModel)
                    .where(DashboardJobModel.id == job_id)
                    .values(
                        status="failed",
                        error=str(e),
                        completed_at=datetime.now(UTC).replace(tzinfo=None),
                    )
                )
                await session.commit()
        except Exception as exc:
            logger.debug("dashboard_job_tracking_failed", error=str(exc))
        logger.error("dashboard_reconsolidate_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reconsolidation failed: {e}")


@router.post("/database/reset")
async def dashboard_database_reset(
    auth: AuthContext = Depends(require_admin_permission),
):
    """Drop all tables and re-run Alembic migrations. Destructive. Admin-only."""
    root = _project_root()
    alembic_ini = root / "alembic.ini"
    if not alembic_ini.is_file():
        raise HTTPException(status_code=500, detail="alembic.ini not found; cannot run migrations")
    try:
        from alembic import command
        from alembic.config import Config

        config = Config(str(alembic_ini))
        if str(root) not in sys.path:
            sys.path.insert(0, str(root))
        command.downgrade(config, "base")
        command.upgrade(config, "head")
        return {"success": True, "message": "Database reset and recreated."}
    except Exception as e:
        logger.exception("database_reset_failed", error=str(e))
        raise HTTPException(status_code=500, detail="Database reset failed.")
