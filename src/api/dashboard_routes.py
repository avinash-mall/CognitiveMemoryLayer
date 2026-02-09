"""Dashboard API routes for monitoring and management."""

import math
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import List, Optional
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, text

from ..core.config import get_settings
from ..storage.connection import DatabaseManager
from ..storage.models import EventLogModel, MemoryRecordModel, SemanticFactModel
from .auth import AuthContext, require_admin_permission
from .schemas import (
    ComponentStatus,
    DashboardComponentsResponse,
    DashboardConsolidateRequest,
    DashboardEventItem,
    DashboardEventListResponse,
    DashboardForgetRequest,
    DashboardMemoryDetail,
    DashboardMemoryListItem,
    DashboardMemoryListResponse,
    DashboardOverview,
    DashboardTenantsResponse,
    DashboardTimelineResponse,
    TenantInfo,
    TimelinePoint,
)

logger = structlog.get_logger()

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])


def _get_db(request: Request) -> DatabaseManager:
    """Get database manager from app state."""
    return request.app.state.db


# ---- Overview ----


@dashboard_router.get("/overview", response_model=DashboardOverview)
async def dashboard_overview(
    tenant_id: Optional[str] = Query(None, description="Filter by tenant (omit for all)"),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Comprehensive dashboard overview with KPIs and breakdowns."""
    try:
        async with db.pg_session() as session:
            # Base filter
            mem_filter = []
            fact_filter = []
            event_filter = []
            if tenant_id:
                mem_filter.append(MemoryRecordModel.tenant_id == tenant_id)
                fact_filter.append(SemanticFactModel.tenant_id == tenant_id)
                event_filter.append(EventLogModel.tenant_id == tenant_id)

            # --- Memory record stats ---
            total_q = select(func.count()).select_from(MemoryRecordModel)
            for f in mem_filter:
                total_q = total_q.where(f)
            total = (await session.execute(total_q)).scalar() or 0

            # Counts by status
            status_q = select(MemoryRecordModel.status, func.count()).group_by(
                MemoryRecordModel.status
            )
            for f in mem_filter:
                status_q = status_q.where(f)
            status_rows = (await session.execute(status_q)).all()
            by_status = {row[0]: row[1] for row in status_rows}

            # Counts by type
            type_q = select(MemoryRecordModel.type, func.count()).group_by(MemoryRecordModel.type)
            for f in mem_filter:
                type_q = type_q.where(f)
            type_rows = (await session.execute(type_q)).all()
            by_type = {row[0]: row[1] for row in type_rows}

            # Averages
            avg_q = select(
                func.avg(MemoryRecordModel.confidence),
                func.avg(MemoryRecordModel.importance),
                func.avg(MemoryRecordModel.access_count),
                func.avg(MemoryRecordModel.decay_rate),
            )
            for f in mem_filter:
                avg_q = avg_q.where(f)
            avg_row = (await session.execute(avg_q)).one_or_none()
            avg_confidence = float(avg_row[0] or 0) if avg_row else 0.0
            avg_importance = float(avg_row[1] or 0) if avg_row else 0.0
            avg_access_count = float(avg_row[2] or 0) if avg_row else 0.0
            avg_decay_rate = float(avg_row[3] or 0) if avg_row else 0.0

            # Labile count
            labile_q = (
                select(func.count())
                .select_from(MemoryRecordModel)
                .where(MemoryRecordModel.labile.is_(True))
            )
            for f in mem_filter:
                labile_q = labile_q.where(f)
            labile_count = (await session.execute(labile_q)).scalar() or 0

            # Temporal range
            time_q = select(
                func.min(MemoryRecordModel.timestamp),
                func.max(MemoryRecordModel.timestamp),
            )
            for f in mem_filter:
                time_q = time_q.where(f)
            time_row = (await session.execute(time_q)).one_or_none()
            oldest = time_row[0] if time_row else None
            newest = time_row[1] if time_row else None

            # Estimated size (rough: avg text length * count * overhead)
            size_q = select(func.avg(func.length(MemoryRecordModel.text)))
            for f in mem_filter:
                size_q = size_q.where(f)
            avg_text_len = (await session.execute(size_q)).scalar() or 0
            estimated_size_mb = round((float(avg_text_len) * total * 2.5) / (1024 * 1024), 2)

            # --- Semantic facts stats ---
            fact_total_q = select(func.count()).select_from(SemanticFactModel)
            for f in fact_filter:
                fact_total_q = fact_total_q.where(f)
            total_facts = (await session.execute(fact_total_q)).scalar() or 0

            fact_current_q = (
                select(func.count())
                .select_from(SemanticFactModel)
                .where(SemanticFactModel.is_current.is_(True))
            )
            for f in fact_filter:
                fact_current_q = fact_current_q.where(f)
            current_facts = (await session.execute(fact_current_q)).scalar() or 0

            fact_cat_q = select(SemanticFactModel.category, func.count()).group_by(
                SemanticFactModel.category
            )
            for f in fact_filter:
                fact_cat_q = fact_cat_q.where(f)
            fact_cat_rows = (await session.execute(fact_cat_q)).all()
            facts_by_category = {row[0]: row[1] for row in fact_cat_rows}

            fact_avg_q = select(
                func.avg(SemanticFactModel.confidence),
                func.avg(SemanticFactModel.evidence_count),
            )
            for f in fact_filter:
                fact_avg_q = fact_avg_q.where(f)
            fact_avg_row = (await session.execute(fact_avg_q)).one_or_none()
            avg_fact_confidence = float(fact_avg_row[0] or 0) if fact_avg_row else 0.0
            avg_evidence_count = float(fact_avg_row[1] or 0) if fact_avg_row else 0.0

            # --- Event stats ---
            event_total_q = select(func.count()).select_from(EventLogModel)
            for f in event_filter:
                event_total_q = event_total_q.where(f)
            total_events = (await session.execute(event_total_q)).scalar() or 0

            event_type_q = select(EventLogModel.event_type, func.count()).group_by(
                EventLogModel.event_type
            )
            for f in event_filter:
                event_type_q = event_type_q.where(f)
            event_type_rows = (await session.execute(event_type_q)).all()
            events_by_type = {row[0]: row[1] for row in event_type_rows}

            event_op_q = (
                select(EventLogModel.operation, func.count())
                .where(EventLogModel.operation.isnot(None))
                .group_by(EventLogModel.operation)
            )
            for f in event_filter:
                event_op_q = event_op_q.where(f)
            event_op_rows = (await session.execute(event_op_q)).all()
            events_by_operation = {row[0]: row[1] for row in event_op_rows}

            return DashboardOverview(
                total_memories=total,
                active_memories=by_status.get("active", 0),
                silent_memories=by_status.get("silent", 0),
                compressed_memories=by_status.get("compressed", 0),
                archived_memories=by_status.get("archived", 0),
                deleted_memories=by_status.get("deleted", 0),
                labile_memories=labile_count,
                by_type=by_type,
                by_status=by_status,
                avg_confidence=round(avg_confidence, 4),
                avg_importance=round(avg_importance, 4),
                avg_access_count=round(avg_access_count, 2),
                avg_decay_rate=round(avg_decay_rate, 4),
                oldest_memory=oldest,
                newest_memory=newest,
                estimated_size_mb=estimated_size_mb,
                total_semantic_facts=total_facts,
                current_semantic_facts=current_facts,
                facts_by_category=facts_by_category,
                avg_fact_confidence=round(avg_fact_confidence, 4),
                avg_evidence_count=round(avg_evidence_count, 2),
                total_events=total_events,
                events_by_type=events_by_type,
                events_by_operation=events_by_operation,
            )
    except Exception as e:
        logger.error("dashboard_overview_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---- Memory List ----


@dashboard_router.get("/memories", response_model=DashboardMemoryListResponse)
async def dashboard_memories(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
    type: Optional[str] = Query(None),
    status: Optional[str] = Query(None),
    search: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    sort_by: str = Query(
        "timestamp",
        pattern="^(timestamp|confidence|importance|access_count|written_at|type|status)$",
    ),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Paginated memory list with filtering and sorting."""
    try:
        async with db.pg_session() as session:
            filters = []
            if tenant_id:
                filters.append(MemoryRecordModel.tenant_id == tenant_id)
            if type:
                filters.append(MemoryRecordModel.type == type)
            if status:
                filters.append(MemoryRecordModel.status == status)
            if search:
                filters.append(MemoryRecordModel.text.ilike(f"%{search}%"))

            # Total count
            count_q = select(func.count()).select_from(MemoryRecordModel)
            for f in filters:
                count_q = count_q.where(f)
            total = (await session.execute(count_q)).scalar() or 0

            # Sorting
            sort_col = getattr(MemoryRecordModel, sort_by, MemoryRecordModel.timestamp)
            order_col = sort_col.desc() if order == "desc" else sort_col.asc()

            # Query
            q = (
                select(MemoryRecordModel)
                .order_by(order_col)
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
            for f in filters:
                q = q.where(f)
            rows = (await session.execute(q)).scalars().all()

            items = [
                DashboardMemoryListItem(
                    id=r.id,
                    tenant_id=r.tenant_id,
                    agent_id=r.agent_id,
                    type=r.type,
                    status=r.status,
                    text=r.text[:300] if r.text else "",
                    key=r.key,
                    namespace=r.namespace,
                    context_tags=r.context_tags or [],
                    confidence=r.confidence or 0.5,
                    importance=r.importance or 0.5,
                    access_count=r.access_count or 0,
                    decay_rate=r.decay_rate or 0.01,
                    labile=r.labile or False,
                    version=r.version or 1,
                    timestamp=r.timestamp,
                    written_at=r.written_at,
                )
                for r in rows
            ]

            return DashboardMemoryListResponse(
                items=items,
                total=total,
                page=page,
                per_page=per_page,
                total_pages=max(1, math.ceil(total / per_page)),
            )
    except Exception as e:
        logger.error("dashboard_memories_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---- Memory Detail ----


@dashboard_router.get("/memories/{memory_id}", response_model=DashboardMemoryDetail)
async def dashboard_memory_detail(
    memory_id: UUID,
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Full detail for a single memory record."""
    try:
        async with db.pg_session() as session:
            q = select(MemoryRecordModel).where(MemoryRecordModel.id == memory_id)
            record = (await session.execute(q)).scalar_one_or_none()
            if not record:
                raise HTTPException(status_code=404, detail="Memory not found")

            # Related events
            event_q = (
                select(EventLogModel)
                .where(EventLogModel.memory_ids.any(memory_id))
                .order_by(EventLogModel.created_at.desc())
                .limit(50)
            )
            events = (await session.execute(event_q)).scalars().all()
            related_events = [
                {
                    "id": str(e.id),
                    "event_type": e.event_type,
                    "operation": e.operation,
                    "created_at": e.created_at.isoformat() if e.created_at else None,
                    "payload": e.payload,
                }
                for e in events
            ]

            return DashboardMemoryDetail(
                id=record.id,
                tenant_id=record.tenant_id,
                agent_id=record.agent_id,
                type=record.type,
                status=record.status,
                text=record.text,
                key=record.key,
                namespace=record.namespace,
                context_tags=record.context_tags or [],
                source_session_id=record.source_session_id,
                entities=record.entities,
                relations=record.relations,
                metadata=record.meta,
                confidence=record.confidence or 0.5,
                importance=record.importance or 0.5,
                access_count=record.access_count or 0,
                last_accessed_at=record.last_accessed_at,
                decay_rate=record.decay_rate or 0.01,
                labile=record.labile or False,
                provenance=record.provenance,
                version=record.version or 1,
                supersedes_id=record.supersedes_id,
                content_hash=record.content_hash,
                timestamp=record.timestamp,
                written_at=record.written_at,
                valid_from=record.valid_from,
                valid_to=record.valid_to,
                related_events=related_events,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_memory_detail_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---- Events ----


@dashboard_router.get("/events", response_model=DashboardEventListResponse)
async def dashboard_events(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
    event_type: Optional[str] = Query(None),
    operation: Optional[str] = Query(None),
    tenant_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Paginated event log."""
    try:
        async with db.pg_session() as session:
            filters = []
            if tenant_id:
                filters.append(EventLogModel.tenant_id == tenant_id)
            if event_type:
                filters.append(EventLogModel.event_type == event_type)
            if operation:
                filters.append(EventLogModel.operation == operation)

            count_q = select(func.count()).select_from(EventLogModel)
            for f in filters:
                count_q = count_q.where(f)
            total = (await session.execute(count_q)).scalar() or 0

            q = (
                select(EventLogModel)
                .order_by(EventLogModel.created_at.desc())
                .offset((page - 1) * per_page)
                .limit(per_page)
            )
            for f in filters:
                q = q.where(f)
            rows = (await session.execute(q)).scalars().all()

            items = [
                DashboardEventItem(
                    id=e.id,
                    tenant_id=e.tenant_id,
                    scope_id=e.scope_id,
                    agent_id=e.agent_id,
                    event_type=e.event_type,
                    operation=e.operation,
                    payload=e.payload,
                    memory_ids=[mid for mid in (e.memory_ids or [])],
                    parent_event_id=e.parent_event_id,
                    created_at=e.created_at,
                )
                for e in rows
            ]

            return DashboardEventListResponse(
                items=items,
                total=total,
                page=page,
                per_page=per_page,
                total_pages=max(1, math.ceil(total / per_page)),
            )
    except Exception as e:
        logger.error("dashboard_events_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---- Timeline ----


@dashboard_router.get("/timeline", response_model=DashboardTimelineResponse)
async def dashboard_timeline(
    days: int = Query(30, ge=1, le=365),
    tenant_id: Optional[str] = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Memory creation timeline aggregated by day."""
    try:
        async with db.pg_session() as session:
            cutoff = datetime.now(timezone.utc).replace(tzinfo=None) - timedelta(days=days)

            filters = [MemoryRecordModel.timestamp >= cutoff]
            if tenant_id:
                filters.append(MemoryRecordModel.tenant_id == tenant_id)

            q = (
                select(
                    func.date_trunc("day", MemoryRecordModel.timestamp).label("day"),
                    func.count().label("cnt"),
                )
                .where(*filters)
                .group_by(text("1"))
                .order_by(text("1"))
            )
            rows = (await session.execute(q)).all()

            points = [
                TimelinePoint(
                    date=row[0].strftime("%Y-%m-%d") if row[0] else "",
                    count=row[1],
                )
                for row in rows
            ]
            total = sum(p.count for p in points)

            return DashboardTimelineResponse(points=points, total=total)
    except Exception as e:
        logger.error("dashboard_timeline_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---- Components Health ----


@dashboard_router.get("/components", response_model=DashboardComponentsResponse)
async def dashboard_components(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Health check for all system components."""
    components: List[ComponentStatus] = []

    # PostgreSQL
    try:
        t0 = time.monotonic()
        async with db.pg_session() as session:
            await session.execute(text("SELECT 1"))

            mem_count = (
                await session.execute(select(func.count()).select_from(MemoryRecordModel))
            ).scalar() or 0
            fact_count = (
                await session.execute(select(func.count()).select_from(SemanticFactModel))
            ).scalar() or 0
            event_count = (
                await session.execute(select(func.count()).select_from(EventLogModel))
            ).scalar() or 0
        latency = (time.monotonic() - t0) * 1000
        settings = get_settings()
        components.append(
            ComponentStatus(
                name="PostgreSQL",
                status="ok",
                latency_ms=round(latency, 2),
                details={
                    "memory_records": mem_count,
                    "semantic_facts": fact_count,
                    "events": event_count,
                    "embedding_dimensions": settings.embedding.dimensions,
                },
            )
        )
    except Exception as e:
        components.append(ComponentStatus(name="PostgreSQL", status="error", error=str(e)))

    # Neo4j
    try:
        t0 = time.monotonic()
        if db.neo4j_driver:
            async with db.neo4j_session() as neo_session:
                result = await neo_session.run("MATCH (n) RETURN count(n) AS cnt")
                record = await result.single()
                node_count = record["cnt"] if record else 0
                result2 = await neo_session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
                record2 = await result2.single()
                rel_count = record2["cnt"] if record2 else 0
            latency = (time.monotonic() - t0) * 1000
            components.append(
                ComponentStatus(
                    name="Neo4j",
                    status="ok",
                    latency_ms=round(latency, 2),
                    details={"nodes": node_count, "relationships": rel_count},
                )
            )
        else:
            components.append(
                ComponentStatus(name="Neo4j", status="unknown", error="Driver not initialized")
            )
    except Exception as e:
        components.append(ComponentStatus(name="Neo4j", status="error", error=str(e)))

    # Redis
    try:
        t0 = time.monotonic()
        if db.redis:
            await db.redis.ping()
            db_size = await db.redis.dbsize()
            info = await db.redis.info("memory")
            used_memory_mb = round(info.get("used_memory", 0) / (1024 * 1024), 2)
            latency = (time.monotonic() - t0) * 1000
            components.append(
                ComponentStatus(
                    name="Redis",
                    status="ok",
                    latency_ms=round(latency, 2),
                    details={
                        "keys": db_size,
                        "used_memory_mb": used_memory_mb,
                    },
                )
            )
        else:
            components.append(
                ComponentStatus(name="Redis", status="unknown", error="Client not initialized")
            )
    except Exception as e:
        components.append(ComponentStatus(name="Redis", status="error", error=str(e)))

    return DashboardComponentsResponse(components=components)


# ---- Tenants ----


@dashboard_router.get("/tenants", response_model=DashboardTenantsResponse)
async def dashboard_tenants(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """List all known tenants with summary counts."""
    try:
        async with db.pg_session() as session:
            # Memory counts per tenant
            mem_q = select(
                MemoryRecordModel.tenant_id,
                func.count().label("mem_count"),
            ).group_by(MemoryRecordModel.tenant_id)
            mem_rows = (await session.execute(mem_q)).all()
            mem_map = {row[0]: row[1] for row in mem_rows}

            # Fact counts per tenant
            fact_q = select(
                SemanticFactModel.tenant_id,
                func.count().label("fact_count"),
            ).group_by(SemanticFactModel.tenant_id)
            fact_rows = (await session.execute(fact_q)).all()
            fact_map = {row[0]: row[1] for row in fact_rows}

            # Event counts per tenant
            event_q = select(
                EventLogModel.tenant_id,
                func.count().label("event_count"),
            ).group_by(EventLogModel.tenant_id)
            event_rows = (await session.execute(event_q)).all()
            event_map = {row[0]: row[1] for row in event_rows}

            # Merge all tenant IDs
            all_tenants = sorted(set(mem_map.keys()) | set(fact_map.keys()) | set(event_map.keys()))

            tenants = [
                TenantInfo(
                    tenant_id=tid,
                    memory_count=mem_map.get(tid, 0),
                    fact_count=fact_map.get(tid, 0),
                    event_count=event_map.get(tid, 0),
                )
                for tid in all_tenants
            ]

            return DashboardTenantsResponse(tenants=tenants)
    except Exception as e:
        logger.error("dashboard_tenants_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---- Management: Consolidation ----


@dashboard_router.post("/consolidate")
async def dashboard_consolidate(
    body: DashboardConsolidateRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Trigger memory consolidation for a tenant."""
    try:
        orchestrator = request.app.state.orchestrator
        user_id = body.user_id or body.tenant_id
        report = await orchestrator.consolidation.consolidate(
            tenant_id=body.tenant_id,
            user_id=user_id,
        )
        return {
            "status": "completed",
            "tenant_id": body.tenant_id,
            "user_id": user_id,
            "episodes_sampled": report.episodes_sampled,
            "clusters_formed": report.clusters_formed,
            "gists_extracted": report.gists_extracted,
            "elapsed_seconds": getattr(report, "elapsed_seconds", None),
        }
    except Exception as e:
        logger.error("dashboard_consolidate_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


# ---- Management: Forgetting ----


@dashboard_router.post("/forget")
async def dashboard_forget(
    body: DashboardForgetRequest,
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Trigger active forgetting for a tenant."""
    try:
        orchestrator = request.app.state.orchestrator
        user_id = body.user_id or body.tenant_id
        report = await orchestrator.forgetting.run_forgetting(
            tenant_id=body.tenant_id,
            user_id=user_id,
            max_memories=body.max_memories,
            dry_run=body.dry_run,
        )
        return {
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
    except Exception as e:
        logger.error("dashboard_forget_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Forgetting failed: {e}")


# ---- Management: Database reset ----


def _project_root() -> Path:
    """Project root (directory containing alembic.ini and migrations/)."""
    return Path(__file__).resolve().parent.parent.parent


@dashboard_router.post("/database/reset")
async def dashboard_database_reset(
    auth: AuthContext = Depends(require_admin_permission),
):
    """Drop all tables and re-run migrations. Uses current .env (e.g. EMBEDDING__DIMENSIONS)."""
    root = _project_root()
    alembic_ini = root / "alembic.ini"
    if not alembic_ini.is_file():
        raise HTTPException(
            status_code=500,
            detail="alembic.ini not found; cannot run migrations",
        )
    env = os.environ.copy()
    # Ensure Python path includes project root for 'src' imports during migrations
    env.setdefault("PYTHONPATH", str(root))
    if "PYTHONPATH" in env and str(root) not in env["PYTHONPATH"].split(os.pathsep):
        env["PYTHONPATH"] = os.pathsep.join([str(root), env["PYTHONPATH"]])

    def run_alembic(*args: str) -> subprocess.CompletedProcess:
        return subprocess.run(
            [sys.executable, "-m", "alembic"] + list(args),
            cwd=str(root),
            capture_output=True,
            text=True,
            env=env,
            timeout=120,
        )

    try:
        down = run_alembic("downgrade", "base")
        if down.returncode != 0:
            logger.error("database_reset_downgrade_failed", stderr=down.stderr, stdout=down.stdout)
            raise HTTPException(
                status_code=500,
                detail=f"alembic downgrade base failed: {down.stderr or down.stdout or 'unknown'}",
            )
        up = run_alembic("upgrade", "head")
        if up.returncode != 0:
            logger.error("database_reset_upgrade_failed", stderr=up.stderr, stdout=up.stdout)
            raise HTTPException(
                status_code=500,
                detail=f"alembic upgrade head failed: {up.stderr or up.stdout or 'unknown'}",
            )
        return {"success": True, "message": "Database reset and recreated."}
    except subprocess.TimeoutExpired as e:
        logger.error("database_reset_timeout", error=str(e))
        raise HTTPException(status_code=500, detail="Migration timed out.")
