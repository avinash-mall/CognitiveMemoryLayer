"""Dashboard API routes for monitoring and management."""

import json
import math
import os
import sys
import time
import uuid as uuid_mod
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from fastapi import APIRouter, Depends, HTTPException, Query, Request
from sqlalchemy import func, select, text, update

from ..core.config import get_settings
from ..storage.connection import DatabaseManager
from ..storage.models import (
    DashboardJobModel,
    EventLogModel,
    MemoryRecordModel,
    SemanticFactModel,
)
from .auth import AuthContext, require_admin_permission
from .schemas import (
    BulkActionRequest,
    ComponentStatus,
    ConfigItem,
    ConfigSection,
    ConfigUpdateRequest,
    DashboardComponentsResponse,
    DashboardConfigResponse,
    DashboardConsolidateRequest,
    DashboardEventItem,
    DashboardEventListResponse,
    DashboardForgetRequest,
    DashboardJobItem,
    DashboardJobsResponse,
    DashboardLabileResponse,
    DashboardMemoryDetail,
    DashboardMemoryListItem,
    DashboardMemoryListResponse,
    DashboardOverview,
    DashboardRateLimitsResponse,
    DashboardReconsolidateRequest,
    DashboardRetrievalRequest,
    DashboardRetrievalResponse,
    DashboardSessionsResponse,
    DashboardTenantsResponse,
    DashboardTimelineResponse,
    GraphEdgeInfo,
    GraphExploreResponse,
    GraphNodeInfo,
    GraphSearchResponse,
    GraphSearchResult,
    GraphStatsResponse,
    HourlyRequestCount,
    RateLimitEntry,
    RequestStatsResponse,
    RetrievalResultItem,
    SessionInfo,
    TenantInfo,
    TenantLabileInfo,
    TimelinePoint,
)

logger = structlog.get_logger()

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])

_REQUEST_COUNT_PREFIX = "dashboard:reqcount:"
_CONFIG_OVERRIDES_KEY = "dashboard:config:overrides"

# Settings that admins may change at runtime (non-secret, non-connection).
_EDITABLE_SETTINGS = {
    "auth.rate_limit_requests_per_minute",
    "debug",
    "app_name",
    "embedding.provider",
    "embedding.model",
    "embedding.dimensions",
    "embedding.local_model",
    "llm.provider",
    "llm.model",
}

# Fields whose values must be masked in the config output.
_SECRET_FIELD_TOKENS = {"key", "password", "secret", "token"}


def _get_db(request: Request) -> DatabaseManager:
    """Get database manager from app state."""
    return request.app.state.db


# ---------------------------------------------------------------------------
# Overview
# ---------------------------------------------------------------------------


@dashboard_router.get("/overview", response_model=DashboardOverview)
async def dashboard_overview(
    tenant_id: str | None = Query(None, description="Filter by tenant (omit for all)"),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Comprehensive dashboard overview: total memories, semantic facts, event counts, storage estimates, and breakdowns by type and status. Optionally filter by tenant."""
    try:
        async with db.pg_session() as session:
            mem_filter: list = []
            fact_filter: list = []
            event_filter: list = []
            if tenant_id:
                mem_filter.append(MemoryRecordModel.tenant_id == tenant_id)
                fact_filter.append(SemanticFactModel.tenant_id == tenant_id)
                event_filter.append(EventLogModel.tenant_id == tenant_id)

            # Memory record stats
            total_q = select(func.count()).select_from(MemoryRecordModel)
            for f in mem_filter:
                total_q = total_q.where(f)
            total = (await session.execute(total_q)).scalar() or 0

            status_q = select(MemoryRecordModel.status, func.count()).group_by(
                MemoryRecordModel.status
            )
            for f in mem_filter:
                status_q = status_q.where(f)
            by_status = {r[0]: r[1] for r in (await session.execute(status_q)).all()}

            type_q = select(MemoryRecordModel.type, func.count()).group_by(MemoryRecordModel.type)
            for f in mem_filter:
                type_q = type_q.where(f)
            by_type = {r[0]: r[1] for r in (await session.execute(type_q)).all()}

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

            labile_q = (
                select(func.count())
                .select_from(MemoryRecordModel)
                .where(MemoryRecordModel.labile.is_(True))
            )
            for f in mem_filter:
                labile_q = labile_q.where(f)
            labile_count = (await session.execute(labile_q)).scalar() or 0

            time_q = select(
                func.min(MemoryRecordModel.timestamp), func.max(MemoryRecordModel.timestamp)
            )
            for f in mem_filter:
                time_q = time_q.where(f)
            time_row = (await session.execute(time_q)).one_or_none()
            oldest = time_row[0] if time_row else None
            newest = time_row[1] if time_row else None

            size_q = select(func.avg(func.length(MemoryRecordModel.text)))
            for f in mem_filter:
                size_q = size_q.where(f)
            avg_text_len = (await session.execute(size_q)).scalar() or 0
            estimated_size_mb = round((float(avg_text_len) * total * 2.5) / (1024 * 1024), 2)

            # Semantic facts
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
            facts_by_category = {r[0]: r[1] for r in (await session.execute(fact_cat_q)).all()}

            fact_avg_q = select(
                func.avg(SemanticFactModel.confidence), func.avg(SemanticFactModel.evidence_count)
            )
            for f in fact_filter:
                fact_avg_q = fact_avg_q.where(f)
            fact_avg_row = (await session.execute(fact_avg_q)).one_or_none()
            avg_fact_confidence = float(fact_avg_row[0] or 0) if fact_avg_row else 0.0
            avg_evidence_count = float(fact_avg_row[1] or 0) if fact_avg_row else 0.0

            # Events
            event_total_q = select(func.count()).select_from(EventLogModel)
            for f in event_filter:
                event_total_q = event_total_q.where(f)
            total_events = (await session.execute(event_total_q)).scalar() or 0

            event_type_q = select(EventLogModel.event_type, func.count()).group_by(
                EventLogModel.event_type
            )
            for f in event_filter:
                event_type_q = event_type_q.where(f)
            events_by_type = {r[0]: r[1] for r in (await session.execute(event_type_q)).all()}

            event_op_q = (
                select(EventLogModel.operation, func.count())
                .where(EventLogModel.operation.isnot(None))
                .group_by(EventLogModel.operation)
            )
            for f in event_filter:
                event_op_q = event_op_q.where(f)
            events_by_operation = {r[0]: r[1] for r in (await session.execute(event_op_q)).all()}

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


# ---------------------------------------------------------------------------
# Memory List
# ---------------------------------------------------------------------------


@dashboard_router.get("/memories", response_model=DashboardMemoryListResponse)
async def dashboard_memories(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
    type: str | None = Query(None),
    status: str | None = Query(None),
    search: str | None = Query(None),
    tenant_id: str | None = Query(None),
    source_session_id: str | None = Query(None),
    sort_by: str = Query(
        "timestamp",
        pattern="^(timestamp|confidence|importance|access_count|written_at|type|status)$",
    ),
    order: str = Query("desc", pattern="^(asc|desc)$"),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Paginated list of memories with filtering by tenant, type, status, and full-text search. Supports sorting and bulk actions."""
    try:
        async with db.pg_session() as session:
            filters: list = []
            if tenant_id:
                filters.append(MemoryRecordModel.tenant_id == tenant_id)
            if type:
                filters.append(MemoryRecordModel.type == type)
            if status:
                filters.append(MemoryRecordModel.status == status)
            if source_session_id:
                filters.append(MemoryRecordModel.source_session_id == source_session_id)
            if search:
                escaped = search.replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_")
                filters.append(MemoryRecordModel.text.ilike(f"%{escaped}%", escape="\\"))

            count_q = select(func.count()).select_from(MemoryRecordModel)
            for f in filters:
                count_q = count_q.where(f)
            total = (await session.execute(count_q)).scalar() or 0

            sort_col = getattr(MemoryRecordModel, sort_by, MemoryRecordModel.timestamp)
            order_col = sort_col.desc() if order == "desc" else sort_col.asc()
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


# ---------------------------------------------------------------------------
# Memory Detail
# ---------------------------------------------------------------------------


@dashboard_router.get("/memories/{memory_id}", response_model=DashboardMemoryDetail)
async def dashboard_memory_detail(
    memory_id: UUID,
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Full detail for a single memory record: text, type, confidence, metadata, related semantic facts, and event history."""
    try:
        async with db.pg_session() as session:
            q = select(MemoryRecordModel).where(MemoryRecordModel.id == memory_id)
            record = (await session.execute(q)).scalar_one_or_none()
            if not record:
                raise HTTPException(status_code=404, detail="Memory not found")
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


# ---------------------------------------------------------------------------
# Bulk Memory Actions
# ---------------------------------------------------------------------------


@dashboard_router.post("/memories/bulk-action")
async def dashboard_bulk_action(
    body: BulkActionRequest,
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Apply an action (e.g. forget, archive) to multiple memories at once. Admin-only."""
    try:
        async with db.pg_session() as session:
            if body.action == "delete":
                new_status = "deleted"
            elif body.action == "archive":
                new_status = "archived"
            elif body.action == "silence":
                new_status = "silent"
            else:
                raise HTTPException(status_code=400, detail=f"Unknown action: {body.action}")

            stmt = (
                update(MemoryRecordModel)
                .where(MemoryRecordModel.id.in_(body.memory_ids))
                .values(status=new_status)
            )
            result = await session.execute(stmt)
            await session.commit()
            return {"success": True, "affected": result.rowcount, "action": body.action}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_bulk_action_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------


@dashboard_router.get("/events", response_model=DashboardEventListResponse)
async def dashboard_events(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
    event_type: str | None = Query(None),
    operation: str | None = Query(None),
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Paginated event log (writes, reads, consolidations, etc.). Filter by tenant, event type, or operation."""
    try:
        async with db.pg_session() as session:
            filters: list = []
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


# ---------------------------------------------------------------------------
# Timeline
# ---------------------------------------------------------------------------


@dashboard_router.get("/timeline", response_model=DashboardTimelineResponse)
async def dashboard_timeline(
    days: int = Query(30, ge=1, le=365),
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Memory creation timeline aggregated by day for charts. Default 30 days, optional tenant filter."""
    try:
        async with db.pg_session() as session:
            cutoff = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=days)
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
                TimelinePoint(date=r[0].strftime("%Y-%m-%d") if r[0] else "", count=r[1])
                for r in rows
            ]
            return DashboardTimelineResponse(points=points, total=sum(p.count for p in points))
    except Exception as e:
        logger.error("dashboard_timeline_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Components Health
# ---------------------------------------------------------------------------


@dashboard_router.get("/components", response_model=DashboardComponentsResponse)
async def dashboard_components(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Health status of all system components: PostgreSQL, Redis, Neo4j, embedding service. Includes latency and connection state."""
    components: list[ComponentStatus] = []

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
                    details={"keys": db_size, "used_memory_mb": used_memory_mb},
                )
            )
        else:
            components.append(
                ComponentStatus(name="Redis", status="unknown", error="Client not initialized")
            )
    except Exception as e:
        components.append(ComponentStatus(name="Redis", status="error", error=str(e)))

    return DashboardComponentsResponse(components=components)


# ---------------------------------------------------------------------------
# Tenants (Enhanced)
# ---------------------------------------------------------------------------


@dashboard_router.get("/tenants", response_model=DashboardTenantsResponse)
async def dashboard_tenants(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """List all tenants with memory counts, active session counts, and last activity. Used for tenant selector in dashboard."""
    try:
        async with db.pg_session() as session:
            # Memory counts + active counts + last memory time per tenant
            mem_q = select(
                MemoryRecordModel.tenant_id,
                func.count().label("mem_count"),
                func.count().filter(MemoryRecordModel.status == "active").label("active_count"),
                func.max(MemoryRecordModel.timestamp).label("last_mem"),
            ).group_by(MemoryRecordModel.tenant_id)
            mem_rows = (await session.execute(mem_q)).all()
            mem_map: dict[str, dict] = {}
            for r in mem_rows:
                mem_map[r[0]] = {"count": r[1], "active": r[2], "last_mem": r[3]}

            # Fact counts per tenant
            fact_q = select(SemanticFactModel.tenant_id, func.count().label("fact_count")).group_by(
                SemanticFactModel.tenant_id
            )
            fact_rows = (await session.execute(fact_q)).all()
            fact_map = {r[0]: r[1] for r in fact_rows}

            # Event counts + last event per tenant
            event_q = select(
                EventLogModel.tenant_id,
                func.count().label("event_count"),
                func.max(EventLogModel.created_at).label("last_evt"),
            ).group_by(EventLogModel.tenant_id)
            event_rows = (await session.execute(event_q)).all()
            event_map: dict[str, dict] = {}
            for r in event_rows:
                event_map[r[0]] = {"count": r[1], "last_evt": r[2]}

            all_tenants = sorted(set(mem_map.keys()) | set(fact_map.keys()) | set(event_map.keys()))

            tenants = [
                TenantInfo(
                    tenant_id=tid,
                    memory_count=mem_map.get(tid, {}).get("count", 0),
                    active_memory_count=mem_map.get(tid, {}).get("active", 0),
                    fact_count=fact_map.get(tid, 0),
                    event_count=event_map.get(tid, {}).get("count", 0),
                    last_memory_at=mem_map.get(tid, {}).get("last_mem"),
                    last_event_at=event_map.get(tid, {}).get("last_evt"),
                )
                for tid in all_tenants
            ]
            return DashboardTenantsResponse(tenants=tenants)
    except Exception as e:
        logger.error("dashboard_tenants_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Sessions
# ---------------------------------------------------------------------------


@dashboard_router.get("/sessions", response_model=DashboardSessionsResponse)
async def dashboard_sessions(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """List active sessions from Redis with memory counts per source_session_id. Optionally filter by tenant."""
    try:
        sessions_list: list[SessionInfo] = []
        redis_sessions: dict[str, SessionInfo] = {}

        # --- Redis: scan session:* keys ---
        if db.redis:
            cursor = 0
            while True:
                cursor, keys = await db.redis.scan(cursor, match="session:*", count=200)
                for key in keys:
                    key_str = key.decode() if isinstance(key, bytes) else key
                    sid = key_str.removeprefix("session:")
                    ttl = await db.redis.ttl(key_str)
                    raw = await db.redis.get(key_str)
                    info = SessionInfo(session_id=sid, ttl_seconds=ttl)
                    if raw:
                        try:
                            data = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
                            info.tenant_id = data.get("tenant_id")
                            info.created_at = (
                                datetime.fromisoformat(data["created_at"])
                                if data.get("created_at")
                                else None
                            )
                            info.expires_at = (
                                datetime.fromisoformat(data["expires_at"])
                                if data.get("expires_at")
                                else None
                            )
                            info.metadata = {
                                k: v
                                for k, v in data.items()
                                if k not in ("tenant_id", "created_at", "expires_at")
                            }
                        except (json.JSONDecodeError, KeyError, ValueError):
                            pass
                    # Apply tenant filter if provided
                    if tenant_id and info.tenant_id and info.tenant_id != tenant_id:
                        continue
                    redis_sessions[sid] = info
                if cursor == 0:
                    break

        # --- DB: memory counts per source_session_id ---
        async with db.pg_session() as session:
            sess_q = (
                select(
                    MemoryRecordModel.source_session_id,
                    func.count().label("cnt"),
                )
                .where(MemoryRecordModel.source_session_id.isnot(None))
                .group_by(MemoryRecordModel.source_session_id)
            )
            if tenant_id:
                sess_q = sess_q.where(MemoryRecordModel.tenant_id == tenant_id)
            rows = (await session.execute(sess_q)).all()
            db_counts = {r[0]: r[1] for r in rows}

        # Merge
        all_sids = set(redis_sessions.keys()) | set(db_counts.keys())
        for sid in sorted(all_sids):
            if sid in redis_sessions:
                info = redis_sessions[sid]
                info.memory_count = db_counts.get(sid, 0)
                sessions_list.append(info)
            else:
                sessions_list.append(
                    SessionInfo(session_id=sid, memory_count=db_counts.get(sid, 0))
                )

        return DashboardSessionsResponse(
            sessions=sessions_list,
            total_active=len(redis_sessions),
            total_memories_with_session=sum(db_counts.values()),
        )
    except Exception as e:
        logger.error("dashboard_sessions_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Rate Limits
# ---------------------------------------------------------------------------


@dashboard_router.get("/ratelimits", response_model=DashboardRateLimitsResponse)
async def dashboard_ratelimits(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Current rate-limit usage per API key from Redis. Shows remaining requests and reset time."""
    settings = get_settings()
    rpm = settings.auth.rate_limit_requests_per_minute
    entries: list[RateLimitEntry] = []

    if db.redis:
        cursor = 0
        while True:
            cursor, keys = await db.redis.scan(cursor, match="ratelimit:*", count=200)
            for key in keys:
                key_str = key.decode() if isinstance(key, bytes) else key
                raw_count = await db.redis.get(key_str)
                ttl = await db.redis.ttl(key_str)
                count = int(raw_count) if raw_count else 0
                suffix = key_str.removeprefix("ratelimit:")
                if suffix.startswith("apikey:"):
                    key_type = "apikey"
                    identifier = suffix.removeprefix("apikey:")[:8] + "..."
                elif suffix.startswith("ip:"):
                    key_type = "ip"
                    identifier = suffix.removeprefix("ip:")
                else:
                    key_type = "other"
                    identifier = suffix[:16]
                entries.append(
                    RateLimitEntry(
                        key_type=key_type,
                        identifier=identifier,
                        current_count=count,
                        limit=rpm,
                        ttl_seconds=max(ttl, 0),
                        utilization_pct=round((count / rpm) * 100, 1) if rpm > 0 else 0.0,
                    )
                )
            if cursor == 0:
                break

    return DashboardRateLimitsResponse(entries=entries, configured_rpm=rpm)


# ---------------------------------------------------------------------------
# Request Stats (hourly counts)
# ---------------------------------------------------------------------------


@dashboard_router.get("/request-stats", response_model=RequestStatsResponse)
async def dashboard_request_stats(
    hours: int = Query(24, ge=1, le=48),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Hourly request counts from Redis counters. Default last 24 hours. For usage charts."""
    points: list[HourlyRequestCount] = []
    total = 0

    if db.redis:
        now = datetime.now(UTC)
        for i in range(hours - 1, -1, -1):
            dt = now - timedelta(hours=i)
            hour_key = dt.strftime("%Y-%m-%d-%H")
            rkey = f"{_REQUEST_COUNT_PREFIX}{hour_key}"
            raw = await db.redis.get(rkey)
            count = int(raw) if raw else 0
            total += count
            points.append(HourlyRequestCount(hour=dt.strftime("%Y-%m-%dT%H:00"), count=count))

    return RequestStatsResponse(points=points, total_last_24h=total)


# ---------------------------------------------------------------------------
# Knowledge Graph
# ---------------------------------------------------------------------------


@dashboard_router.get("/graph/stats", response_model=GraphStatsResponse)
async def dashboard_graph_stats(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Knowledge graph statistics from Neo4j: total nodes, edges, entity type distribution, and tenants with graph data."""
    if not db.neo4j_driver:
        return GraphStatsResponse()
    try:
        async with db.neo4j_session() as session:
            r1 = await session.run("MATCH (n:Entity) RETURN count(n) AS cnt")
            rec1 = await r1.single()
            total_nodes = rec1["cnt"] if rec1 else 0

            r2 = await session.run("MATCH ()-[r]->() RETURN count(r) AS cnt")
            rec2 = await r2.single()
            total_edges = rec2["cnt"] if rec2 else 0

            r3 = await session.run(
                "MATCH (n:Entity) RETURN n.entity_type AS t, count(*) AS c ORDER BY c DESC LIMIT 20"
            )
            entity_types = {rec["t"]: rec["c"] async for rec in r3 if rec["t"]}

            r4 = await session.run("MATCH (n:Entity) RETURN DISTINCT n.tenant_id AS tid")
            tenants_with_graph = [rec["tid"] async for rec in r4 if rec["tid"]]

        return GraphStatsResponse(
            total_nodes=total_nodes,
            total_edges=total_edges,
            entity_types=entity_types,
            tenants_with_graph=sorted(tenants_with_graph),
        )
    except Exception as e:
        logger.error("dashboard_graph_stats_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.get("/graph/overview", response_model=GraphExploreResponse)
async def dashboard_graph_overview(
    tenant_id: str = Query(..., description="Tenant ID"),
    scope_id: str | None = Query(None, description="Scope ID (defaults to tenant_id)"),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Sample subgraph for a tenant: finds the highest-degree entity and returns its 2-hop neighborhood. Used to auto-load the graph on page visit."""
    if not db.neo4j_driver:
        return GraphExploreResponse()
    scope_id = scope_id or tenant_id
    try:
        async with db.neo4j_session() as session:
            # Find entity with most relationships in this tenant
            r0 = await session.run(
                """
                MATCH (n:Entity {tenant_id: $tenant_id, scope_id: $scope_id})
                WITH n, COUNT { (n)--() } AS deg
                ORDER BY deg DESC
                LIMIT 1
                RETURN n.entity AS entity
                """,
                tenant_id=tenant_id,
                scope_id=scope_id,
            )
            rec0 = await r0.single()
            if not rec0 or not rec0["entity"]:
                return GraphExploreResponse()

            entity = rec0["entity"]
            depth = 2

            # Same neighborhood query as explore
            query = f"""
            MATCH path = (start:Entity {{
                tenant_id: $tenant_id, scope_id: $scope_id, entity: $entity
            }})-[*1..{min(depth, 5)}]-(neighbor:Entity)
            WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
            UNWIND relationships(path) AS rel
            WITH DISTINCT neighbor, rel, startNode(rel) AS sn, endNode(rel) AS en
            RETURN
                collect(DISTINCT {{
                    entity: neighbor.entity,
                    entity_type: neighbor.entity_type,
                    properties: properties(neighbor)
                }}) AS neighbors,
                collect(DISTINCT {{
                    source: sn.entity,
                    target: en.entity,
                    predicate: type(rel),
                    confidence: coalesce(rel.confidence, 0),
                    properties: properties(rel)
                }}) AS rels
            """
            result = await session.run(query, tenant_id=tenant_id, scope_id=scope_id, entity=entity)
            record = await result.single()

            nodes: list[GraphNodeInfo] = [
                GraphNodeInfo(id=entity, entity=entity, entity_type="center")
            ]
            edges: list[GraphEdgeInfo] = []
            seen_nodes = {entity}

            if record:
                for n in record["neighbors"] or []:
                    ent = n.get("entity", "")
                    if ent and ent not in seen_nodes:
                        seen_nodes.add(ent)
                        props = dict(n.get("properties") or {})
                        props.pop("tenant_id", None)
                        props.pop("scope_id", None)
                        nodes.append(
                            GraphNodeInfo(
                                id=ent,
                                entity=ent,
                                entity_type=n.get("entity_type", "unknown"),
                                properties=props,
                            )
                        )
                seen_edges = set()
                for r in record["rels"] or []:
                    src = r.get("source", "")
                    tgt = r.get("target", "")
                    pred = r.get("predicate", "RELATED_TO")
                    edge_key = f"{src}-{pred}-{tgt}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        props = dict(r.get("properties") or {})
                        props.pop("created_at", None)
                        props.pop("updated_at", None)
                        edges.append(
                            GraphEdgeInfo(
                                source=src,
                                target=tgt,
                                predicate=pred,
                                confidence=float(r.get("confidence", 0)),
                                properties=props,
                            )
                        )

        return GraphExploreResponse(nodes=nodes, edges=edges, center_entity=entity)
    except Exception as e:
        logger.error("dashboard_graph_overview_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.get("/graph/explore", response_model=GraphExploreResponse)
async def dashboard_graph_explore(
    tenant_id: str = Query(..., description="Tenant ID"),
    entity: str = Query(..., description="Center entity name"),
    scope_id: str | None = Query(None, description="Scope ID (defaults to tenant_id)"),
    depth: int = Query(2, ge=1, le=5),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Explore the neighborhood of a specific entity in the knowledge graph. Specify tenant, entity name, scope, and depth (1-5). Returns nodes and edges for visualization."""
    if not db.neo4j_driver:
        return GraphExploreResponse(center_entity=entity)
    scope_id = scope_id or tenant_id
    try:
        async with db.neo4j_session() as session:
            # Get nodes + edges in one query
            query = f"""
            MATCH path = (start:Entity {{
                tenant_id: $tenant_id, scope_id: $scope_id, entity: $entity
            }})-[*1..{min(depth, 5)}]-(neighbor:Entity)
            WHERE neighbor.tenant_id = $tenant_id AND neighbor.scope_id = $scope_id
            UNWIND relationships(path) AS rel
            WITH DISTINCT neighbor, rel, startNode(rel) AS sn, endNode(rel) AS en
            RETURN
                collect(DISTINCT {{
                    entity: neighbor.entity,
                    entity_type: neighbor.entity_type,
                    properties: properties(neighbor)
                }}) AS neighbors,
                collect(DISTINCT {{
                    source: sn.entity,
                    target: en.entity,
                    predicate: type(rel),
                    confidence: coalesce(rel.confidence, 0),
                    properties: properties(rel)
                }}) AS rels
            """
            result = await session.run(query, tenant_id=tenant_id, scope_id=scope_id, entity=entity)
            record = await result.single()

            nodes: list[GraphNodeInfo] = [
                GraphNodeInfo(id=entity, entity=entity, entity_type="center")
            ]
            edges: list[GraphEdgeInfo] = []
            seen_nodes = {entity}

            if record:
                for n in record["neighbors"] or []:
                    ent = n.get("entity", "")
                    if ent and ent not in seen_nodes:
                        seen_nodes.add(ent)
                        props = dict(n.get("properties") or {})
                        props.pop("tenant_id", None)
                        props.pop("scope_id", None)
                        nodes.append(
                            GraphNodeInfo(
                                id=ent,
                                entity=ent,
                                entity_type=n.get("entity_type", "unknown"),
                                properties=props,
                            )
                        )
                seen_edges = set()
                for r in record["rels"] or []:
                    src = r.get("source", "")
                    tgt = r.get("target", "")
                    pred = r.get("predicate", "RELATED_TO")
                    edge_key = f"{src}-{pred}-{tgt}"
                    if edge_key not in seen_edges:
                        seen_edges.add(edge_key)
                        props = dict(r.get("properties") or {})
                        props.pop("created_at", None)
                        props.pop("updated_at", None)
                        edges.append(
                            GraphEdgeInfo(
                                source=src,
                                target=tgt,
                                predicate=pred,
                                confidence=float(r.get("confidence", 0)),
                                properties=props,
                            )
                        )

        return GraphExploreResponse(nodes=nodes, edges=edges, center_entity=entity)
    except Exception as e:
        logger.error("dashboard_graph_explore_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@dashboard_router.get("/graph/search", response_model=GraphSearchResponse)
async def dashboard_graph_search(
    query: str = Query(..., min_length=1, description="Entity name pattern"),
    tenant_id: str | None = Query(None),
    limit: int = Query(25, ge=1, le=100),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Search entities in the knowledge graph by name pattern (case-insensitive contains). Returns matching entities with type and tenant."""
    if not db.neo4j_driver:
        return GraphSearchResponse()
    try:
        async with db.neo4j_session() as session:
            cypher = """
            MATCH (n:Entity)
            WHERE toLower(n.entity) CONTAINS toLower($pattern)
            """
            params: dict[str, Any] = {"pattern": query, "lim": limit}
            if tenant_id:
                cypher += " AND n.tenant_id = $tenant_id"
                params["tenant_id"] = tenant_id
            cypher += " RETURN n.entity AS entity, n.entity_type AS entity_type, n.tenant_id AS tid, n.scope_id AS sid LIMIT $lim"
            result = await session.run(cypher, **params)
            results = [
                GraphSearchResult(
                    entity=rec["entity"],
                    entity_type=rec["entity_type"] or "",
                    tenant_id=rec["tid"] or "",
                    scope_id=rec["sid"] or "",
                )
                async for rec in result
            ]
        return GraphSearchResponse(results=results)
    except Exception as e:
        logger.error("dashboard_graph_search_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _is_secret(field_name: str) -> bool:
    lower = field_name.lower()
    return any(tok in lower for tok in _SECRET_FIELD_TOKENS)


def _mask_value(value: Any) -> str:
    return "****" if value else ""


@dashboard_router.get("/config", response_model=DashboardConfigResponse)
async def dashboard_config(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Read-only configuration snapshot. Secrets are masked. Includes app, embedding, LLM, and storage settings."""
    settings = get_settings()

    # Load overrides from Redis
    overrides: dict[str, Any] = {}
    if db.redis:
        try:
            raw = await db.redis.get(_CONFIG_OVERRIDES_KEY)
            if raw:
                overrides = json.loads(raw.decode() if isinstance(raw, bytes) else raw)
        except Exception:
            pass

    sections: list[ConfigSection] = []

    # Application
    app_items = [
        ConfigItem(
            key="app_name",
            value=settings.app_name,
            default_value="CognitiveMemoryLayer",
            is_editable=True,
            source="env" if os.environ.get("APP_NAME") else "default",
            description="Application name",
        ),
        ConfigItem(
            key="debug",
            value=settings.debug,
            default_value=False,
            is_editable=True,
            source="env" if os.environ.get("DEBUG") else "default",
            description="Debug mode",
        ),
        ConfigItem(
            key="cors_origins",
            value=settings.cors_origins,
            default_value=None,
            is_editable=False,
            description="CORS allowed origins",
        ),
    ]
    sections.append(ConfigSection(name="Application", items=app_items))

    # Database
    db_s = settings.database
    db_items = [
        ConfigItem(
            key="database.postgres_url",
            value=(
                _mask_value(db_s.postgres_url) if _is_secret("postgres_url") else db_s.postgres_url
            ),
            default_value="postgresql+asyncpg://memory:memory@localhost/memory",
            is_secret=False,
            description="PostgreSQL connection URL",
        ),
        ConfigItem(
            key="database.neo4j_url",
            value=db_s.neo4j_url,
            default_value="bolt://localhost:7687",
            description="Neo4j connection URL",
        ),
        ConfigItem(
            key="database.neo4j_user",
            value=db_s.neo4j_user,
            default_value="neo4j",
            description="Neo4j username",
        ),
        ConfigItem(
            key="database.neo4j_password",
            value="****" if db_s.neo4j_password else "",
            default_value="",
            is_secret=True,
            description="Neo4j password",
        ),
        ConfigItem(
            key="database.redis_url",
            value=db_s.redis_url,
            default_value="redis://localhost:6379",
            description="Redis connection URL",
        ),
    ]
    sections.append(ConfigSection(name="Database", items=db_items))

    # Embedding
    emb = settings.embedding
    emb_items = [
        ConfigItem(
            key="embedding.provider",
            value=emb.provider,
            default_value="openai",
            is_editable=True,
            description="Embedding provider (openai, local, openai_compatible, ollama)",
        ),
        ConfigItem(
            key="embedding.model",
            value=emb.model,
            default_value="text-embedding-3-small",
            is_editable=True,
            description="Embedding model name",
        ),
        ConfigItem(
            key="embedding.dimensions",
            value=emb.dimensions,
            default_value=1536,
            is_editable=True,
            description="Embedding vector dimensions",
        ),
        ConfigItem(
            key="embedding.local_model",
            value=emb.local_model,
            default_value="all-MiniLM-L6-v2",
            is_editable=True,
            description="Local embedding model name",
        ),
        ConfigItem(
            key="embedding.api_key",
            value="****" if emb.api_key else "",
            default_value="",
            is_secret=True,
            description="Embedding API key",
        ),
        ConfigItem(
            key="embedding.base_url",
            value=emb.base_url or "",
            default_value="",
            description="Custom embedding endpoint URL",
        ),
    ]
    sections.append(ConfigSection(name="Embedding", items=emb_items))

    # LLM
    llm = settings.llm
    llm_items = [
        ConfigItem(
            key="llm.provider",
            value=llm.provider,
            default_value="openai",
            is_editable=True,
            description="LLM provider (openai, openai_compatible, ollama, gemini, claude)",
        ),
        ConfigItem(
            key="llm.model",
            value=llm.model,
            default_value="gpt-4o-mini",
            is_editable=True,
            description="LLM model name",
        ),
        ConfigItem(
            key="llm.api_key",
            value="****" if llm.api_key else "",
            default_value="",
            is_secret=True,
            description="LLM API key",
        ),
        ConfigItem(
            key="llm.base_url",
            value=llm.base_url or "",
            default_value="",
            description="Custom LLM endpoint URL",
        ),
    ]
    sections.append(ConfigSection(name="LLM", items=llm_items))

    # Auth
    auth_s = settings.auth
    auth_items = [
        ConfigItem(
            key="auth.api_key",
            value="****" if auth_s.api_key else "",
            default_value="",
            is_secret=True,
            description="Standard API key",
        ),
        ConfigItem(
            key="auth.admin_api_key",
            value="****" if auth_s.admin_api_key else "",
            default_value="",
            is_secret=True,
            description="Admin API key",
        ),
        ConfigItem(
            key="auth.default_tenant_id",
            value=auth_s.default_tenant_id,
            default_value="default",
            description="Default tenant ID",
        ),
        ConfigItem(
            key="auth.rate_limit_requests_per_minute",
            value=auth_s.rate_limit_requests_per_minute,
            default_value=60,
            is_editable=True,
            description="Rate limit (requests per minute, 0=disabled)",
        ),
    ]
    sections.append(ConfigSection(name="Auth", items=auth_items))

    # Apply override source labels
    for section in sections:
        for item in section.items:
            if item.key in overrides:
                item.source = "override"
                if not item.is_secret:
                    item.value = overrides[item.key]

    return DashboardConfigResponse(sections=sections)


@dashboard_router.put("/config")
async def dashboard_config_update(
    body: ConfigUpdateRequest,
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Update editable config settings at runtime. Stored in Redis (non-persistent across restarts). Only non-secret, non-connection fields."""
    if not db.redis:
        raise HTTPException(
            status_code=503, detail="Redis not available; cannot store config overrides"
        )

    # Validate only editable keys
    for key in body.updates:
        if key not in _EDITABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"Setting '{key}' is not editable")

    # Load existing overrides
    try:
        raw = await db.redis.get(_CONFIG_OVERRIDES_KEY)
        overrides = json.loads(raw.decode() if isinstance(raw, bytes) else raw) if raw else {}
    except Exception:
        overrides = {}

    overrides.update(body.updates)
    await db.redis.set(_CONFIG_OVERRIDES_KEY, json.dumps(overrides))
    return {"success": True, "overrides": overrides}


# ---------------------------------------------------------------------------
# Labile / Reconsolidation
# ---------------------------------------------------------------------------


@dashboard_router.get("/labile", response_model=DashboardLabileResponse)
async def dashboard_labile(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Labile (short-term) memory overview: DB labile counts, Redis scope and session counts. Optionally filter by tenant."""
    try:
        # DB labile counts
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

        # Redis labile scopes
        redis_scope_map: dict[str, dict[str, int]] = {}  # tenant -> {scopes, sessions, memories}
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

                    # Count sessions in this scope
                    scope_sessions = await db.redis.lrange(key_str, 0, -1)
                    session_count = len(scope_sessions)
                    redis_scope_map[tid]["sessions"] += session_count

                    # Count memories across sessions
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


# ---------------------------------------------------------------------------
# Retrieval / Read Test
# ---------------------------------------------------------------------------


@dashboard_router.post("/retrieval", response_model=DashboardRetrievalResponse)
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
                from ..retrieval.packet_builder import MemoryPacketBuilder

                builder = MemoryPacketBuilder()
                llm_context = builder.to_llm_context(packet, max_tokens=2000)
            except Exception:
                pass

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


# ---------------------------------------------------------------------------
# Job History
# ---------------------------------------------------------------------------


@dashboard_router.get("/jobs", response_model=DashboardJobsResponse)
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
                        id=j.id,
                        job_type=j.job_type,
                        tenant_id=j.tenant_id,
                        user_id=j.user_id,
                        dry_run=j.dry_run or False,
                        status=j.status,
                        result=j.result,
                        error=j.error,
                        started_at=j.started_at,
                        completed_at=j.completed_at,
                        duration_seconds=duration,
                    )
                )
            return DashboardJobsResponse(items=items, total=total)
    except Exception as e:
        logger.error("dashboard_jobs_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


# ---------------------------------------------------------------------------
# Management: Consolidation (with job tracking)
# ---------------------------------------------------------------------------


@dashboard_router.post("/consolidate")
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

    # Record job start
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
    except Exception:
        pass  # Non-critical: proceed even if job tracking fails

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
        # Update job record
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
        except Exception:
            pass
        return result_data
    except Exception as e:
        # Update job record with error
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
        except Exception:
            pass
        logger.error("dashboard_consolidate_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


# ---------------------------------------------------------------------------
# Management: Forgetting (with job tracking)
# ---------------------------------------------------------------------------


@dashboard_router.post("/forget")
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

    # Record job start
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
    except Exception:
        pass

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
        except Exception:
            pass
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
        except Exception:
            pass
        logger.error("dashboard_forget_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Forgetting failed: {e}")


# ---------------------------------------------------------------------------
# Management: Reconsolidation (release labile state)
# ---------------------------------------------------------------------------


@dashboard_router.post("/reconsolidate")
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
    except Exception:
        pass

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
        except Exception:
            pass
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
        except Exception:
            pass
        logger.error("dashboard_reconsolidate_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Reconsolidation failed: {e}")


# ---------------------------------------------------------------------------
# Management: Database reset
# ---------------------------------------------------------------------------


def _project_root() -> Path:
    """Project root (directory containing alembic.ini and migrations/)."""
    return Path(__file__).resolve().parent.parent.parent


@dashboard_router.post("/database/reset")
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


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------


@dashboard_router.get("/export/memories")
async def dashboard_export_memories(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Export memories as JSON for download. Optionally filter by tenant. Returns a streaming response."""
    try:
        async with db.pg_session() as session:
            q = select(MemoryRecordModel).order_by(MemoryRecordModel.timestamp.desc()).limit(10000)
            if tenant_id:
                q = q.where(MemoryRecordModel.tenant_id == tenant_id)
            rows = (await session.execute(q)).scalars().all()
            data = [
                {
                    "id": str(r.id),
                    "tenant_id": r.tenant_id,
                    "type": r.type,
                    "status": r.status,
                    "text": r.text,
                    "key": r.key,
                    "confidence": r.confidence,
                    "importance": r.importance,
                    "access_count": r.access_count,
                    "timestamp": r.timestamp.isoformat() if r.timestamp else None,
                }
                for r in rows
            ]
            from starlette.responses import JSONResponse

            return JSONResponse(
                content=data,
                headers={"Content-Disposition": "attachment; filename=memories_export.json"},
            )
    except Exception as e:
        logger.error("dashboard_export_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))
