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
from ..core.env_file import get_env_path, update_env
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
    GraphNeo4jConfigResponse,
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

# Settings that admins may edit (non-secret). Persisted to .env.
_EDITABLE_SETTINGS = frozenset(
    {
        "app_name",
        "debug",
        "cors_origins",
        "embedding.provider",
        "embedding.model",
        "embedding.dimensions",
        "embedding.local_model",
        "embedding.base_url",
        "llm.provider",
        "llm.model",
        "llm.base_url",
        "llm_internal.provider",
        "llm_internal.model",
        "llm_internal.base_url",
        "auth.default_tenant_id",
        "auth.rate_limit_requests_per_minute",
        "chunker.tokenizer",
        "chunker.chunk_size",
        "chunker.overlap_percent",
        "retrieval.episode_relevance_threshold",
        "retrieval.max_episodes_when_constraints",
        "retrieval.max_episodes_default",
        "retrieval.max_constraint_tokens",
        "retrieval.default_step_timeout_ms",
        "retrieval.total_timeout_ms",
        "retrieval.graph_timeout_ms",
        "retrieval.fact_timeout_ms",
        "retrieval.hnsw_ef_search",
        "retrieval.reranker.recency_weight",
        "retrieval.reranker.relevance_weight",
        "retrieval.reranker.confidence_weight",
        "retrieval.reranker.active_constraint_bonus",
        "features.stable_keys_enabled",
        "features.write_time_facts_enabled",
        "features.batch_embeddings_enabled",
        "features.store_async",
        "features.cached_embeddings_enabled",
        "features.retrieval_timeouts_enabled",
        "features.skip_if_found_cross_group",
        "features.db_dependency_counts",
        "features.bounded_state_enabled",
        "features.hnsw_ef_search_tuning",
        "features.constraint_extraction_enabled",
        "features.use_llm_constraint_extractor",
        "features.use_llm_write_time_facts",
        "features.use_llm_query_classifier_only",
        "features.use_llm_salience_refinement",
        "features.use_llm_pii_redaction",
        "features.use_llm_write_gate_importance",
        "features.use_llm_memory_type",
        "features.use_llm_confidence",
        "features.use_llm_context_tags",
        "features.use_llm_decay_rate",
        "features.use_llm_conflict_detection_only",
        "features.use_llm_constraint_reranker",
    }
)

# Config key -> env var for .env persistence
_CONFIG_KEY_TO_ENV: dict[str, str] = {
    "app_name": "APP_NAME",
    "debug": "DEBUG",
    "cors_origins": "CORS_ORIGINS",
    "database.postgres_url": "DATABASE__POSTGRES_URL",
    "database.neo4j_url": "DATABASE__NEO4J_URL",
    "database.neo4j_browser_url": "DATABASE__NEO4J_BROWSER_URL",
    "database.neo4j_user": "DATABASE__NEO4J_USER",
    "database.neo4j_password": "DATABASE__NEO4J_PASSWORD",
    "database.redis_url": "DATABASE__REDIS_URL",
    "embedding.provider": "EMBEDDING__PROVIDER",
    "embedding.model": "EMBEDDING__MODEL",
    "embedding.dimensions": "EMBEDDING__DIMENSIONS",
    "embedding.local_model": "EMBEDDING__LOCAL_MODEL",
    "embedding.api_key": "EMBEDDING__API_KEY",
    "embedding.base_url": "EMBEDDING__BASE_URL",
    "llm.provider": "LLM__PROVIDER",
    "llm.model": "LLM__MODEL",
    "llm.api_key": "LLM__API_KEY",
    "llm.base_url": "LLM__BASE_URL",
    "llm_internal.provider": "LLM_INTERNAL__PROVIDER",
    "llm_internal.model": "LLM_INTERNAL__MODEL",
    "llm_internal.base_url": "LLM_INTERNAL__BASE_URL",
    "llm_internal.api_key": "LLM_INTERNAL__API_KEY",
    "auth.api_key": "AUTH__API_KEY",
    "auth.admin_api_key": "AUTH__ADMIN_API_KEY",
    "auth.default_tenant_id": "AUTH__DEFAULT_TENANT_ID",
    "auth.rate_limit_requests_per_minute": "AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE",
    "chunker.tokenizer": "CHUNKER__TOKENIZER",
    "chunker.chunk_size": "CHUNKER__CHUNK_SIZE",
    "chunker.overlap_percent": "CHUNKER__OVERLAP_PERCENT",
    "retrieval.episode_relevance_threshold": "RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD",
    "retrieval.max_episodes_when_constraints": "RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS",
    "retrieval.max_episodes_default": "RETRIEVAL__MAX_EPISODES_DEFAULT",
    "retrieval.max_constraint_tokens": "RETRIEVAL__MAX_CONSTRAINT_TOKENS",
    "retrieval.default_step_timeout_ms": "RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS",
    "retrieval.total_timeout_ms": "RETRIEVAL__TOTAL_TIMEOUT_MS",
    "retrieval.graph_timeout_ms": "RETRIEVAL__GRAPH_TIMEOUT_MS",
    "retrieval.fact_timeout_ms": "RETRIEVAL__FACT_TIMEOUT_MS",
    "retrieval.hnsw_ef_search": "RETRIEVAL__HNSW_EF_SEARCH",
    "retrieval.reranker.recency_weight": "RETRIEVAL__RERANKER__RECENCY_WEIGHT",
    "retrieval.reranker.relevance_weight": "RETRIEVAL__RERANKER__RELEVANCE_WEIGHT",
    "retrieval.reranker.confidence_weight": "RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT",
    "retrieval.reranker.active_constraint_bonus": "RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS",
}
for _fk in (
    "stable_keys_enabled",
    "write_time_facts_enabled",
    "batch_embeddings_enabled",
    "store_async",
    "cached_embeddings_enabled",
    "retrieval_timeouts_enabled",
    "skip_if_found_cross_group",
    "db_dependency_counts",
    "bounded_state_enabled",
    "hnsw_ef_search_tuning",
    "constraint_extraction_enabled",
    "use_llm_constraint_extractor",
    "use_llm_write_time_facts",
    "use_llm_query_classifier_only",
    "use_llm_salience_refinement",
    "use_llm_pii_redaction",
    "use_llm_write_gate_importance",
    "use_llm_memory_type",
    "use_llm_confidence",
    "use_llm_context_tags",
    "use_llm_decay_rate",
    "use_llm_conflict_detection_only",
    "use_llm_constraint_reranker",
):
    _CONFIG_KEY_TO_ENV[f"features.{_fk}"] = f"FEATURES__{_fk.upper()}"

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


@dashboard_router.get("/graph/neo4j-config", response_model=GraphNeo4jConfigResponse)
async def dashboard_graph_neo4j_config(
    auth: AuthContext = Depends(require_admin_permission),
):
    """Return Neo4j connection config for browser (neovis.js). Admin-only. Use DATABASE__NEO4J_BROWSER_URL when Neo4j is not reachable at DATABASE__NEO4J_URL from the browser (e.g. Docker: bolt://localhost:7687)."""
    settings = get_settings()
    db = settings.database
    server_url = db.neo4j_browser_url or db.neo4j_url
    return GraphNeo4jConfigResponse(
        server_url=server_url,
        server_user=db.neo4j_user,
        server_password=db.neo4j_password or "",
    )


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _is_secret(field_name: str) -> bool:
    lower = field_name.lower()
    return any(tok in lower for tok in _SECRET_FIELD_TOKENS)


def _mask_value(value: Any) -> str:
    return "****" if value else ""


def _config_source(env_var: str) -> str:
    """Return 'env' if env var is set, else 'default'."""
    return "env" if os.environ.get(env_var) is not None else "default"


@dashboard_router.get("/config", response_model=DashboardConfigResponse)
async def dashboard_config(
    auth: AuthContext = Depends(require_admin_permission),
    db: DatabaseManager = Depends(_get_db),
):
    """Configuration snapshot. Secrets are masked. Changes persist to .env; restart required for most settings."""
    settings = get_settings()
    sections: list[ConfigSection] = []

    # Application
    app_items = [
        ConfigItem(
            key="debug",
            value=settings.app_name,
            default_value="CognitiveMemoryLayer",
            is_editable=True,
            source=_config_source("APP_NAME"),
            description="Application name. Used in responses and UI.",
            requires_restart=True,
            is_required=False,
            env_var="APP_NAME",
        ),
        ConfigItem(
            key="debug",
            value=settings.debug,
            default_value=False,
            is_editable=True,
            source=_config_source("DEBUG"),
            description="Enable debug mode. Exposes verbose 500 error details. Use false in production.",
            requires_restart=True,
            is_required=False,
            env_var="DEBUG",
            options=["true", "false"],
        ),
        ConfigItem(
            key="cors_origins",
            value=settings.cors_origins,
            default_value=None,
            is_editable=True,
            source=_config_source("CORS_ORIGINS"),
            description="Comma-separated CORS allowed origins. Unset uses default list; DEBUG=true allows '*'.",
            requires_restart=True,
            is_required=False,
            env_var="CORS_ORIGINS",
        ),
    ]
    sections.append(ConfigSection(name="Application", items=app_items))

    # Database
    db_s = settings.database
    db_items = [
        ConfigItem(
            key="database.postgres_url",
            value=db_s.postgres_url,
            default_value="postgresql+asyncpg://memory:memory@localhost/memory",
            is_editable=False,
            source=_config_source("DATABASE__POSTGRES_URL"),
            description="PostgreSQL connection URL. Required. Format: postgresql+asyncpg://user:pass@host:port/db",
            requires_restart=True,
            is_required=True,
            env_var="DATABASE__POSTGRES_URL",
        ),
        ConfigItem(
            key="database.neo4j_url",
            value=db_s.neo4j_url,
            default_value="bolt://localhost:7687",
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_URL"),
            description="Neo4j Bolt URL for knowledge graph. Optional.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_URL",
        ),
        ConfigItem(
            key="database.neo4j_user",
            value=db_s.neo4j_user,
            default_value="neo4j",
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_USER"),
            description="Neo4j username. Optional.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_USER",
        ),
        ConfigItem(
            key="database.neo4j_password",
            value="****" if db_s.neo4j_password else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("DATABASE__NEO4J_PASSWORD"),
            description="Neo4j password. Set when Neo4j is secured.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__NEO4J_PASSWORD",
        ),
        ConfigItem(
            key="database.redis_url",
            value=db_s.redis_url,
            default_value="redis://localhost:6379",
            is_editable=False,
            source=_config_source("DATABASE__REDIS_URL"),
            description="Redis URL for cache, Celery broker, rate limits.",
            requires_restart=True,
            is_required=False,
            env_var="DATABASE__REDIS_URL",
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
            source=_config_source("EMBEDDING__PROVIDER"),
            description="Provider: openai | local | openai_compatible | ollama. Affects vector generation.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING__PROVIDER",
            options=["openai", "local", "openai_compatible", "ollama"],
        ),
        ConfigItem(
            key="embedding.model",
            value=emb.model,
            default_value="text-embedding-3-small",
            is_editable=True,
            source=_config_source("EMBEDDING__MODEL"),
            description="Model name. Must match provider. Changing impacts all new embeddings.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING__MODEL",
        ),
        ConfigItem(
            key="embedding.dimensions",
            value=emb.dimensions,
            default_value=1536,
            is_editable=True,
            source=_config_source("EMBEDDING__DIMENSIONS"),
            description="Vector dimension. Must match model and DB schema. Changing requires migration.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING__DIMENSIONS",
        ),
        ConfigItem(
            key="embedding.local_model",
            value=emb.local_model,
            default_value="all-MiniLM-L6-v2",
            is_editable=True,
            source=_config_source("EMBEDDING__LOCAL_MODEL"),
            description="Model for provider=local (sentence-transformers).",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING__LOCAL_MODEL",
        ),
        ConfigItem(
            key="embedding.api_key",
            value="****" if emb.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("EMBEDDING__API_KEY"),
            description="API key for embedding provider. Fallback: OPENAI_API_KEY.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING__API_KEY",
        ),
        ConfigItem(
            key="embedding.base_url",
            value=emb.base_url or "",
            default_value="",
            is_editable=True,
            source=_config_source("EMBEDDING__BASE_URL"),
            description="OpenAI-compatible endpoint URL. For ollama/openai_compatible.",
            requires_restart=True,
            is_required=False,
            env_var="EMBEDDING__BASE_URL",
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
            source=_config_source("LLM__PROVIDER"),
            description="Provider: openai | openai_compatible | ollama | gemini | claude.",
            requires_restart=True,
            is_required=False,
            env_var="LLM__PROVIDER",
            options=["openai", "openai_compatible", "ollama", "gemini", "claude"],
        ),
        ConfigItem(
            key="llm.model",
            value=llm.model,
            default_value="gpt-4o-mini",
            is_editable=True,
            source=_config_source("LLM__MODEL"),
            description="Model name. Used for consolidation, classifier, conflict detection.",
            requires_restart=True,
            is_required=False,
            env_var="LLM__MODEL",
        ),
        ConfigItem(
            key="llm.api_key",
            value="****" if llm.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("LLM__API_KEY"),
            description="LLM API key. Fallback: OPENAI_API_KEY.",
            requires_restart=True,
            is_required=False,
            env_var="LLM__API_KEY",
        ),
        ConfigItem(
            key="llm.base_url",
            value=llm.base_url or "",
            default_value="",
            is_editable=True,
            source=_config_source("LLM__BASE_URL"),
            description="OpenAI-compatible endpoint. For ollama/openai_compatible.",
            requires_restart=True,
            is_required=False,
            env_var="LLM__BASE_URL",
        ),
    ]
    sections.append(ConfigSection(name="LLM", items=llm_items))

    # LLM Internal
    llmi = settings.llm_internal
    llmi_items = [
        ConfigItem(
            key="llm_internal.provider",
            value=llmi.provider or "(uses llm)",
            default_value=None,
            is_editable=True,
            source=_config_source("LLM_INTERNAL__PROVIDER"),
            description="Separate LLM for internal tasks (~1-10 calls/turn). Unset uses LLM__*.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__PROVIDER",
            options=["(uses llm)", "openai", "openai_compatible", "ollama", "gemini", "claude"],
        ),
        ConfigItem(
            key="llm_internal.model",
            value=llmi.model or "(uses llm)",
            default_value=None,
            is_editable=True,
            source=_config_source("LLM_INTERNAL__MODEL"),
            description="Model for internal LLM.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__MODEL",
        ),
        ConfigItem(
            key="llm_internal.base_url",
            value=llmi.base_url or "",
            default_value="",
            is_editable=True,
            source=_config_source("LLM_INTERNAL__BASE_URL"),
            description="Base URL for internal LLM.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__BASE_URL",
        ),
        ConfigItem(
            key="llm_internal.api_key",
            value="****" if llmi.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("LLM_INTERNAL__API_KEY"),
            description="API key for internal LLM.",
            requires_restart=True,
            is_required=False,
            env_var="LLM_INTERNAL__API_KEY",
        ),
    ]
    sections.append(ConfigSection(name="LLM Internal", items=llmi_items))

    # Auth
    auth_s = settings.auth
    auth_items = [
        ConfigItem(
            key="auth.api_key",
            value="****" if auth_s.api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("AUTH__API_KEY"),
            description="API key for memory operations (X-API-Key). Required.",
            requires_restart=True,
            is_required=True,
            env_var="AUTH__API_KEY",
        ),
        ConfigItem(
            key="auth.admin_api_key",
            value="****" if auth_s.admin_api_key else "",
            default_value="",
            is_secret=True,
            is_editable=False,
            source=_config_source("AUTH__ADMIN_API_KEY"),
            description="Admin API key for dashboard, consolidate, run_forgetting.",
            requires_restart=True,
            is_required=False,
            env_var="AUTH__ADMIN_API_KEY",
        ),
        ConfigItem(
            key="auth.default_tenant_id",
            value=auth_s.default_tenant_id,
            default_value="default",
            is_editable=True,
            source=_config_source("AUTH__DEFAULT_TENANT_ID"),
            description="Default tenant when X-Tenant-ID not sent.",
            requires_restart=True,
            is_required=False,
            env_var="AUTH__DEFAULT_TENANT_ID",
        ),
        ConfigItem(
            key="auth.rate_limit_requests_per_minute",
            value=auth_s.rate_limit_requests_per_minute,
            default_value=60,
            is_editable=True,
            source=_config_source("AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE"),
            description="Per-tenant rate limit. 0=disabled. Increase for bulk eval.",
            requires_restart=False,
            is_required=False,
            env_var="AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE",
        ),
    ]
    sections.append(ConfigSection(name="Auth", items=auth_items))

    # Chunker
    chk = settings.chunker
    chk_items = [
        ConfigItem(
            key="chunker.tokenizer",
            value=chk.tokenizer,
            default_value="google/flan-t5-base",
            is_editable=True,
            source=_config_source("CHUNKER__TOKENIZER"),
            description="Hugging Face tokenizer ID for semchunk.",
            requires_restart=True,
            is_required=False,
            env_var="CHUNKER__TOKENIZER",
        ),
        ConfigItem(
            key="chunker.chunk_size",
            value=chk.chunk_size,
            default_value=500,
            is_editable=True,
            source=_config_source("CHUNKER__CHUNK_SIZE"),
            description="Max tokens per chunk. Align with embedding model max input.",
            requires_restart=True,
            is_required=False,
            env_var="CHUNKER__CHUNK_SIZE",
        ),
        ConfigItem(
            key="chunker.overlap_percent",
            value=chk.overlap_percent,
            default_value=0.15,
            is_editable=True,
            source=_config_source("CHUNKER__OVERLAP_PERCENT"),
            description="Chunk overlap ratio 0-1 (e.g. 0.15 = 15%).",
            requires_restart=True,
            is_required=False,
            env_var="CHUNKER__OVERLAP_PERCENT",
        ),
    ]
    sections.append(ConfigSection(name="Chunker", items=chk_items))

    # Retrieval
    ret = settings.retrieval
    rer = ret.reranker
    ret_items = [
        ConfigItem(
            key="retrieval.episode_relevance_threshold",
            value=ret.episode_relevance_threshold,
            default_value=0.5,
            is_editable=True,
            source=_config_source("RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD"),
            description="Min relevance for episodes in context.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__EPISODE_RELEVANCE_THRESHOLD",
        ),
        ConfigItem(
            key="retrieval.max_episodes_when_constraints",
            value=ret.max_episodes_when_constraints,
            default_value=3,
            is_editable=True,
            source=_config_source("RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS"),
            description="Max episodes when constraints exist.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__MAX_EPISODES_WHEN_CONSTRAINTS",
        ),
        ConfigItem(
            key="retrieval.max_episodes_default",
            value=ret.max_episodes_default,
            default_value=5,
            is_editable=True,
            source=_config_source("RETRIEVAL__MAX_EPISODES_DEFAULT"),
            description="Max episodes when no constraints.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__MAX_EPISODES_DEFAULT",
        ),
        ConfigItem(
            key="retrieval.max_constraint_tokens",
            value=ret.max_constraint_tokens,
            default_value=400,
            is_editable=True,
            source=_config_source("RETRIEVAL__MAX_CONSTRAINT_TOKENS"),
            description="Token budget for active constraints.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__MAX_CONSTRAINT_TOKENS",
        ),
        ConfigItem(
            key="retrieval.default_step_timeout_ms",
            value=ret.default_step_timeout_ms,
            default_value=500,
            is_editable=True,
            source=_config_source("RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS"),
            description="Per-step timeout (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__DEFAULT_STEP_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.total_timeout_ms",
            value=ret.total_timeout_ms,
            default_value=2000,
            is_editable=True,
            source=_config_source("RETRIEVAL__TOTAL_TIMEOUT_MS"),
            description="Total retrieval budget (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__TOTAL_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.graph_timeout_ms",
            value=ret.graph_timeout_ms,
            default_value=1000,
            is_editable=True,
            source=_config_source("RETRIEVAL__GRAPH_TIMEOUT_MS"),
            description="Graph step timeout (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__GRAPH_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.fact_timeout_ms",
            value=ret.fact_timeout_ms,
            default_value=200,
            is_editable=True,
            source=_config_source("RETRIEVAL__FACT_TIMEOUT_MS"),
            description="Fact lookup timeout (ms).",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__FACT_TIMEOUT_MS",
        ),
        ConfigItem(
            key="retrieval.hnsw_ef_search",
            value=ret.hnsw_ef_search,
            default_value=64,
            is_editable=True,
            source=_config_source("RETRIEVAL__HNSW_EF_SEARCH"),
            description="pgvector HNSW ef_search. Higher = recall, slower.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__HNSW_EF_SEARCH",
        ),
        ConfigItem(
            key="retrieval.reranker.recency_weight",
            value=rer.recency_weight,
            default_value=0.1,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__RECENCY_WEIGHT"),
            description="Recency weight in reranking.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__RECENCY_WEIGHT",
        ),
        ConfigItem(
            key="retrieval.reranker.relevance_weight",
            value=rer.relevance_weight,
            default_value=0.5,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__RELEVANCE_WEIGHT"),
            description="Relevance weight in reranking.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__RELEVANCE_WEIGHT",
        ),
        ConfigItem(
            key="retrieval.reranker.confidence_weight",
            value=rer.confidence_weight,
            default_value=0.2,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT"),
            description="Confidence weight in reranking.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__CONFIDENCE_WEIGHT",
        ),
        ConfigItem(
            key="retrieval.reranker.active_constraint_bonus",
            value=rer.active_constraint_bonus,
            default_value=0.2,
            is_editable=True,
            source=_config_source("RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS"),
            description="Bonus for active constraints.",
            requires_restart=True,
            is_required=False,
            env_var="RETRIEVAL__RERANKER__ACTIVE_CONSTRAINT_BONUS",
        ),
    ]
    sections.append(ConfigSection(name="Retrieval", items=ret_items))

    # Features
    feat = settings.features
    feat_items = [
        ConfigItem(
            key="features.stable_keys_enabled",
            value=feat.stable_keys_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__STABLE_KEYS_ENABLED"),
            description="SHA256-based stable keys for consolidation.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__STABLE_KEYS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.write_time_facts_enabled",
            value=feat.write_time_facts_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__WRITE_TIME_FACTS_ENABLED"),
            description="Populate semantic store at write time.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__WRITE_TIME_FACTS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.batch_embeddings_enabled",
            value=feat.batch_embeddings_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__BATCH_EMBEDDINGS_ENABLED"),
            description="Single embed_batch() per turn.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__BATCH_EMBEDDINGS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.store_async",
            value=feat.store_async,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__STORE_ASYNC"),
            description="Enqueue turn writes to Redis. Reduces latency; requires Redis.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__STORE_ASYNC",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.cached_embeddings_enabled",
            value=feat.cached_embeddings_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__CACHED_EMBEDDINGS_ENABLED"),
            description="Cache embeddings in Redis.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__CACHED_EMBEDDINGS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.retrieval_timeouts_enabled",
            value=feat.retrieval_timeouts_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__RETRIEVAL_TIMEOUTS_ENABLED"),
            description="Per-step asyncio.wait_for timeouts.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__RETRIEVAL_TIMEOUTS_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.skip_if_found_cross_group",
            value=feat.skip_if_found_cross_group,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__SKIP_IF_FOUND_CROSS_GROUP"),
            description="Skip remaining steps on fact hit.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__SKIP_IF_FOUND_CROSS_GROUP",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.db_dependency_counts",
            value=feat.db_dependency_counts,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__DB_DEPENDENCY_COUNTS"),
            description="DB-side aggregation for forgetting.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__DB_DEPENDENCY_COUNTS",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.bounded_state_enabled",
            value=feat.bounded_state_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__BOUNDED_STATE_ENABLED"),
            description="LRU+TTL state maps in working memory.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__BOUNDED_STATE_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.hnsw_ef_search_tuning",
            value=feat.hnsw_ef_search_tuning,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__HNSW_EF_SEARCH_TUNING"),
            description="Query-time HNSW ef_search tuning.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__HNSW_EF_SEARCH_TUNING",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.constraint_extraction_enabled",
            value=feat.constraint_extraction_enabled,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__CONSTRAINT_EXTRACTION_ENABLED"),
            description="Extract goals, values, policies at write time.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__CONSTRAINT_EXTRACTION_ENABLED",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_constraint_extractor",
            value=feat.use_llm_constraint_extractor,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR"),
            description="LLM for constraint extraction (vs rule-based).",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_write_time_facts",
            value=feat.use_llm_write_time_facts,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_WRITE_TIME_FACTS"),
            description="LLM for write-time fact extraction.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_WRITE_TIME_FACTS",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_query_classifier_only",
            value=feat.use_llm_query_classifier_only,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY"),
            description="Always use LLM for query classification; skip fast pattern.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_salience_refinement",
            value=feat.use_llm_salience_refinement,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_SALIENCE_REFINEMENT"),
            description="LLM salience in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_SALIENCE_REFINEMENT",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_pii_redaction",
            value=feat.use_llm_pii_redaction,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_PII_REDACTION"),
            description="LLM PII spans in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_PII_REDACTION",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_write_gate_importance",
            value=feat.use_llm_write_gate_importance,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE"),
            description="LLM importance in write gate.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_WRITE_GATE_IMPORTANCE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_memory_type",
            value=feat.use_llm_memory_type,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_MEMORY_TYPE"),
            description="LLM memory_type in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_MEMORY_TYPE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_confidence",
            value=feat.use_llm_confidence,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONFIDENCE"),
            description="LLM confidence in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONFIDENCE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_context_tags",
            value=feat.use_llm_context_tags,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONTEXT_TAGS"),
            description="LLM context_tags when caller does not provide tags.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONTEXT_TAGS",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_decay_rate",
            value=feat.use_llm_decay_rate,
            default_value=True,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_DECAY_RATE"),
            description="LLM decay_rate in unified extractor.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_DECAY_RATE",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_conflict_detection_only",
            value=feat.use_llm_conflict_detection_only,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY"),
            description="Always use LLM for conflict detection.",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONFLICT_DETECTION_ONLY",
            options=["true", "false"],
        ),
        ConfigItem(
            key="features.use_llm_constraint_reranker",
            value=feat.use_llm_constraint_reranker,
            default_value=False,
            is_editable=True,
            source=_config_source("FEATURES__USE_LLM_CONSTRAINT_RERANKER"),
            description="LLM for constraint reranking (1 call per read).",
            requires_restart=True,
            is_required=False,
            env_var="FEATURES__USE_LLM_CONSTRAINT_RERANKER",
            options=["true", "false"],
        ),
    ]
    sections.append(ConfigSection(name="Features", items=feat_items))

    return DashboardConfigResponse(sections=sections)


def _validate_config_updates(updates: dict[str, Any]) -> list[str]:
    """Validate config updates. Returns list of error messages."""
    errors: list[str] = []
    for key, val in updates.items():
        if key == "embedding.dimensions":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("embedding.dimensions must be a positive integer")
        elif key == "chunker.chunk_size":
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append("chunker.chunk_size must be a positive integer")
        elif key == "chunker.overlap_percent":
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append("chunker.overlap_percent must be between 0 and 1")
        elif key == "auth.rate_limit_requests_per_minute":
            if not isinstance(val, (int, float)) or val < 0:
                errors.append("auth.rate_limit_requests_per_minute must be non-negative")
        elif key.startswith("retrieval.") and "timeout" in key:
            if not isinstance(val, (int, float)) or val <= 0:
                errors.append(f"{key} must be a positive number")
        elif key.startswith("retrieval.reranker."):
            if not isinstance(val, (int, float)) or val < 0 or val > 1:
                errors.append(f"{key} must be between 0 and 1")
    return errors


@dashboard_router.put("/config")
async def dashboard_config_update(
    body: ConfigUpdateRequest,
    auth: AuthContext = Depends(require_admin_permission),
):
    """Update editable config settings. Persisted to .env. Restart required for most changes."""
    for key in body.updates:
        if key not in _EDITABLE_SETTINGS:
            raise HTTPException(status_code=400, detail=f"Setting '{key}' is not editable")

    errors = _validate_config_updates(body.updates)
    if errors:
        raise HTTPException(status_code=400, detail="; ".join(errors))

    env_updates: dict[str, Any] = {}
    for key, val in body.updates.items():
        env_var = _CONFIG_KEY_TO_ENV.get(key)
        if env_var:
            if key == "embedding.dimensions":
                env_updates[env_var] = int(val)
            elif key == "cors_origins":
                if isinstance(val, list):
                    env_updates[env_var] = ",".join(str(v) for v in val)
                elif val is None or val == "":
                    env_updates[env_var] = ""
                else:
                    env_updates[env_var] = str(val)
            else:
                env_updates[env_var] = val

    try:
        env_path = get_env_path()
        if not env_path.parent.exists():
            raise HTTPException(
                status_code=503,
                detail="Project root not found; cannot persist to .env",
            )
        update_env(env_updates)
    except OSError as e:
        logger.warning("config_persist_env_failed", path=str(get_env_path()), error=str(e))
        raise HTTPException(
            status_code=503,
            detail=f"Cannot write to .env: {e}",
        ) from e

    return {
        "success": True,
        "message": "Settings saved to .env. Restart the server for changes to take effect.",
    }


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
