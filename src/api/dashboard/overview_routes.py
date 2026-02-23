"""Dashboard overview, timeline, components, tenants, sessions, rate limits, request stats."""

import json
import time
from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, text

from ...core.config import get_embedding_dimensions, get_settings
from ...storage.models import EventLogModel, MemoryRecordModel, SemanticFactModel
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    _REQUEST_COUNT_PREFIX,
    ComponentStatus,
    DashboardComponentsResponse,
    DashboardOverview,
    DashboardRateLimitsResponse,
    DashboardSessionsResponse,
    DashboardTenantsResponse,
    DashboardTimelineResponse,
    HourlyRequestCount,
    RateLimitEntry,
    RequestStatsResponse,
    SessionInfo,
    TenantInfo,
    TimelinePoint,
    _get_db,
    logger,
)

router = APIRouter()


@router.get("/overview", response_model=DashboardOverview)
async def dashboard_overview(
    tenant_id: str | None = Query(None, description="Filter by tenant (omit for all)"),
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
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


@router.get("/timeline", response_model=DashboardTimelineResponse)
async def dashboard_timeline(
    days: int = Query(30, ge=1, le=365),
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
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


@router.get("/components", response_model=DashboardComponentsResponse)
async def dashboard_components(
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
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
        components.append(
            ComponentStatus(
                name="PostgreSQL",
                status="ok",
                latency_ms=round(latency, 2),
                details={
                    "memory_records": mem_count,
                    "semantic_facts": fact_count,
                    "events": event_count,
                    "embedding_dimensions": get_embedding_dimensions(),
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


@router.get("/tenants", response_model=DashboardTenantsResponse)
async def dashboard_tenants(
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
):
    """List all tenants with memory counts, active session counts, and last activity. Used for tenant selector in dashboard."""
    try:
        async with db.pg_session() as session:
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

            fact_q = select(SemanticFactModel.tenant_id, func.count().label("fact_count")).group_by(
                SemanticFactModel.tenant_id
            )
            fact_rows = (await session.execute(fact_q)).all()
            fact_map = {r[0]: r[1] for r in fact_rows}

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


@router.get("/sessions", response_model=DashboardSessionsResponse)
async def dashboard_sessions(
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
):
    """List active sessions from Redis with memory counts per source_session_id. Optionally filter by tenant."""
    try:
        sessions_list: list[SessionInfo] = []
        redis_sessions: dict[str, SessionInfo] = {}

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
                    if tenant_id and info.tenant_id and info.tenant_id != tenant_id:
                        continue
                    redis_sessions[sid] = info
                if cursor == 0:
                    break

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


@router.get("/ratelimits", response_model=DashboardRateLimitsResponse)
async def dashboard_ratelimits(
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
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


@router.get("/request-stats", response_model=RequestStatsResponse)
async def dashboard_request_stats(
    hours: int = Query(24, ge=1, le=48),
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
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
