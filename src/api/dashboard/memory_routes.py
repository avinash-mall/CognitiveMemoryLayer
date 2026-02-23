"""Dashboard memory list, detail, bulk actions, export."""

import math
from datetime import datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select, update

from ...storage.connection import DatabaseManager
from ...storage.models import EventLogModel, MemoryRecordModel
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    BulkActionRequest,
    DashboardMemoryDetail,
    DashboardMemoryListItem,
    DashboardMemoryListResponse,
    _get_db,
    logger,
)

router = APIRouter()


@router.get("/memories", response_model=DashboardMemoryListResponse)
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
                    id=cast("UUID", r.id),
                    tenant_id=cast("str", r.tenant_id),
                    agent_id=cast("str | None", r.agent_id),
                    type=cast("str", r.type),
                    status=cast("str", r.status),
                    text=(cast("str", r.text))[:300] if r.text else "",
                    key=cast("str | None", r.key),
                    namespace=cast("str | None", r.namespace),
                    context_tags=list(cast("list", r.context_tags) or []),
                    confidence=cast("float", r.confidence or 0.5),
                    importance=cast("float", r.importance or 0.5),
                    access_count=cast("int", r.access_count or 0),
                    decay_rate=cast("float", r.decay_rate or 0.01),
                    labile=cast("bool", r.labile or False),
                    version=cast("int", r.version or 1),
                    timestamp=cast("datetime", r.timestamp),
                    written_at=cast("datetime", r.written_at),
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


@router.get("/memories/{memory_id}", response_model=DashboardMemoryDetail)
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
                id=cast("UUID", record.id),
                tenant_id=cast("str", record.tenant_id),
                agent_id=cast("str | None", record.agent_id),
                type=cast("str", record.type),
                status=cast("str", record.status),
                text=cast("str", record.text),
                key=cast("str | None", record.key),
                namespace=cast("str | None", record.namespace),
                context_tags=list(cast("list", record.context_tags) or []),
                source_session_id=cast("str | None", record.source_session_id),
                entities=cast("list", record.entities),
                relations=cast("list", record.relations),
                metadata=cast("dict", record.meta),
                confidence=cast("float", record.confidence or 0.5),
                importance=cast("float", record.importance or 0.5),
                access_count=cast("int", record.access_count or 0),
                last_accessed_at=cast("datetime | None", record.last_accessed_at),
                decay_rate=cast("float", record.decay_rate or 0.01),
                labile=cast("bool", record.labile or False),
                provenance=cast("dict", record.provenance),
                version=cast("int", record.version or 1),
                supersedes_id=cast("UUID | None", record.supersedes_id),
                content_hash=cast("str | None", record.content_hash),
                timestamp=cast("datetime", record.timestamp),
                written_at=cast("datetime", record.written_at),
                valid_from=cast("datetime | None", record.valid_from),
                valid_to=cast("datetime | None", record.valid_to),
                related_events=related_events,
            )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_memory_detail_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/memories/bulk-action")
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
            affected = getattr(result, "rowcount", 0) or 0
            return {"success": True, "affected": affected, "action": body.action}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_bulk_action_error", error=str(e))
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/export/memories")
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
