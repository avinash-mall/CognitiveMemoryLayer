"""Dashboard event log routes."""

import math
from datetime import datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy import func, select

from ...storage.models import EventLogModel
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    DashboardEventItem,
    DashboardEventListResponse,
    _get_db,
    logger,
)

router = APIRouter()


@router.get("/events", response_model=DashboardEventListResponse)
async def dashboard_events(
    page: int = Query(1, ge=1),
    per_page: int = Query(25, ge=1, le=200),
    event_type: str | None = Query(None),
    operation: str | None = Query(None),
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
    db=Depends(_get_db),
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
                    id=cast("UUID", e.id),
                    tenant_id=cast("str", e.tenant_id),
                    scope_id=cast("str", e.scope_id),
                    agent_id=cast("str | None", e.agent_id),
                    event_type=cast("str", e.event_type),
                    operation=cast("str | None", e.operation),
                    payload=cast("dict", e.payload),
                    memory_ids=list(cast("list", e.memory_ids) or []),
                    parent_event_id=cast("UUID | None", e.parent_event_id),
                    created_at=cast("datetime", e.created_at),
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
