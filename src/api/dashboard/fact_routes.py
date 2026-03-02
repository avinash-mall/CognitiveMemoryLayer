"""Dashboard semantic facts routes: list, detail, invalidate."""

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from pydantic import BaseModel, Field

from ...storage.connection import DatabaseManager
from ..auth import AuthContext, require_admin_permission
from ._shared import logger

router = APIRouter()


class FactItem(BaseModel):
    """Serialised semantic fact for dashboard display."""

    id: str
    tenant_id: str
    category: str
    key: str
    value: str
    confidence: float
    evidence_count: int
    is_current: bool
    version: int
    created_at: str | None = None
    updated_at: str | None = None


class FactListResponse(BaseModel):
    items: list[FactItem] = Field(default_factory=list)
    total: int = 0


@router.get("/facts")
async def dashboard_facts(
    request: Request,
    auth: AuthContext = Depends(require_admin_permission),
    tenant_id: str = Query(None),
    category: str = Query(None),
    current_only: bool = Query(True),
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
) -> FactListResponse:
    """List semantic facts with optional filters."""
    db: DatabaseManager = request.app.state.db
    try:
        from sqlalchemy import func, select

        from ...storage.models import SemanticFactModel

        async with db.pg_session() as session:
            q = select(SemanticFactModel)
            if tenant_id:
                q = q.where(SemanticFactModel.tenant_id == tenant_id)
            elif auth.tenant_id:
                q = q.where(SemanticFactModel.tenant_id == auth.tenant_id)
            if category:
                q = q.where(SemanticFactModel.category == category)
            if current_only:
                q = q.where(SemanticFactModel.is_current.is_(True))
            count_q = select(func.count()).select_from(q.subquery())
            total = (await session.execute(count_q)).scalar() or 0
            q = q.order_by(SemanticFactModel.updated_at.desc()).offset(offset).limit(limit)
            rows = (await session.execute(q)).scalars().all()
            items = [
                FactItem(
                    id=str(r.id),
                    tenant_id=str(r.tenant_id),
                    category=str(r.category),
                    key=str(r.key),
                    value=str(r.value),
                    confidence=float(r.confidence),
                    evidence_count=int(r.evidence_count),
                    is_current=bool(r.is_current),
                    version=int(r.version),
                    created_at=r.created_at.isoformat() if r.created_at else None,
                    updated_at=r.updated_at.isoformat() if r.updated_at else None,
                )
                for r in rows
            ]
            return FactListResponse(items=items, total=total)
    except Exception as e:
        logger.error("dashboard_facts_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to list facts: {e}")


@router.post("/facts/{fact_id}/invalidate")
async def dashboard_invalidate_fact(
    request: Request,
    fact_id: str,
    auth: AuthContext = Depends(require_admin_permission),
) -> dict:
    """Invalidate (mark as not current) a specific semantic fact."""
    db: DatabaseManager = request.app.state.db
    try:
        from datetime import UTC, datetime

        from sqlalchemy import update

        from ...storage.models import SemanticFactModel

        async with db.pg_session() as session:
            result = await session.execute(
                update(SemanticFactModel)
                .where(SemanticFactModel.id == fact_id)
                .values(
                    is_current=False,
                    valid_to=datetime.now(UTC).replace(tzinfo=None),
                )
            )
            await session.commit()
            if getattr(result, "rowcount", 0) == 0:
                raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")
            return {"success": True, "message": f"Fact {fact_id} invalidated"}
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_invalidate_fact_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to invalidate fact: {e}")
