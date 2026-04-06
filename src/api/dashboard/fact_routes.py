"""Dashboard semantic facts routes: list, detail, invalidate."""

from datetime import datetime
from typing import cast
from uuid import UUID

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from starlette.responses import JSONResponse

from ...storage.connection import DatabaseManager
from ...storage.models import SemanticFactModel
from ..auth import AuthContext, require_admin_permission
from ._shared import (
    DashboardFactDetail,
    DashboardFactEvidenceItem,
    DashboardFactEvidenceResponse,
    FactItem,
    FactListResponse,
    logger,
)

router = APIRouter()


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


@router.get("/facts/{fact_id}", response_model=DashboardFactDetail)
async def dashboard_fact_detail(
    request: Request,
    fact_id: str,
    auth: AuthContext = Depends(require_admin_permission),
) -> DashboardFactDetail:
    """Return fact detail with lineage and supersession context."""
    _ = auth
    db: DatabaseManager = request.app.state.db
    try:
        from sqlalchemy import select

        async with db.pg_session() as session:
            row = (
                await session.execute(
                    select(SemanticFactModel).where(SemanticFactModel.id == fact_id)
                )
            ).scalar_one_or_none()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")

        fact_store = request.app.state.orchestrator.neocortical.facts
        lineage = await fact_store.get_fact_lineage(row.tenant_id, fact_id=str(row.id))
        superseded_by = await fact_store.get_superseded_chain(row.tenant_id, str(row.id))

        return DashboardFactDetail(
            id=str(row.id),
            tenant_id=str(row.tenant_id),
            category=str(row.category),
            key=str(row.key),
            subject=str(row.subject),
            predicate=str(row.predicate),
            value=row.value,
            value_type=str(row.value_type),
            context_tags=list(row.context_tags or []),
            confidence=float(row.confidence),
            evidence_count=int(row.evidence_count),
            evidence_ids=list(row.evidence_ids or []),
            valid_from=cast("datetime | None", row.valid_from),
            valid_to=cast("datetime | None", row.valid_to),
            is_current=bool(row.is_current),
            created_at=cast("datetime | None", row.created_at),
            updated_at=cast("datetime | None", row.updated_at),
            version=int(row.version),
            supersedes_id=str(row.supersedes_id) if row.supersedes_id else None,
            lineage=lineage,
            superseded_by=superseded_by,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_fact_detail_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load fact detail: {e}")


@router.get("/facts/{fact_id}/evidence", response_model=DashboardFactEvidenceResponse)
async def dashboard_fact_evidence(
    request: Request,
    fact_id: str,
    auth: AuthContext = Depends(require_admin_permission),
) -> DashboardFactEvidenceResponse:
    """Return evidence memories backing a fact."""
    _ = auth
    db: DatabaseManager = request.app.state.db
    try:
        from sqlalchemy import select

        async with db.pg_session() as session:
            row = (
                await session.execute(
                    select(SemanticFactModel).where(SemanticFactModel.id == fact_id)
                )
            ).scalar_one_or_none()
            if row is None:
                raise HTTPException(status_code=404, detail=f"Fact {fact_id} not found")

        evidence_ids = list(row.evidence_ids or [])
        valid_memory_ids: list[UUID] = []
        missing_evidence_ids: list[str] = []
        for evidence_id in evidence_ids:
            try:
                valid_memory_ids.append(UUID(evidence_id))
            except (TypeError, ValueError):
                missing_evidence_ids.append(str(evidence_id))

        records = []
        if valid_memory_ids:
            records = await request.app.state.orchestrator.hippocampal.store.get_by_ids_batch(
                valid_memory_ids
            )
        found_ids = {str(record.id) for record in records}
        missing_evidence_ids.extend(
            [
                evidence_id
                for evidence_id in evidence_ids
                if evidence_id not in found_ids and evidence_id not in missing_evidence_ids
            ]
        )

        evidence = [
            DashboardFactEvidenceItem(
                id=record.id,
                text=record.text,
                type=record.type.value,
                status=record.status.value,
                confidence=record.confidence,
                importance=record.importance,
                source_session_id=record.source_session_id,
                timestamp=record.timestamp,
                written_at=record.written_at,
                supersedes_id=record.supersedes_id,
                metadata=record.metadata,
            )
            for record in records
        ]

        return DashboardFactEvidenceResponse(
            fact_id=fact_id,
            evidence=evidence,
            missing_evidence_ids=missing_evidence_ids,
        )
    except HTTPException:
        raise
    except Exception as e:
        logger.error("dashboard_fact_evidence_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to load fact evidence: {e}")


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


@router.get("/facts/export")
async def dashboard_export_facts(
    request: Request,
    tenant_id: str | None = Query(None),
    auth: AuthContext = Depends(require_admin_permission),
) -> JSONResponse:
    """Export semantic facts as JSON for download. Optionally filter by tenant."""
    db: DatabaseManager = request.app.state.db
    try:
        from sqlalchemy import select

        from ...storage.models import SemanticFactModel

        async with db.pg_session() as session:
            q = select(SemanticFactModel).order_by(SemanticFactModel.updated_at.desc()).limit(10000)
            if tenant_id:
                q = q.where(SemanticFactModel.tenant_id == tenant_id)
            rows = (await session.execute(q)).scalars().all()
            data = [
                {
                    "id": str(r.id),
                    "tenant_id": r.tenant_id,
                    "category": r.category,
                    "key": r.key,
                    "value": r.value,
                    "confidence": float(r.confidence),
                    "evidence_count": int(r.evidence_count),
                    "is_current": bool(r.is_current),
                    "version": int(r.version),
                    "created_at": r.created_at.isoformat() if r.created_at else None,
                    "updated_at": r.updated_at.isoformat() if r.updated_at else None,
                }
                for r in rows
            ]
            return JSONResponse(
                content=data,
                headers={"Content-Disposition": "attachment; filename=facts_export.json"},
            )
    except Exception as e:
        logger.error("dashboard_export_facts_error", error=str(e))
        raise HTTPException(status_code=500, detail=f"Failed to export facts: {e}")
