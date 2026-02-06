"""Admin API routes for consolidation, forgetting, and user management."""

from fastapi import APIRouter, Depends, Request

from .auth import AuthContext, require_admin_permission
from ..memory.orchestrator import MemoryOrchestrator

admin_router = APIRouter(prefix="/admin", tags=["admin"])


def get_orchestrator(request: Request) -> MemoryOrchestrator:
    """Get memory orchestrator from app state."""
    return request.app.state.orchestrator


@admin_router.post("/consolidate/{user_id}")
async def trigger_consolidation(
    user_id: str,
    auth: AuthContext = Depends(require_admin_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Manually trigger consolidation for a user."""
    report = await orchestrator.consolidation.consolidate(
        tenant_id=auth.tenant_id,
        user_id=user_id,
    )
    return {
        "status": "consolidation_completed",
        "user_id": user_id,
        "episodes_sampled": report.episodes_sampled,
        "clusters_formed": report.clusters_formed,
        "gists_extracted": report.gists_extracted,
    }


@admin_router.post("/forget/{user_id}")
async def trigger_forgetting(
    user_id: str,
    dry_run: bool = True,
    auth: AuthContext = Depends(require_admin_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Manually trigger forgetting for a user."""
    report = await orchestrator.forgetting.run_forgetting(
        tenant_id=auth.tenant_id,
        user_id=user_id,
        dry_run=dry_run,
    )
    return {
        "status": "forgetting_completed",
        "user_id": user_id,
        "dry_run": dry_run,
        "memories_scanned": report.memories_scanned,
        "operations_applied": report.result.operations_applied,
    }
