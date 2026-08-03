"""Admin API routes for consolidation, forgetting, and user management."""

from fastapi import APIRouter, Depends, HTTPException, Request

from ..memory.orchestrator import MemoryOrchestrator
from .auth import AuthContext, require_admin_permission

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
    """Manually trigger episodic-to-semantic consolidation for a user. Samples episodes, clusters them, extracts gists, and migrates to semantic facts. Admin-only."""
    try:
        report = await orchestrator.consolidation.consolidate(
            tenant_id=auth.tenant_id,
            user_id=user_id,
        )
        return {
            "status": "consolidation_completed",
            "user_id": user_id,
            "episodes_sampled": report.episodes_sampled,
            "clusters_formed": report.clusters_formed,
            "clusters_skipped_no_recurrence": report.clusters_skipped_no_recurrence,
            "gists_extracted": report.gists_extracted,
            "details_recovered": report.details_recovered,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Consolidation failed: {e}")


@admin_router.get("/consolidation/status")
async def consolidation_status(
    auth: AuthContext = Depends(require_admin_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Whether the background consolidation sweep is running, and how often it has fired.

    Trigger counts rather than config: this subsystem shipped with a documented 6-hour
    interval that had no caller, and reading the diff is what missed that three times.
    """
    return orchestrator.consolidation.status


@admin_router.post("/forget/{user_id}")
async def trigger_forgetting(
    user_id: str,
    dry_run: bool = True,
    auth: AuthContext = Depends(require_admin_permission),
    orchestrator: MemoryOrchestrator = Depends(get_orchestrator),
):
    """Manually trigger active forgetting for a user. Use dry_run=true to preview without applying. Admin-only."""
    try:
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
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Forgetting failed: {e}")
