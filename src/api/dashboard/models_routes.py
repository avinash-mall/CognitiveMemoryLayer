"""Dashboard route for model pack status."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, Field

from ..auth import AuthContext, require_admin_permission

router = APIRouter()


class ModelStatusResponse(BaseModel):
    available: bool = False
    families: list[str] = Field(default_factory=list)
    task_models: list[str] = Field(default_factory=list)
    load_errors: list[str] = Field(default_factory=list)
    models_dir: str = ""


@router.get("/models/status")
async def dashboard_models_status(
    auth: AuthContext = Depends(require_admin_permission),
) -> ModelStatusResponse:
    """Return loaded model families, task models, and any load errors."""
    try:
        from ...utils.modelpack import get_modelpack_runtime

        mp = get_modelpack_runtime()
        _ = mp.available  # triggers lazy load
        return ModelStatusResponse(
            available=mp.available,
            families=sorted(mp._models.keys()),
            task_models=sorted(mp._task_models.keys()),
            load_errors=mp._load_errors[:10],
            models_dir=str(mp.models_dir),
        )
    except Exception as exc:
        return ModelStatusResponse(
            available=False,
            load_errors=[f"Failed to load modelpack: {exc}"],
        )
