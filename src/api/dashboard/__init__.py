"""Dashboard API package. Aggregates all dashboard route modules."""

from fastapi import APIRouter

from ._shared import _get_db
from .config_routes import router as config_router
from .events_routes import router as events_router
from .fact_routes import router as fact_router
from .graph_routes import router as graph_router
from .jobs_routes import router as jobs_router
from .memory_routes import router as memory_router
from .overview_routes import router as overview_router

dashboard_router = APIRouter(prefix="/dashboard", tags=["dashboard"])

dashboard_router.include_router(overview_router, tags=[])
dashboard_router.include_router(memory_router, tags=[])
dashboard_router.include_router(events_router, tags=[])
dashboard_router.include_router(graph_router, tags=[])
dashboard_router.include_router(config_router, tags=[])
dashboard_router.include_router(jobs_router, tags=[])
dashboard_router.include_router(fact_router, tags=[])

__all__ = ["_get_db", "dashboard_router"]
