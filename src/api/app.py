"""FastAPI application factory and entry point."""

import pathlib
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from ..core.config import get_settings
from ..storage.connection import DatabaseManager
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .routes import router
from .admin_routes import admin_router
from .dashboard_routes import dashboard_router

# Resolve dashboard static files directory
_DASHBOARD_DIR = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    get_settings()
    db_manager = DatabaseManager.get_instance()
    app.state.db = db_manager

    from ..memory.orchestrator import MemoryOrchestrator

    app.state.orchestrator = await MemoryOrchestrator.create(db_manager)

    yield

    await db_manager.close()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    get_settings()

    app = FastAPI(
        title="Cognitive Memory Layer",
        description="Neuro-inspired memory system for LLMs",
        version="1.0.0",
        lifespan=lifespan,
    )

    settings = get_settings()
    if settings.cors_origins is not None:
        origins = settings.cors_origins
    elif settings.debug:
        origins = ["*"]
    else:
        origins = ["http://localhost:3000", "http://localhost:8080"]
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.add_middleware(RateLimitMiddleware)
    app.add_middleware(RequestLoggingMiddleware)

    app.include_router(router, prefix="/api/v1")
    app.include_router(admin_router, prefix="/api/v1")
    app.include_router(dashboard_router, prefix="/api/v1")

    # Prometheus metrics endpoint
    from prometheus_client import make_asgi_app

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Dashboard SPA: serve static files and catch-all for client-side routing
    if _DASHBOARD_DIR.is_dir():
        app.mount(
            "/dashboard/static", StaticFiles(directory=str(_DASHBOARD_DIR)), name="dashboard-static"
        )

        @app.get("/dashboard/{rest_of_path:path}")
        async def dashboard_spa(rest_of_path: str = ""):
            """Serve the dashboard SPA index.html for all dashboard routes."""
            index = _DASHBOARD_DIR / "index.html"
            if index.is_file():
                return FileResponse(str(index))
            return {"error": "Dashboard not found"}

        @app.get("/dashboard")
        async def dashboard_root():
            """Serve the dashboard root."""
            index = _DASHBOARD_DIR / "index.html"
            if index.is_file():
                return FileResponse(str(index))
            return {"error": "Dashboard not found"}

    return app


app = create_app()
