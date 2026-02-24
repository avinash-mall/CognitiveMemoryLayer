"""FastAPI application factory and entry point."""

import pathlib
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.base import BaseHTTPMiddleware

from ..core.config import get_settings, validate_embedding_dimensions
from ..core.exceptions import MemoryAccessDenied, MemoryNotFoundError
from ..storage.connection import DatabaseManager
from .admin_routes import admin_router
from .dashboard import dashboard_router
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .routes import router

# Resolve dashboard static files directory
_DASHBOARD_DIR = pathlib.Path(__file__).resolve().parent.parent / "dashboard" / "static"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator:
    """Application lifespan handler."""
    settings = get_settings()

    # Validate embedding dimensions match DB schema at startup
    validate_embedding_dimensions(settings)

    db_manager = await DatabaseManager.create()
    app.state.db = db_manager

    from ..memory.orchestrator import MemoryOrchestrator

    app.state.orchestrator = await MemoryOrchestrator.create(db_manager)

    yield

    await db_manager.close()


def create_app() -> FastAPI:
    """Create FastAPI application."""
    app = FastAPI(
        title="Cognitive Memory Layer",
        description="Neuro-inspired memory system for LLMs",
        version="1.0.0",
        lifespan=lifespan,
    )

    @app.exception_handler(MemoryNotFoundError)
    async def memory_not_found_handler(_request: Request, exc: MemoryNotFoundError) -> JSONResponse:
        return JSONResponse(status_code=404, content={"detail": str(exc)})

    @app.exception_handler(MemoryAccessDenied)
    async def memory_access_denied_handler(
        _request: Request, exc: MemoryAccessDenied
    ) -> JSONResponse:
        return JSONResponse(status_code=403, content={"detail": str(exc)})

    settings = get_settings()
    if settings.cors_origins is not None:
        origins = settings.cors_origins
    elif settings.debug:
        origins = ["*"]
    else:
        origins = ["http://localhost:3000", "http://localhost:8080"]

    # CORS spec: credentials are incompatible with wildcard origins
    allow_credentials = "*" not in origins

    class CSRFCheckMiddleware(BaseHTTPMiddleware):
        """Require X-Requested-With for dashboard state-changing requests."""

        async def dispatch(self, request: Request, call_next):
            if (
                request.url.path.startswith("/api/v1/dashboard")
                and request.method in ("POST", "PUT", "DELETE", "PATCH")
                and request.headers.get("X-Requested-With") != "XMLHttpRequest"
            ):
                return JSONResponse(
                    status_code=403,
                    content={"detail": "Missing CSRF header"},
                )
            return await call_next(request)

    app.add_middleware(CSRFCheckMiddleware)
    app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    rpm = getattr(settings.auth, "rate_limit_requests_per_minute", 60)
    if rpm > 0:
        app.add_middleware(RateLimitMiddleware, requests_per_minute=rpm)
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
            """Serves the dashboard SPA (index.html) for client-side routing. All dashboard routes (e.g. /dashboard/#graph) are handled by this catch-all."""
            index = _DASHBOARD_DIR / "index.html"
            if index.is_file():
                return FileResponse(str(index))
            return {"error": "Dashboard not found"}

    return app


app = create_app()
