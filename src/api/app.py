"""FastAPI application factory and entry point."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from ..core.config import get_settings
from ..storage.connection import DatabaseManager
from .middleware import RateLimitMiddleware, RequestLoggingMiddleware
from .routes import router
from .admin_routes import admin_router


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
    origins = settings.cors_origins if settings.cors_origins is not None else ["https://yourdomain.com"]
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

    # Prometheus metrics endpoint
    from prometheus_client import make_asgi_app

    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    return app


app = create_app()
