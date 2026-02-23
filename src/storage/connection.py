"""Database connection manager for PostgreSQL, Neo4j, and Redis."""

import threading
from collections.abc import AsyncGenerator
from contextlib import asynccontextmanager
from typing import Any

import redis.asyncio as redis
import structlog
from neo4j import AsyncDriver, AsyncGraphDatabase
from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)

from ..core.config import ensure_asyncpg_url, get_settings

_logger = structlog.get_logger(__name__)

# Pool sizing: max persistent connections and extra under peak load
_PG_POOL_SIZE = 20
_PG_MAX_OVERFLOW = 10


class DatabaseManager:
    """Singleton manager for all database connections."""

    _instance: "DatabaseManager | None" = None
    _lock: "threading.Lock" = threading.Lock()

    pg_engine: AsyncEngine | None
    pg_session_factory: async_sessionmaker[AsyncSession] | None
    neo4j_driver: AsyncDriver | None
    redis: Any

    @classmethod
    async def create(cls) -> "DatabaseManager":
        """Async factory that guarantees clean rollback on partial failure."""
        instance = cls.__new__(cls)
        instance.pg_engine = None
        instance.pg_session_factory = None
        instance.neo4j_driver = None
        instance.redis = None
        try:
            instance._init_connections()
        except Exception:
            await instance.close()
            raise
        return instance

    def _init_connections(self) -> None:
        """Synchronous connection setup (engines, drivers, pools)."""
        settings = get_settings()
        db = settings.database
        self.pg_engine = create_async_engine(
            ensure_asyncpg_url(db.postgres_url),
            pool_size=_PG_POOL_SIZE,
            max_overflow=_PG_MAX_OVERFLOW,
            pool_pre_ping=True,
        )
        self.pg_session_factory = async_sessionmaker(
            self.pg_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        if not (db.neo4j_password or "").strip() and "localhost" not in (db.neo4j_url or ""):
            raise ValueError(
                "Neo4j password is required. Set DATABASE__NEO4J_PASSWORD in environment."
            )
        self.neo4j_driver = AsyncGraphDatabase.driver(
            db.neo4j_url,
            auth=(db.neo4j_user, db.neo4j_password or ""),
        )
        self.redis = redis.from_url(db.redis_url)

    def __init__(self) -> None:
        """Synchronous constructor. Prefer create() for async lifespan to get proper cleanup on failure."""
        self.pg_engine = None
        self.pg_session_factory = None
        self.neo4j_driver = None
        self.redis = None
        try:
            self._init_connections()
        except Exception as e:
            import asyncio

            async def _cleanup():
                if self.pg_engine is not None:
                    await self.pg_engine.dispose()
                if self.neo4j_driver is not None:
                    await self.neo4j_driver.close()
                if self.redis is not None:
                    await self.redis.aclose()

            try:
                loop = asyncio.get_running_loop()
                loop.create_task(_cleanup())
            except RuntimeError:
                _logger.error(
                    "db_manager_init_failed",
                    msg="Initialization failed without active event loop; cleanup not performed automatically.",
                    error=str(e),
                    exc_info=True,
                )
            raise

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        with cls._lock:
            if cls._instance is None:
                cls._instance = cls()
        return cls._instance

    @asynccontextmanager
    async def pg_session(self) -> AsyncGenerator[AsyncSession, None]:
        if self.pg_session_factory is None:
            raise RuntimeError("DatabaseManager not initialized or PostgreSQL disabled")
        session = self.pg_session_factory()
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def neo4j_session(self):
        if self.neo4j_driver is None:
            raise RuntimeError("DatabaseManager is closed; cannot create Neo4j session")
        session = self.neo4j_driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def close(self) -> None:
        """Close all database connections safely."""
        if self.pg_engine:
            await self.pg_engine.dispose()
            self.pg_engine = None
            self.pg_session_factory = None
        if self.neo4j_driver:
            await self.neo4j_driver.close()
            self.neo4j_driver = None
        if self.redis:
            await self.redis.aclose()
            self.redis = None
        # Reset singleton so next get_instance() creates a fresh manager (avoids using closed driver).
        with self._lock:
            if DatabaseManager._instance is self:
                DatabaseManager._instance = None
