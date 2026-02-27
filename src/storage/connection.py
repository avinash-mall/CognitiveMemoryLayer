"""Database connection manager for PostgreSQL, Neo4j, and Redis."""

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


def _attach_pool_metrics(engine: AsyncEngine) -> None:
    """Attach SQLAlchemy pool checkout/checkin metrics listeners."""
    try:
        from sqlalchemy import event

        from ..utils.metrics import (
            DB_POOL_CHECKED_OUT,
            DB_POOL_CHECKINS_TOTAL,
            DB_POOL_CHECKOUTS_TOTAL,
            DB_POOL_INVALIDATIONS_TOTAL,
        )
    except Exception:
        return

    @event.listens_for(engine.sync_engine, "checkout")
    def _on_checkout(*_args, **_kwargs) -> None:
        DB_POOL_CHECKOUTS_TOTAL.inc()
        DB_POOL_CHECKED_OUT.inc()

    @event.listens_for(engine.sync_engine, "checkin")
    def _on_checkin(*_args, **_kwargs) -> None:
        DB_POOL_CHECKINS_TOTAL.inc()
        DB_POOL_CHECKED_OUT.dec()

    @event.listens_for(engine.sync_engine, "invalidate")
    def _on_invalidate(*_args, **_kwargs) -> None:
        DB_POOL_INVALIDATIONS_TOTAL.inc()


class DatabaseManager:
    """Manager for database connections (explicit lifecycle)."""

    pg_engine: AsyncEngine | None
    pg_session_factory: async_sessionmaker[AsyncSession] | None
    neo4j_driver: AsyncDriver | None
    redis: Any

    @classmethod
    async def create(cls) -> "DatabaseManager":
        """Async factory that guarantees clean rollback on partial failure."""
        instance = cls()
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
        _attach_pool_metrics(self.pg_engine)
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
        """Initialize empty handles. Use ``await DatabaseManager.create()``."""
        self.pg_engine = None
        self.pg_session_factory = None
        self.neo4j_driver = None
        self.redis = None

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
