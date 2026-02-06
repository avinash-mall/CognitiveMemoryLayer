"""Database connection manager for PostgreSQL, Neo4j, and Redis."""

from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..core.config import ensure_asyncpg_url, get_settings


class DatabaseManager:
    """Singleton manager for all database connections."""

    _instance: "DatabaseManager | None" = None

    def __init__(self) -> None:
        settings = get_settings()
        self.pg_engine = None
        self.pg_session_factory = None
        self.neo4j_driver = None
        self.redis = None

        try:
            # PostgreSQL
            self.pg_engine = create_async_engine(
                ensure_asyncpg_url(settings.database.postgres_url),
                pool_size=20,
                max_overflow=10,
                pool_pre_ping=True,
            )
            self.pg_session_factory = async_sessionmaker(
                self.pg_engine,
                class_=AsyncSession,
                expire_on_commit=False,
            )

            # Neo4j (require password when connecting to a non-trivial URL)
            db = settings.database
            if not (db.neo4j_password or "").strip() and "localhost" not in (db.neo4j_url or ""):
                raise ValueError(
                    "Neo4j password is required. Set DATABASE__NEO4J_PASSWORD in environment."
                )
            self.neo4j_driver = AsyncGraphDatabase.driver(
                db.neo4j_url,
                auth=(db.neo4j_user, db.neo4j_password or ""),
            )

            # Redis
            self.redis = redis.from_url(settings.database.redis_url)
        except Exception:
            # Clean up any already-created resources on partial init failure
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
                asyncio.run(_cleanup())
            raise

    @classmethod
    def get_instance(cls) -> "DatabaseManager":
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    @asynccontextmanager
    async def pg_session(self) -> AsyncGenerator[AsyncSession, None]:
        session = self.pg_session_factory()
        try:
            yield session
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()

    @asynccontextmanager
    async def neo4j_session(self):
        session = self.neo4j_driver.session()
        try:
            yield session
        finally:
            await session.close()

    async def close(self) -> None:
        await self.pg_engine.dispose()
        await self.neo4j_driver.close()
        await self.redis.aclose()
