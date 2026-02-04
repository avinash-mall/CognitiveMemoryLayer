"""Database connection manager for PostgreSQL, Neo4j, and Redis."""
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import redis.asyncio as redis
from neo4j import AsyncGraphDatabase
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

from ..core.config import get_settings


class DatabaseManager:
    """Singleton manager for all database connections."""

    _instance: "DatabaseManager | None" = None

    def __init__(self) -> None:
        settings = get_settings()

        # PostgreSQL
        self.pg_engine = create_async_engine(
            settings.database.postgres_url,
            pool_size=20,
            max_overflow=10,
            pool_pre_ping=True,
        )
        self.pg_session_factory = async_sessionmaker(
            self.pg_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )

        # Neo4j
        self.neo4j_driver = AsyncGraphDatabase.driver(
            settings.database.neo4j_url,
            auth=(
                settings.database.neo4j_user,
                settings.database.neo4j_password,
            ),
        )

        # Redis
        self.redis = redis.from_url(settings.database.redis_url)

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
        await self.redis.close()
