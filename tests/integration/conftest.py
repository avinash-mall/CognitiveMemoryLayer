"""Integration test fixtures; optionally use Testcontainers for Postgres and Neo4j.

When testcontainers is installed and Docker socket is available, Postgres and Neo4j
containers are started (module-scoped). When running inside Docker (e.g. docker-compose)
or USE_ENV_DB=1, DATABASE__POSTGRES_URL from env is used instead (no testcontainers).
Otherwise the root conftest's pg_engine/db_session are used.
"""

from __future__ import annotations

import asyncio
import os
from typing import AsyncGenerator, Generator

import pytest

try:
    from testcontainers.neo4j import Neo4jContainer
    from testcontainers.postgres import PostgresContainer

    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False


def _asyncpg_url(postgres_url: str) -> str:
    """Convert postgresql:// URL to postgresql+asyncpg://."""
    if postgres_url.startswith("postgresql://"):
        return postgres_url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return postgres_url


def _use_env_postgres() -> bool:
    """Use env Postgres (e.g. docker-compose) instead of testcontainers."""
    if os.environ.get("USE_ENV_DB"):
        return True
    # Inside Docker there is usually no Docker socket; use compose Postgres.
    if os.environ.get("DATABASE__POSTGRES_URL"):
        if os.path.exists("/.dockerenv"):
            return True
        try:
            # Docker socket typical paths
            if not os.path.exists("/var/run/docker.sock") and not os.path.exists(
                "//var/run/docker.sock"
            ):
                return True
        except OSError:
            return True
    return False


if HAS_TESTCONTAINERS and not _use_env_postgres():

    @pytest.fixture(scope="module")
    def postgres_container() -> Generator:
        """Start PostgreSQL container for integration tests."""
        with PostgresContainer("pgvector/pgvector:pg16") as postgres:
            yield postgres

    @pytest.fixture(scope="module")
    def neo4j_container() -> Generator:
        """Start Neo4j container for integration tests."""
        with Neo4jContainer("neo4j:5") as neo4j:
            yield neo4j

    @pytest.fixture(scope="module")
    def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
        """Event loop for module-scoped async fixtures."""
        loop = asyncio.new_event_loop()
        yield loop
        loop.close()

    @pytest.fixture(scope="module")
    async def pg_engine(postgres_container: PostgresContainer):
        """Override root pg_engine with Testcontainers Postgres for integration tests."""
        from sqlalchemy.ext.asyncio import create_async_engine

        url = _asyncpg_url(postgres_container.get_connection_url())
        engine = create_async_engine(url, pool_pre_ping=True)

        from src.storage.models import Base

        async with engine.begin() as conn:
            await conn.run_sync(Base.metadata.create_all)

        yield engine
        await engine.dispose()

    @pytest.fixture
    async def db_session(pg_engine) -> AsyncGenerator:
        """Override root db_session for integration tests; rollback after test."""
        from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker

        async_session = async_sessionmaker(
            pg_engine,
            class_=AsyncSession,
            expire_on_commit=False,
        )
        async with async_session() as session:
            yield session
            await session.rollback()
