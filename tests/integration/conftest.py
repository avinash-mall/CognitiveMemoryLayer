"""Integration test fixtures; optionally use Testcontainers for Postgres and Neo4j.

When testcontainers is installed and Docker socket is available, Postgres and Neo4j
containers are started (module-scoped). When running inside Docker (e.g. docker-compose)
or USE_ENV_DB=1, DATABASE__POSTGRES_URL from env is used instead (no testcontainers).
Otherwise the root conftest's pg_engine/db_session are used.
"""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator

import pytest

try:
    from testcontainers.neo4j import Neo4jContainer
    from testcontainers.postgres import PostgresContainer

    HAS_TESTCONTAINERS = True
except ImportError:
    HAS_TESTCONTAINERS = False


def _use_env_postgres() -> bool:
    """Use env Postgres (e.g. docker-compose / CI services) instead of testcontainers.

    Returns True when the caller has explicitly provided a Postgres URL via
    DATABASE__POSTGRES_URL or the USE_ENV_DB flag.  In those situations we
    should honour the external database rather than spinning up testcontainers.
    """
    if os.environ.get("USE_ENV_DB"):
        return True
    return bool(os.environ.get("DATABASE__POSTGRES_URL"))


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

    @pytest.fixture
    async def pg_engine(postgres_container: PostgresContainer):
        """Override root pg_engine with Testcontainers Postgres for integration tests.

        Function-scoped so each test gets an engine on its own event loop, avoiding
        "Future attached to a different loop" when disposing or rolling back.
        """
        from sqlalchemy.ext.asyncio import create_async_engine

        from src.core.config import ensure_asyncpg_url

        url = ensure_asyncpg_url(postgres_container.get_connection_url())
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
