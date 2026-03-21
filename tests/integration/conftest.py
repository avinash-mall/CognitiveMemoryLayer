"""Integration test fixtures; optionally use Testcontainers for Postgres and Neo4j.

When testcontainers is installed and Docker socket is available, Postgres and Neo4j
containers are started (module-scoped). When running inside Docker (e.g. docker-compose)
or USE_ENV_DB=1, DATABASE__POSTGRES_URL from env is used instead (no testcontainers).
Otherwise the root conftest's pg_engine/db_session are used.
"""

from __future__ import annotations

import os
import warnings
from collections.abc import AsyncGenerator, Generator
from uuid import uuid4

import pytest
import requests

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


def _get_live_api_url() -> str:
    base_url = os.environ.get("CML_TEST_URL") or os.environ.get("CML_BASE_URL") or ""
    return base_url.rstrip("/")


@pytest.fixture
def live_api_server() -> dict[str, str]:
    """Resolve and verify the live API used by the request-based integration tests."""
    base_url = _get_live_api_url()
    api_key = os.environ.get("CML_TEST_API_KEY") or os.environ.get("CML_API_KEY") or ""
    admin_api_key = os.environ.get("CML_ADMIN_API_KEY") or api_key
    if not base_url or not api_key:
        raise RuntimeError(
            "Live API integration tests require CML_BASE_URL/CML_API_KEY (or CML_TEST_URL/CML_TEST_API_KEY)."
        )
    try:
        response = requests.get(f"{base_url}/api/v1/health", timeout=3)
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        raise RuntimeError(
            f"CML server not reachable at {base_url}; start the API before running live integration tests"
        ) from exc
    if response.status_code != 200:
        raise RuntimeError(
            f"CML server health check failed at {base_url} with status {response.status_code}"
        )
    return {
        "base_url": base_url,
        "api_key": api_key,
        "admin_api_key": admin_api_key,
    }


@pytest.fixture
def live_api_test_context(live_api_server: dict[str, str]):
    """Provide isolated tenant/session ids and best-effort cleanup for live API tests."""
    tenant_id = f"it-live-{uuid4().hex[:12]}"
    session_id = f"it-session-{uuid4().hex[:12]}"
    payload = {
        **live_api_server,
        "tenant_id": tenant_id,
        "session_id": session_id,
        "headers": {
            "X-API-Key": live_api_server["api_key"],
            "X-Tenant-Id": tenant_id,
        },
        "admin_headers": {
            "X-API-Key": live_api_server["admin_api_key"],
            "X-Tenant-Id": tenant_id,
        },
    }
    yield payload

    try:
        response = requests.delete(
            f"{live_api_server['base_url']}/api/v1/memory/all",
            headers=payload["admin_headers"],
            timeout=10,
        )
        if response.status_code not in {200, 404}:
            warnings.warn(
                f"live API cleanup returned {response.status_code}: {response.text[:200]}",
                stacklevel=1,
            )
    except requests.RequestException as exc:  # pragma: no cover - network dependent
        warnings.warn(f"live API cleanup failed: {exc}", stacklevel=1)


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
