"""Pytest fixtures for Phase 1 and beyond."""

import asyncio
from datetime import datetime
from typing import AsyncGenerator, Generator
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Allow importing src when run from project root or with PYTHONPATH
try:
    from src.storage.models import Base  # noqa: F401
except ImportError:
    import sys
    from pathlib import Path

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _asyncpg_url(url: str) -> str:
    """Ensure URL uses async driver for create_async_engine (avoids psycopg2 when both installed)."""
    if url.startswith("postgresql://") and "+asyncpg" not in url:
        return url.replace("postgresql://", "postgresql+asyncpg://", 1)
    return url


def _get_postgres_url() -> str:
    from src.core.config import get_settings

    return _asyncpg_url(get_settings().database.postgres_url)


@pytest.fixture(scope="session")
def event_loop() -> Generator[asyncio.AbstractEventLoop, None, None]:
    """Create an event loop for the test session."""
    loop = asyncio.new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def sample_memory_record():
    """Sample memory record for testing. Holistic: context_tags, source_session_id."""
    from src.core.schemas import MemoryRecord, Provenance
    from src.core.enums import MemoryType, MemorySource, MemoryStatus

    return MemoryRecord(
        id=uuid4(),
        tenant_id="test-tenant",
        context_tags=["personal"],
        source_session_id="test-session",
        type=MemoryType.EPISODIC_EVENT,
        text="Test memory content",
        confidence=0.8,
        importance=0.7,
        timestamp=datetime.utcnow(),
        written_at=datetime.utcnow(),
        access_count=5,
        status=MemoryStatus.ACTIVE,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )


@pytest.fixture
def sample_chunk():
    """Sample semantic chunk for testing."""
    from src.memory.working.models import SemanticChunk, ChunkType

    return SemanticChunk(
        id="test-chunk-1",
        text="I prefer vegetarian food",
        chunk_type=ChunkType.PREFERENCE,
        salience=0.8,
        confidence=0.9,
        entities=["vegetarian", "food"],
    )


@pytest.fixture
def mock_llm():
    """Mock LLM client for tests."""
    llm = AsyncMock()
    llm.complete = AsyncMock(return_value='{"result": "test"}')
    llm.complete_json = AsyncMock(return_value={"result": "test"})
    return llm


@pytest.fixture
def mock_embeddings():
    """Mock embedding client for tests."""
    client = MagicMock()
    client.dimensions = 1536
    client.embed = AsyncMock(
        return_value=MagicMock(
            embedding=[0.1] * 1536,
            model="test-model",
            dimensions=1536,
            tokens_used=10,
        )
    )
    client.embed_batch = AsyncMock(return_value=[])
    return client


@pytest.fixture
async def pg_engine():
    """Create an async engine for PostgreSQL (from env settings)."""
    url = _get_postgres_url()
    engine = create_async_engine(url, pool_pre_ping=True)
    yield engine
    await engine.dispose()


@pytest.fixture
async def db_session(pg_engine) -> AsyncGenerator[AsyncSession, None]:
    """Provide an async DB session for tests. Tables must exist (migrations run)."""
    async_session = async_sessionmaker(
        pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    async with async_session() as session:
        yield session
        await session.rollback()


@pytest.fixture
async def pg_session_factory(pg_engine):
    """Provide an async session factory for tests that need PostgresMemoryStore (Phase 3)."""
    async_session = async_sessionmaker(
        pg_engine,
        class_=AsyncSession,
        expire_on_commit=False,
    )
    yield async_session


@pytest.fixture
async def event_log_repo(db_session: AsyncSession):
    """Provide EventLogRepository with a live session."""
    from src.storage.event_log import EventLogRepository

    return EventLogRepository(db_session)
