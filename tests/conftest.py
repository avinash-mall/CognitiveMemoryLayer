"""Pytest fixtures for Phase 1 and beyond."""

from collections.abc import AsyncGenerator
from datetime import datetime
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest
from sqlalchemy.ext.asyncio import AsyncSession, async_sessionmaker, create_async_engine

# Load repo .env so tests read AUTH__API_KEY, etc. (no hardcoded fallbacks in tests)
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

# Allow importing src when run from project root or with PYTHONPATH
try:
    import src.storage.models  # noqa: F401 - ensure module loaded for fixtures
except ImportError:
    import sys

    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _get_postgres_url() -> str:
    from src.core.config import ensure_asyncpg_url, get_settings

    return ensure_asyncpg_url(get_settings().database.postgres_url)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    """Auto-clear the settings LRU cache after each test to prevent pollution (MED-47)."""
    yield
    try:
        from src.core.config import get_settings

        get_settings.cache_clear()
    except Exception:
        pass


@pytest.fixture
def sample_memory_record():
    """Sample memory record for testing. Holistic: context_tags, source_session_id."""
    from src.core.enums import MemorySource, MemoryStatus, MemoryType
    from src.core.schemas import MemoryRecord, Provenance

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
    from src.memory.working.models import ChunkType, SemanticChunk

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
    """Mock embedding client for tests. Dimensions from .env (get_settings().embedding.dimensions)."""
    from src.core.config import get_settings

    dims = get_settings().embedding.dimensions
    client = MagicMock()
    client.dimensions = dims
    client.embed = AsyncMock(
        return_value=MagicMock(
            embedding=[0.1] * dims,
            model="test-model",
            dimensions=dims,
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
