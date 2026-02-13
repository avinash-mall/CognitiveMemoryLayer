"""Shared test fixtures for cognitive-memory-layer."""

from __future__ import annotations

import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

# Load repo-root .env so tests read CML_* / AUTH__* (no hardcoded fallbacks)
try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parents[3]
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass

# Test config values from env; fallback so tests run (no skip)
_TEST_API_KEY = os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY") or "test-key"
_TEST_BASE_URL = (
    os.environ.get("CML_BASE_URL") or os.environ.get("MEMORY_API_URL") or "http://localhost:8000"
).strip()

# --- Configuration Fixtures ---


@pytest.fixture
def test_config() -> CMLConfig:
    """Create a test configuration from .env (with fallbacks so tests run)."""
    return CMLConfig(
        api_key=_TEST_API_KEY,
        base_url=_TEST_BASE_URL,
        tenant_id=os.environ.get("CML_TENANT_ID", "default"),
        timeout=10.0,
        max_retries=0,
    )


@pytest.fixture
def mock_config() -> CMLConfig:
    """Config for unit tests (no real server); API key from env (with fallback)."""
    return CMLConfig(
        api_key=_TEST_API_KEY,
        base_url="http://mock-server:8000",
        tenant_id="mock-tenant",
        max_retries=0,
    )


@pytest.fixture
def cml_config() -> CMLConfig:
    """Config from .env for unit tests (with fallbacks so tests run)."""
    return CMLConfig(
        api_key=_TEST_API_KEY,
        base_url=_TEST_BASE_URL,
        tenant_id="test-tenant",
        timeout=10.0,
        max_retries=0,
    )


# --- Client Fixtures ---


@pytest.fixture
def sync_client(test_config: CMLConfig) -> Generator[CognitiveMemoryLayer, None, None]:
    """Create a sync client for testing."""
    client = CognitiveMemoryLayer(config=test_config)
    yield client
    client.close()


@pytest_asyncio.fixture
async def async_client(
    test_config: CMLConfig,
) -> AsyncGenerator[AsyncCognitiveMemoryLayer, None]:
    """Create an async client for testing."""
    client = AsyncCognitiveMemoryLayer(config=test_config)
    yield client
    await client.close()


# --- Mock Response Helpers ---


def make_write_response(
    success: bool = True,
    memory_id: str = "00000000-0000-0000-0000-000000000001",
    chunks_created: int = 1,
    message: str = "Stored 1 memory chunks",
) -> dict:
    """Create a mock write response."""
    return {
        "success": success,
        "memory_id": memory_id,
        "chunks_created": chunks_created,
        "message": message,
    }


def make_read_response(
    query: str = "test query",
    memories: list | None = None,
    total_count: int = 1,
    elapsed_ms: float = 42.0,
) -> dict:
    """Create a mock read response."""
    if memories is None:
        memories = [
            {
                "id": "00000000-0000-0000-0000-000000000001",
                "text": "User prefers vegetarian food",
                "type": "preference",
                "confidence": 0.9,
                "relevance": 0.95,
                "timestamp": "2025-01-01T00:00:00Z",
                "metadata": {},
            }
        ]
    return {
        "query": query,
        "memories": memories,
        "facts": [],
        "preferences": memories,
        "episodes": [],
        "llm_context": "## Preferences\n- User prefers vegetarian food",
        "total_count": total_count,
        "elapsed_ms": elapsed_ms,
    }


def make_turn_response(
    memory_context: str = "## Preferences\n- vegetarian",
    memories_retrieved: int = 3,
    memories_stored: int = 1,
    reconsolidation_applied: bool = False,
) -> dict:
    """Create a mock turn response."""
    return {
        "memory_context": memory_context,
        "memories_retrieved": memories_retrieved,
        "memories_stored": memories_stored,
        "reconsolidation_applied": reconsolidation_applied,
    }


def make_stats_response(
    total_memories: int = 10,
    active_memories: int = 8,
    silent_memories: int = 1,
    archived_memories: int = 1,
    by_type: dict | None = None,
    avg_confidence: float = 0.85,
    avg_importance: float = 0.75,
    estimated_size_mb: float = 0.1,
) -> dict:
    """Create a mock stats response."""
    if by_type is None:
        by_type = {}
    return {
        "total_memories": total_memories,
        "active_memories": active_memories,
        "silent_memories": silent_memories,
        "archived_memories": archived_memories,
        "by_type": by_type,
        "avg_confidence": avg_confidence,
        "avg_importance": avg_importance,
        "oldest_memory": None,
        "newest_memory": None,
        "estimated_size_mb": estimated_size_mb,
    }
