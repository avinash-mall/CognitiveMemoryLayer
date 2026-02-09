"""Shared test fixtures for py-cml."""

from __future__ import annotations

import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig


# --- Configuration Fixtures ---


@pytest.fixture
def test_config() -> CMLConfig:
    """Create a test configuration."""
    return CMLConfig(
        api_key="test-api-key",
        base_url="http://localhost:8000",
        tenant_id="test-tenant",
        timeout=10.0,
        max_retries=0,
    )


@pytest.fixture
def mock_config() -> CMLConfig:
    """Config for unit tests (no real server)."""
    return CMLConfig(
        api_key="mock-key",
        base_url="http://mock-server:8000",
        tenant_id="mock-tenant",
        max_retries=0,
    )


# --- Client Fixtures ---


@pytest.fixture
def sync_client(test_config: CMLConfig) -> pytest.Generator[CognitiveMemoryLayer, None, None]:
    """Create a sync client for testing."""
    client = CognitiveMemoryLayer(config=test_config)
    yield client
    client.close()


@pytest_asyncio.fixture
async def async_client(
    test_config: CMLConfig,
) -> pytest.AsyncGenerator[AsyncCognitiveMemoryLayer, None]:
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
