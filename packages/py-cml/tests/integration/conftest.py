"""Integration test fixtures (require running CML server)."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

INTEGRATION_URL = os.environ.get("CML_TEST_URL", "http://localhost:8000")
INTEGRATION_KEY = os.environ.get("CML_TEST_API_KEY", "test-key")
INTEGRATION_TENANT = f"test-{os.getpid()}"


@pytest.fixture(scope="session")
def integration_config() -> CMLConfig:
    """Session-scoped config for integration tests."""
    return CMLConfig(
        api_key=INTEGRATION_KEY,
        base_url=INTEGRATION_URL,
        tenant_id=INTEGRATION_TENANT,
        timeout=30.0,
        max_retries=1,
    )


def _server_reachable(config: CMLConfig) -> bool:
    """Try health() once; return True if server is reachable."""
    try:
        client = CognitiveMemoryLayer(config=config)
        client.health()
        client.close()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def live_client(
    integration_config: CMLConfig,
) -> pytest.AsyncGenerator[AsyncCognitiveMemoryLayer, None]:
    """Async client against live server; teardown tries delete_all then close."""
    if not _server_reachable(integration_config):
        pytest.skip("CML server not reachable; set CML_TEST_URL and run server")
    client = AsyncCognitiveMemoryLayer(config=integration_config)
    yield client
    try:
        await client.delete_all(confirm=True)
    except Exception:
        pass
    await client.close()


@pytest.fixture
def live_sync_client(
    integration_config: CMLConfig,
) -> pytest.Generator[CognitiveMemoryLayer, None, None]:
    """Sync client against live server; teardown tries delete_all then close."""
    if not _server_reachable(integration_config):
        pytest.skip("CML server not reachable; set CML_TEST_URL and run server")
    client = CognitiveMemoryLayer(config=integration_config)
    yield client
    try:
        client.delete_all(confirm=True)
    except Exception:
        pass
    client.close()
