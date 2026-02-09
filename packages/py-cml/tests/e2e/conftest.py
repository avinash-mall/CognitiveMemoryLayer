"""E2E fixtures: reuse integration live server (same config and live_client)."""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

INTEGRATION_URL = os.environ.get("CML_TEST_URL", "http://localhost:8000")
INTEGRATION_KEY = os.environ.get("CML_TEST_API_KEY", "test-key")
INTEGRATION_TENANT = f"test-e2e-{os.getpid()}"


@pytest.fixture(scope="session")
def integration_config() -> CMLConfig:
    """Session-scoped config for e2e (same as integration)."""
    return CMLConfig(
        api_key=INTEGRATION_KEY,
        base_url=INTEGRATION_URL,
        tenant_id=INTEGRATION_TENANT,
        timeout=30.0,
        max_retries=1,
    )


def _server_reachable(config: CMLConfig) -> bool:
    try:
        client = CognitiveMemoryLayer(config=config)
        client.health()
        client.close()
        return True
    except Exception:
        return False


@pytest_asyncio.fixture
async def live_client(integration_config: CMLConfig):
    """Async client against live server (e2e)."""
    if not _server_reachable(integration_config):
        pytest.skip("CML server not reachable for e2e")
    client = AsyncCognitiveMemoryLayer(config=integration_config)
    yield client
    try:
        await client.delete_all(confirm=True)
    except Exception:
        pass
    await client.close()

