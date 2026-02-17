"""Integration test fixtures (require running CML server)."""

from __future__ import annotations

import contextlib
import os
from collections.abc import AsyncGenerator, Generator
from pathlib import Path

import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

# Load repo .env then read URL and key (no hardcoded fallbacks)
try:
    from dotenv import load_dotenv

    _root = Path(__file__).resolve().parents[4]
    load_dotenv(_root / ".env")
except Exception:
    pass

# Use env when set; fallback so tests run (no skip) and fail at first request if server is down
INTEGRATION_URL = (
    os.environ.get("CML_TEST_URL")
    or os.environ.get("CML_BASE_URL")
    or os.environ.get("MEMORY_API_URL")
    or "http://localhost:8000"
).strip()
INTEGRATION_KEY = (
    os.environ.get("CML_TEST_API_KEY") or os.environ.get("AUTH__API_KEY") or "test-key"
)
INTEGRATION_ADMIN_KEY = os.environ.get("AUTH__ADMIN_API_KEY") or INTEGRATION_KEY
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
        admin_api_key=INTEGRATION_ADMIN_KEY,
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
) -> AsyncGenerator[AsyncCognitiveMemoryLayer, None]:
    """Async client against live server; teardown tries delete_all then close."""
    if not _server_reachable(integration_config):
        pytest.skip(
            f"CML server not reachable at {integration_config.base_url} (start server and re-run)"
        )
    client = AsyncCognitiveMemoryLayer(config=integration_config)
    yield client
    with contextlib.suppress(Exception):
        await client.delete_all(confirm=True)
    await client.close()


@pytest.fixture
def live_sync_client(
    integration_config: CMLConfig,
) -> Generator[CognitiveMemoryLayer, None, None]:
    """Sync client against live server; teardown tries delete_all then close."""
    if not _server_reachable(integration_config):
        pytest.skip(
            f"CML server not reachable at {integration_config.base_url} (start server and re-run)"
        )
    client = CognitiveMemoryLayer(config=integration_config)
    yield client
    with contextlib.suppress(Exception):
        client.delete_all(confirm=True)
    client.close()
