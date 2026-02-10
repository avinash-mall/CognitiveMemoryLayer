"""E2E fixtures: reuse integration live server (same config and live_client)."""

from __future__ import annotations

import contextlib
import os
from pathlib import Path

import pytest
import pytest_asyncio

from cml import AsyncCognitiveMemoryLayer, CognitiveMemoryLayer
from cml.config import CMLConfig

# Use CML_TEST_API_KEY if set; else try repo .env AUTH__API_KEY so server and tests share one key
INTEGRATION_URL = os.environ.get("CML_TEST_URL", "http://localhost:8000")
_key = os.environ.get("CML_TEST_API_KEY")
if not _key:
    try:
        from dotenv import load_dotenv

        _root = Path(__file__).resolve().parents[4]
        load_dotenv(_root / ".env")
        _key = os.environ.get("AUTH__API_KEY", "test-key")
    except Exception:
        _key = "test-key"
INTEGRATION_KEY = _key
INTEGRATION_ADMIN_KEY = os.environ.get("AUTH__ADMIN_API_KEY") or INTEGRATION_KEY
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
        admin_api_key=INTEGRATION_ADMIN_KEY,
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
    with contextlib.suppress(Exception):
        await client.delete_all(confirm=True)
    await client.close()
