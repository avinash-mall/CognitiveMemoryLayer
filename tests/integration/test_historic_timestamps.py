"""Integration tests for timestamp preservation in write flow."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.storage.postgres import PostgresMemoryStore


@pytest.mark.asyncio
async def test_timestamp_preservation(pg_session_factory):
    """Ingest via API with historical timestamp; verify episodic store preserves timestamp."""
    from src.core.config import get_settings

    s = get_settings()
    api_key = s.auth.api_key or s.auth.admin_api_key or "test-key"

    with TestClient(app) as client:
        tenant_id = "test-tenant-timestamp"
        session_id = "test-session-timestamp"
        historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()

        response = client.post(
            "/api/v1/memory/write",
            headers={"X-Tenant-Id": tenant_id, "X-API-Key": api_key},
            json={
                "content": "I moved to New York in January 2023.",
                "session_id": session_id,
                "timestamp": historical_date,
            },
        )
        assert response.status_code == 200

    episodic = PostgresMemoryStore(pg_session_factory)
    records = await episodic.scan(tenant_id)
    assert len(records) >= 1, "Expected at least one record from write"
    for r in records:
        assert r.timestamp.year == 2023, f"Record timestamp {r.timestamp} should be 2023"
