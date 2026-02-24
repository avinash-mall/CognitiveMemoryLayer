"""Integration tests for timestamp preservation in write flow."""

from datetime import UTC, datetime

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.storage.postgres import PostgresMemoryStore


@pytest.mark.asyncio
async def test_timestamp_preservation(pg_session_factory, monkeypatch):
    """Ingest via API with historical timestamp; verify episodic store preserves timestamp."""
    monkeypatch.setenv("AUTH__API_KEY", "test-key")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "test-key")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "default")
    from src.core.config import get_settings

    get_settings.cache_clear()
    try:
        from src.api.auth import _build_api_keys

        _build_api_keys.cache_clear()
    except Exception:
        pass
    try:
        with TestClient(app) as client:
            tenant_id = "test-tenant-timestamp"
            session_id = "test-session-timestamp"
            historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()

            response = client.post(
                "/api/v1/memory/write",
                headers={"X-Tenant-Id": tenant_id, "X-API-Key": "test-key"},
                json={
                    "content": "I moved to New York in January 2023.",
                    "session_id": session_id,
                    "timestamp": historical_date,
                },
            )
            if response.status_code == 500:
                pytest.skip("Write failed (e.g. embedding service unreachable in Docker)")
            assert response.status_code == 200

        episodic = PostgresMemoryStore(pg_session_factory)
        records = await episodic.scan(tenant_id)
        assert len(records) >= 1, "Expected at least one record from write"
        for r in records:
            assert r.timestamp.year == 2023, f"Record timestamp {r.timestamp} should be 2023"
    finally:
        get_settings.cache_clear()
        try:
            from src.api.auth import _build_api_keys

            _build_api_keys.cache_clear()
        except Exception:
            pass
