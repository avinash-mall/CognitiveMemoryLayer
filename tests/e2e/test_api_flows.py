"""E2E tests for full API memory lifecycle."""
import os

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings


@pytest.fixture
def client(monkeypatch):
    """Test client with auth headers. Auth keys from config (set for tests)."""
    monkeypatch.setenv("AUTH__API_KEY", "demo-key-123")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-456")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "demo")
    get_settings.cache_clear()
    try:
        with TestClient(
            app,
            headers={
                "X-API-Key": "demo-key-123",
                "X-Tenant-ID": "demo",
            },
        ) as c:
            yield c
    finally:
        get_settings.cache_clear()


@pytest.mark.skipif(
    not os.environ.get("OPENAI_API_KEY"),
    reason="OpenAI API key required for full E2E (embeddings)",
)
def test_full_memory_lifecycle(client):
    """Test write -> read -> update -> forget flow."""
    scope_id = "e2e-test-session"

    write_resp = client.post(
        "/api/v1/memory/write",
        json={
            "scope": "session",
            "scope_id": scope_id,
            "content": "I prefer vegetarian food and I live in Paris",
        },
    )
    assert write_resp.status_code == 200
    write_data = write_resp.json()
    assert write_data["success"] is True

    read_resp = client.post(
        "/api/v1/memory/read",
        json={
            "scope": "session",
            "scope_id": scope_id,
            "query": "What food do I like?",
        },
    )
    assert read_resp.status_code == 200
    read_data = read_resp.json()
    assert "total_count" in read_data
    assert "memories" in read_data

    if read_data["memories"]:
        memory_id = read_data["memories"][0]["id"]
        update_resp = client.post(
            "/api/v1/memory/update",
            json={
                "scope": "session",
                "scope_id": scope_id,
                "memory_id": str(memory_id),
                "feedback": "correct",
            },
        )
        assert update_resp.status_code == 200

    forget_resp = client.post(
        "/api/v1/memory/forget",
        json={
            "scope": "session",
            "scope_id": scope_id,
            "query": "vegetarian",
        },
    )
    assert forget_resp.status_code == 200


def test_unauthorized_access():
    """Test that unauthorized requests are rejected."""
    with TestClient(app) as client_no_auth:
        resp = client_no_auth.post(
            "/api/v1/memory/write",
            json={
                "scope": "session",
                "scope_id": "test",
                "content": "test",
            },
        )
        assert resp.status_code == 401


def test_health_response_structure(client):
    """Test health returns expected response structure."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
