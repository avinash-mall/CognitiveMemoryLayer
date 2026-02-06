"""Integration tests for Phase 9 REST API flow."""

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


@pytest.fixture
def admin_client(monkeypatch):
    """Test client with admin auth (MED-39: use yield + cleanup)."""
    monkeypatch.setenv("AUTH__API_KEY", "demo-key-123")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-456")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "admin")
    get_settings.cache_clear()
    try:
        with TestClient(
            app,
            headers={
                "X-API-Key": "admin-key-456",
                "X-Tenant-ID": "admin",
            },
        ) as c:
            yield c
    finally:
        get_settings.cache_clear()


def test_health_check(client):
    """Health endpoint should return 200 without auth."""
    # Health is under /api/v1 from router - actually the router has prefix /api/v1
    # and health is at router.get("/health") so it's /api/v1/health
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "healthy"


def test_health_no_auth():
    """Health check works without API key."""
    with TestClient(app) as client_no_auth:
        resp = client_no_auth.get("/api/v1/health")
    assert resp.status_code == 200


def test_write_requires_auth():
    """Write should require API key."""
    with TestClient(app) as client_no_auth:
        resp = client_no_auth.post(
            "/api/v1/memory/write",
            json={
                "content": "test",
                "session_id": "s1",
            },
        )
    assert resp.status_code == 401


def test_read_requires_auth():
    """Read should require API key."""
    with TestClient(app) as client_no_auth:
        resp = client_no_auth.post(
            "/api/v1/memory/read",
            json={
                "query": "test",
            },
        )
    assert resp.status_code == 401


def test_stats_requires_auth():
    """Stats should require API key."""
    with TestClient(app) as client_no_auth:
        resp = client_no_auth.get("/api/v1/memory/stats")
    assert resp.status_code == 401
