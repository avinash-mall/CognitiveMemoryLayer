"""Integration tests for dashboard API (real app and DB)."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings


@pytest.fixture
def admin_client(monkeypatch):
    """Test client with admin API key for dashboard access."""
    monkeypatch.setenv("AUTH__API_KEY", "demo-key-123")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-456")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "demo")
    get_settings.cache_clear()
    try:
        with TestClient(
            app,
            headers={
                "X-API-Key": "admin-key-456",
                "X-Tenant-ID": "demo",
            },
        ) as c:
            yield c
    finally:
        get_settings.cache_clear()


class TestDashboardIntegration:
    """Dashboard endpoints with real app state (db from lifespan)."""

    def test_dashboard_overview_returns_200(self, admin_client: TestClient):
        """GET /dashboard/overview returns 200 and overview structure."""
        resp = admin_client.get("/api/v1/dashboard/overview")
        assert resp.status_code == 200
        data = resp.json()
        assert "total_memories" in data
        assert "total_events" in data
        assert "by_type" in data
        assert "by_status" in data

    def test_dashboard_overview_with_tenant_filter(self, admin_client: TestClient):
        """GET /dashboard/overview?tenant_id=... returns 200."""
        resp = admin_client.get("/api/v1/dashboard/overview?tenant_id=demo")
        assert resp.status_code == 200

    def test_dashboard_memories_returns_200(self, admin_client: TestClient):
        """GET /dashboard/memories returns paginated list."""
        resp = admin_client.get("/api/v1/dashboard/memories?page=1&per_page=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data
        assert "page" in data
        assert data["per_page"] == 5

    def test_dashboard_events_returns_200(self, admin_client: TestClient):
        """GET /dashboard/events returns paginated list."""
        resp = admin_client.get("/api/v1/dashboard/events?page=1&per_page=5")
        assert resp.status_code == 200
        data = resp.json()
        assert "items" in data
        assert "total" in data

    def test_dashboard_timeline_returns_200(self, admin_client: TestClient):
        """GET /dashboard/timeline returns points and total."""
        resp = admin_client.get("/api/v1/dashboard/timeline?days=30")
        assert resp.status_code == 200
        data = resp.json()
        assert "points" in data
        assert "total" in data

    def test_dashboard_components_returns_200(self, admin_client: TestClient):
        """GET /dashboard/components returns component status list."""
        resp = admin_client.get("/api/v1/dashboard/components")
        assert resp.status_code == 200
        data = resp.json()
        assert "components" in data
        assert len(data["components"]) >= 1

    def test_dashboard_tenants_returns_200(self, admin_client: TestClient):
        """GET /dashboard/tenants returns tenants list."""
        resp = admin_client.get("/api/v1/dashboard/tenants")
        assert resp.status_code == 200
        data = resp.json()
        assert "tenants" in data

    def test_dashboard_memory_detail_404_for_unknown_id(self, admin_client: TestClient):
        """GET /dashboard/memories/{id} returns 404 for non-existent memory."""
        resp = admin_client.get("/api/v1/dashboard/memories/00000000-0000-0000-0000-000000000001")
        assert resp.status_code == 404
