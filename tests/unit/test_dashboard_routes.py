"""Unit tests for dashboard API routes (auth and response shape with mocked DB)."""

import os
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Request
from fastapi.testclient import TestClient

from src.api.app import app
from src.api.dashboard import _get_db
from src.core.config import get_settings

# CSRF middleware requires this for dashboard POST/PUT/DELETE/PATCH
DASHBOARD_POST_HEADERS = {"X-Requested-With": "XMLHttpRequest"}


def _env_auth():
    """Read auth from .env (with fallback so tests run)."""
    api_key = os.environ.get("AUTH__API_KEY") or "test-key"
    admin_key = os.environ.get("AUTH__ADMIN_API_KEY") or api_key
    tenant = os.environ.get("AUTH__DEFAULT_TENANT_ID", "default")
    return api_key, admin_key, tenant


@pytest.fixture
def admin_headers(monkeypatch):
    """Set up auth environment from .env and return headers for admin API key."""
    api_key, admin_key, tenant = _env_auth()
    monkeypatch.setenv("AUTH__API_KEY", api_key)
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", admin_key)
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", tenant)
    get_settings.cache_clear()
    try:
        yield {
            "X-API-Key": admin_key,
            "X-Tenant-ID": tenant,
            **DASHBOARD_POST_HEADERS,
        }
    finally:
        get_settings.cache_clear()


@pytest.fixture
def user_headers(monkeypatch):
    """Set up auth environment with distinct admin key so user key returns 403 on dashboard."""
    api_key, _, tenant = _env_auth()
    monkeypatch.setenv("AUTH__API_KEY", api_key)
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-for-tests")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", tenant)
    get_settings.cache_clear()
    try:
        yield {
            "X-API-Key": api_key,
            "X-Tenant-ID": tenant,
            **DASHBOARD_POST_HEADERS,
        }
    finally:
        get_settings.cache_clear()


def _make_mock_db_session():
    """Build a mock async session that returns empty/zero for dashboard queries."""
    result = MagicMock()
    result.scalar = MagicMock(return_value=0)
    result.one_or_none = MagicMock(return_value=None)
    result.all = MagicMock(return_value=[])
    result.scalars = MagicMock(return_value=MagicMock(all=MagicMock(return_value=[])))

    session = MagicMock()
    session.execute = AsyncMock(return_value=result)
    return session


def _make_mock_db():
    """Build a mock DatabaseManager with pg_session returning empty results."""
    mock_session = _make_mock_db_session()

    class AsyncSessionCM:
        async def __aenter__(self):
            return mock_session

        async def __aexit__(self, *args):
            pass

    mock_db = MagicMock()
    mock_db.pg_session = MagicMock(return_value=AsyncSessionCM())
    mock_db.neo4j_driver = None
    mock_db.redis = None
    return mock_db


@pytest.fixture
def mock_db():
    """Mock DatabaseManager for dashboard routes."""
    return _make_mock_db()


def _override_get_db(mock_db):
    """Set dependency override for _get_db; caller must pop it in finally."""

    def _mock_get_db(request: Request):
        return mock_db

    app.dependency_overrides[_get_db] = _mock_get_db


class TestDashboardAuth:
    """Tests for dashboard endpoint authentication and authorization."""

    def test_overview_without_api_key_returns_401(self, monkeypatch):
        """Dashboard overview requires API key."""
        monkeypatch.setenv("AUTH__API_KEY", "k")
        monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-k")
        monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "t")
        get_settings.cache_clear()
        try:
            with TestClient(app) as client:
                resp = client.get(
                    "/api/v1/dashboard/overview", headers={"X-Tenant-ID": "unit-dash-no-key"}
                )
            assert resp.status_code == 401
        finally:
            get_settings.cache_clear()

    def test_overview_with_user_key_returns_403(self, user_headers):
        """Dashboard overview requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.get("/api/v1/dashboard/overview")
        assert resp.status_code == 403

    def test_memories_with_user_key_returns_403(self, user_headers):
        """Dashboard memories list requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.get("/api/v1/dashboard/memories")
        assert resp.status_code == 403

    def test_events_with_user_key_returns_403(self, user_headers):
        """Dashboard events list requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.get("/api/v1/dashboard/events")
        assert resp.status_code == 403

    def test_timeline_with_user_key_returns_403(self, user_headers):
        """Dashboard timeline requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.get("/api/v1/dashboard/timeline")
        assert resp.status_code == 403

    def test_components_with_user_key_returns_403(self, user_headers):
        """Dashboard components health requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.get("/api/v1/dashboard/components")
        assert resp.status_code == 403

    def test_tenants_with_user_key_returns_403(self, user_headers):
        """Dashboard tenants list requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.get("/api/v1/dashboard/tenants")
        assert resp.status_code == 403

    def test_consolidate_with_user_key_returns_403(self, user_headers):
        """Dashboard consolidate requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.post(
                "/api/v1/dashboard/consolidate",
                json={"tenant_id": "t1"},
            )
        assert resp.status_code == 403

    def test_forget_with_user_key_returns_403(self, user_headers):
        """Dashboard forget requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.post(
                "/api/v1/dashboard/forget",
                json={"tenant_id": "t1", "dry_run": True},
            )
        assert resp.status_code == 403

    def test_reconsolidate_with_user_key_returns_403(self, user_headers):
        """Dashboard reconsolidate requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.post(
                "/api/v1/dashboard/reconsolidate",
                json={"tenant_id": "t1"},
            )
        assert resp.status_code == 403

    def test_database_reset_with_user_key_returns_403(self, user_headers):
        """Dashboard database reset requires admin key."""
        with TestClient(app, headers=user_headers) as client:
            resp = client.post("/api/v1/dashboard/database/reset")
        assert resp.status_code == 403

    def test_dashboard_post_without_csrf_header_returns_403(self, admin_headers):
        """Dashboard POST without X-Requested-With returns 403 (CSRF protection)."""
        headers_without_csrf = {k: v for k, v in admin_headers.items() if k != "X-Requested-With"}
        with TestClient(app, headers=headers_without_csrf) as client:
            resp = client.post(
                "/api/v1/dashboard/reconsolidate",
                json={"tenant_id": "t1"},
            )
        assert resp.status_code == 403
        assert resp.json().get("detail") == "Missing CSRF header"


class TestDashboardWithAdminAndMockDb:
    """Tests for dashboard endpoints with admin key and mocked DB (no real Postgres)."""

    def test_overview_returns_200_and_shape(self, admin_headers, mock_db):
        """Dashboard overview returns 200 and DashboardOverview shape with mocked DB."""
        _override_get_db(mock_db)
        try:
            with TestClient(app, headers=admin_headers) as client:
                resp = client.get("/api/v1/dashboard/overview")
            assert resp.status_code == 200
            data = resp.json()
            assert "total_memories" in data
            assert "active_memories" in data
            assert "by_type" in data
            assert "by_status" in data
            assert "total_events" in data
            assert "total_semantic_facts" in data
            assert data["total_memories"] == 0
            assert data["total_events"] == 0
        finally:
            app.dependency_overrides.pop(_get_db, None)

    def test_memories_returns_200_and_pagination_shape(self, admin_headers, mock_db):
        """Dashboard memories returns 200 and paginated list shape."""
        _override_get_db(mock_db)
        try:
            with TestClient(app, headers=admin_headers) as client:
                resp = client.get("/api/v1/dashboard/memories?page=1&per_page=10")
            assert resp.status_code == 200
            data = resp.json()
            assert "items" in data
            assert "total" in data
            assert "page" in data
            assert "per_page" in data
            assert "total_pages" in data
            assert data["items"] == []
            assert data["total"] == 0
        finally:
            app.dependency_overrides.pop(_get_db, None)

    def test_events_returns_200_and_pagination_shape(self, admin_headers, mock_db):
        """Dashboard events returns 200 and paginated list shape."""
        _override_get_db(mock_db)
        try:
            with TestClient(app, headers=admin_headers) as client:
                resp = client.get("/api/v1/dashboard/events?page=1&per_page=10")
            assert resp.status_code == 200
            data = resp.json()
            assert "items" in data
            assert "total" in data
            assert "page" in data
            assert "total_pages" in data
        finally:
            app.dependency_overrides.pop(_get_db, None)

    def test_timeline_returns_200_and_shape(self, admin_headers, mock_db):
        """Dashboard timeline returns 200 and timeline shape."""
        _override_get_db(mock_db)
        try:
            with TestClient(app, headers=admin_headers) as client:
                resp = client.get("/api/v1/dashboard/timeline?days=7")
            assert resp.status_code == 200
            data = resp.json()
            assert "points" in data
            assert "total" in data
            assert isinstance(data["points"], list)
        finally:
            app.dependency_overrides.pop(_get_db, None)

    def test_tenants_returns_200_and_shape(self, admin_headers, mock_db):
        """Dashboard tenants returns 200 and tenants list shape."""
        _override_get_db(mock_db)
        try:
            with TestClient(app, headers=admin_headers) as client:
                resp = client.get("/api/v1/dashboard/tenants")
            assert resp.status_code == 200
            data = resp.json()
            assert "tenants" in data
            assert isinstance(data["tenants"], list)
        finally:
            app.dependency_overrides.pop(_get_db, None)

    def test_components_returns_200_and_shape(self, admin_headers, mock_db):
        """Dashboard components returns 200 and component status list."""
        _override_get_db(mock_db)
        try:
            with TestClient(app, headers=admin_headers) as client:
                resp = client.get("/api/v1/dashboard/components")
            assert resp.status_code == 200
            data = resp.json()
            assert "components" in data
            assert isinstance(data["components"], list)
            names = [c["name"] for c in data["components"]]
            assert "PostgreSQL" in names
        finally:
            app.dependency_overrides.pop(_get_db, None)

    def test_reconsolidate_returns_200_and_shape(self, admin_headers, mock_db):
        """Dashboard reconsolidate returns 200 and result shape when orchestrator is mocked."""
        mock_labile = MagicMock()
        mock_labile.release_all_for_tenant = AsyncMock(return_value=3)
        mock_recon = MagicMock()
        mock_recon.labile_tracker = mock_labile
        mock_orchestrator = MagicMock()
        mock_orchestrator.reconsolidation = mock_recon

        # Session for job tracking: add, commit, execute (use shared helper so execute is AsyncMock)
        mock_session = _make_mock_db_session()
        mock_session.add = MagicMock()
        mock_session.commit = AsyncMock()

        class AsyncSessionCM:
            async def __aenter__(self):
                return mock_session

            async def __aexit__(self, *args):
                pass

        mock_db.pg_session = MagicMock(return_value=AsyncSessionCM())
        mock_db.redis = None  # so rate-limit middleware uses in-memory fallback (no connection)
        # Route uses request.app.state.db (not Depends), so set state explicitly

        _override_get_db(mock_db)
        orig_orch = getattr(app.state, "orchestrator", None)
        orig_db = getattr(app.state, "db", None)
        try:
            with TestClient(app, headers=admin_headers) as client:
                # Override state after client start (lifespan may have set real db/orchestrator)
                app.state.orchestrator = mock_orchestrator
                app.state.db = mock_db
                resp = client.post(
                    "/api/v1/dashboard/reconsolidate",
                    json={"tenant_id": "t1"},
                )
                assert resp.status_code == 200
                data = resp.json()
                assert data.get("status") == "completed"
                assert data.get("tenant_id") == "t1"
                assert "sessions_released" in data
                assert isinstance(data["sessions_released"], int)
        finally:
            app.state.orchestrator = orig_orch
            app.state.db = orig_db
            app.dependency_overrides.pop(_get_db, None)


class TestDashboardSchemas:
    """Tests for dashboard request/response schemas."""

    def test_dashboard_consolidate_request(self):
        """DashboardConsolidateRequest validates tenant_id and optional user_id."""
        from src.api.schemas import DashboardConsolidateRequest

        req = DashboardConsolidateRequest(tenant_id="t1")
        assert req.tenant_id == "t1"
        assert req.user_id is None

        req2 = DashboardConsolidateRequest(tenant_id="t2", user_id="u2")
        assert req2.user_id == "u2"

    def test_dashboard_forget_request(self):
        """DashboardForgetRequest has expected defaults."""
        from src.api.schemas import DashboardForgetRequest

        req = DashboardForgetRequest(tenant_id="t1")
        assert req.tenant_id == "t1"
        assert req.dry_run is True
        assert req.max_memories == 5000

        req2 = DashboardForgetRequest(tenant_id="t2", dry_run=False, max_memories=100)
        assert req2.dry_run is False
        assert req2.max_memories == 100

    def test_dashboard_reconsolidate_request(self):
        """DashboardReconsolidateRequest validates tenant_id and optional user_id."""
        from src.api.schemas import DashboardReconsolidateRequest

        req = DashboardReconsolidateRequest(tenant_id="t1")
        assert req.tenant_id == "t1"
        assert req.user_id is None

        req2 = DashboardReconsolidateRequest(tenant_id="t2", user_id="u2")
        assert req2.user_id == "u2"
