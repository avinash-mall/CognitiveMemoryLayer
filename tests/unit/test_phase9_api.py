"""Unit tests for Phase 9 REST API components."""

from fastapi.testclient import TestClient

from src.api.app import app
from src.api.auth import (
    AuthContext,
    _build_api_keys,
    get_auth_context,
)
from src.api.schemas import (
    ReadMemoryRequest,
    WriteMemoryRequest,
)


class TestAuthConfig:
    """Tests for config-based auth (_build_api_keys)."""

    def test_build_api_keys_empty_when_no_config(self, monkeypatch):
        monkeypatch.setenv("AUTH__API_KEY", "")
        monkeypatch.setenv("AUTH__ADMIN_API_KEY", "")
        from src.core.config import get_settings

        get_settings.cache_clear()
        keys = _build_api_keys()
        assert keys == {}

    def test_build_api_keys_from_config(self, monkeypatch):
        monkeypatch.setenv("AUTH__API_KEY", "demo-key-123")
        monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-456")
        monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "demo")
        from src.core.config import get_settings

        get_settings.cache_clear()
        keys = _build_api_keys()
        assert "demo-key-123" in keys
        assert keys["demo-key-123"].tenant_id == "demo"
        assert keys["demo-key-123"].can_write
        assert not keys["demo-key-123"].can_admin
        assert "admin-key-456" in keys
        assert keys["admin-key-456"].can_admin


class TestSchemas:
    """Tests for API schemas."""

    def test_write_memory_request(self):
        req = WriteMemoryRequest(
            content="Hello",
            context_tags=["conversation"],
            session_id="s1",
        )
        assert req.session_id == "s1"
        assert req.content == "Hello"
        assert req.metadata == {}

    def test_read_memory_request_defaults(self):
        req = ReadMemoryRequest(
            query="test",
            context_filter=["conversation"],
        )
        assert req.max_results == 10
        assert req.format == "packet"


class TestAuthContextPermissions:
    """Tests for get_auth_context and permission dependencies via dependency override."""

    def test_require_write_permission_returns_403_when_can_write_false(self):
        """When AuthContext has can_write=False, write endpoint returns 403."""

        async def no_write_context():
            return AuthContext(
                tenant_id="t1",
                can_read=True,
                can_write=False,
                can_admin=False,
            )

        app.dependency_overrides[get_auth_context] = no_write_context
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/memory/write",
                    json={"content": "test"},
                    headers={"X-API-Key": "any", "X-Tenant-ID": "t1"},
                )
                assert resp.status_code == 403
                assert "Write permission" in resp.json().get("detail", "")
        finally:
            app.dependency_overrides.pop(get_auth_context, None)

    def test_require_admin_permission_returns_403_when_can_admin_false(self):
        """When AuthContext has can_admin=False, dashboard admin endpoint returns 403."""

        async def no_admin_context():
            return AuthContext(
                tenant_id="t1",
                can_read=True,
                can_write=True,
                can_admin=False,
            )

        app.dependency_overrides[get_auth_context] = no_admin_context
        try:
            with TestClient(app) as client:
                resp = client.get(
                    "/api/v1/dashboard/overview",
                    headers={"X-API-Key": "any", "X-Tenant-ID": "t1"},
                )
                assert resp.status_code == 403
                assert "Admin permission" in resp.json().get("detail", "")
        finally:
            app.dependency_overrides.pop(get_auth_context, None)
