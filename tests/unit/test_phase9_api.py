"""Unit tests for Phase 9 REST API components."""
import pytest
from fastapi import FastAPI
from fastapi.security import APIKeyHeader
from fastapi.testclient import TestClient

from src.api.auth import (
    AuthContext,
    _build_api_keys,
    api_key_header,
)
from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.schemas import (
    WriteMemoryRequest,
    ReadMemoryRequest,
    MemoryItem,
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
