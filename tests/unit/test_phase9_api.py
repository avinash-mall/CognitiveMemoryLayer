"""Unit tests for Phase 9 REST API components."""
import pytest
from fastapi import FastAPI
from fastapi.security import APIKeyHeader
from fastapi.testclient import TestClient

from src.api.auth import (
    AuthContext,
    AuthService,
    get_auth_context,
    api_key_header,
)
from src.api.middleware import RateLimitMiddleware, RequestLoggingMiddleware
from src.api.schemas import (
    WriteMemoryRequest,
    ReadMemoryRequest,
    MemoryItem,
)


class TestAuthService:
    """Tests for AuthService."""

    def test_validate_key_valid(self):
        svc = AuthService()
        ctx = svc.validate_key("demo-key-123")
        assert ctx is not None
        assert ctx.tenant_id == "demo"
        assert ctx.can_write

    def test_validate_key_invalid(self):
        svc = AuthService()
        assert svc.validate_key("invalid-key") is None

    def test_validate_key_admin(self):
        svc = AuthService()
        ctx = svc.validate_key("admin-key-456")
        assert ctx is not None
        assert ctx.can_admin

    def test_check_permission(self):
        svc = AuthService()
        ctx = svc.validate_key("demo-key-123")
        assert svc.check_permission(ctx, "read")
        assert svc.check_permission(ctx, "write")
        assert not svc.check_permission(ctx, "admin")


class TestSchemas:
    """Tests for API schemas."""

    def test_write_memory_request(self):
        req = WriteMemoryRequest(user_id="u1", content="Hello")
        assert req.user_id == "u1"
        assert req.content == "Hello"
        assert req.metadata == {}

    def test_read_memory_request_defaults(self):
        req = ReadMemoryRequest(user_id="u1", query="test")
        assert req.max_results == 10
        assert req.format == "packet"
