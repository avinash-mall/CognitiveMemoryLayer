"""Unit tests for API routes and middleware."""

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings


@pytest.fixture
def auth_headers(monkeypatch):
    """Set up auth environment and return headers."""
    monkeypatch.setenv("AUTH__API_KEY", "test-key-123")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-456")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "test-tenant")
    get_settings.cache_clear()
    try:
        yield {
            "X-API-Key": "test-key-123",
            "X-Tenant-ID": "test-tenant",
        }
    finally:
        get_settings.cache_clear()


class TestHealthEndpoint:
    """Tests for health check endpoint."""

    def test_health_returns_200(self, auth_headers):
        """Health check returns 200 OK."""
        with TestClient(app, headers=auth_headers) as client:
            resp = client.get("/api/v1/health")
            assert resp.status_code == 200

    def test_health_returns_healthy_status(self, auth_headers):
        """Health check returns healthy status."""
        with TestClient(app, headers=auth_headers) as client:
            resp = client.get("/api/v1/health")
            data = resp.json()
            assert data["status"] == "healthy"
            assert "timestamp" in data


class TestAuthMiddleware:
    """Tests for authentication middleware."""

    def test_missing_api_key_returns_401(self, monkeypatch):
        """Requests without API key get 401."""
        monkeypatch.setenv("AUTH__API_KEY", "required-key")
        monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "t1")
        get_settings.cache_clear()
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/memory/write",
                    json={"content": "test"},
                )
                assert resp.status_code == 401
        finally:
            get_settings.cache_clear()

    def test_invalid_api_key_returns_401(self, monkeypatch):
        """Requests with invalid API key get 401."""
        monkeypatch.setenv("AUTH__API_KEY", "valid-key")
        monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "t1")
        get_settings.cache_clear()
        try:
            with TestClient(app, headers={"X-API-Key": "wrong-key"}) as client:
                resp = client.post(
                    "/api/v1/memory/write",
                    json={"content": "test"},
                )
                assert resp.status_code == 401
        finally:
            get_settings.cache_clear()


class TestAPISchemas:
    """Tests for API request/response schemas."""

    def test_write_memory_request_validation(self):
        """WriteMemoryRequest validates required fields."""
        from src.api.schemas import WriteMemoryRequest

        req = WriteMemoryRequest(content="Test content")
        assert req.content == "Test content"
        assert req.context_tags is None  # Optional, not []
        assert req.session_id is None
        assert req.metadata == {}

    def test_write_memory_request_with_all_fields(self):
        """WriteMemoryRequest accepts all optional fields."""
        from src.api.schemas import WriteMemoryRequest

        req = WriteMemoryRequest(
            content="Test",
            context_tags=["conversation"],
            session_id="s1",
            memory_type="episodic_event",
            metadata={"key": "value"},
        )
        assert req.context_tags == ["conversation"]
        assert req.session_id == "s1"

    def test_read_memory_request_defaults(self):
        """ReadMemoryRequest has sensible defaults."""
        from src.api.schemas import ReadMemoryRequest

        req = ReadMemoryRequest(query="test query")
        assert req.max_results == 10
        assert req.format == "packet"

    def test_read_memory_request_custom_values(self):
        """ReadMemoryRequest accepts custom values."""
        from src.api.schemas import ReadMemoryRequest

        req = ReadMemoryRequest(
            query="test",
            max_results=5,
            context_filter=["personal"],
            format="list",
        )
        assert req.max_results == 5
        assert req.format == "list"

    def test_forget_request_defaults(self):
        """ForgetRequest has sensible defaults."""
        from src.api.schemas import ForgetRequest

        req = ForgetRequest()
        assert req.action == "delete"
        assert req.memory_ids is None
        assert req.query is None

    def test_update_memory_request_validation(self):
        """UpdateMemoryRequest validates fields."""
        from src.api.schemas import UpdateMemoryRequest

        req = UpdateMemoryRequest(memory_id="123e4567-e89b-12d3-a456-426614174000")
        assert req.feedback is None
        assert req.text is None

    def test_process_turn_request_structure(self):
        """ProcessTurnRequest has expected structure."""
        from src.api.schemas import ProcessTurnRequest

        req = ProcessTurnRequest(
            user_message="Hello",
            assistant_response="Hi there!",
            session_id="s1",
        )
        assert req.user_message == "Hello"
        assert req.assistant_response == "Hi there!"

    def test_create_session_request_defaults(self):
        """CreateSessionRequest has defaults."""
        from src.api.schemas import CreateSessionRequest

        req = CreateSessionRequest()
        assert req.metadata == {}
        assert req.ttl_hours == 24  # Default is 24 hours


class TestMemoryItem:
    """Tests for MemoryItem response schema."""

    def test_memory_item_structure(self):
        """MemoryItem has expected fields."""
        from src.api.schemas import MemoryItem
        from datetime import datetime

        item = MemoryItem(
            id="123e4567-e89b-12d3-a456-426614174000",
            text="Test memory",
            type="semantic_fact",
            confidence=0.9,
            relevance=0.85,  # Not relevance_score
            timestamp=datetime.now(),  # Not created_at
        )
        assert item.text == "Test memory"
        assert item.confidence == 0.9
        assert item.relevance == 0.85


class TestResponseSchemas:
    """Tests for response schemas."""

    def test_write_memory_response(self):
        """WriteMemoryResponse structure."""
        from src.api.schemas import WriteMemoryResponse

        resp = WriteMemoryResponse(
            success=True,
            memory_id="123e4567-e89b-12d3-a456-426614174000",
            chunks_created=2,
            message="Stored 2 chunks",
        )
        assert resp.success is True
        assert resp.chunks_created == 2

    def test_read_memory_response(self):
        """ReadMemoryResponse structure."""
        from src.api.schemas import ReadMemoryResponse

        resp = ReadMemoryResponse(
            query="test",
            memories=[],
            total_count=0,
            elapsed_ms=10.5,
        )
        assert resp.total_count == 0
        assert resp.memories == []
        assert resp.elapsed_ms == 10.5

    def test_forget_response(self):
        """ForgetResponse structure."""
        from src.api.schemas import ForgetResponse

        resp = ForgetResponse(
            success=True,
            affected_count=5,
            # No 'action' field in ForgetResponse
        )
        assert resp.affected_count == 5
        assert resp.success is True
