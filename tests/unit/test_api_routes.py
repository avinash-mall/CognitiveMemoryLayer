"""Unit tests for API routes and middleware."""

import os
from datetime import UTC

import pytest
from fastapi import Request
from fastapi.testclient import TestClient
from pydantic import ValidationError

from src.api.app import app
from src.api.auth import _build_api_keys
from src.core.config import get_settings


@pytest.fixture
def auth_headers(monkeypatch):
    """Set up auth environment from .env and return headers."""
    api_key = os.environ.get("AUTH__API_KEY") or "[REDACTED]"
    admin_key = os.environ.get("AUTH__ADMIN_API_KEY") or api_key
    tenant = os.environ.get("AUTH__DEFAULT_TENANT_ID", "default")
    monkeypatch.setenv("AUTH__API_KEY", api_key)
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", admin_key)
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", tenant)
    get_settings.cache_clear()
    _build_api_keys.cache_clear()
    try:
        yield {
            "X-API-Key": api_key,
            "X-Tenant-ID": tenant,
        }
    finally:
        get_settings.cache_clear()
        _build_api_keys.cache_clear()


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
        _build_api_keys.cache_clear()
        try:
            with TestClient(app) as client:
                resp = client.post(
                    "/api/v1/memory/write",
                    json={"content": "test"},
                    headers={"X-Tenant-ID": "unit-auth-missing-key"},
                )
                assert resp.status_code == 401
        finally:
            get_settings.cache_clear()
            _build_api_keys.cache_clear()

    def test_invalid_api_key_returns_401(self, monkeypatch):
        """Requests with invalid API key get 401."""
        monkeypatch.setenv("AUTH__API_KEY", "valid-key")
        monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "t1")
        get_settings.cache_clear()
        _build_api_keys.cache_clear()
        try:
            with TestClient(
                app, headers={"X-API-Key": "wrong-key", "X-Tenant-ID": "unit-auth-invalid-key"}
            ) as client:
                resp = client.post(
                    "/api/v1/memory/write",
                    json={"content": "test"},
                )
                assert resp.status_code == 401
        finally:
            get_settings.cache_clear()
            _build_api_keys.cache_clear()


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

    def test_write_memory_request_content_max_length_rejected(self):
        """WriteMemoryRequest rejects content exceeding 100,000 characters."""
        from src.api.schemas import WriteMemoryRequest

        with pytest.raises(ValidationError) as exc_info:
            WriteMemoryRequest(content="x" * 100_001)
        assert "content" in str(exc_info.value) or "100000" in str(exc_info.value).lower()

    def test_write_memory_request_content_min_length_rejected(self):
        """WriteMemoryRequest rejects empty content."""
        from src.api.schemas import WriteMemoryRequest

        with pytest.raises(ValidationError) as exc_info:
            WriteMemoryRequest(content="")
        assert "content" in str(exc_info.value)

    def test_write_memory_request_content_at_limit_accepted(self):
        """WriteMemoryRequest accepts content at exactly 100,000 characters."""
        from src.api.schemas import WriteMemoryRequest

        req = WriteMemoryRequest(content="x" * 100_000)
        assert len(req.content) == 100_000

    def test_write_memory_request_with_all_fields(self):
        """WriteMemoryRequest accepts all optional fields."""
        from datetime import datetime

        from src.api.schemas import WriteMemoryRequest

        ts = datetime(2025, 1, 15, 12, 0, 0, tzinfo=UTC)
        req = WriteMemoryRequest(
            content="Test",
            context_tags=["conversation"],
            session_id="s1",
            memory_type="episodic_event",
            metadata={"key": "value"},
            namespace="ns1",
            timestamp=ts,
        )
        assert req.context_tags == ["conversation"]
        assert req.session_id == "s1"
        assert req.namespace == "ns1"
        assert req.timestamp == ts

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

    def test_read_memory_request_accepts_user_timezone(self):
        """ReadMemoryRequest accepts optional user_timezone for timezone-aware retrieval."""
        from src.api.schemas import ReadMemoryRequest

        req = ReadMemoryRequest(query="today?", user_timezone="America/New_York")
        assert req.user_timezone == "America/New_York"
        req_default = ReadMemoryRequest(query="test")
        assert req_default.user_timezone is None

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

    def test_process_turn_request_accepts_user_timezone(self):
        """ProcessTurnRequest accepts optional user_timezone for retrieval in the turn."""
        from src.api.schemas import ProcessTurnRequest

        req = ProcessTurnRequest(
            user_message="Hi",
            user_timezone="Europe/London",
        )
        assert req.user_timezone == "Europe/London"
        req_default = ProcessTurnRequest(user_message="Hi")
        assert req_default.user_timezone is None

    def test_create_session_request_defaults(self):
        """CreateSessionRequest has defaults."""
        from src.api.schemas import CreateSessionRequest

        req = CreateSessionRequest()
        assert req.metadata == {}
        assert req.ttl_hours == 24  # Default is 24 hours


class TestUpdateExceptionHandlers:
    """Tests for update endpoint 404/403 exception handlers."""

    def test_update_nonexistent_memory_returns_404(self, auth_headers):
        """POST /memory/update with non-existent memory_id returns 404."""
        from unittest.mock import AsyncMock, MagicMock

        from src.api.routes import get_orchestrator
        from src.core.exceptions import MemoryNotFoundError

        mock_orch = MagicMock()
        mock_orch.update = AsyncMock(
            side_effect=MemoryNotFoundError(memory_id="00000000-0000-0000-0000-000000000001")
        )

        def override_orchestrator(request: Request):
            return mock_orch

        app.dependency_overrides[get_orchestrator] = override_orchestrator
        try:
            with TestClient(app, headers=auth_headers) as client:
                resp = client.post(
                    "/api/v1/memory/update",
                    json={
                        "memory_id": "00000000-0000-0000-0000-000000000001",
                        "feedback": "correct",
                    },
                )
            assert resp.status_code == 404
            assert "detail" in resp.json()
        finally:
            app.dependency_overrides.pop(get_orchestrator, None)

    def test_update_other_tenant_memory_returns_403(self, auth_headers):
        """POST /memory/update with memory from other tenant returns 403."""
        from unittest.mock import AsyncMock, MagicMock

        from src.api.routes import get_orchestrator
        from src.core.exceptions import MemoryAccessDenied

        mock_orch = MagicMock()
        mock_orch.update = AsyncMock(
            side_effect=MemoryAccessDenied("Memory does not belong to tenant")
        )

        def override_orchestrator(request: Request):
            return mock_orch

        app.dependency_overrides[get_orchestrator] = override_orchestrator
        try:
            with TestClient(app, headers=auth_headers) as client:
                resp = client.post(
                    "/api/v1/memory/update",
                    json={
                        "memory_id": "12345678-1234-1234-1234-123456789012",
                        "feedback": "correct",
                    },
                )
            assert resp.status_code == 403
            assert "detail" in resp.json()
        finally:
            app.dependency_overrides.pop(get_orchestrator, None)


class TestMemoryItem:
    """Tests for MemoryItem response schema."""

    def test_memory_item_structure(self):
        """MemoryItem has expected fields."""
        from datetime import datetime

        from src.api.schemas import MemoryItem

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
        assert resp.eval_outcome is None
        assert resp.eval_reason is None

    def test_write_memory_response_eval_mode(self):
        """WriteMemoryResponse accepts eval_outcome and eval_reason when X-Eval-Mode used."""
        from src.api.schemas import WriteMemoryResponse

        resp = WriteMemoryResponse(
            success=True,
            memory_id=None,
            chunks_created=0,
            message="Skipped",
            eval_outcome="skipped",
            eval_reason="Below novelty threshold: 0.10 < 0.20",
        )
        assert resp.eval_outcome == "skipped"
        assert resp.eval_reason == "Below novelty threshold: 0.10 < 0.20"

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
