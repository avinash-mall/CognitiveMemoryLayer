"""E2E tests for full API memory lifecycle."""

import json
import os

import pytest
from fastapi.testclient import TestClient

from src.api.app import app
from src.core.config import get_settings


class _MockLLMClient:
    """Mock LLM for e2e tests so no real API/Ollama is required."""

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: str | None = None,
    ) -> str:
        # Return valid chunk JSON so SemanticChunker parses it
        text = prompt.split("Text to chunk:")[-1] if "Text to chunk:" in prompt else prompt
        text = text.strip()[:500]
        return json.dumps(
            [
                {"type": "statement", "text": text, "salience": 0.7, "confidence": 0.8},
            ]
        )

    async def complete_json(self, prompt: str, schema=None, temperature: float = 0.0):
        return {"result": "ok"}


@pytest.fixture
def client(monkeypatch):
    """Test client with auth headers. Uses mock embeddings and mock LLM when no real API keys."""
    monkeypatch.setenv("AUTH__API_KEY", "demo-key-123")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key-456")
    monkeypatch.setenv("AUTH__DEFAULT_TENANT_ID", "demo")
    get_settings.cache_clear()
    mock_llm = _MockLLMClient()
    monkeypatch.setattr(
        "src.memory.orchestrator.get_internal_llm_client",
        lambda: mock_llm,
    )
    if not os.environ.get("OPENAI_API_KEY"):
        from src.utils.embeddings import MockEmbeddingClient

        dims = get_settings().embedding.dimensions
        mock_emb = MockEmbeddingClient(dimensions=dims)
        monkeypatch.setattr(
            "src.memory.orchestrator.get_embedding_client",
            lambda: mock_emb,
        )
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


def test_full_memory_lifecycle(client):
    """Test write -> read -> update -> forget flow."""
    session_id = "e2e-test-session"

    write_resp = client.post(
        "/api/v1/memory/write",
        json={
            "content": "I prefer vegetarian food and I live in Paris",
            "session_id": session_id,
            "context_tags": ["conversation", "preference"],
        },
    )
    assert write_resp.status_code == 200
    write_data = write_resp.json()
    assert write_data["success"] is True

    read_resp = client.post(
        "/api/v1/memory/read",
        json={
            "query": "What food do I like?",
            "context_filter": ["conversation", "preference"],
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
                "memory_id": str(memory_id),
                "feedback": "correct",
            },
        )
        assert update_resp.status_code == 200

    forget_resp = client.post(
        "/api/v1/memory/forget",
        json={
            "query": "vegetarian",
        },
    )
    assert forget_resp.status_code == 200


def test_unauthorized_access():
    """Test that unauthorized requests are rejected (401). Use a distinct tenant so rate limit doesn't apply."""
    with TestClient(app) as client_no_auth:
        resp = client_no_auth.post(
            "/api/v1/memory/write",
            json={"content": "test"},
            headers={
                "X-Tenant-ID": "e2e-unauth"
            },  # distinct tenant so rate limit bucket is separate
        )
        assert (
            resp.status_code == 401
        ), f"Expected 401 Unauthorized, got {resp.status_code}: {resp.json()}"


def test_health_response_structure(client):
    """Test health returns expected response structure."""
    resp = client.get("/api/v1/health")
    assert resp.status_code == 200
    data = resp.json()
    assert "status" in data
    assert data["status"] == "healthy"
    assert "timestamp" in data
