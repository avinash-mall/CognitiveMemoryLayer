"""Unit tests for mock embedding client."""

import pytest

from src.core.config import get_settings
from src.utils.embeddings import MockEmbeddingClient, get_embedding_client


@pytest.mark.asyncio
async def test_mock_embedding_dimensions():
    client = MockEmbeddingClient(dimensions=64)
    assert client.dimensions == 64


@pytest.mark.asyncio
async def test_mock_embed_deterministic():
    client = MockEmbeddingClient(dimensions=16)
    r1 = await client.embed("hello")
    r2 = await client.embed("hello")
    assert r1.embedding == r2.embedding
    assert len(r1.embedding) == 16
    assert r1.model == "mock"


@pytest.mark.asyncio
async def test_mock_embed_batch():
    client = MockEmbeddingClient(dimensions=8)
    results = await client.embed_batch(["a", "b"])
    assert len(results) == 2
    assert len(results[0].embedding) == 8
    assert results[0].embedding != results[1].embedding


def test_get_embedding_client_openai_without_key_falls_back_to_mock(monkeypatch):
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai")
    monkeypatch.delenv("EMBEDDING_INTERNAL__API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("EMBEDDING_INTERNAL__BASE_URL", raising=False)
    get_settings.cache_clear()
    client = get_embedding_client()
    assert isinstance(client, MockEmbeddingClient)
