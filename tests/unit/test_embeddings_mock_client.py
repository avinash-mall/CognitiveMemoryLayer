"""Unit tests for mock embedding client.

Cache and get_embedding_client() behavior are covered in test_embeddings_client_cache.py.
"""

import pytest

from src.utils.embeddings import MockEmbeddingClient, clear_embedding_client_cache


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


def test_clear_embedding_client_cache_callable():
    """clear_embedding_client_cache() is callable and does not raise (smoke test)."""
    clear_embedding_client_cache()
