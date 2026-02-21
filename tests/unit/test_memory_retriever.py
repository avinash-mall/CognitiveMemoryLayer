"""Unit tests for MemoryRetriever (embedding reuse, etc.)."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils.embeddings import EmbeddingResult


@pytest.mark.asyncio
async def test_retrieve_embeds_query_once(mock_embeddings):
    """MemoryRetriever.retrieve() calls embeddings.embed exactly once per query,
    passing the embedding to HybridRetriever to avoid duplicate embedding calls.
    """
    from src.retrieval.memory_retriever import MemoryRetriever

    # Ensure mock returns proper EmbeddingResult
    mock_embeddings.embed = AsyncMock(
        return_value=EmbeddingResult(
            embedding=[0.1] * 1536,
            model="test",
            dimensions=1536,
            tokens_used=10,
        )
    )

    hippocampal = MagicMock()
    hippocampal.embeddings = mock_embeddings
    hippocampal.search = AsyncMock(return_value=[])

    neocortical = MagicMock()
    neocortical.facts = MagicMock()
    neocortical.facts.get_facts_by_category = AsyncMock(return_value=[])
    neocortical.get_fact = AsyncMock(return_value=None)
    neocortical.text_search = AsyncMock(return_value=[])

    retriever = MemoryRetriever(
        hippocampal=hippocampal,
        neocortical=neocortical,
        llm_client=None,
    )

    # Decision-style query triggers both VECTOR and CONSTRAINTS steps
    await retriever.retrieve("tenant-1", "Should I order the lobster?")

    # Embeddings.embed must be called exactly once
    assert mock_embeddings.embed.await_count == 1
    mock_embeddings.embed.assert_called_once_with("Should I order the lobster?")
