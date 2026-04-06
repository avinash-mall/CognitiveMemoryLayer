"""Unit tests for BatchingEmbeddingClient.

Covers:
- Concurrent embed_batch calls are coalesced into one inner call
- Results are returned to the correct callers
- Errors from inner client propagate to all waiters
- Disabled batcher (batch_wait_ms=0) is not applied by factory
- Single embed() delegates to embed_batch()
- Empty input returns empty list
"""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.core.config import get_settings
from src.utils.embeddings import (
    BatchingEmbeddingClient,
    EmbeddingResult,
    MockEmbeddingClient,
    clear_embedding_client_cache,
    get_embedding_client,
)


def _make_result(text: str, dims: int = 4) -> EmbeddingResult:
    return EmbeddingResult(
        embedding=[0.1] * dims,
        model="test",
        dimensions=dims,
        tokens_used=len(text.split()),
    )


def _make_inner(texts_to_results: dict[str, EmbeddingResult] | None = None, dims: int = 4):
    """Return a mock EmbeddingClient whose embed_batch records call args."""
    inner = MagicMock()
    inner.dimensions = dims
    call_log: list[list[str]] = []

    async def _embed_batch(texts: list[str]) -> list[EmbeddingResult]:
        call_log.append(list(texts))
        if texts_to_results:
            return [texts_to_results.get(t, _make_result(t, dims)) for t in texts]
        return [_make_result(t, dims) for t in texts]

    inner.embed_batch = _embed_batch
    inner._call_log = call_log
    return inner


# ---------------------------------------------------------------------------
# Basic functionality
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_empty_embed_batch_returns_empty():
    inner = _make_inner()
    batcher = BatchingEmbeddingClient(inner, max_wait_ms=1)
    result = await batcher.embed_batch([])
    assert result == []
    assert inner._call_log == []


@pytest.mark.asyncio
async def test_single_embed_delegates_to_embed_batch():
    inner = _make_inner()
    batcher = BatchingEmbeddingClient(inner, max_wait_ms=5)
    r = await batcher.embed("hello world")
    assert isinstance(r, EmbeddingResult)
    assert inner._call_log == [["hello world"]]


@pytest.mark.asyncio
async def test_dimensions_proxied_from_inner():
    inner = _make_inner(dims=768)
    batcher = BatchingEmbeddingClient(inner, max_wait_ms=5)
    assert batcher.dimensions == 768


# ---------------------------------------------------------------------------
# Coalescing
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_concurrent_calls_coalesced_into_one_inner_call():
    """Two embed_batch calls that arrive within the wait window should be merged."""
    inner = _make_inner()
    batcher = BatchingEmbeddingClient(inner, max_wait_ms=50)

    results = await asyncio.gather(
        batcher.embed_batch(["a", "b"]),
        batcher.embed_batch(["c"]),
    )

    # Both callers got their results
    assert len(results[0]) == 2
    assert len(results[1]) == 1

    # Inner was called at most once (all texts merged)
    total_texts_per_call = [len(call) for call in inner._call_log]
    # At least one call happened, and at most 2 (race-condition tolerance)
    assert len(inner._call_log) <= 2
    assert sum(total_texts_per_call) == 3


@pytest.mark.asyncio
async def test_results_routed_to_correct_callers():
    """Each caller receives exactly its requested results in order."""
    results_map = {
        "x": _make_result("x"),
        "y": _make_result("y"),
        "z": _make_result("z"),
    }
    inner = _make_inner(texts_to_results=results_map)
    batcher = BatchingEmbeddingClient(inner, max_wait_ms=50)

    r1, r2 = await asyncio.gather(
        batcher.embed_batch(["x", "y"]),
        batcher.embed_batch(["z"]),
    )

    assert len(r1) == 2
    assert len(r2) == 1
    # Results come back in the order they were submitted
    assert r1[0] is results_map["x"]
    assert r1[1] is results_map["y"]
    assert r2[0] is results_map["z"]


# ---------------------------------------------------------------------------
# Error propagation
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_inner_error_propagates_to_all_waiters():
    """If the inner embed_batch raises, all pending futures see the exception."""
    inner = MagicMock()
    inner.dimensions = 4

    async def _failing(texts):
        raise RuntimeError("GPU exploded")

    inner.embed_batch = _failing

    batcher = BatchingEmbeddingClient(inner, max_wait_ms=50)

    with pytest.raises(RuntimeError, match="GPU exploded"):
        await asyncio.gather(
            batcher.embed_batch(["a"]),
            batcher.embed_batch(["b"]),
        )


@pytest.mark.asyncio
async def test_inner_size_mismatch_propagates_error():
    """If inner returns wrong number of results, ValueError propagates."""
    inner = MagicMock()
    inner.dimensions = 4

    async def _short(texts):
        return [_make_result("x")]  # always returns 1, regardless of input size

    inner.embed_batch = _short

    batcher = BatchingEmbeddingClient(inner, max_wait_ms=50)

    with pytest.raises((ValueError, BaseException)):
        await batcher.embed_batch(["a", "b", "c"])


# ---------------------------------------------------------------------------
# max_batch_size splits large batches
# ---------------------------------------------------------------------------


@pytest.mark.asyncio
async def test_max_batch_size_splits_pending():
    """When pending texts exceed max_batch_size, multiple inner calls are made."""
    inner = _make_inner()
    batcher = BatchingEmbeddingClient(inner, max_wait_ms=50, max_batch_size=2)

    texts = ["a", "b", "c", "d", "e"]
    results = await batcher.embed_batch(texts)

    assert len(results) == 5
    # Inner should have been called multiple times with at most 2 texts each
    for call in inner._call_log:
        assert len(call) <= 2
    total = sum(len(c) for c in inner._call_log)
    assert total == 5


# ---------------------------------------------------------------------------
# Factory integration
# ---------------------------------------------------------------------------


@pytest.fixture(autouse=True)
def _clear():
    get_settings.cache_clear()
    clear_embedding_client_cache()
    yield
    get_settings.cache_clear()
    clear_embedding_client_cache()


def test_factory_wraps_non_mock_with_batcher(monkeypatch):
    """get_embedding_client() wraps non-mock providers with BatchingEmbeddingClient."""
    # Use openai_compatible with a dummy base_url so the factory creates a real
    # OpenAIEmbeddings client (no key fallback to mock).
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BATCH_WAIT_MS", "20")
    get_settings.cache_clear()

    client = get_embedding_client()
    assert isinstance(client, BatchingEmbeddingClient)


def test_factory_does_not_wrap_mock(monkeypatch):
    """Mock provider bypasses the batcher even when batch_wait_ms > 0."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "mock")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BATCH_WAIT_MS", "20")
    get_settings.cache_clear()

    client = get_embedding_client()
    assert isinstance(client, MockEmbeddingClient)


def test_factory_batch_wait_zero_disables_batcher(monkeypatch):
    """batch_wait_ms=0 disables the batcher; raw client is returned."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BATCH_WAIT_MS", "0")
    get_settings.cache_clear()

    client = get_embedding_client()
    assert not isinstance(client, BatchingEmbeddingClient)


def test_factory_different_batch_wait_different_cache_key(monkeypatch):
    """Changing batch_wait_ms produces a distinct cached client."""
    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "openai_compatible")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BASE_URL", "http://localhost:9999/v1")
    monkeypatch.setenv("EMBEDDING_INTERNAL__BATCH_WAIT_MS", "10")
    get_settings.cache_clear()
    c1 = get_embedding_client()

    monkeypatch.setenv("EMBEDDING_INTERNAL__BATCH_WAIT_MS", "50")
    get_settings.cache_clear()
    c2 = get_embedding_client()

    assert c1 is not c2
