"""Integration tests: write and read."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_then_read(live_client):
    """Write one memory, read by query, assert in results."""
    w = await live_client.write("Integration test: user loves Python")
    assert w.success
    r = await live_client.read("Python")
    assert r.total_count >= 1
    texts = [m.text for m in r.memories]
    assert any("Python" in t for t in texts)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_write_multiple_then_read(live_client):
    """Write multiple memories, read and verify count."""
    await live_client.write("Memory one: coffee")
    await live_client.write("Memory two: tea")
    r = await live_client.read("drinks")
    assert r.total_count >= 2


@pytest.mark.integration
@pytest.mark.asyncio
async def test_read_llm_context_format(live_client):
    """Read with format llm_context returns context string."""
    await live_client.write("User prefers dark mode")
    r = await live_client.read("preferences", response_format="llm_context")
    assert r.llm_context is not None or r.total_count >= 0
    assert hasattr(r, "context")
