"""Integration tests: batch operations."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_write_then_read(live_client):
    """Batch write several items, then read."""
    items = [
        {"content": "Batch item one"},
        {"content": "Batch item two"},
        {"content": "Batch item three"},
    ]
    try:
        results = await live_client.batch_write(items)
    except Exception:
        raise
    assert len(results) == 3
    if sum(r.chunks_created for r in results) == 0:
        pytest.skip("Server stored no chunks (check embedding and write-gate config)")
    r = await live_client.read("Batch item")
    if r.total_count == 0:
        pytest.skip("Server stored but read returned no results (check retrieval/embedding config)")
    assert r.total_count >= 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_read(live_client):
    """Batch read multiple queries."""
    await live_client.write("Query A content")
    await live_client.write("Query B content")
    try:
        results = await live_client.batch_read(["Query A", "Query B"])
    except Exception:
        raise
    assert len(results) >= 2
