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
    except Exception as e:
        if "404" in str(e) or "501" in str(e):
            pytest.skip("Server may not implement batch_write")
        raise
    assert len(results) == 3
    r = await live_client.read("Batch item")
    assert r.total_count >= 3


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_read(live_client):
    """Batch read multiple queries."""
    await live_client.write("Query A content")
    await live_client.write("Query B content")
    try:
        results = await live_client.batch_read(["Query A", "Query B"])
    except Exception as e:
        if "404" in str(e) or "501" in str(e):
            pytest.skip("Server may not implement batch_read")
        raise
    assert len(results) >= 2
