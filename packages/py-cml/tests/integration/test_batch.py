"""Integration tests: batch operations."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_batch_write_then_read(live_client):
    """Batch write several items, then read."""
    items = [
        {"content": "Batch item: user enjoys hiking in the mountains on weekends"},
        {"content": "Batch item: user drinks coffee every morning before work"},
        {"content": "Batch item: user is learning Python programming language"},
    ]
    try:
        results = await live_client.batch_write(items)
    except Exception:
        raise
    assert len(results) == 3
    chunks_created = [r.chunks_created for r in results]
    assert sum(chunks_created) > 0
    assert sum(1 for c in chunks_created if c > 0) >= 2, (
        f"Expected at least 2 batch writes to succeed, got: {chunks_created}"
    )
    r = await live_client.read("Batch item")
    assert r.total_count >= 2


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
