"""Integration tests: stats."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stats_returns_response(live_client):
    """Write then stats; assert counts present."""
    await live_client.write("Stats test memory")
    s = await live_client.stats()
    assert s is not None
    assert hasattr(s, "total_memories")
    assert hasattr(s, "active_memories")
    assert s.total_memories >= 0
