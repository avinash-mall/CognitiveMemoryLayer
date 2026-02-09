"""Integration tests: stats."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_stats_returns_response(live_client):
    """Write then stats; assert counts present."""
    await live_client.write("Stats test memory")
    try:
        s = await live_client.stats()
    except Exception as e:
        if "404" in str(e) or "501" in str(e):
            pytest.skip("Server may not implement stats")
        raise
    assert s is not None
    assert hasattr(s, "total_memories")
    assert hasattr(s, "active_memories")
    assert s.total_memories >= 0
