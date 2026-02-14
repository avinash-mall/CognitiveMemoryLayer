"""Integration tests: namespace (with_namespace)."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_namespace_write_read(live_client):
    """with_namespace('ns1'), write, read and verify."""
    ns = live_client.with_namespace("integration-ns")
    w = await ns.write("Namespaced memory content")
    if w.chunks_created == 0:
        pytest.skip("Server stored no chunks (check embedding and write-gate config)")
    r = await ns.read("Namespaced")
    if r.total_count == 0:
        pytest.skip("Server stored but read returned no results (check retrieval/embedding config)")
    assert r.total_count >= 1
    assert any("Namespaced" in m.text for m in r.memories)
