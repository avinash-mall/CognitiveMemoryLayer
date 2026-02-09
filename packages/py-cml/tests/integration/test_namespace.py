"""Integration tests: namespace (with_namespace)."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_with_namespace_write_read(live_client):
    """with_namespace('ns1'), write, read and verify."""
    ns = live_client.with_namespace("integration-ns")
    await ns.write("Namespaced memory content")
    r = await ns.read("Namespaced")
    assert r.total_count >= 1
    assert any("Namespaced" in m.text for m in r.memories)
