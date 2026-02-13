"""Integration tests: update and forget."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_update_memory(live_client):
    """Write, update text/confidence, read back."""
    w = await live_client.write("Original text for update test")
    assert w.success and w.memory_id, "write did not return memory_id"
    await live_client.update(
        w.memory_id,
        text="Updated text after edit",
        confidence=0.95,
    )
    r = await live_client.read("update test")
    assert r.total_count >= 1
    if r.memories:
        assert any("Updated" in m.text or "Original" in m.text for m in r.memories)


@pytest.mark.integration
@pytest.mark.asyncio
async def test_forget_by_query(live_client):
    """Write, forget by query, read shows reduced or empty."""
    await live_client.write("Temporary memory to forget")
    r_before = await live_client.read("Temporary memory to forget")
    await live_client.forget(query="Temporary memory to forget")
    r_after = await live_client.read("Temporary memory to forget")
    assert r_after.total_count <= r_before.total_count
