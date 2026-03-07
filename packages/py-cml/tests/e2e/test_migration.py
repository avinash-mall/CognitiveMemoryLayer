"""E2E: export/import between embedded and server."""

from __future__ import annotations

import asyncio

import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_export_import_embedded_to_server(live_client, tmp_path):
    """Export from embedded then import into the live client."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer
    from cml.embedded_utils import export_memories_async, import_memories_async

    async with EmbeddedCognitiveMemoryLayer() as emb:
        await emb.write("User loves Python and prefers it for scripting")
        await emb.write("User lives in Paris and works as a developer")
        out_path = tmp_path / "export.jsonl"
        count = await export_memories_async(emb, str(out_path), format="jsonl")
        assert count >= 2
    imported = await import_memories_async(live_client, str(out_path))
    assert imported >= 2
    # Retry read: server indexing can be eventually consistent
    r = None
    for _ in range(15):
        r = await live_client.read("Python and Paris")
        if r.total_count >= 2:
            break
        await asyncio.sleep(2.0)
    assert r is not None
    assert r.total_count >= 2, (
        f"Server did not return imported memories after retries (got {r.total_count})"
    )
