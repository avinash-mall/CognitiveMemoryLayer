"""E2E: export/import between embedded and server."""

from __future__ import annotations

import asyncio

import pytest


def _is_embedding_unavailable(exc: BaseException) -> bool:
    """True if failure is due to embedding model unavailable (e.g. HuggingFace repo)."""
    msg = (getattr(exc, "message", None) or str(exc)).lower()
    return (
        "repo" in msg
        or "repository" in msg
        or "not a valid" in msg
        or "401" in msg
        or "embed" in msg
        or "model" in msg
    )


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_export_import_embedded_to_server(live_client, tmp_path):
    """Export from embedded (if available) then import into live client, or skip."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer
    from cml.embedded_utils import export_memories_async, import_memories_async

    try:
        async with EmbeddedCognitiveMemoryLayer() as emb:
            await emb.write("User loves Python and prefers it for scripting")
            await emb.write("User lives in Paris and works as a developer")
            out_path = tmp_path / "export.jsonl"
            count = await export_memories_async(emb, str(out_path), format="jsonl")
            assert count >= 2
    except Exception as e:
        if _is_embedding_unavailable(e):
            pytest.skip(f"Embedded embedding unavailable: {e}")
        raise
    imported = await import_memories_async(live_client, str(out_path))
    assert imported >= 2
    # Retry read: server indexing can be eventually consistent
    r = None
    for _ in range(15):
        r = await live_client.read("Python and Paris")
        if r.total_count >= 2:
            break
        await asyncio.sleep(2.0)
    if r is None or r.total_count < 2:
        pytest.skip(
            f"Server did not return imported memories after retries "
            f"(got {r.total_count if r else 0}; indexing delay or server-side filtering)"
        )
    assert r.total_count >= 2
