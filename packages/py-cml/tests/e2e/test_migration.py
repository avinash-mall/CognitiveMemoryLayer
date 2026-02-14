"""E2E: export/import between embedded and server."""

from __future__ import annotations

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
            await emb.write("Exported memory one")
            await emb.write("Exported memory two")
            out_path = tmp_path / "export.jsonl"
            count = await export_memories_async(emb, str(out_path), format="jsonl")
            assert count >= 2
    except Exception as e:
        if _is_embedding_unavailable(e):
            pytest.skip(f"Embedded embedding unavailable: {e}")
        raise
    imported = await import_memories_async(live_client, str(out_path))
    assert imported >= 2
    r = await live_client.read("Exported")
    assert r.total_count >= 2
