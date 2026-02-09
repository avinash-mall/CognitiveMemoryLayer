"""Export/import utilities for embedded mode migration."""

from __future__ import annotations

import asyncio
from typing import Literal

from cml.async_client import AsyncCognitiveMemoryLayer
from cml.client import CognitiveMemoryLayer
from cml.embedded import EmbeddedCognitiveMemoryLayer


async def export_memories_async(
    source: EmbeddedCognitiveMemoryLayer,
    output_path: str,
    format: Literal["json", "jsonl"] = "jsonl",
) -> int:
    """Export all memories from an embedded instance to a file."""
    if not isinstance(source, EmbeddedCognitiveMemoryLayer):
        raise TypeError("source must be EmbeddedCognitiveMemoryLayer")
    source._ensure_initialized()
    store = source._orchestrator.hippocampal.store
    records = await store.scan(
        tenant_id=source._config.tenant_id,
        limit=100_000,
        offset=0,
    )
    count = 0
    if format == "jsonl":
        with open(output_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(rec.model_dump_json() + "\n")
                count += 1
    else:
        import json

        data = [rec.model_dump(mode="json") for rec in records]
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        count = len(data)
    return count


def export_memories(
    source: EmbeddedCognitiveMemoryLayer,
    output_path: str,
    format: Literal["json", "jsonl"] = "jsonl",
) -> int:
    """Synchronous wrapper for export_memories_async."""
    return asyncio.run(export_memories_async(source, output_path, format))


async def import_memories_async(
    target: CognitiveMemoryLayer | AsyncCognitiveMemoryLayer | EmbeddedCognitiveMemoryLayer,
    input_path: str,
) -> int:
    """Import memories from a JSONL or JSON file into any client."""
    from cml.embedded import EmbeddedCognitiveMemoryLayer

    count = 0
    with open(input_path, encoding="utf-8") as f:
        content = f.read()
    if content.strip().startswith("["):
        import json

        data = json.loads(content)
        lines = [json.dumps(item) for item in data]
    else:
        lines = [line.strip() for line in content.split("\n") if line.strip()]

    import json

    for line in lines:
        obj = json.loads(line)
        text = obj.get("text", "")
        meta = obj.get("metadata", {})

        if isinstance(target, (AsyncCognitiveMemoryLayer, EmbeddedCognitiveMemoryLayer)):
            await target.write(text, metadata=meta)
        else:
            target.write(text, metadata=meta)
        count += 1
    return count


def import_memories(
    target: CognitiveMemoryLayer | AsyncCognitiveMemoryLayer | EmbeddedCognitiveMemoryLayer,
    input_path: str,
) -> int:
    """Synchronous wrapper for import_memories_async."""
    return asyncio.run(import_memories_async(target, input_path))
