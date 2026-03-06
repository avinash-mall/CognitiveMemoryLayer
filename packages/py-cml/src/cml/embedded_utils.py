"""Export/import utilities for embedded mode migration."""

from __future__ import annotations

import asyncio
from typing import Any, Literal

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
    from datetime import datetime

    for line in lines:
        obj = json.loads(line)
        text = obj.get("text", "")
        meta = obj.get("metadata", {})

        write_kwargs: dict[str, Any] = {"metadata": meta}

        mem_type = obj.get("type")
        if mem_type:
            write_kwargs["memory_type"] = mem_type

        ts = obj.get("timestamp")
        if ts:
            try:
                write_kwargs["timestamp"] = datetime.fromisoformat(str(ts))
            except (ValueError, TypeError):
                pass

        ctx_tags = obj.get("context_tags")
        if ctx_tags and isinstance(ctx_tags, list):
            write_kwargs["context_tags"] = ctx_tags

        ns = obj.get("namespace")
        if ns:
            write_kwargs["namespace"] = ns

        session_id = obj.get("source_session_id")
        if session_id:
            write_kwargs["session_id"] = session_id

        confidence = obj.get("confidence")
        if confidence is not None:
            meta.setdefault("_imported_confidence", confidence)

        if isinstance(target, (AsyncCognitiveMemoryLayer, EmbeddedCognitiveMemoryLayer)):
            await target.write(text, **write_kwargs)
        else:
            target.write(text, **write_kwargs)
        count += 1
    return count


def import_memories(
    target: CognitiveMemoryLayer | AsyncCognitiveMemoryLayer | EmbeddedCognitiveMemoryLayer,
    input_path: str,
) -> int:
    """Synchronous wrapper for import_memories_async."""
    return asyncio.run(import_memories_async(target, input_path))
