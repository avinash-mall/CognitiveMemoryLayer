"""Async py-cml example with gather, batch reads, and streaming."""

from __future__ import annotations

import asyncio

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import AsyncCognitiveMemoryLayer
from cml.models import MemoryType

EXAMPLE_META = {
    "name": "async_example",
    "kind": "python",
    "summary": "Async gather writes, batch reads, and SSE streaming.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 90,
}


async def main_async() -> int:
    print_header("CML Async Example")
    try:
        async with AsyncCognitiveMemoryLayer(config=build_cml_config()) as memory:
            writes = [
                ("User likes hiking in the mountains.", MemoryType.PREFERENCE),
                ("User prefers morning workouts.", MemoryType.PREFERENCE),
                ("User is training for a marathon.", MemoryType.SEMANTIC_FACT),
            ]
            results = await asyncio.gather(
                *[
                    memory.write(text, session_id="examples-async", memory_type=memory_type)
                    for text, memory_type in writes
                ]
            )
            print(
                f"Stored {sum(1 for result in results if result.success)}/{len(results)} memories"
            )

            query_results = await memory.batch_read(
                ["exercise habits", "outdoor activities"],
                max_results=3,
                response_format="packet",
            )
            for result in query_results:
                print(f"Batch read '{result.query}' -> {result.total_count} results")

            streamed = []
            async for item in memory.read_stream("user preferences", max_results=3):
                streamed.append(item.text)
            print(f"Streamed {len(streamed)} items")
            for text in streamed:
                print(f"  - {text}")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
