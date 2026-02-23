"""Async usage of py-cml with asyncio.

Set CML_API_KEY and CML_BASE_URL in .env (same value as server AUTH__API_KEY for local dev).
"""

import asyncio
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from cml import AsyncCognitiveMemoryLayer


def _memory_config():
    base_url = (os.environ.get("CML_BASE_URL") or "").strip() or "http://localhost:8000"
    return {
        "api_key": os.environ.get("CML_API_KEY"),
        "base_url": base_url,
    }


async def demo_basic():
    """Basic async: write, read, batch_read."""
    async with AsyncCognitiveMemoryLayer(**_memory_config()) as memory:
        await asyncio.gather(
            memory.write("User likes hiking in the mountains"),
            memory.write("User prefers morning workouts"),
            memory.write("User is training for a marathon"),
        )
        result = await memory.read("exercise habits")
        print(f"Found {result.total_count} memories about exercise")
        for mem in result.memories:
            print(f"  - {mem.text}")
        results = await memory.batch_read(
            ["exercise habits", "outdoor activities", "fitness goals"]
        )
        for r in results:
            print(f"\nQuery: {r.query} -> {r.total_count} results")


async def demo_concurrent():
    """Concurrent writes and reads."""
    session_id = "async-demo-session"
    items = [
        ("User's favorite color is blue", "preference"),
        ("User works as a software developer", "semantic_fact"),
        ("User prefers dark mode interfaces", "preference"),
    ]
    async with AsyncCognitiveMemoryLayer(**_memory_config()) as memory:
        tasks = [
            memory.write(content, session_id=session_id, memory_type=mtype)
            for content, mtype in items
        ]
        results = await asyncio.gather(*tasks)
        print(f"Stored {sum(1 for r in results if r.success)}/{len(items)} memories")
        queries = ["user preferences", "user job"]
        read_results = await asyncio.gather(
            *[memory.read(q, response_format="packet") for q in queries]
        )
        for q, r in zip(queries, read_results, strict=False):
            print(f"  '{q}': {r.total_count} memories")


async def demo_pipeline():
    """Processing pipeline with per-session writes and reads."""
    async with AsyncCognitiveMemoryLayer(**_memory_config()) as memory:
        sessions = ["session-1", "session-2"]

        async def process(sid: str):
            await memory.write(
                f"Test memory for {sid}",
                session_id=sid,
                memory_type="episodic_event",
            )
            r = await memory.read("test memory")
            return sid, r.total_count

        results = await asyncio.gather(*[process(s) for s in sessions])
        for sid, count in results:
            print(f"  {sid}: {count} memories")


async def main():
    print("=" * 60)
    print("Async Usage - Cognitive Memory Layer (py-cml)")
    print("=" * 60)
    try:
        await demo_basic()
        print("\n--- Concurrent ---")
        await demo_concurrent()
        print("\n--- Pipeline ---")
        await demo_pipeline()
        print("\n" + "=" * 60)
        print("Done.")
    except Exception as e:
        if "Connect" in str(type(e).__name__) or "Connection" in str(e):
            print(
                "\n✗ Could not connect. Start API: docker compose -f docker/docker-compose.yml up api"
            )
        else:
            raise


if __name__ == "__main__":
    asyncio.run(main())
