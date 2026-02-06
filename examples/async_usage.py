"""
Async Usage Example - Cognitive Memory Layer

This example demonstrates asynchronous usage of the memory system,
ideal for high-performance applications, web servers, and concurrent operations.

Uses the holistic API: content, session_id, memory_type, query, format (tenant from API key).

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api

    2. Install httpx:
       pip install httpx
"""

import asyncio
import os
import sys

# Allow running from project root or examples/
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from memory_client import AsyncCognitiveMemoryClient


async def demo_basic_async():
    """Basic async operations."""
    print("\n--- Basic Async Operations ---\n")

    async with AsyncCognitiveMemoryClient() as client:
        # Health check
        health = await client.health()
        print(f"API Status: {health['status']}")

        # Write a memory (content, session_id, memory_type; tenant from API key)
        result = await client.write(
            content="User prefers async programming patterns.",
            session_id="async-demo-session",
            memory_type="preference",
        )
        print(f"Write result: {result}")

        # Read memories (query, format; tenant from API key)
        memories = await client.read(
            query="programming preferences",
            format="llm_context",
        )
        print(f"Found {memories.total_count} memories")
        if memories.llm_context:
            print(f"Context: {memories.llm_context[:200]}...")


async def demo_concurrent_writes():
    """Demonstrate concurrent memory writes."""
    print("\n--- Concurrent Writes ---\n")

    session_id = "concurrent-demo-session"
    memories_to_store = [
        ("User's favorite color is blue", "preference"),
        ("User works as a software developer", "semantic_fact"),
        ("User is learning Rust programming", "episodic_event"),
        ("User prefers dark mode interfaces", "preference"),
        ("User lives in San Francisco", "semantic_fact"),
    ]

    async with AsyncCognitiveMemoryClient() as client:
        tasks = [
            client.write(
                content=content,
                session_id=session_id,
                memory_type=mtype,
            )
            for content, mtype in memories_to_store
        ]
        print(f"Writing {len(tasks)} memories concurrently...")
        results = await asyncio.gather(*tasks)
        success_count = sum(1 for r in results if r.success)
        print(f"✓ Successfully stored {success_count}/{len(tasks)} memories")


async def demo_concurrent_reads():
    """Demonstrate concurrent memory reads."""
    print("\n--- Concurrent Reads ---\n")

    session_id = "concurrent-demo-session"
    queries = [
        "What is the user's job?",
        "What are the user's preferences?",
        "Where does the user live?",
        "What is the user learning?",
    ]

    async with AsyncCognitiveMemoryClient() as client:
        tasks = [
            client.read(query=q, format="packet")
            for q in queries
        ]
        print(f"Executing {len(tasks)} queries concurrently...")
        results = await asyncio.gather(*tasks)
        for query, result in zip(queries, results):
            print(f"\nQuery: '{query}'")
            print(f"  Found: {result.total_count} memories ({result.elapsed_ms:.1f}ms)")
            if result.memories:
                print(f"  Top: {result.memories[0].text[:50]}...")


async def demo_pipeline():
    """Demonstrate a processing pipeline with memory."""
    print("\n--- Processing Pipeline ---\n")

    async with AsyncCognitiveMemoryClient() as client:
        sessions = ["session-1", "session-2", "session-3"]

        async def process_session(sid: str):
            await client.write(
                content=f"This is a test memory for {sid}",
                session_id=sid,
                memory_type="episodic_event",
            )
            result = await client.read(query="test memory")
            return sid, result.total_count

        print(f"Processing {len(sessions)} sessions concurrently...")
        tasks = [process_session(sid) for sid in sessions]
        results = await asyncio.gather(*tasks)
        for session_id, count in results:
            print(f"  {session_id}: {count} memories")


async def demo_batch_processing():
    """Demonstrate batch processing patterns."""
    print("\n--- Batch Processing ---\n")

    session_id = "batch-demo-session"
    items = [f"Memory item {i}: Important information #{i}" for i in range(20)]

    async with AsyncCognitiveMemoryClient() as client:
        batch_size = 5
        print(f"Processing {len(items)} items in batches of {batch_size}...")
        for i in range(0, len(items), batch_size):
            batch = items[i : i + batch_size]
            tasks = [
                client.write(content=item, session_id=session_id)
                for item in batch
            ]
            results = await asyncio.gather(*tasks)
            success = sum(1 for r in results if r.success)
            print(f"  Batch {i // batch_size + 1}: {success}/{len(batch)} stored")
        try:
            stats = await client.stats()
            print(f"\nTotal memories stored: {stats.total_memories}")
        except Exception:
            print(f"\nBatch complete: {len(items)} items submitted.")


async def demo_with_timeout():
    """Demonstrate timeout handling."""
    print("\n--- Timeout Handling ---\n")

    async with AsyncCognitiveMemoryClient(timeout=5.0) as client:
        try:
            result = await asyncio.wait_for(client.health(), timeout=2.0)
            print(f"✓ Health check completed: {result['status']}")
        except asyncio.TimeoutError:
            print("✗ Request timed out")


async def main():
    """Run all async demos."""
    print("=" * 60)
    print("Cognitive Memory Layer - Async Usage Examples")
    print("=" * 60)

    try:
        await demo_basic_async()
        await demo_concurrent_writes()
        await demo_concurrent_reads()
        await demo_pipeline()
        await demo_batch_processing()
        await demo_with_timeout()

        print("\n" + "=" * 60)
        print("All async demos completed!")
        print("=" * 60)

    except Exception as e:
        if "ConnectError" in str(type(e)):
            print("\n✗ Could not connect to API. Start it with:")
            print("  docker compose -f docker/docker-compose.yml up api")
        else:
            raise


if __name__ == "__main__":
    asyncio.run(main())
