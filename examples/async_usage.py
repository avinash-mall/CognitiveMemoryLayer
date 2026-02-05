"""
Async Usage Example - Cognitive Memory Layer

This example demonstrates asynchronous usage of the memory system,
ideal for high-performance applications, web servers, and concurrent operations.

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api
    
    2. Install httpx:
       pip install httpx
"""

import asyncio
from memory_client import AsyncCognitiveMemoryClient


async def demo_basic_async():
    """Basic async operations."""
    print("\n--- Basic Async Operations ---\n")
    
    async with AsyncCognitiveMemoryClient() as client:
        # Health check
        health = await client.health()
        print(f"API Status: {health['status']}")
        
        # Write a memory
        result = await client.write(
            scope="session",
            scope_id="async-demo-session",
            content="User prefers async programming patterns.",
            memory_type="preference"
        )
        print(f"Write result: {result}")
        
        # Read memories
        memories = await client.read(
            scope="session",
            scope_id="async-demo-session",
            query="programming preferences",
            format="llm_context"
        )
        print(f"Found {memories.total_count} memories")
        if memories.llm_context:
            print(f"Context: {memories.llm_context[:200]}...")


async def demo_concurrent_writes():
    """Demonstrate concurrent memory writes."""
    print("\n--- Concurrent Writes ---\n")
    
    scope = "session"
    scope_id = "concurrent-demo-session"
    
    memories_to_store = [
        ("User's favorite color is blue", "preference"),
        ("User works as a software developer", "semantic_fact"),
        ("User is learning Rust programming", "episodic_event"),
        ("User prefers dark mode interfaces", "preference"),
        ("User lives in San Francisco", "semantic_fact"),
    ]
    
    async with AsyncCognitiveMemoryClient() as client:
        # Create all write tasks
        tasks = [
            client.write(scope=scope, scope_id=scope_id, content=content, memory_type=mtype)
            for content, mtype in memories_to_store
        ]
        
        # Execute concurrently
        print(f"Writing {len(tasks)} memories concurrently...")
        results = await asyncio.gather(*tasks)
        
        # Report results
        success_count = sum(1 for r in results if r.success)
        print(f"✓ Successfully stored {success_count}/{len(tasks)} memories")


async def demo_concurrent_reads():
    """Demonstrate concurrent memory reads."""
    print("\n--- Concurrent Reads ---\n")
    
    scope = "session"
    scope_id = "concurrent-demo-session"
    
    queries = [
        "What is the user's job?",
        "What are the user's preferences?",
        "Where does the user live?",
        "What is the user learning?",
    ]
    
    async with AsyncCognitiveMemoryClient() as client:
        # Create all read tasks
        tasks = [
            client.read(scope=scope, scope_id=scope_id, query=q, format="packet")
            for q in queries
        ]
        
        # Execute concurrently
        print(f"Executing {len(tasks)} queries concurrently...")
        results = await asyncio.gather(*tasks)
        
        # Report results
        for query, result in zip(queries, results):
            print(f"\nQuery: '{query}'")
            print(f"  Found: {result.total_count} memories ({result.elapsed_ms:.1f}ms)")
            if result.memories:
                print(f"  Top: {result.memories[0].text[:50]}...")


async def demo_pipeline():
    """Demonstrate a processing pipeline with memory."""
    print("\n--- Processing Pipeline ---\n")
    
    async with AsyncCognitiveMemoryClient() as client:
        # Simulate processing multiple sessions
        sessions = ["session-1", "session-2", "session-3"]
        
        async def process_session(session_id: str):
            """Process a single session: write and read."""
            # Write some data
            await client.write(
                scope="session",
                scope_id=session_id,
                content=f"This is a test memory for {session_id}",
                memory_type="episodic_event"
            )
            
            # Read it back
            result = await client.read(
                scope="session",
                scope_id=session_id,
                query="test memory"
            )
            
            return session_id, result.total_count
        
        # Process all sessions concurrently
        print(f"Processing {len(sessions)} sessions concurrently...")
        tasks = [process_session(sid) for sid in sessions]
        results = await asyncio.gather(*tasks)
        
        for session_id, count in results:
            print(f"  {session_id}: {count} memories")


async def demo_batch_processing():
    """Demonstrate batch processing patterns."""
    print("\n--- Batch Processing ---\n")
    
    scope = "session"
    scope_id = "batch-demo-session"
    
    # Large batch of items to process
    items = [f"Memory item {i}: Important information #{i}" for i in range(20)]
    
    async with AsyncCognitiveMemoryClient() as client:
        # Process in batches of 5
        batch_size = 5
        
        print(f"Processing {len(items)} items in batches of {batch_size}...")
        
        for i in range(0, len(items), batch_size):
            batch = items[i:i + batch_size]
            
            # Create tasks for this batch
            tasks = [
                client.write(scope=scope, scope_id=scope_id, content=item)
                for item in batch
            ]
            
            # Wait for batch to complete
            results = await asyncio.gather(*tasks)
            success = sum(1 for r in results if r.success)
            
            print(f"  Batch {i // batch_size + 1}: {success}/{len(batch)} stored")
        
        # Verify total stored
        stats_response = await client._request("GET", f"/memory/stats/session/{scope_id}")
        print(f"\nTotal memories stored: {stats_response.get('total_memories', 0)}")


async def demo_with_timeout():
    """Demonstrate timeout handling."""
    print("\n--- Timeout Handling ---\n")
    
    async with AsyncCognitiveMemoryClient(timeout=5.0) as client:
        try:
            # This should work
            result = await asyncio.wait_for(
                client.health(),
                timeout=2.0
            )
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
