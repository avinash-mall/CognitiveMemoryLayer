"""
Basic Usage Example - Cognitive Memory Layer

This example demonstrates the fundamental operations:
1. Storing memories
2. Retrieving memories
3. Updating memories
4. Forgetting memories

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
       docker compose -f docker/docker-compose.yml up api
    
    2. Install httpx:
       pip install httpx
"""

from memory_client import CognitiveMemoryClient


def main():
    # Initialize the client
    client = CognitiveMemoryClient(
        base_url="http://localhost:8000",
        api_key="demo-key-123"
    )
    
    user_id = "example-user-001"
    
    print("=" * 60)
    print("Cognitive Memory Layer - Basic Usage Example")
    print("=" * 60)
    
    # =========================================
    # 1. Store different types of memories
    # =========================================
    print("\n1. STORING MEMORIES")
    print("-" * 40)
    
    # Store a semantic fact
    result = client.write(
        user_id=user_id,
        content="The user's name is Alice and she works as a software engineer at TechCorp.",
        memory_type="semantic_fact"
    )
    print(f"✓ Stored semantic fact: {result.message}")
    
    # Store a preference
    result = client.write(
        user_id=user_id,
        content="User prefers Python over JavaScript for backend development.",
        memory_type="preference"
    )
    print(f"✓ Stored preference: {result.message}")
    
    # Store a constraint (important - never auto-forgotten)
    result = client.write(
        user_id=user_id,
        content="User is allergic to shellfish - never recommend seafood restaurants with shellfish.",
        memory_type="constraint"
    )
    print(f"✓ Stored constraint: {result.message}")
    
    # Store an episodic event
    result = client.write(
        user_id=user_id,
        content="On 2024-01-15, user mentioned they are planning to learn Rust this year.",
        memory_type="episodic_event"
    )
    print(f"✓ Stored episodic event: {result.message}")
    
    # Store a hypothesis (uncertain, needs confirmation)
    result = client.write(
        user_id=user_id,
        content="User might be interested in machine learning based on their questions about PyTorch.",
        memory_type="hypothesis"
    )
    print(f"✓ Stored hypothesis: {result.message}")
    
    # =========================================
    # 2. Retrieve memories
    # =========================================
    print("\n2. RETRIEVING MEMORIES")
    print("-" * 40)
    
    # Basic retrieval
    result = client.read(
        user_id=user_id,
        query="What programming languages does the user like?"
    )
    print(f"\nQuery: 'What programming languages does the user like?'")
    print(f"Found {result.total_count} relevant memories ({result.elapsed_ms:.1f}ms)")
    for mem in result.memories[:3]:
        print(f"  - [{mem.type}] {mem.text[:60]}... (confidence: {mem.confidence:.0%})")
    
    # Retrieve with LLM-ready context
    result = client.read(
        user_id=user_id,
        query="Tell me about the user",
        format="llm_context"
    )
    print(f"\nQuery: 'Tell me about the user' (LLM context format)")
    print("LLM Context:")
    print("-" * 40)
    if result.llm_context:
        # Print first 500 chars
        print(result.llm_context[:500])
        if len(result.llm_context) > 500:
            print("...")
    
    # Filter by memory type
    result = client.read(
        user_id=user_id,
        query="dietary restrictions",
        memory_types=["constraint", "preference"]
    )
    print(f"\nQuery: 'dietary restrictions' (constraints & preferences only)")
    print(f"Found {result.total_count} memories")
    for mem in result.memories:
        print(f"  - [{mem.type}] {mem.text}")
    
    # =========================================
    # 3. Update memories with feedback
    # =========================================
    print("\n3. UPDATING MEMORIES")
    print("-" * 40)
    
    # Get a memory to update
    result = client.read(user_id=user_id, query="machine learning hypothesis")
    
    if result.memories:
        memory_id = result.memories[0].id
        print(f"Memory to update: {result.memories[0].text[:50]}...")
        
        # Confirm the hypothesis (increases confidence)
        update_result = client.update(
            user_id=user_id,
            memory_id=memory_id,
            feedback="correct"
        )
        print(f"✓ Marked as correct (reinforced): version {update_result.get('version')}")
    
    # =========================================
    # 4. Get memory statistics
    # =========================================
    print("\n4. MEMORY STATISTICS")
    print("-" * 40)
    
    stats = client.stats(user_id)
    print(f"User: {stats.user_id}")
    print(f"Total memories: {stats.total_memories}")
    print(f"Active memories: {stats.active_memories}")
    print(f"Average confidence: {stats.avg_confidence:.0%}")
    print(f"By type: {stats.by_type}")
    
    # =========================================
    # 5. Forget memories
    # =========================================
    print("\n5. FORGETTING MEMORIES")
    print("-" * 40)
    
    # Archive old memories (soft delete, can be recovered)
    result = client.forget(
        user_id=user_id,
        query="planning to learn Rust",
        action="archive"
    )
    print(f"✓ Archived {result.get('affected_count', 0)} memories about learning Rust")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    client.close()


if __name__ == "__main__":
    main()
