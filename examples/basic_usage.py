"""
Basic Usage Example - Cognitive Memory Layer.
Set AUTH__API_KEY and MEMORY_API_URL (or CML_BASE_URL) in .env before running.

This example demonstrates the fundamental operations:
1. Storing memories
2. Retrieving memories
3. Updating memories
4. Forgetting memories

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
       docker compose -f docker/docker-compose.yml up api
    
    2. Install httpx and python-dotenv:
       pip install httpx python-dotenv
"""

import os
from pathlib import Path

try:
    from dotenv import load_dotenv
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from memory_client import CognitiveMemoryClient


def main():
    base_url = (os.environ.get("MEMORY_API_URL") or os.environ.get("CML_BASE_URL") or "").strip()
    if not base_url:
        raise SystemExit("Set MEMORY_API_URL or CML_BASE_URL in .env")
    client = CognitiveMemoryClient(
        base_url=base_url,
        api_key=os.environ.get("AUTH__API_KEY", ""),
    )
    
    session_id = "example-session-001"

    print("=" * 60)
    print("Cognitive Memory Layer - Basic Usage Example")
    print("=" * 60)

    # =========================================
    # 1. Store different types of memories (holistic: tenant from auth)
    # =========================================
    print("\n1. STORING MEMORIES")
    print("-" * 40)

    result = client.write(
        "The user's name is Alice and she works as a software engineer at TechCorp.",
        session_id=session_id,
        context_tags=["personal"],
        memory_type="semantic_fact",
    )
    print(f"✓ Stored semantic fact: {result.message}")

    result = client.write(
        "User prefers Python over JavaScript for backend development.",
        session_id=session_id,
        context_tags=["preference"],
        memory_type="preference",
    )
    print(f"✓ Stored preference: {result.message}")

    result = client.write(
        "User is allergic to shellfish - never recommend seafood restaurants with shellfish.",
        session_id=session_id,
        context_tags=["constraint"],
        memory_type="constraint",
    )
    print(f"✓ Stored constraint: {result.message}")

    result = client.write(
        "On 2024-01-15, user mentioned they are planning to learn Rust this year.",
        session_id=session_id,
        context_tags=["conversation"],
        memory_type="episodic_event",
    )
    print(f"✓ Stored episodic event: {result.message}")
    
    # Store a hypothesis (uncertain, needs confirmation)
    result = client.write(
        "User might be interested in machine learning based on their questions about PyTorch.",
        session_id=session_id,
        context_tags=["conversation"],
        memory_type="hypothesis",
    )
    print(f"✓ Stored hypothesis: {result.message}")

    # =========================================
    # 2. Retrieve memories
    # =========================================
    print("\n2. RETRIEVING MEMORIES")
    print("-" * 40)

    result = client.read("What programming languages does the user like?")
    print(f"\nQuery: 'What programming languages does the user like?'")
    print(f"Found {result.total_count} relevant memories ({result.elapsed_ms:.1f}ms)")
    for mem in result.memories[:3]:
        print(f"  - [{mem.type}] {mem.text[:60]}... (confidence: {mem.confidence:.0%})")
    
    result = client.read("Tell me about the user", format="llm_context")
    print(f"\nQuery: 'Tell me about the user' (LLM context format)")
    print("LLM Context:")
    print("-" * 40)
    if result.llm_context:
        # Print first 500 chars
        print(result.llm_context[:500])
        if len(result.llm_context) > 500:
            print("...")
    
    result = client.read(
        "dietary restrictions",
        memory_types=["constraint", "preference"],
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
    
    result = client.read("machine learning hypothesis")
    if result.memories:
        memory_id = result.memories[0].id
        print(f"Memory to update: {result.memories[0].text[:50]}...")
        update_result = client.update(memory_id=memory_id, feedback="correct")
        print(f"✓ Marked as correct (reinforced): version {update_result.get('version')}")
    
    # =========================================
    # 4. Get memory statistics
    # =========================================
    print("\n4. MEMORY STATISTICS")
    print("-" * 40)
    
    stats = client.stats()
    print(f"Total memories: {stats.total_memories}")
    print(f"Active memories: {stats.active_memories}")
    print(f"Average confidence: {stats.avg_confidence:.0%}")
    print(f"By type: {stats.by_type}")
    
    # =========================================
    # 5. Forget memories
    # =========================================
    print("\n5. FORGETTING MEMORIES")
    print("-" * 40)
    
    result = client.forget(query="planning to learn Rust", action="archive")
    print(f"✓ Archived {result.get('affected_count', 0)} memories about learning Rust")
    
    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)
    
    client.close()


if __name__ == "__main__":
    main()
