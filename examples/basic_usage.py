"""
Basic Usage Example - Cognitive Memory Layer.

Uses the py-cml package. Set CML_API_KEY and CML_BASE_URL in .env before running.

Demonstrates: write, read, update, forget, stats.

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
       docker compose -f docker/docker-compose.yml up api
    2. pip install cognitive-memory-layer  # or from repo root: pip install -e .
"""

import os
from pathlib import Path
from uuid import UUID

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from cml import CognitiveMemoryLayer
from cml.models.enums import MemoryType


def main():
    base_url = (os.environ.get("CML_BASE_URL") or "").strip() or "http://localhost:8000"
    with CognitiveMemoryLayer(
        api_key=os.environ.get("CML_API_KEY"),
        base_url=base_url,
    ) as memory:
        _run_basic_usage(memory, "example-session-001")


def _run_basic_usage(memory: CognitiveMemoryLayer, session_id: str) -> None:

    print("=" * 60)
    print("Cognitive Memory Layer - Basic Usage Example")
    print("=" * 60)

    # 1. Store different types of memories
    print("\n1. STORING MEMORIES")
    print("-" * 40)
    memory.write(
        "The user's name is Alice and she works as a software engineer at TechCorp.",
        session_id=session_id,
        context_tags=["personal"],
        memory_type=MemoryType.SEMANTIC_FACT,
    )
    print("✓ Stored semantic fact")
    memory.write(
        "User prefers Python over JavaScript for backend development.",
        session_id=session_id,
        context_tags=["preference"],
        memory_type=MemoryType.PREFERENCE,
    )
    print("✓ Stored preference")
    memory.write(
        "User is allergic to shellfish - never recommend seafood restaurants with shellfish.",
        session_id=session_id,
        context_tags=["constraint"],
        memory_type=MemoryType.CONSTRAINT,
    )
    print("✓ Stored constraint")
    memory.write(
        "On 2024-01-15, user mentioned they are planning to learn Rust this year.",
        session_id=session_id,
        context_tags=["conversation"],
        memory_type=MemoryType.EPISODIC_EVENT,
    )
    print("✓ Stored episodic event")
    memory.write(
        "User might be interested in machine learning based on their questions about PyTorch.",
        session_id=session_id,
        context_tags=["conversation"],
        memory_type=MemoryType.HYPOTHESIS,
    )
    print("✓ Stored hypothesis")

    # 2. Retrieve memories
    print("\n2. RETRIEVING MEMORIES")
    print("-" * 40)
    result = memory.read("What programming languages does the user like?")
    print("\nQuery: 'What programming languages does the user like?'")
    print(f"Found {result.total_count} relevant memories ({result.elapsed_ms:.1f}ms)")
    for mem in result.memories[:3]:
        print(f"  - [{mem.type}] {mem.text[:60]}... (confidence: {mem.confidence:.0%})")
    result = memory.read(
        "Tell me about the user",
        response_format="llm_context",
    )
    print("\nQuery: 'Tell me about the user' (LLM context format)")
    print("LLM Context:")
    print("-" * 40)
    ctx = result.context or result.llm_context
    if ctx:
        print(ctx[:500] + ("..." if len(ctx) > 500 else ""))
    result = memory.read(
        "dietary restrictions",
        memory_types=[MemoryType.CONSTRAINT, MemoryType.PREFERENCE],
    )
    print("\nQuery: 'dietary restrictions' (constraints & preferences only)")
    print(f"Found {result.total_count} memories")
    for mem in result.memories:
        print(f"  - [{mem.type}] {mem.text}")
    if result.constraints:
        print("  Constraints (server-extracted when enabled):")
        for c in result.constraints:
            print(f"    - {c.text}")

    # 3. Update memories with feedback
    print("\n3. UPDATING MEMORIES")
    print("-" * 40)
    result = memory.read("machine learning hypothesis")
    if result.memories:
        mem = result.memories[0]
        mid = mem.id if isinstance(mem.id, UUID) else UUID(str(mem.id))
        memory.update(memory_id=mid, feedback="correct")
        print(f"✓ Marked as correct (reinforced): {mem.text[:50]}...")

    # 4. Memory statistics
    print("\n4. MEMORY STATISTICS")
    print("-" * 40)
    stats = memory.stats()
    print(f"Total memories: {stats.total_memories}")
    print(f"Active memories: {stats.active_memories}")
    print(f"Average confidence: {stats.avg_confidence:.0%}")
    print(f"By type: {stats.by_type}")

    # 5. Forget memories
    print("\n5. FORGETTING MEMORIES")
    print("-" * 40)
    resp = memory.forget(query="planning to learn Rust", action="archive")
    print(f"✓ Archived {resp.affected_count} memories about learning Rust")

    print("\n" + "=" * 60)
    print("Example completed successfully!")
    print("=" * 60)


if __name__ == "__main__":
    main()

