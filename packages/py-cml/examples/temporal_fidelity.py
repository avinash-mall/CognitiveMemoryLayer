"""
Temporal Fidelity Example

Demonstrates how to use the timestamp parameter to store memories with
specific event times, enabling historical data replay and temporal reasoning.

This is particularly useful for:
- Benchmark evaluations (e.g., Locomo) that replay historical conversations
- Data migration from other systems with preserved timestamps
- Testing temporal reasoning and memory consolidation over time
"""

from datetime import datetime, timezone, timedelta
from cml import CognitiveMemoryLayer


def main():
    # Initialize the client (reads CML_API_KEY and CML_BASE_URL from .env)
    memory = CognitiveMemoryLayer()

    print("=" * 60)
    print("Temporal Fidelity Example")
    print("=" * 60)

    # Example 1: Store historical memories with specific timestamps
    print("\n1. Storing historical memories with specific timestamps...")

    # Simulate a conversation from 6 months ago
    six_months_ago = datetime.now(timezone.utc) - timedelta(days=180)
    
    memory.write(
        "User mentioned they prefer dark mode for all applications",
        timestamp=six_months_ago,
        session_id="historical_session_1",
        context_tags=["preference", "ui"]
    )
    print(f"   ✓ Stored memory from {six_months_ago.date()}")

    # Simulate another memory from 3 months ago
    three_months_ago = datetime.now(timezone.utc) - timedelta(days=90)
    
    memory.write(
        "User switched to light mode because of eye strain",
        timestamp=three_months_ago,
        session_id="historical_session_2",
        context_tags=["preference", "ui", "health"]
    )
    print(f"   ✓ Stored memory from {three_months_ago.date()}")

    # Example 2: Process historical conversation turns
    print("\n2. Processing historical conversation turns...")

    one_month_ago = datetime.now(timezone.utc) - timedelta(days=30)
    
    result = memory.turn(
        user_message="What's my preferred theme setting?",
        assistant_response="Based on your recent feedback, you prefer light mode to reduce eye strain.",
        timestamp=one_month_ago,
        session_id="historical_session_3"
    )
    print(f"   ✓ Processed turn from {one_month_ago.date()}")
    print(f"   ✓ Stored {result.chunks_created} chunks")

    # Example 3: Store current memories (timestamp defaults to now)
    print("\n3. Storing current memories (timestamp defaults to now)...")

    memory.write(
        "User is currently testing the temporal fidelity feature",
        session_id="current_session",
        context_tags=["testing"]
    )
    print("   ✓ Stored memory with current timestamp (default)")

    # Example 4: Retrieve memories and see temporal ordering
    print("\n4. Retrieving memories to verify temporal ordering...")

    memories = memory.read(
        "user theme preference",
        max_results=5
    )

    print(f"\n   Found {len(memories.memories)} relevant memories:")
    for mem in memories.memories:
        print(f"   - [{mem.timestamp.date()}] {mem.text[:60]}...")
        print(f"     Relevance: {mem.relevance:.2f}")

    # Example 5: Benchmark/evaluation scenario
    print("\n5. Benchmark evaluation scenario (Locomo-style)...")
    print("   Simulating historical conversation replay...")

    # Simulate a multi-turn conversation from a specific date
    session_date = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)
    
    conversation_turns = [
        ("Hello, I'm new here", "Welcome! How can I help you today?"),
        ("I need help setting up my profile", "I'd be happy to help. What would you like to configure?"),
        ("I prefer dark themes", "Got it! I'll remember that you prefer dark themes."),
    ]

    for i, (user_msg, assistant_msg) in enumerate(conversation_turns):
        # Each turn happens 5 minutes after the previous one
        turn_time = session_date + timedelta(minutes=i * 5)
        
        memory.turn(
            user_message=user_msg,
            assistant_response=assistant_msg,
            timestamp=turn_time,
            session_id="benchmark_session"
        )
        print(f"   ✓ Turn {i+1} at {turn_time.strftime('%H:%M')}")

    print("\n6. Verifying temporal fidelity...")
    
    # Retrieve memories from the benchmark session
    benchmark_memories = memory.read(
        "user preferences",
        max_results=10
    )

    print(f"   Retrieved {len(benchmark_memories.memories)} memories")
    print("   Timestamps are preserved for accurate temporal reasoning!")

    # Get statistics
    print("\n7. Memory statistics...")
    stats = memory.stats()
    print(f"   Total memories: {stats.total_memories}")
    print(f"   Active memories: {stats.active_memories}")

    print("\n" + "=" * 60)
    print("Temporal fidelity example completed!")
    print("=" * 60)
    print("\nKey takeaways:")
    print("- Use 'timestamp' parameter to store historical data with correct event times")
    print("- Timestamps default to 'now' when not provided (backward compatible)")
    print("- Essential for benchmark evaluations and data migration")
    print("- Enables accurate temporal reasoning and memory consolidation")

    memory.close()


if __name__ == "__main__":
    main()
