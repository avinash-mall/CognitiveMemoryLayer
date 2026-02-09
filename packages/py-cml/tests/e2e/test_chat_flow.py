"""E2E: multi-turn chat flow."""

import pytest


@pytest.mark.e2e
@pytest.mark.asyncio
async def test_multi_turn_conversation(live_client):
    """Several turns with session_id; assert turn responses have memory_context or counts."""
    session_id = "e2e-chat-session"
    r1 = await live_client.turn(
        user_message="Hi, my name is Sam.",
        session_id=session_id,
    )
    assert r1 is not None
    assert hasattr(r1, "memory_context")
    assert hasattr(r1, "memories_retrieved") and hasattr(r1, "memories_stored")

    r2 = await live_client.turn(
        user_message="What do you know about me?",
        session_id=session_id,
    )
    assert r2 is not None

    await live_client.turn(
        user_message="I also love hiking.",
        session_id=session_id,
    )
    r3 = await live_client.turn(
        user_message="What are my hobbies?",
        session_id=session_id,
    )
    assert r3 is not None
    assert r3.memory_context is not None or r3.memories_retrieved >= 0
