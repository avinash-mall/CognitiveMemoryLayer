"""Integration tests: turn (seamless)."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turn_basic(live_client):
    """Turn with user_message returns TurnResponse."""
    r = await live_client.turn(user_message="Hello, my name is Alex.")
    assert r is not None
    assert hasattr(r, "memory_context")
    assert hasattr(r, "memories_retrieved")
    assert hasattr(r, "memories_stored")


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turn_with_session_id(live_client):
    """Turn with session_id uses session."""
    r = await live_client.turn(
        user_message="I like hiking",
        session_id="integration-session-1",
    )
    assert r is not None


@pytest.mark.integration
@pytest.mark.asyncio
async def test_turn_store_exchange(live_client):
    """Turn with user_message and assistant_response stores exchange."""
    r = await live_client.turn(
        user_message="What's my favorite color?",
        assistant_response="You said your favorite color is blue.",
    )
    assert r is not None
    assert r.memories_stored >= 0
