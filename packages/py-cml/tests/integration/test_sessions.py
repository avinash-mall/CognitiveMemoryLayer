"""Integration tests: sessions."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_session(live_client):
    """Create session returns session_id."""
    sess = await live_client.create_session(name="integration-test-session")
    assert sess is not None
    assert sess.session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_session_context(live_client):
    """Create session, write with session_id, get_session_context."""
    sess = await live_client.create_session(name="context-test")
    await live_client.write("Session-scoped memory", session_id=sess.session_id)
    ctx = await live_client.get_session_context(sess.session_id)
    assert ctx is not None
    assert ctx.session_id == sess.session_id
