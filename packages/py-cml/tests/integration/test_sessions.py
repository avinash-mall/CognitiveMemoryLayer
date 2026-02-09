"""Integration tests: sessions."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_create_session(live_client):
    """Create session returns session_id."""
    try:
        sess = await live_client.create_session(name="integration-test-session")
    except Exception as e:
        if "404" in str(e) or "501" in str(e):
            pytest.skip("Server may not implement create_session")
        raise
    assert sess is not None
    assert sess.session_id


@pytest.mark.integration
@pytest.mark.asyncio
async def test_get_session_context(live_client):
    """Create session, write with session_id, get_session_context."""
    try:
        sess = await live_client.create_session(name="context-test")
    except Exception as e:
        if "404" in str(e) or "501" in str(e):
            pytest.skip("Server may not implement sessions")
        raise
    await live_client.write("Session-scoped memory", session_id=sess.session_id)
    try:
        ctx = await live_client.get_session_context(sess.session_id)
    except Exception as e:
        if "404" in str(e) or "501" in str(e):
            pytest.skip("Server may not implement get_session_context")
        raise
    assert ctx is not None
    assert ctx.session_id == sess.session_id
