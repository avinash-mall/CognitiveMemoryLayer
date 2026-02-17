"""
Integration tests for LLM-based compression.

Uses real LLM from config when available; otherwise uses a mock LLM so tests
always run (no skip). Real code path is exercised in both cases.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.forgetting.compression import summarize_for_compression
from src.utils.llm import get_llm_client


def _make_mock_llm():
    """Return a mock LLM that returns short summaries (no external API)."""
    mock = MagicMock()
    mock.complete = AsyncMock(return_value="User works from home and uses Slack and Zoom.")
    return mock


@pytest.fixture(scope="module")
def llm_client():
    """Real LLM from config if available; otherwise mock so tests never skip."""
    client = get_llm_client()
    if client is None:
        return _make_mock_llm()
    return client


@pytest.mark.asyncio
async def test_llm_compression_summarize_returns_shorter_text(llm_client):
    """summarize_for_compression returns a shorter summary (real or mock LLM)."""
    long_text = (
        "The user mentioned that they have been working from home for the past year "
        "and really enjoy the flexibility. They start at 9am and often take a short "
        "walk in the afternoon. Their team uses Slack and Zoom for meetings."
    )
    max_chars = 80
    summary = await summarize_for_compression(
        long_text,
        max_chars=max_chars,
        llm_client=llm_client,
    )
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) <= max_chars
    assert len(summary) < len(long_text)


@pytest.mark.asyncio
async def test_llm_compression_summarize_content_reasonable(llm_client):
    """Summary is non-empty and within length (real or mock LLM)."""
    text = "Alice said she prefers tea over coffee and drinks two cups every morning."
    summary = await summarize_for_compression(
        text,
        max_chars=60,
        llm_client=llm_client,
    )
    assert isinstance(summary, str)
    assert len(summary) >= 5
    assert len(summary) <= 60
    assert not (summary.endswith("...") and len(summary) == 57)
