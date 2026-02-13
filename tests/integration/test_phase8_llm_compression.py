"""
Integration tests for LLM-based compression using the default configured LLM (.env).

These tests are skipped when the LLM is not available:
  - Not configured: set LLM__PROVIDER and LLM__BASE_URL or LLM__API_KEY in .env.
  - Unreachable or rate-limited: tests skip instead of fail (e.g. 429, connection errors).
"""

import pytest

from src.forgetting.compression import summarize_for_compression
from src.utils.llm import get_llm_client

SKIP_NO_LLM = "Default LLM not configured for completion: set LLM__PROVIDER and LLM__BASE_URL or LLM__API_KEY in .env."


def _skip_if_llm_unavailable(reason: str) -> None:
    """Skip the current test when LLM is unavailable (rate limit, connection, etc.)."""
    pytest.skip(reason)


@pytest.fixture(scope="module")
def llm_client():
    """Default LLM client from config; skip if not configured for completion."""
    client = get_llm_client()
    if client is None:
        pytest.skip(SKIP_NO_LLM)
    return client


async def _ensure_llm_reachable(llm_client):
    """Ensure default LLM is reachable; skip test if rate-limited, connection error, or other API failure."""
    try:
        await llm_client.complete("Hi", max_tokens=5)
    except Exception as e:  # noqa: BLE001
        _skip_if_llm_unavailable(f"LLM unavailable or rate-limited: {type(e).__name__}: {e}")


@pytest.mark.asyncio
@pytest.mark.requires_llm
async def test_llm_compression_summarize_returns_shorter_text(llm_client):
    """Default LLM: summarize_for_compression returns a shorter summary."""
    await _ensure_llm_reachable(llm_client)
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
    # Summary should be meaningfully shorter than original
    assert len(summary) < len(long_text)


@pytest.mark.asyncio
@pytest.mark.requires_llm
async def test_llm_compression_summarize_content_reasonable(llm_client):
    """Default LLM: summary is non-empty and looks like a sentence."""
    await _ensure_llm_reachable(llm_client)
    text = "Alice said she prefers tea over coffee and drinks two cups every morning."
    summary = await summarize_for_compression(
        text,
        max_chars=60,
        llm_client=llm_client,
    )
    assert isinstance(summary, str)
    assert len(summary) >= 5
    assert len(summary) <= 60
    # Should not be raw truncation with ellipsis at exactly max_chars-3
    assert not (summary.endswith("...") and len(summary) == 57)
