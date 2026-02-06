"""
Integration tests for LLM-based compression using a real vLLM server.

These tests are skipped unless a vLLM OpenAI-compatible endpoint is available:
  - Set LLM__BASE_URL (or LLM__VLLM_BASE_URL / VLLM_BASE_URL) to the vLLM endpoint, e.g. http://vllm:8000/v1.
  - Start vLLM first, e.g.:
      docker compose -f docker/docker-compose.yml --profile vllm up -d vllm
    then:
      docker compose run --rm -e LLM__PROVIDER=vllm -e LLM__BASE_URL=http://vllm:8000/v1 app \\
        pytest tests/integration/test_phase8_vllm_compression.py -v
  - Without a configured and reachable vLLM server, the tests are skipped with a clear message.
"""

import os

import pytest

from src.forgetting.compression import summarize_for_compression
from src.utils.llm import OpenAICompatibleClient


def _vllm_base_url() -> str | None:
    """Base URL for vLLM from env (LLM__BASE_URL or legacy vars)."""
    return (
        os.environ.get("LLM__BASE_URL")
        or os.environ.get("LLM__VLLM_BASE_URL")
        or os.environ.get("VLLM_BASE_URL")
    )


SKIP_NO_VLLM = (
    "vLLM not configured: set LLM__BASE_URL (or LLM__VLLM_BASE_URL / VLLM_BASE_URL) to the vLLM "
    "OpenAI-compatible endpoint (e.g. http://vllm:8000/v1) and ensure the vLLM server is running."
)


@pytest.fixture(scope="module")
def vllm_client():
    """OpenAI-compatible client for vLLM when configured; otherwise skip with a clear message."""
    base_url = _vllm_base_url()
    if not base_url:
        pytest.skip(SKIP_NO_VLLM)
    return OpenAICompatibleClient(base_url=base_url)


async def _skip_if_vllm_unreachable(vllm_client):
    """Skip if vLLM server is not reachable (e.g. connection refused or model still loading)."""
    try:
        await vllm_client.complete("Hi", max_tokens=5)
    except Exception as e:
        pytest.skip(f"vLLM server unreachable: {e}")


@pytest.mark.asyncio
async def test_vllm_summarize_for_compression_returns_shorter_text(vllm_client):
    """Real vLLM: summarize_for_compression returns a shorter summary."""
    await _skip_if_vllm_unreachable(vllm_client)
    long_text = (
        "The user mentioned that they have been working from home for the past year "
        "and really enjoy the flexibility. They start at 9am and often take a short "
        "walk in the afternoon. Their team uses Slack and Zoom for meetings."
    )
    max_chars = 80
    summary = await summarize_for_compression(
        long_text,
        max_chars=max_chars,
        llm_client=vllm_client,
    )
    assert isinstance(summary, str)
    assert len(summary) > 0
    assert len(summary) <= max_chars
    # Summary should be meaningfully shorter than original
    assert len(summary) < len(long_text)


@pytest.mark.asyncio
async def test_vllm_summarize_for_compression_content_reasonable(vllm_client):
    """Real vLLM: summary is non-empty and looks like a sentence."""
    await _skip_if_vllm_unreachable(vllm_client)
    text = "Alice said she prefers tea over coffee and drinks two cups every morning."
    summary = await summarize_for_compression(
        text,
        max_chars=60,
        llm_client=vllm_client,
    )
    assert isinstance(summary, str)
    assert len(summary) >= 5
    assert len(summary) <= 60
    # Should not be raw truncation with ellipsis at exactly max_chars-3
    assert not (summary.endswith("...") and len(summary) == 57)
