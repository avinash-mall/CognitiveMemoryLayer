"""Compression (summarization) helpers for active forgetting."""

from typing import Protocol

from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

# System prompt for one-sentence gist extraction (fits small models / short summary)
COMPRESSION_SYSTEM = (
    "You summarize text into one short sentence (under 100 characters). "
    "Keep only the main fact or idea. Output nothing else."
)


class SummarizerBackend(Protocol):
    """Async summarizer backend contract (e.g., Hugging Face summarizer API)."""

    async def summarize(self, text: str, *, max_chars: int | None = None) -> str: ...


def _truncate(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    return text[: max_chars - 3].rstrip() + "..."


async def summarize_for_compression(
    text: str,
    max_chars: int = 100,
    llm_client: LLMClient | None = None,
    summarizer_backend: SummarizerBackend | None = None,
) -> str:
    """
    Produce a compressed (summarized) version of text for forgetting.

    Preference order when text exceeds ``max_chars``:
    1) shared summarizer backend API (if provided)
    2) LLM completion fallback (if provided)
    3) deterministic truncation
    """
    cleaned = " ".join((text or "").split())
    if not cleaned:
        return ""

    if len(cleaned) <= max_chars:
        return cleaned

    if summarizer_backend is not None:
        try:
            summary = await summarizer_backend.summarize(cleaned, max_chars=max_chars)
            summary = " ".join((summary or "").split())
            if summary:
                return _truncate(summary, max_chars)
        except Exception as exc:
            logger.warning("compression_summarizer_backend_failed", extra={"error": str(exc)})

    if llm_client is None:
        return _truncate(cleaned, max_chars)

    try:
        prompt = f"Summarize in one short sentence (max {max_chars} chars):\n\n{cleaned[:2000]}"
        summary = await llm_client.complete(
            prompt,
            temperature=0.0,
            max_tokens=80,
            system_prompt=COMPRESSION_SYSTEM,
        )
        summary = " ".join((summary or "").split())
        if summary:
            return _truncate(summary, max_chars)
        return _truncate(cleaned, max_chars)
    except Exception:
        return _truncate(cleaned, max_chars)
