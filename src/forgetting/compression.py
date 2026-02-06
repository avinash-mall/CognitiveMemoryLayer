"""LLM-based compression (summarization) for active forgetting."""

from typing import Optional

from ..utils.llm import LLMClient

# System prompt for one-sentence gist extraction (fits small models like Llama 3.2 1B)
COMPRESSION_SYSTEM = (
    "You summarize text into one short sentence (under 100 characters). "
    "Keep only the main fact or idea. Output nothing else."
)


async def summarize_for_compression(
    text: str,
    max_chars: int = 100,
    llm_client: Optional[LLMClient] = None,
) -> str:
    """
    Produce a compressed (summarized) version of text for forgetting.
    If llm_client is provided and text is longer than max_chars, use LLM; else truncate.
    """
    if len(text) <= max_chars:
        return text
    if llm_client is None:
        return text[: max_chars - 3] + "..."
    try:
        prompt = f"Summarize in one short sentence (max {max_chars} chars):\n\n{text[:2000]}"
        summary = await llm_client.complete(
            prompt,
            temperature=0.0,
            max_tokens=80,
            system_prompt=COMPRESSION_SYSTEM,
        )
        summary = summary.strip()
        if len(summary) > max_chars:
            summary = summary[: max_chars - 3] + "..."
        return summary or text[: max_chars - 3] + "..."
    except Exception:
        return text[: max_chars - 3] + "..."
