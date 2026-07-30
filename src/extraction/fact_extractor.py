"""Fact extraction from conversation text (for reconsolidation)."""

import json
from dataclasses import dataclass

from ..utils.llm import JSON_ARRAY_SYSTEM_PROMPT, LLMClient
from ..utils.parsing import strip_markdown_fences

FACT_EXTRACTION_PROMPT = """Extract durable, generalizable facts from this conversation turn.
Focus on: preferences, identity details, relationships, beliefs, and stated facts.
Ignore: greetings, questions, transient chat.

Conversation:
{text}

Return a JSON array of facts. Each fact: {{"text": "...", "type": "semantic_fact"}}.
Types: semantic_fact, preference, identity, relationship.
Return only the JSON array, no other text."""


@dataclass
class ExtractedFact:
    """A single extracted fact from text."""

    text: str
    type: str = "semantic_fact"


class LLMFactExtractor:
    """LLM-based fact extraction for reconsolidation.

    Returns [] when ``llm_client`` is None, which is also the heuristic-mode
    behaviour — so there is no separate no-op base class to instantiate.
    """

    def __init__(self, llm_client: LLMClient | None) -> None:
        self.llm = llm_client

    async def extract(self, text: str) -> list[ExtractedFact]:
        """Extract facts from text using LLM. Returns [] when LLM disabled."""
        if not text or not text.strip() or self.llm is None:
            return []
        prompt = FACT_EXTRACTION_PROMPT.format(text=text.strip())
        try:
            response = await self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=500,
                system_prompt=JSON_ARRAY_SYSTEM_PROMPT,
            )
            data = json.loads(strip_markdown_fences(response))
            if not isinstance(data, list):
                data = [data]
            return [
                ExtractedFact(
                    text=item.get("text", ""),
                    type=item.get("type", "semantic_fact"),
                )
                for item in data
                if item.get("text")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []
