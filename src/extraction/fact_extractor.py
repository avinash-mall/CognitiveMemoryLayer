"""Fact extraction from conversation text (for reconsolidation)."""
import json
from dataclasses import dataclass
from typing import List, Optional

from ..utils.llm import LLMClient

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


class FactExtractor:
    """
    Extracts facts from conversation text for reconsolidation.
    Default no-op; use LLMFactExtractor for LLM-based extraction.
    """

    async def extract(self, text: str) -> List[ExtractedFact]:
        """Extract facts from text. Default: no extraction."""
        return []


class LLMFactExtractor(FactExtractor):
    """LLM-based fact extraction using the same client as summarization (e.g. vLLM)."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def extract(self, text: str) -> List[ExtractedFact]:
        """Extract facts from text using LLM."""
        if not text or not text.strip():
            return []
        prompt = FACT_EXTRACTION_PROMPT.format(text=text.strip())
        try:
            response = await self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=500,
                system_prompt="You are a JSON generator. Always respond with a valid JSON array only, no markdown.",
            )
            raw = response.strip()
            if raw.startswith("```"):
                lines = raw.split("\n")
                if lines[0].startswith("```"):
                    lines = lines[1:]
                if lines and lines[-1].strip() == "```":
                    lines = lines[:-1]
                raw = "\n".join(lines)
            data = json.loads(raw)
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
