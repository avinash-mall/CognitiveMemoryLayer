"""Entity extraction from text (LLM-based)."""
import json
from typing import List, Optional

from ..core.schemas import EntityMention
from ..utils.llm import LLMClient

ENTITY_EXTRACTION_PROMPT = """Extract named entities from the following text.

Text: {text}

For each entity, provide:
1. The exact text as it appears
2. A normalized/canonical form
3. The entity type (PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY, PRODUCT, EVENT, CONCEPT, PREFERENCE, ATTRIBUTE)

Return JSON array:
[
  {{"text": "Paris", "normalized": "Paris, France", "type": "LOCATION"}},
  {{"text": "next Monday", "normalized": "2026-02-09", "type": "DATE"}}
]

Extract ALL meaningful entities. Return only the JSON array, no other text."""


class EntityExtractor:
    """Extracts named entities from text using LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def extract(
        self,
        text: str,
        context: Optional[str] = None,
    ) -> List[EntityMention]:
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        try:
            response = await self.llm.complete(
                prompt, temperature=0.0, max_tokens=500
            )
            data = json.loads(response)
            if not isinstance(data, list):
                data = [data]
            return [
                EntityMention(
                    text=e.get("text", ""),
                    normalized=e.get("normalized", e.get("text", "")),
                    entity_type=e.get("type", "CONCEPT"),
                )
                for e in data
                if e.get("text")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    async def extract_batch(
        self, texts: List[str]
    ) -> List[List[EntityMention]]:
        import asyncio
        return await asyncio.gather(*[self.extract(t) for t in texts])
