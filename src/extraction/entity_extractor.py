"""Entity extraction from text (LLM-based)."""

import json

from ..core.schemas import EntityMention
from ..utils.llm import LLMClient


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code block fences from LLM output (LOW-12)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        # Remove first line (```json or ```) and last line (```)
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


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
        context: str | None = None,
    ) -> list[EntityMention]:
        prompt = ENTITY_EXTRACTION_PROMPT.format(text=text)
        if context:
            prompt = f"Context: {context}\n\n{prompt}"
        try:
            response = await self.llm.complete(prompt, temperature=0.0, max_tokens=500)
            data = json.loads(_strip_markdown_fences(response))
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

    async def extract_batch(self, texts: list[str]) -> list[list[EntityMention]]:
        """Extract entities from multiple texts in a single LLM call.

        Each text is labelled [0], [1], ... in the prompt so results can be
        reassembled in order.  Falls back to individual calls if the batch
        response cannot be parsed.
        """
        if not texts:
            return []
        if len(texts) == 1:
            return [await self.extract(texts[0])]

        # Build a single prompt with all texts
        sections = "\n\n".join(f"[{i}] {text.strip()}" for i, text in enumerate(texts))
        batch_prompt = (
            "Extract named entities from each of the following numbered texts.\n\n"
            + sections
            + "\n\nReturn a JSON object mapping each index (as a string key) to an array "
            'of entities.  Each entity: {"text": ..., "normalized": ..., "type": ...}.\n'
            'Example: {"0": [{"text": "Paris", "normalized": "Paris, France", '
            '"type": "LOCATION"}], "1": []}\n'
            "Return ONLY valid JSON, no other text."
        )

        try:
            raw = await self.llm.complete(batch_prompt, temperature=0.0, max_tokens=2000)
            data = json.loads(_strip_markdown_fences(raw))
            if not isinstance(data, dict):
                raise ValueError("Expected dict")

            results: list[list[EntityMention]] = []
            for i in range(len(texts)):
                entries = data.get(str(i), [])
                if not isinstance(entries, list):
                    entries = []
                results.append(
                    [
                        EntityMention(
                            text=e.get("text", ""),
                            normalized=str(e.get("normalized") or e.get("text", "")),
                            entity_type=e.get("type", "CONCEPT"),
                        )
                        for e in entries
                        if isinstance(e, dict) and e.get("text")
                    ]
                )
            return results
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: individual calls (original behaviour)
            import asyncio

            return list(await asyncio.gather(*[self.extract(t) for t in texts]))
