"""Relation (OpenIE-style) extraction from text."""
import json
import re
from typing import List, Optional

from ..core.schemas import Relation
from ..utils.llm import LLMClient

RELATION_EXTRACTION_PROMPT = """Extract relationships from the following text using Open Information Extraction.

Text: {text}

For each relationship, identify:
1. Subject (who/what)
2. Predicate (the relationship/action)
3. Object (who/what is affected)

Return JSON array of triples:
[
  {{"subject": "John", "predicate": "lives_in", "object": "Paris", "confidence": 0.9}},
  {{"subject": "user", "predicate": "prefers", "object": "vegetarian food", "confidence": 0.85}}
]

Rules:
- Normalize predicates to snake_case
- Assign confidence based on how explicit the relationship is
- Use "user" as subject for first-person statements

Return only the JSON array."""


class RelationExtractor:
    """Extracts relation triples from text using LLM."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def extract(
        self,
        text: str,
        entities: Optional[List[str]] = None,
    ) -> List[Relation]:
        prompt = RELATION_EXTRACTION_PROMPT.format(text=text)
        if entities:
            prompt += f"\n\nKnown entities: {', '.join(entities)}"
        try:
            response = await self.llm.complete(
                prompt, temperature=0.0, max_tokens=500
            )
            data = json.loads(response)
            if not isinstance(data, list):
                data = [data]
            return [
                Relation(
                    subject=r.get("subject", ""),
                    predicate=self._normalize_predicate(r.get("predicate", "")),
                    object=r.get("object", ""),
                    confidence=float(r.get("confidence", 0.8)),
                )
                for r in data
                if r.get("subject") and r.get("predicate") and r.get("object")
            ]
        except (json.JSONDecodeError, KeyError, TypeError):
            return []

    def _normalize_predicate(self, predicate: str) -> str:
        normalized = re.sub(r"[\s\-]+", "_", predicate.lower())
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)
        return normalized
