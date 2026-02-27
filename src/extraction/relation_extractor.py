"""Relation extraction from text.

Primary path:
- LLM OpenIE extraction when an LLM client is available.

Fallback path:
- spaCy dependency-based relation extraction for non-LLM mode.
"""

import asyncio
import json
import re

from ..core.schemas import Relation
from ..utils.llm import LLMClient
from ..utils.ner import extract_relations


def _strip_markdown_fences(text: str) -> str:
    """Strip markdown code block fences from LLM output (LOW-12)."""
    text = text.strip()
    if text.startswith("```"):
        lines = text.split("\n")
        if lines[-1].strip() == "```":
            lines = lines[1:-1]
        elif lines[0].startswith("```"):
            lines = lines[1:]
        text = "\n".join(lines).strip()
    return text


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
    """Extracts relation triples from text using LLM with spaCy fallback."""

    def __init__(self, llm_client: LLMClient | None = None) -> None:
        self.llm = llm_client

    async def extract(
        self,
        text: str,
        entities: list[str] | None = None,
    ) -> list[Relation]:
        if not self.llm:
            return self._spacy_extract(text)

        prompt = RELATION_EXTRACTION_PROMPT.format(text=text)
        if entities:
            prompt += f"\n\nKnown entities: {', '.join(entities)}"
        try:
            response = await self.llm.complete(prompt, temperature=0.0, max_tokens=500)
            data = json.loads(_strip_markdown_fences(response))
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
            return self._spacy_extract(text)

    def _spacy_extract(self, text: str) -> list[Relation]:
        return [
            Relation(
                subject=r.subject,
                predicate=self._normalize_predicate(r.predicate),
                object=r.object,
                confidence=r.confidence,
            )
            for r in extract_relations(text)
        ]

    async def extract_batch(self, items: list[tuple[str, list[str]]]) -> list[list[Relation]]:
        """Extract relations from multiple (text, entities) pairs in a single LLM call.

        Each pair is labelled [0], [1], ... in the prompt so results can be
        reassembled in order.  Falls back to individual calls if the batch
        response cannot be parsed.
        """
        if not items:
            return []
        if not self.llm:
            return [self._spacy_extract(text) for text, _entities in items]
        if len(items) == 1:
            text, entities = items[0]
            return [await self.extract(text, entities=entities)]

        sections = "\n\n".join(
            f"[{i}] {text.strip()}"
            + (f"\n  Known entities: {', '.join(entities)}" if entities else "")
            for i, (text, entities) in enumerate(items)
        )
        batch_prompt = (
            "Extract relation triples from each of the following numbered texts using "
            "Open Information Extraction.\n\n"
            + sections
            + "\n\nReturn a JSON object mapping each index (as a string key) to an array "
            "of triples.  Each triple: "
            '{"subject": ..., "predicate": ..., "object": ..., "confidence": 0.0-1.0}.\n'
            'Example: {"0": [{"subject": "user", "predicate": "prefers", '
            '"object": "vegetarian food", "confidence": 0.85}], "1": []}\n'
            "Return ONLY valid JSON, no other text."
        )
        try:
            raw = await self.llm.complete(batch_prompt, temperature=0.0, max_tokens=2000)
            data = json.loads(_strip_markdown_fences(raw))
            if not isinstance(data, dict):
                raise ValueError("Expected dict")

            results: list[list[Relation]] = []
            for i in range(len(items)):
                entries = data.get(str(i), [])
                if not isinstance(entries, list):
                    entries = []
                results.append(
                    [
                        Relation(
                            subject=r.get("subject", ""),
                            predicate=self._normalize_predicate(r.get("predicate", "")),
                            object=r.get("object", ""),
                            confidence=float(r.get("confidence", 0.8)),
                        )
                        for r in entries
                        if r.get("subject") and r.get("predicate") and r.get("object")
                    ]
                )
            return results
        except (json.JSONDecodeError, ValueError, KeyError, TypeError):
            # Fallback: individual calls (original behaviour)
            tasks = [self.extract(text, entities=entities) for text, entities in items]
            return list(await asyncio.gather(*tasks))

    def _normalize_predicate(self, predicate: str) -> str:
        normalized = re.sub(r"[\s\-]+", "_", predicate.lower())
        normalized = re.sub(r"[^a-z0-9_]", "", normalized)
        return normalized
