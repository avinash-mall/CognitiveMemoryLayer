"""Fact extraction from conversation text (for reconsolidation)."""
from dataclasses import dataclass
from typing import List


@dataclass
class ExtractedFact:
    """A single extracted fact from text."""

    text: str
    type: str = "semantic_fact"


class FactExtractor:
    """
    Extracts facts from conversation text for reconsolidation.
    Override extract() for LLM-based or NER-based extraction.
    """

    async def extract(self, text: str) -> List[ExtractedFact]:
        """Extract facts from text. Default: no extraction."""
        return []
