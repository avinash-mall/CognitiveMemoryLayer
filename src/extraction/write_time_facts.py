"""Write-time fact extraction: populate the semantic store at write time.

Uses rule-based patterns (no LLM) to keep latency minimal on the hot
write path.  Only extracts high-confidence, well-structured facts.
Write-time facts receive lower initial confidence (0.6) than
consolidation-derived facts (0.8) so that the consolidation pipeline
can still refine them.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..memory.neocortical.schemas import FactCategory

if TYPE_CHECKING:
    from ..memory.working.models import SemanticChunk


@dataclass
class ExtractedFact:
    """A structured fact extracted at write-time."""

    key: str
    category: FactCategory
    predicate: str
    value: str
    confidence: float


# ── Rule-based patterns ─────────────────────────────────────────────
# Each tuple: (compiled regex, key_template, category, confidence_boost)

_PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], str, FactCategory, float]] = [
    # "I prefer/like/love/enjoy/hate/dislike X"
    (
        re.compile(
            r"\b(?:i|my)\s+(?:prefer|like|love|enjoy|hate|dislike)\s+(.+)",
            re.IGNORECASE,
        ),
        "user:preference:{pred}",
        FactCategory.PREFERENCE,
        0.7,
    ),
    # "My favorite X is Y"
    (
        re.compile(
            r"\b(?:my|the)\s+(?:favorite|favourite)\s+(\w+)\s+(?:is|are)\s+(.+)",
            re.IGNORECASE,
        ),
        "user:preference:{pred}",
        FactCategory.PREFERENCE,
        0.75,
    ),
]

_IDENTITY_PATTERNS: list[tuple[re.Pattern[str], str, FactCategory, float]] = [
    # "My name is X" / "I'm X" / "Call me X"
    (
        re.compile(
            r"\b(?:my name is|i'?m|call me|i am)\s+" r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.IGNORECASE,
        ),
        "user:identity:name",
        FactCategory.IDENTITY,
        0.85,
    ),
    # "I live in X" / "I'm from X" / "I moved to X"
    (
        re.compile(
            r"\b(?:i live in|i'?m from|i moved to|i'm based in)\s+"
            r"([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
            re.IGNORECASE,
        ),
        "user:location:current_city",
        FactCategory.LOCATION,
        0.7,
    ),
    # "I work as X" / "I'm a X" / "My job is X"
    (
        re.compile(
            r"\b(?:i work as|i'?m a|my job is|my occupation is)\s+(.+)",
            re.IGNORECASE,
        ),
        "user:occupation:role",
        FactCategory.OCCUPATION,
        0.7,
    ),
]

# Known predicate keywords for preference sub-categorisation
_PREDICATE_KEYWORDS: dict[str, list[str]] = {
    "cuisine": ["food", "restaurant", "eat", "cook", "meal", "cuisine", "dish"],
    "music": ["music", "song", "band", "listen", "genre", "artist"],
    "color": ["color", "colour"],
    "language": ["language", "speak"],
    "sport": ["sport", "play", "team", "game", "exercise"],
    "movie": ["movie", "film", "cinema", "watch"],
    "book": ["book", "read", "author", "novel"],
}

# Confidence baseline for write-time facts
_WRITE_TIME_CONFIDENCE_BASE: float = 0.6


class WriteTimeFactExtractor:
    """Extract structured facts from chunks at write-time.

    Processes preference, fact, and constraint chunk types.
    Uses purely rule-based pattern matching — no LLM calls.
    """

    def extract(self, chunk: SemanticChunk) -> list[ExtractedFact]:
        """Extract facts from a single chunk.

        Returns an empty list for chunk types that don't contain facts.
        """
        from ..memory.working.models import ChunkType

        fact_bearing_types = {
            ChunkType.PREFERENCE,
            ChunkType.FACT,
            ChunkType.CONSTRAINT,
        }
        if chunk.chunk_type not in fact_bearing_types:
            return []

        facts: list[ExtractedFact] = []
        text = chunk.text.strip()

        # Try preference patterns
        for pattern, key_template, category, conf_boost in _PREFERENCE_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            # Handle two-group patterns (e.g. "favorite X is Y")
            if match.lastindex and match.lastindex >= 2:
                value = match.group(2).strip().rstrip(".")
                predicate = match.group(1).strip().lower()
            else:
                value = match.group(1).strip().rstrip(".")
                predicate = _derive_predicate(value)
            key = key_template.format(pred=predicate)
            facts.append(
                ExtractedFact(
                    key=key,
                    category=category,
                    predicate=predicate,
                    value=value,
                    confidence=_WRITE_TIME_CONFIDENCE_BASE * conf_boost,
                )
            )

        # Try identity patterns
        for pattern, key_str, category, conf_boost in _IDENTITY_PATTERNS:
            match = pattern.search(text)
            if not match:
                continue
            value = match.group(1).strip()
            predicate = key_str.split(":")[-1]
            facts.append(
                ExtractedFact(
                    key=key_str,
                    category=category,
                    predicate=predicate,
                    value=value,
                    confidence=_WRITE_TIME_CONFIDENCE_BASE * conf_boost,
                )
            )

        return facts


def _derive_predicate(value: str) -> str:
    """Derive a predicate name from the preference value.

    Matches against known keyword groups first, falling back to a stable
    hash when no group matches.
    """
    value_lower = value.lower()
    for predicate, keywords in _PREDICATE_KEYWORDS.items():
        if any(kw in value_lower for kw in keywords):
            return predicate
    return hashlib.sha256(value_lower.encode()).hexdigest()[:12]
