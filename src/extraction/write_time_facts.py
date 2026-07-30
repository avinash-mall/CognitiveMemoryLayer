"""Write-time fact extraction for non-LLM write paths.

LLM extraction remains in the unified extractor; this is the regex-only
heuristic used when the LLM is disabled.
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


_PREDICATE_KEYWORDS: dict[str, list[str]] = {
    "cuisine": ["food", "restaurant", "eat", "cook", "meal", "cuisine", "dish"],
    "music": ["music", "song", "band", "listen", "genre", "artist"],
    "color": ["color", "colour"],
    "language": ["language", "speak"],
    "sport": ["sport", "play", "team", "game", "exercise"],
    "movie": ["movie", "film", "cinema", "watch"],
    "book": ["book", "read", "author", "novel"],
    "hobby": ["hobby", "hobbies", "enjoy doing", "spare time", "free time"],
    "pet": ["pet", "pets", "dog", "cat", "animal"],
}

_WRITE_TIME_CONFIDENCE_BASE: float = 0.6
_DIRECT_NAME_PATTERNS = (
    r"\bmy name is\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?=\s+(?:and|but)\b|$)",
    r"\bcall me\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?=\s+(?:and|but)\b|$)",
    r"\bi(?:'m| am)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})(?=\s*(?:[,.!?]|$|\band\b|\bbut\b))",
)


class WriteTimeFactExtractor:
    """Extract structured facts from chunks at write-time (regex heuristics)."""

    def extract(self, chunk: SemanticChunk) -> list[ExtractedFact]:
        from ..memory.working.models import ChunkType

        fact_bearing_types = {
            ChunkType.PREFERENCE,
            ChunkType.FACT,
            ChunkType.CONSTRAINT,
            ChunkType.STATEMENT,
        }
        if chunk.chunk_type not in fact_bearing_types:
            return []

        text = chunk.text.strip()
        if not text:
            return []

        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str]] = set()
        self._extract_facts(text, facts, seen)
        return facts

    def _append_fact(
        self,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
        *,
        key: str,
        category: FactCategory,
        predicate: str,
        value: str,
        confidence_boost: float,
    ) -> None:
        clean_value = " ".join(value.strip().strip(".").split())
        if not clean_value:
            return
        dedupe_key = (key, clean_value.lower())
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        facts.append(
            ExtractedFact(
                key=key,
                category=category,
                predicate=predicate,
                value=clean_value,
                confidence=min(1.0, _WRITE_TIME_CONFIDENCE_BASE * confidence_boost),
            )
        )

    def _extract_facts(
        self,
        text: str,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        """Regex-based extraction across preference/identity/location/occupation."""
        normalized = " ".join(text.strip().split())
        if not normalized:
            return

        hobby_match = re.search(
            r"\bmy hobbies are\s+(.+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if hobby_match:
            value = self._clean_match_value(hobby_match.group(1))
            if value:
                self._append_fact(
                    facts,
                    seen,
                    key="user:preference:hobby",
                    category=FactCategory.PREFERENCE,
                    predicate="hobby",
                    value=value,
                    confidence_boost=0.78,
                )

        pref = re.search(
            r"\b(?:i|we)\s+(?:really\s+|also\s+|just\s+|still\s+)*(?:prefer|like|love|enjoy|hate|dislike)\s+(.+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if pref:
            obj = self._clean_match_value(pref.group(1))
            if obj:
                if "hobb" in normalized.lower():
                    obj = re.sub(r"\s+as hobbies?\b", "", obj, flags=re.IGNORECASE).strip()
                    predicate = "hobby"
                else:
                    predicate = _derive_predicate(obj)
                self._append_fact(
                    facts,
                    seen,
                    key=f"user:preference:{predicate}",
                    category=FactCategory.PREFERENCE,
                    predicate=predicate,
                    value=obj,
                    confidence_boost=0.75,
                )

        favorite = re.search(
            r"\bmy favou?rite\s+(.+?)\s+is\s+(.+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if favorite:
            descriptor = self._clean_match_value(favorite.group(1))
            value = self._clean_match_value(favorite.group(2))
            if descriptor and value:
                predicate = _derive_predicate(descriptor)
                self._append_fact(
                    facts,
                    seen,
                    key=f"user:preference:{predicate}",
                    category=FactCategory.PREFERENCE,
                    predicate=predicate,
                    value=value,
                    confidence_boost=0.78,
                )

        for pattern in _DIRECT_NAME_PATTERNS:
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            self._append_fact(
                facts,
                seen,
                key="user:identity:name",
                category=FactCategory.IDENTITY,
                predicate="name",
                value=self._clean_match_value(match.group(1)),
                confidence_boost=0.9,
            )
            break

        for pattern in (
            r"\bi live in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"\bi(?: am|'m)\s+from\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"\bi(?: am|'m)\s+based in\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
            r"\bi moved to\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
        ):
            match = re.search(pattern, normalized, flags=re.IGNORECASE)
            if not match:
                continue
            self._append_fact(
                facts,
                seen,
                key="user:location:current_city",
                category=FactCategory.LOCATION,
                predicate="current_city",
                value=self._clean_match_value(match.group(1)),
                confidence_boost=0.78,
            )
            break

        match = re.search(
            r"\bi(?:'m| am)\s+(?:an?|the)\s+(.+)",
            normalized,
            flags=re.IGNORECASE,
        )
        if match:
            role_text = self._clean_match_value(match.group(1))
            if role_text and "favorite" not in role_text.lower():
                self._append_fact(
                    facts,
                    seen,
                    key="user:occupation:role",
                    category=FactCategory.OCCUPATION,
                    predicate="role",
                    value=role_text,
                    confidence_boost=0.78,
                )

    @staticmethod
    def _clean_match_value(value: str) -> str:
        return value.strip().strip(".,!?;:")


def _derive_predicate(value: str) -> str:
    """Derive a predicate name from the preference value."""
    value_lower = value.lower()
    for predicate, keywords in _PREDICATE_KEYWORDS.items():
        if any(kw in value_lower for kw in keywords):
            return predicate
    return hashlib.sha256(value_lower.encode()).hexdigest()[:12]
