"""Write-time fact extraction: populate the semantic store at write time.

LLM path remains in unified extractor.
Non-LLM path uses spaCy parse + NER only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..memory.neocortical.schemas import FactCategory
from ..utils.ner import parse_text

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

_PREFERENCE_LEMMAS = {"prefer", "like", "love", "enjoy", "hate", "dislike"}
_FIRST_PERSON_TOKENS = {"i", "me", "my", "mine", "myself"}
_WRITE_TIME_CONFIDENCE_BASE: float = 0.6


class WriteTimeFactExtractor:
    """Extract structured facts from chunks at write-time."""

    def extract(self, chunk: SemanticChunk) -> list[ExtractedFact]:
        from ..memory.working.models import ChunkType

        fact_bearing_types = {
            ChunkType.PREFERENCE,
            ChunkType.FACT,
            ChunkType.CONSTRAINT,
        }
        if chunk.chunk_type not in fact_bearing_types:
            return []

        text = chunk.text.strip()
        if not text:
            return []

        doc = parse_text(text)
        if doc is None:
            return []

        facts: list[ExtractedFact] = []
        seen: set[tuple[str, str]] = set()

        self._extract_preference_facts(doc, facts, seen)
        self._extract_identity_facts(doc, facts, seen)
        self._extract_location_facts(doc, facts, seen)
        self._extract_occupation_facts(doc, facts, seen)

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
        clean_value = value.strip().strip(".")
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

    def _extract_preference_facts(
        self,
        doc: Any,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        for token in doc:
            if token.lemma_.lower() not in _PREFERENCE_LEMMAS:
                continue
            if token.pos_ not in {"VERB", "AUX"}:
                continue

            obj = ""
            for child in token.children:
                if child.dep_ in {"dobj", "attr", "oprd", "pobj", "xcomp"}:
                    obj = " ".join(tok.text for tok in child.subtree).strip()
                    if obj:
                        break
            if not obj:
                sent = token.sent.text.strip()
                marker = token.text
                idx = sent.lower().find(marker.lower())
                if idx >= 0:
                    obj = sent[idx + len(marker) :].strip()
            if not obj:
                continue

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

    def _extract_identity_facts(
        self,
        doc: Any,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        for ent in doc.ents:
            if ent.label_ != "PERSON" or not ent.text.strip():
                continue
            sent = ent.sent
            if not self._has_first_person(sent):
                continue
            lemmas = {t.lemma_.lower() for t in sent}
            if lemmas & {"be", "call", "name"}:
                self._append_fact(
                    facts,
                    seen,
                    key="user:identity:name",
                    category=FactCategory.IDENTITY,
                    predicate="name",
                    value=ent.text,
                    confidence_boost=0.9,
                )
                return

    def _extract_location_facts(
        self,
        doc: Any,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        for ent in doc.ents:
            if ent.label_ not in {"GPE", "LOC"}:
                continue
            sent = ent.sent
            if not self._has_first_person(sent):
                continue
            lemmas = {t.lemma_.lower() for t in sent}
            if lemmas & {"live", "be", "move", "base", "come"}:
                self._append_fact(
                    facts,
                    seen,
                    key="user:location:current_city",
                    category=FactCategory.LOCATION,
                    predicate="current_city",
                    value=ent.text,
                    confidence_boost=0.78,
                )
                return

    def _extract_occupation_facts(
        self,
        doc: Any,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        for sent in doc.sents:
            if not self._has_first_person(sent):
                continue

            orgs = [
                ent.text.strip() for ent in sent.ents if ent.label_ == "ORG" and ent.text.strip()
            ]
            lemmas = {t.lemma_.lower() for t in sent}
            if orgs and "work" in lemmas:
                self._append_fact(
                    facts,
                    seen,
                    key="user:occupation:role",
                    category=FactCategory.OCCUPATION,
                    predicate="role",
                    value=f"works at {orgs[0]}",
                    confidence_boost=0.72,
                )
                return

            for token in sent:
                if token.dep_ in {"attr", "oprd"} and token.pos_ in {"NOUN", "PROPN"}:
                    self._append_fact(
                        facts,
                        seen,
                        key="user:occupation:role",
                        category=FactCategory.OCCUPATION,
                        predicate="role",
                        value=" ".join(t.text for t in token.subtree),
                        confidence_boost=0.78,
                    )
                    return

    @staticmethod
    def _has_first_person(sent: Any) -> bool:
        return any(t.text.lower() in _FIRST_PERSON_TOKENS for t in sent)


def _derive_predicate(value: str) -> str:
    """Derive a predicate name from the preference value."""
    value_lower = value.lower()
    for predicate, keywords in _PREDICATE_KEYWORDS.items():
        if any(kw in value_lower for kw in keywords):
            return predicate
    return hashlib.sha256(value_lower.encode()).hexdigest()[:12]
