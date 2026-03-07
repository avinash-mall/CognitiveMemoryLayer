"""Write-time fact extraction for non-LLM write paths.

LLM extraction remains in the unified extractor. When available, the
fact_extraction_structured token model is the primary structured-fact path.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from ..memory.neocortical.schemas import FactCategory
from ..utils.modelpack import get_modelpack_runtime
from ..utils.ner import parse_text
from .fact_span_adapter import build_structured_fact_records, span_prediction_confidence

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
_FIRST_PERSON_TOKENS = {"i", "my", "mine", "myself"}
_WRITE_TIME_CONFIDENCE_BASE: float = 0.6
_HASH_SUFFIX_RE = re.compile(r"^[0-9a-f]{12}$")
_DIRECT_NAME_PATTERNS = (
    r"\bmy name is\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?=\s+(?:and|but)\b|$)",
    r"\bcall me\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)(?=\s+(?:and|but)\b|$)",
    r"\bi(?:'m| am)\s+([A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+){0,2})(?=\s*(?:[,.!?]|$|\band\b|\bbut\b))",
)


class WriteTimeFactExtractor:
    """Extract structured facts from chunks at write-time."""

    def __init__(self) -> None:
        self.modelpack = get_modelpack_runtime()

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

        # --- model path: structured span extraction ---
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)(
                "fact_extraction_structured"
            ):
                span_pred = self.modelpack.predict_spans("fact_extraction_structured", text)
                if span_pred is not None and span_pred.spans:
                    records = build_structured_fact_records(
                        text,
                        span_pred,
                        derive_predicate=_derive_predicate,
                        label_to_category=_label_to_category,
                        confidence=span_prediction_confidence(span_pred, multiplier=0.75),
                    )
                    model_facts = [
                        ExtractedFact(
                            key=record.key,
                            category=record.category,
                            predicate=record.predicate,
                            value=record.value,
                            confidence=record.confidence,
                        )
                        for record in records
                    ]
                    for fact in model_facts:
                        if self._keep_model_fact(fact):
                            self._append_model_fact(facts, seen, fact)
        except Exception:
            pass

        # --- heuristic path (regex / spaCy) ---
        doc = parse_text(text)
        if doc is None or not _doc_supports_dependency_parse(doc):
            self._extract_facts_without_nlp(text, facts, seen)
            return facts

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

    def _append_model_fact(
        self,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
        fact: ExtractedFact,
    ) -> None:
        clean_value = " ".join(fact.value.strip().strip(".").split())
        if fact.key == "user:preference:hobby":
            clean_value = re.sub(r"\s+as hobbies?\b", "", clean_value, flags=re.IGNORECASE).strip()
        if not clean_value:
            return
        dedupe_key = (fact.key, clean_value.lower())
        if dedupe_key in seen:
            return
        seen.add(dedupe_key)
        facts.append(
            ExtractedFact(
                key=fact.key,
                category=fact.category,
                predicate=fact.predicate,
                value=clean_value,
                confidence=fact.confidence,
            )
        )

    def _extract_preference_facts(
        self,
        doc: Any,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        for sent in doc.sents:
            sent_text = sent.text.strip()
            hobby_match = re.search(
                r"\bmy hobbies are\s+(.+)",
                sent_text,
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

            match = re.search(
                r"\bmy favou?rite\s+(.+?)\s+is\s+(.+)",
                sent_text,
                flags=re.IGNORECASE,
            )
            if not match:
                continue
            descriptor = self._clean_match_value(match.group(1))
            value = self._clean_match_value(match.group(2))
            if not descriptor or not value:
                continue
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

            if "hobb" in token.sent.text.lower():
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

    def _extract_identity_facts(
        self,
        doc: Any,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        for sent in doc.sents:
            sent_text = sent.text.strip()
            for pattern in _DIRECT_NAME_PATTERNS:
                match = re.search(pattern, sent_text, flags=re.IGNORECASE)
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

            sent_text = sent.text.strip()
            match = re.search(
                r"\bi(?:'m| am)\s+(?:an?|the)\s+(.+)",
                sent_text,
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
                    return

    @staticmethod
    def _has_first_person(sent: Any) -> bool:
        return any(t.text.lower() in _FIRST_PERSON_TOKENS for t in sent)

    def _extract_facts_without_nlp(
        self,
        text: str,
        facts: list[ExtractedFact],
        seen: set[tuple[str, str]],
    ) -> None:
        """Best-effort fallback extraction when spaCy model is unavailable."""
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

    @staticmethod
    def _clean_match_value(value: str) -> str:
        return value.strip().strip(".,!?;:")

    @staticmethod
    def _keep_model_fact(fact: ExtractedFact) -> bool:
        if fact.category == FactCategory.IDENTITY:
            return False
        if fact.category == FactCategory.LOCATION:
            return fact.key.startswith("user:location:") and bool(fact.value.strip())
        if fact.category == FactCategory.OCCUPATION:
            return fact.key == "user:occupation:role" and bool(fact.value.strip())
        if fact.category == FactCategory.PREFERENCE:
            if not fact.key.startswith("user:preference:"):
                return False
            predicate = fact.key.removeprefix("user:preference:")
            if not predicate:
                return False
            if not _HASH_SUFFIX_RE.fullmatch(predicate):
                return True
            words = re.findall(r"[a-z0-9]+", fact.value.lower())
            return len(words) >= 2 and len(fact.value.strip()) >= 8
        return True


_LABEL_CATEGORY_MAP: dict[str, FactCategory] = {
    "preference": FactCategory.PREFERENCE,
    "identity": FactCategory.IDENTITY,
    "location": FactCategory.LOCATION,
    "occupation": FactCategory.OCCUPATION,
    "attribute": FactCategory.ATTRIBUTE,
    "goal": FactCategory.GOAL,
    "value": FactCategory.VALUE,
    "state": FactCategory.STATE,
    "causal": FactCategory.CAUSAL,
    "policy": FactCategory.POLICY,
}


def _label_to_category(label: str) -> FactCategory:
    return _LABEL_CATEGORY_MAP.get(label.lower(), FactCategory.CUSTOM)


def _derive_predicate(value: str) -> str:
    """Derive a predicate name from the preference value."""
    value_lower = value.lower()
    for predicate, keywords in _PREDICATE_KEYWORDS.items():
        if any(kw in value_lower for kw in keywords):
            return predicate
    return hashlib.sha256(value_lower.encode()).hexdigest()[:12]


def _doc_supports_dependency_parse(doc: Any) -> bool:
    try:
        return bool(doc.has_annotation("DEP"))
    except Exception:
        return False
