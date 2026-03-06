"""spaCy-based NER/relational helpers for non-LLM fallback paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

spacy: Any
try:  # pragma: no cover - import guard
    import spacy as _spacy

    spacy = _spacy
except Exception:  # pragma: no cover - optional dependency
    spacy = None


_DEFAULT_MODEL_CANDIDATES = (
    os.getenv("NER__MODEL", "").strip(),
    os.getenv("SPACY_MODEL", "").strip(),
    "en_core_web_sm",
)

_ENTITY_LABEL_MAP = {
    "PERSON": "PERSON",
    "ORG": "ORGANIZATION",
    "NORP": "CONCEPT",
    "FAC": "LOCATION",
    "GPE": "LOCATION",
    "LOC": "LOCATION",
    "PRODUCT": "PRODUCT",
    "EVENT": "EVENT",
    "WORK_OF_ART": "CONCEPT",
    "LAW": "CONCEPT",
    "LANGUAGE": "ATTRIBUTE",
    "DATE": "DATE",
    "TIME": "TIME",
    "MONEY": "MONEY",
    "QUANTITY": "ATTRIBUTE",
    "ORDINAL": "ATTRIBUTE",
    "CARDINAL": "ATTRIBUTE",
}

_PII_ENTITY_TYPES = {"PERSON", "ORGANIZATION", "LOCATION"}

_COREFERENCE_ALIAS_MAP = {
    "i": "user",
    "me": "user",
    "my": "user",
    "mine": "user",
    "myself": "user",
    "we": "user",
    "us": "user",
    "our": "user",
    "ours": "user",
    "ourselves": "user",
    "you": "assistant",
    "your": "assistant",
    "yours": "assistant",
}

_ENTITY_ALIAS_MAP = {
    "nyc": "new york city",
    "new york": "new york city",
    "new york, ny": "new york city",
    "sf": "san francisco",
    "sfo": "san francisco",
    "la": "los angeles",
    "u.s.": "united states",
    "u.s.a.": "united states",
    "usa": "united states",
    "uk": "united kingdom",
    "u.k.": "united kingdom",
}

_ENTITY_CANONICAL_TYPES = {"PERSON", "ORGANIZATION", "LOCATION", "CONCEPT", "ATTRIBUTE"}
_WHITESPACE_RE = re.compile(r"\s+")
_NON_WORD_RE = re.compile(r"[^\w\s\.\-]+")

_US_STREET_SUFFIX = (
    "street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|"
    "court|ct|way|terrace|ter|place|pl|parkway|pkwy"
)

_PII_REGEX_PATTERNS: dict[str, str] = {
    "EMAIL": r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
    "PHONE": r"(?<!\w)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\w)",
    "PHONE_INTL": r"(?<!\w)\+\d{1,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){2,5}\d{2,4}(?!\w)",
    "ADDRESS_US": (
        rf"\b\d{{1,6}}[A-Za-z]?\s+[A-Za-z0-9][A-Za-z0-9\s\.\-]{{1,50}}?\s+(?:{_US_STREET_SUFFIX})\b"
        r"(?:,\s*[A-Za-z\.\-\s]+)?(?:,\s*[A-Z]{2}\s+\d{5}(?:-\d{4})?)?"
    ),
    "ADDRESS_UK": (
        r"\b\d{1,4}[A-Za-z]?\s+[A-Za-z][A-Za-z0-9\s\.\-]{1,50}?\s+"
        r"(?:road|rd|street|st|lane|ln|avenue|ave|close|cl|drive|dr)\b"
        r"(?:,\s*[A-Za-z\.\-\s]+)?(?:,\s*[A-Z]{1,2}\d[A-Z\d]?\s*\d[A-Z]{2})?"
    ),
    "SSN": r"\b\d{3}-\d{2}-\d{4}\b",
    "CREDIT_CARD": r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
    "IP_ADDRESS": r"\b\d{1,3}(?:\.\d{1,3}){3}\b",
}

_PII_REGEX_COMPILED = {
    pii_type: re.compile(pattern, re.IGNORECASE)
    for pii_type, pattern in _PII_REGEX_PATTERNS.items()
}


@dataclass(frozen=True)
class NEREntity:
    text: str
    normalized: str
    entity_type: str
    start_char: int
    end_char: int


@dataclass(frozen=True)
class NERRelation:
    subject: str
    predicate: str
    object: str
    confidence: float


def normalize_entity_name(text: str, entity_type: str | None = None) -> str:
    """Normalize entities for alias/coreference matching across turns."""
    raw = text.strip()
    if not raw:
        return ""

    collapsed = _WHITESPACE_RE.sub(" ", raw)
    cleaned = _NON_WORD_RE.sub(" ", collapsed).strip().lower()
    cleaned = _WHITESPACE_RE.sub(" ", cleaned)

    if entity_type in {None, "PERSON", "CONCEPT", "ATTRIBUTE"}:
        mapped_coref = _COREFERENCE_ALIAS_MAP.get(cleaned)
        if mapped_coref:
            return mapped_coref

    mapped_alias = _ENTITY_ALIAS_MAP.get(cleaned)
    if mapped_alias:
        return mapped_alias

    if entity_type is None or entity_type in _ENTITY_CANONICAL_TYPES:
        return cleaned

    return collapsed


def normalize_scope_values(values: list[str]) -> list[str]:
    """Canonicalize and deduplicate scope values while preserving order."""
    out: list[str] = []
    seen: set[str] = set()
    for value in values:
        normalized = normalize_entity_name(value)
        if not normalized:
            continue
        if normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


@lru_cache(maxsize=1)
def get_nlp() -> Any | None:
    if spacy is None:
        logger.warning("spacy_not_installed", extra={"hint": "pip install spacy"})
        return None
    last_error = ""
    for candidate in _DEFAULT_MODEL_CANDIDATES:
        model = candidate.strip()
        if not model:
            continue
        try:
            nlp = spacy.load(model)
            logger.info("spacy_model_loaded", extra={"model": model})
            return nlp
        except Exception as exc:
            last_error = str(exc)
    logger.warning(
        "spacy_model_load_failed",
        extra={
            "candidates": [m for m in _DEFAULT_MODEL_CANDIDATES if m],
            "error": last_error,
        },
    )
    return None


def parse_text(text: str) -> Any | None:
    clean = text.strip()
    if not clean:
        return None
    nlp = get_nlp()
    if nlp is None:
        return None
    try:
        return nlp(clean)
    except Exception:
        return None


def extract_entities(text: str, *, max_entities: int = 64) -> list[NEREntity]:
    doc = parse_text(text)
    if doc is None:
        return []
    out: list[NEREntity] = []
    seen: set[tuple[int, int, str]] = set()
    for ent in doc.ents:
        if len(out) >= max_entities:
            break
        mapped = _ENTITY_LABEL_MAP.get(ent.label_, "CONCEPT")
        key = (int(ent.start_char), int(ent.end_char), mapped)
        if key in seen:
            continue
        seen.add(key)
        text_value = ent.text.strip()
        if not text_value:
            continue
        out.append(
            NEREntity(
                text=text_value,
                normalized=normalize_entity_name(text_value, mapped),
                entity_type=mapped,
                start_char=int(ent.start_char),
                end_char=int(ent.end_char),
            )
        )
    return out


def extract_pii_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []

    for pii_type, pattern in _PII_REGEX_COMPILED.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), pii_type))

    for ent in extract_entities(text):
        if ent.entity_type in _PII_ENTITY_TYPES:
            spans.append((ent.start_char, ent.end_char, ent.entity_type))

    spans.sort(key=lambda item: (item[0], item[1], item[2]))
    deduped: list[tuple[int, int, str]] = []
    seen: set[tuple[int, int, str]] = set()
    for span in spans:
        if span in seen:
            continue
        seen.add(span)
        deduped.append(span)
    return deduped


def contains_pii_entities(text: str) -> bool:
    return bool(extract_pii_spans(text))


def _predicate(token: Any) -> str:
    raw = getattr(token, "lemma_", "") or getattr(token, "text", "") or "related_to"
    pred = re.sub(r"[^a-z0-9_]+", "_", raw.lower()).strip("_")
    return pred or "related_to"


def _entity_for_token(token: Any, doc_entities: list[NEREntity]) -> str:
    char_idx = int(getattr(token, "idx", 0))
    for ent in doc_entities:
        if ent.start_char <= char_idx < ent.end_char:
            return ent.normalized
    noun_chunks = list(getattr(token.doc, "noun_chunks", []))
    for chunk in noun_chunks:
        if int(chunk.start) <= int(token.i) < int(chunk.end):
            text = str(chunk.text).strip()
            if text:
                return text
    return str(getattr(token, "text", "")).strip()


def extract_relations(
    text: str,
    *,
    max_relations: int = 32,
) -> list[NERRelation]:
    try:
        from .modelpack import get_modelpack_runtime

        mp = get_modelpack_runtime()
        if getattr(mp, "has_task_model", lambda _: False)("fact_extraction_structured"):
            span_pred = mp.predict_spans("fact_extraction_structured", text)
            if span_pred is not None and span_pred.spans:
                span_relations: list[NERRelation] = []
                for span in span_pred.spans:
                    start, end, label = span[0], span[1], span[2]
                    span_text = text[start:end] if start < len(text) and end <= len(text) else ""
                    if span_text and label:
                        span_relations.append(
                            NERRelation(
                                subject=span_text,
                                predicate=label,
                                object="",
                                confidence=0.85,
                            )
                        )
                    if len(span_relations) >= max_relations:
                        break
                if span_relations:
                    return span_relations
    except Exception:
        pass  # fall through to existing dependency-based extraction

    doc = parse_text(text)
    if doc is None:
        return []
    doc_entities = extract_entities(text)
    relations: list[NERRelation] = []
    seen: set[tuple[str, str, str]] = set()

    for token in doc:
        if len(relations) >= max_relations:
            break
        if token.pos_ not in {"VERB", "AUX"}:
            continue

        subjects = [c for c in token.children if c.dep_ in {"nsubj", "nsubjpass", "csubj"}]
        objects = [
            c for c in token.children if c.dep_ in {"dobj", "pobj", "attr", "dative", "oprd"}
        ]

        for prep in (c for c in token.children if c.dep_ == "prep"):
            objects.extend([c for c in prep.children if c.dep_ == "pobj"])

        if not subjects or not objects:
            continue

        pred = _predicate(token)
        for subj in subjects:
            subj_text = _entity_for_token(subj, doc_entities)
            if not subj_text:
                continue
            for obj in objects:
                obj_text = _entity_for_token(obj, doc_entities)
                if not obj_text or obj_text == subj_text:
                    continue
                key = (subj_text.lower(), pred, obj_text.lower())
                if key in seen:
                    continue
                seen.add(key)
                relations.append(
                    NERRelation(
                        subject=subj_text,
                        predicate=pred,
                        object=obj_text,
                        confidence=0.65,
                    )
                )
                if len(relations) >= max_relations:
                    break
            if len(relations) >= max_relations:
                break
    return relations


def extract_entity_texts(text: str, *, max_entities: int = 16) -> list[str]:
    return [e.text for e in extract_entities(text, max_entities=max_entities)]
