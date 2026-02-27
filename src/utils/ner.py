"""spaCy-based NER/relational helpers for non-LLM fallback paths."""

from __future__ import annotations

import os
import re
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)

try:  # pragma: no cover - import guard
    import spacy
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
                normalized=text_value,
                entity_type=mapped,
                start_char=int(ent.start_char),
                end_char=int(ent.end_char),
            )
        )
    return out


def extract_pii_spans(text: str) -> list[tuple[int, int, str]]:
    spans: list[tuple[int, int, str]] = []
    for ent in extract_entities(text):
        if ent.entity_type in _PII_ENTITY_TYPES:
            spans.append((ent.start_char, ent.end_char, ent.entity_type))
    return spans


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
