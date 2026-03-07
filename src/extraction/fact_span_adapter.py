"""Shared adapters for fact_extraction_structured span predictions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable


@dataclass(frozen=True)
class StructuredSpanMatch:
    """Normalized fact span extracted from a token model prediction."""

    start: int
    end: int
    label: str
    value: str


@dataclass(frozen=True)
class StructuredFactRecord:
    """Intermediate structured fact record shared across write/reconsolidation paths."""

    key: str
    category: Any
    predicate: str
    value: str
    confidence: float


@dataclass(frozen=True)
class StructuredRelationRecord:
    """Intermediate user-centric relation record for model-backed fallback extraction."""

    subject: str
    predicate: str
    object: str
    confidence: float


def _span_rows(span_pred: Any) -> tuple[tuple[int, int, str], ...]:
    rows = getattr(span_pred, "spans", span_pred)
    if not isinstance(rows, (list, tuple)):
        return ()
    out: list[tuple[int, int, str]] = []
    for item in rows:
        if not isinstance(item, (list, tuple)) or len(item) < 3:
            continue
        try:
            start = int(item[0])
            end = int(item[1])
        except (TypeError, ValueError):
            continue
        label = str(item[2] or "").strip().lower()
        if start < 0 or end <= start or not label:
            continue
        out.append((start, end, label))
    return tuple(out)


def span_prediction_confidence(span_pred: Any, *, multiplier: float = 1.0) -> float:
    """Return a clipped confidence score for a span prediction payload."""
    try:
        confidence = float(getattr(span_pred, "confidence", 1.0))
    except (TypeError, ValueError):
        confidence = 1.0
    return max(0.0, min(1.0, confidence * multiplier))


def extract_structured_span_matches(text: str, span_pred: Any) -> list[StructuredSpanMatch]:
    """Normalize and dedupe raw task-model spans against the source text."""
    clean_text = str(text or "")
    seen: set[tuple[str, str]] = set()
    matches: list[StructuredSpanMatch] = []

    for start, end, label in _span_rows(span_pred):
        if start >= len(clean_text) or end > len(clean_text):
            continue
        value = clean_text[start:end].strip().strip(".")
        if not value:
            continue
        dedupe_key = (label, value.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        matches.append(
            StructuredSpanMatch(
                start=start,
                end=end,
                label=label,
                value=value,
            )
        )

    matches.sort(key=lambda item: (item.start, item.end, item.label, item.value.casefold()))
    return matches


def build_structured_fact_records(
    text: str,
    span_pred: Any,
    *,
    derive_predicate: Callable[[str], str],
    label_to_category: Callable[[str], Any],
    confidence: float,
) -> list[StructuredFactRecord]:
    """Convert span predictions into structured fact records."""
    records: list[StructuredFactRecord] = []
    seen: set[tuple[str, str]] = set()

    for match in extract_structured_span_matches(text, span_pred):
        predicate = derive_predicate(match.value)
        category = label_to_category(match.label)
        category_value = getattr(category, "value", str(category))
        key = f"user:{category_value}:{predicate}"
        dedupe_key = (key, match.value.casefold())
        if dedupe_key in seen:
            continue
        seen.add(dedupe_key)
        records.append(
            StructuredFactRecord(
                key=key,
                category=category,
                predicate=predicate,
                value=match.value,
                confidence=confidence,
            )
        )
    return records


def build_user_relation_records(
    text: str,
    span_pred: Any,
    *,
    subject: str = "user",
    confidence: float = 0.85,
    max_relations: int | None = None,
) -> list[StructuredRelationRecord]:
    """Convert span predictions into user-centric relation triples."""
    relations: list[StructuredRelationRecord] = []
    for match in extract_structured_span_matches(text, span_pred):
        relations.append(
            StructuredRelationRecord(
                subject=subject,
                predicate=match.label,
                object=match.value,
                confidence=confidence,
            )
        )
        if max_relations is not None and len(relations) >= max_relations:
            break
    return relations


def build_reconsolidation_fact_dicts(text: str, span_pred: Any) -> list[dict[str, str]]:
    """Convert span predictions into reconsolidation fact dictionaries."""
    facts: list[dict[str, str]] = []
    for match in extract_structured_span_matches(text, span_pred):
        facts.append({"text": match.value, "type": match.label or "semantic_fact"})
    return facts
