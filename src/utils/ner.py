"""Pure-Python entity normalization and regex PII helpers.

The spaCy NER machinery was removed; entity/relation extraction now comes from
the LLM unified extractor. What remains is used by constraint scope matching
and the PII redactor.
"""

from __future__ import annotations

import re

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


def extract_pii_spans(text: str) -> list[tuple[int, int, str]]:
    """Regex-based PII span detection (emails, phones, addresses, SSNs, cards, IPs)."""
    spans: list[tuple[int, int, str]] = []
    for pii_type, pattern in _PII_REGEX_COMPILED.items():
        for match in pattern.finditer(text):
            spans.append((match.start(), match.end(), pii_type))

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
