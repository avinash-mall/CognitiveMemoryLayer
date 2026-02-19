"""Unified write-path LLM extractor: constraints, facts, salience, importance in one call.

When any of the write-path LLM feature flags are enabled, this extractor produces
constraints, facts, salience, importance (and optionally PII spans) in a single
LLM call instead of multiple rule-based passes.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..memory.neocortical.schemas import FactCategory
from ..memory.working.models import SemanticChunk
from ..utils.llm import LLMClient
from .constraint_extractor import ConstraintObject
from .write_time_facts import ExtractedFact

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class PIISpan:
    """A span of text that contains PII."""

    start: int
    end: int
    pii_type: str  # e.g. "email", "phone", "ssn"


@dataclass
class UnifiedExtractionResult:
    """Result of unified write-path extraction."""

    constraints: list[ConstraintObject] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    salience: float = 0.5
    importance: float = 0.5
    pii_spans: list[PIISpan] = field(default_factory=list)
    contains_secrets: bool = False


# ---------------------------------------------------------------------------
# Prompt and schema
# ---------------------------------------------------------------------------

_UNIFIED_PROMPT = """Analyze this text chunk from a user conversation and extract structured information.

Text: {text}

Chunk type: {chunk_type}

Return a JSON object with these fields (use empty arrays/0.0 when none apply):
- "constraints": array of objects with: constraint_type (one of: goal, value, state, causal, policy, preference), subject (usually "user"), description, scope (array of strings), confidence (0.0-1.0)
- "facts": array of objects with: key (e.g. "user:preference:cuisine"), category (one of: preference, identity, location, occupation, relationship, attribute), predicate, value, confidence (0.0-1.0)
- "salience": float 0.0-1.0 (how important/central is this to the user's memory)
- "importance": float 0.0-1.0 (how much should this be stored/retrieved)
- "pii_spans": optional array of objects with start, end (character offsets), pii_type (email, phone, ssn, etc.) - only if PII detected
- "contains_secrets": optional boolean - true if text contains API keys, passwords, tokens, etc.

Return ONLY valid JSON, no markdown or explanation.
"""

_CATEGORY_MAP = {
    "preference": FactCategory.PREFERENCE,
    "identity": FactCategory.IDENTITY,
    "location": FactCategory.LOCATION,
    "occupation": FactCategory.OCCUPATION,
    "relationship": FactCategory.RELATIONSHIP,
    "attribute": FactCategory.ATTRIBUTE,
    "temporal": FactCategory.TEMPORAL,
    "custom": FactCategory.CUSTOM,
}


def _parse_json_from_response(response: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]|\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class UnifiedWritePathExtractor:
    """Single LLM call to extract constraints, facts, salience, importance, PII."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def extract(self, chunk: SemanticChunk) -> UnifiedExtractionResult:
        """Extract constraints, facts, salience, importance, and optionally PII in one LLM call."""
        text = getattr(chunk, "text", None)
        if not isinstance(text, str) or not text.strip():
            return UnifiedExtractionResult(
                salience=getattr(chunk, "salience", 0.5) or 0.5,
                importance=0.5,
            )

        chunk_type = getattr(chunk, "chunk_type", None)
        chunk_type_str = str(chunk_type) if chunk_type else "statement"

        prompt = _UNIFIED_PROMPT.format(text=text.strip(), chunk_type=chunk_type_str)
        try:
            raw = await self._llm.complete_json(prompt, temperature=0.0)
        except Exception:
            # On LLM failure, return defaults
            return UnifiedExtractionResult(
                salience=getattr(chunk, "salience", 0.5) or 0.5,
                importance=0.5,
            )

        if not isinstance(raw, dict):
            raw = _parse_json_from_response(str(raw)) if isinstance(raw, str) else {}

        return self._parse_result(raw, chunk)

    def _parse_result(
        self,
        data: dict[str, Any],
        chunk: SemanticChunk,
    ) -> UnifiedExtractionResult:
        """Parse LLM JSON into UnifiedExtractionResult."""
        constraints: list[ConstraintObject] = []
        for item in data.get("constraints") or []:
            if not isinstance(item, dict):
                continue
            ctype = item.get("constraint_type", "preference")
            if ctype not in ("goal", "value", "state", "causal", "policy", "preference"):
                ctype = "preference"
            constraints.append(
                ConstraintObject(
                    constraint_type=ctype,
                    subject=item.get("subject", "user"),
                    description=item.get("description", chunk.text[:500]),
                    scope=item.get("scope") if isinstance(item.get("scope"), list) else [],
                    activation="",
                    status="active",
                    confidence=float(item.get("confidence", 0.7)),
                    valid_from=chunk.timestamp,
                    valid_to=None,
                    provenance=[chunk.source_turn_id] if chunk.source_turn_id else [],
                )
            )

        facts: list[ExtractedFact] = []
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            cat_str = (item.get("category") or "preference").lower()
            category = _CATEGORY_MAP.get(cat_str, FactCategory.PREFERENCE)
            facts.append(
                ExtractedFact(
                    key=item.get("key", "user:preference:unknown"),
                    category=category,
                    predicate=item.get("predicate", "unknown"),
                    value=str(item.get("value", "")),
                    confidence=float(item.get("confidence", 0.6)),
                )
            )

        salience = float(data.get("salience", 0.5))
        salience = max(0.0, min(1.0, salience))
        importance = float(data.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        pii_spans: list[PIISpan] = []
        for span in data.get("pii_spans") or []:
            if isinstance(span, dict) and "start" in span and "end" in span:
                pii_spans.append(
                    PIISpan(
                        start=int(span["start"]),
                        end=int(span["end"]),
                        pii_type=str(span.get("pii_type", "unknown")),
                    )
                )

        contains_secrets = bool(data.get("contains_secrets", False))

        return UnifiedExtractionResult(
            constraints=constraints,
            facts=facts,
            salience=salience,
            importance=importance,
            pii_spans=pii_spans,
            contains_secrets=contains_secrets,
        )
