"""Constraint extraction: detect and structure latent constraints at write time.

Rule-based extractor that identifies goals, values, states, causal reasoning,
and policies from semantic chunks.  Produces structured ConstraintObject
instances stored in MemoryRecord.metadata["constraints"].

Design note: rule-based first for low latency on the hot write path.
An LLM-based extractor can be layered on top behind a feature flag later.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ..memory.working.models import SemanticChunk

# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------


@dataclass
class ConstraintObject:
    """A structured latent constraint extracted from user input."""

    constraint_type: str  # "goal", "state", "value", "causal", "preference", "policy"
    subject: str  # "user" or extracted speaker name
    description: str  # canonical constraint text
    scope: list[str] = field(default_factory=list)  # domain/topic tags
    activation: str = ""  # trigger conditions (free text)
    status: str = "active"
    confidence: float = 0.7
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    provenance: list[str] = field(default_factory=list)  # source turn IDs

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON-safe storage in MemoryRecord.metadata."""
        d = asdict(self)
        for key in ("valid_from", "valid_to"):
            val = d.get(key)
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        return d


# ---------------------------------------------------------------------------
# Pattern definitions -- each tuple is (compiled_regex, constraint_type, confidence_boost)
# ---------------------------------------------------------------------------

_GOAL_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (
        re.compile(
            r"\b(?:i'?m trying to|i want to|my goal is|i'?m working toward|i aim to)\b", re.I
        ),
        0.1,
    ),
    (
        re.compile(r"\b(?:i'?m preparing for|i'?m focused on|i'?m committed to|i plan to)\b", re.I),
        0.1,
    ),
    (re.compile(r"\b(?:i hope to|i intend to|i'?m striving|working on)\b", re.I), 0.05),
]

_VALUE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:i value|it'?s important (?:to me |that )|i care about)\b", re.I), 0.1),
    (re.compile(r"\b(?:i believe in|i strongly feel|matters? (?:a lot )?to me)\b", re.I), 0.1),
    (re.compile(r"\b(?:i prioriti[sz]e|for me,? the most important)\b", re.I), 0.05),
]

_STATE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:i'?m currently|i'?m dealing with|i'?m going through)\b", re.I), 0.1),
    (re.compile(r"\b(?:i'?m anxious about|i'?m stressed about|i'?m worried about)\b", re.I), 0.1),
    (re.compile(r"\b(?:i'?m struggling with|right now i|at the moment i)\b", re.I), 0.05),
]

_CAUSAL_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:because|since|that'?s why|the reason is)\b", re.I), 0.05),
    (re.compile(r"\b(?:in order to|so that|to make sure|to avoid)\b", re.I), 0.1),
    (re.compile(r"\b(?:due to|as a result|consequently)\b", re.I), 0.05),
]

_POLICY_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:i never|i always|i don'?t|i won'?t|i refuse to)\b", re.I), 0.1),
    (re.compile(r"\b(?:i must|i have to|i need to avoid|i can'?t)\b", re.I), 0.1),
    (re.compile(r"\b(?:i should|i shouldn'?t|i'?m not allowed to)\b", re.I), 0.05),
]

_PREFERENCE_PATTERNS: list[tuple[re.Pattern[str], float]] = [
    (re.compile(r"\b(?:i prefer|i like|i love|i enjoy|i hate|i dislike)\b", re.I), 0.1),
    (re.compile(r"\b(?:my favorite|my favourite)\b", re.I), 0.08),
]

_ALL_PATTERNS: dict[str, list[tuple[re.Pattern[str], float]]] = {
    "goal": _GOAL_PATTERNS,
    "value": _VALUE_PATTERNS,
    "state": _STATE_PATTERNS,
    "causal": _CAUSAL_PATTERNS,
    "policy": _POLICY_PATTERNS,
    "preference": _PREFERENCE_PATTERNS,
}

# Minimum cumulative confidence boost to consider a constraint detected
_MIN_CONFIDENCE_THRESHOLD = 0.05


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class ConstraintExtractor:
    """Rule-based constraint extractor for semantic chunks."""

    def __init__(self, base_confidence: float = 0.65) -> None:
        self._base_confidence = base_confidence

    def extract(self, chunk: SemanticChunk) -> list[ConstraintObject]:
        """Extract zero or more constraint objects from a single chunk."""
        raw = getattr(chunk, "text", None)
        if not isinstance(raw, str):
            return []
        text = raw.strip()
        if not text:
            return []

        constraints: list[ConstraintObject] = []

        for ctype, patterns in _ALL_PATTERNS.items():
            conf_boost = 0.0
            matched = False
            for pattern, boost in patterns:
                if pattern.search(text):
                    conf_boost += boost
                    matched = True

            if not matched or conf_boost < _MIN_CONFIDENCE_THRESHOLD:
                continue

            confidence = min(1.0, self._base_confidence + conf_boost)

            # Derive scope from entities if available
            scope = list(chunk.entities) if chunk.entities else []

            constraints.append(
                ConstraintObject(
                    constraint_type=ctype,
                    subject=self._extract_subject(chunk),
                    description=text,
                    scope=scope,
                    activation="",
                    status="active",
                    confidence=confidence,
                    valid_from=chunk.timestamp,
                    provenance=[chunk.source_turn_id] if chunk.source_turn_id else [],
                )
            )

        return constraints

    def extract_batch(self, chunks: list[SemanticChunk]) -> list[ConstraintObject]:
        """Extract constraints from multiple chunks."""
        results: list[ConstraintObject] = []
        for chunk in chunks:
            results.extend(self.extract(chunk))
        return results

    # ------------------------------------------------------------------
    # Supersession helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def detect_supersession(
        old: ConstraintObject,
        new: ConstraintObject,
        llm_client=None,  # kept for backward compat; no longer used
    ) -> bool:
        """Return True if *new* should supersede *old* using fast heuristics.

        The previous implementation called an LLM per existing constraint on
        every write, causing one LLM call per existing constraint of the same
        type on the hot write path.  This replacement covers the same cases
        with zero LLM calls:

        1. Type must match and old must be active.
        2. Scope check: overlapping scopes (or both scopeless) → supersede.
        3. Description similarity: if descriptions share >50 % word overlap the
           new constraint is considered a topical replacement for the old one.
        """
        if old.constraint_type != new.constraint_type:
            return False
        if old.status != "active":
            return False

        # Scope-based check (original logic preserved)
        if old.scope and new.scope:
            if set(old.scope) & set(new.scope):
                return True
        elif not old.scope and not new.scope:
            return True

        # Fallback: word-overlap on descriptions — catches same-topic rewrites
        # even when scopes differ (e.g. scope drifted but subject is the same).
        stop = {
            "i",
            "a",
            "an",
            "the",
            "is",
            "are",
            "was",
            "were",
            "to",
            "of",
            "and",
            "or",
            "my",
            "me",
            "it",
            "in",
            "on",
            "at",
            "for",
        }
        w_old = set(old.description.lower().split()) - stop
        w_new = set(new.description.lower().split()) - stop
        if w_old and w_new:
            overlap = len(w_old & w_new) / len(w_old | w_new)
            if overlap > 0.5:
                return True

        return False

    @staticmethod
    def constraint_fact_key(constraint: ConstraintObject) -> str:
        """Generate a stable semantic-fact key for a constraint.

        Format: ``user:{type}:{scope_hash}``
        """
        scope_str = ",".join(sorted(constraint.scope)) if constraint.scope else "general"
        scope_hash = hashlib.sha256(scope_str.encode()).hexdigest()[:12]
        return f"user:{constraint.constraint_type}:{scope_hash}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _extract_subject(chunk: SemanticChunk) -> str:
        """Determine the subject of the constraint (usually 'user')."""
        text = chunk.text
        # Check for "Speaker: text" format (LoCoMo ingestion)
        colon_idx = text.find(":")
        if 0 < colon_idx < 30:
            candidate = text[:colon_idx].strip()
            # Only accept if it looks like a name (no spaces after first word is fine)
            if candidate and candidate[0].isupper() and " said" not in candidate.lower():
                return candidate.lower()
        return "user"
