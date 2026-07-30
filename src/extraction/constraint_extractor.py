"""Constraint extraction via regex heuristics + chunk metadata (LLM path lives in the unified extractor)."""

from __future__ import annotations

import hashlib
import re
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ..memory.working.models import SemanticChunk
from ..utils.ner import normalize_scope_values


@dataclass
class ConstraintObject:
    """A structured latent constraint extracted from user input."""

    constraint_type: str
    subject: str
    description: str
    scope: list[str] = field(default_factory=list)
    activation: str = ""
    status: str = "active"
    confidence: float = 0.7
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    provenance: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Serialise for JSON-safe storage in MemoryRecord.metadata."""
        d = asdict(self)
        for key in ("valid_from", "valid_to"):
            val = d.get(key)
            if isinstance(val, datetime):
                d[key] = val.isoformat()
        return d


_CHUNK_TYPE_TO_CONSTRAINT: dict[str, str] = {
    "constraint": "policy",
    "preference": "preference",
}

_HEURISTIC_CONSTRAINT_PATTERNS: tuple[tuple[str, float, tuple[str, ...]], ...] = (
    (
        "policy",
        0.88,
        (
            r"\bi\s+(?:never|always|must|must not|can't|cannot|won't|will not)\b",
            r"\bi\s+(?:refuse|avoid)\s+to\b",
            r"\bi\s+do\s+not\b",
            r"\bi\s+don't\b",
            r"\bpersonal rules?\b",
            r"\bresearch ethics?\b",
            r"\bpolicy\b",
        ),
    ),
    (
        "value",
        0.84,
        (
            r"\bi\s+value\b",
            r"\bimportant to me\b",
            r"\bi\s+believe in\b",
            r"\bi\s+prioriti[sz]e\b",
            r"\babove convenience\b",
        ),
    ),
    (
        "state",
        0.8,
        (
            r"\bi'?m currently\b",
            r"\bright now\b",
            r"\bi'?m stressed\b",
            r"\bi'?m worried\b",
            r"\bi'?m dealing with\b",
            r"\bi'?m mentoring\b",
        ),
    ),
    (
        "causal",
        0.8,
        (
            r"\bbecause of\b",
            r"\bthe reason i\b",
            r"\bdue to\b",
            r"\bin order to\b",
            r"\bso that\b",
        ),
    ),
    (
        "goal",
        0.84,
        (
            r"\bi'?m trying to\b",
            r"\bmy goal is to\b",
            r"\bi'?m working toward\b",
            r"\bi plan to\b",
            r"\bi aim to\b",
            r"\bpublication target\b",
            r"\bworking toward\b",
            r"\btarget\b",
        ),
    ),
)

_SUPERSESSION_PROMPT = """Determine whether NEW constraint supersedes OLD constraint.

OLD:
{old_desc}

NEW:
{new_desc}

Rules:
- supersedes=true only when NEW is the same scope/topic and replaces OLD.
- supersedes=false when they can both remain active.

Return JSON only:
{{"supersedes": false, "confidence": 0.0-1.0}}"""


class ConstraintExtractor:
    """Constraint extractor backed by regex heuristics and chunk type."""

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

        ctype, confidence = self._classify_constraint_type(text, chunk)
        if ctype is None:
            return []

        scope = self._extract_scope(text, chunk.entities)
        return [
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
        ]

    def extract_batch(self, chunks: list[SemanticChunk]) -> list[ConstraintObject]:
        """Extract constraints from multiple chunks."""
        results: list[ConstraintObject] = []
        for chunk in chunks:
            results.extend(self.extract(chunk))
        return results

    def _classify_constraint_type(
        self,
        text: str,
        chunk: SemanticChunk,
    ) -> tuple[str | None, float]:
        heuristic = self._heuristic_constraint_type(text)
        if heuristic and heuristic[1] >= 0.85:
            return heuristic
        chunk_type = getattr(getattr(chunk, "chunk_type", None), "value", "")
        mapped = _CHUNK_TYPE_TO_CONSTRAINT.get(str(chunk_type).lower())
        if mapped:
            return mapped, self._base_confidence
        if heuristic:
            return heuristic
        return None, self._base_confidence

    # ------------------------------------------------------------------
    # Supersession helpers
    # ------------------------------------------------------------------

    @staticmethod
    async def detect_supersession(
        old: ConstraintObject,
        new: ConstraintObject,
        llm_client=None,
    ) -> bool:
        """Return True when NEW supersedes OLD (type/scope pre-filters + LLM)."""
        if old.constraint_type != new.constraint_type:
            return False
        if old.status != "active":
            return False
        old_scope = set(normalize_scope_values(list(old.scope or [])))
        new_scope = set(normalize_scope_values(list(new.scope or [])))
        if old_scope and new_scope and old_scope.isdisjoint(new_scope):
            return False

        if llm_client is None:
            return False

        try:
            payload = await llm_client.complete_json(
                _SUPERSESSION_PROMPT.format(old_desc=old.description, new_desc=new.description),
                temperature=0.0,
            )
            supersedes = bool(payload.get("supersedes", False))
            confidence = float(payload.get("confidence", 0.0))
            return supersedes and confidence >= 0.55
        except Exception:
            return False

    @staticmethod
    def constraint_fact_key(constraint: ConstraintObject) -> str:
        """Generate a stable semantic-fact key for a constraint.

        Format: ``user:{type}:{scope_hash}:{desc_hash}``

        Includes a normalized description hash so distinct constraints of the
        same type and scope coexist rather than colliding.
        """
        canonical_scope = normalize_scope_values(list(constraint.scope or []))
        scope_str = ",".join(sorted(canonical_scope)) if canonical_scope else "general"
        scope_hash = hashlib.sha256(scope_str.encode()).hexdigest()[:12]
        desc_normalized = re.sub(r"\s+", " ", (constraint.description or "").strip().lower())
        desc_hash = hashlib.sha256(desc_normalized.encode()).hexdigest()[:12]
        return f"user:{constraint.constraint_type}:{scope_hash}:{desc_hash}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _heuristic_constraint_type(self, text: str) -> tuple[str, float] | None:
        lowered = text.strip().lower()
        if not lowered:
            return None
        for ctype, confidence, patterns in _HEURISTIC_CONSTRAINT_PATTERNS:
            if any(re.search(pattern, lowered) for pattern in patterns):
                return ctype, max(self._base_confidence, confidence)
        return None

    def _extract_scope(self, text: str, chunk_entities: list[str] | None = None) -> list[str]:
        _ = text
        out: list[str] = []
        if chunk_entities:
            out.extend([str(e).strip() for e in chunk_entities if str(e).strip()])
        return normalize_scope_values(out)[:8]

    @staticmethod
    def _extract_subject(chunk: SemanticChunk) -> str:
        """Determine the subject of the constraint (usually 'user')."""
        text = chunk.text
        colon_idx = text.find(":")
        if 0 < colon_idx < 30:
            candidate = text[:colon_idx].strip()
            if candidate and candidate[0].isupper() and " said" not in candidate.lower():
                return candidate.lower()
        return "user"
