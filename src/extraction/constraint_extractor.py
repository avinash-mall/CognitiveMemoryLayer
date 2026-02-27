"""Constraint extraction with modelpack + NER (no rule-pattern heuristics)."""

from __future__ import annotations

import hashlib
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any

from ..memory.working.models import SemanticChunk
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime
from ..utils.ner import extract_entities


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


_ALLOWED_CONSTRAINT_TYPES = {
    "goal",
    "state",
    "value",
    "causal",
    "preference",
    "policy",
}

_CHUNK_TYPE_TO_CONSTRAINT: dict[str, str] = {
    "constraint": "policy",
    "preference": "preference",
}

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
    """Constraint extractor backed by modelpack + NER."""

    def __init__(
        self,
        base_confidence: float = 0.65,
        modelpack: ModelPackRuntime | None = None,
    ) -> None:
        self._base_confidence = base_confidence
        self._modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

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
        if self._modelpack.available:
            pred = self._modelpack.predict_single("constraint_type", text)
            if pred and pred.label:
                label = pred.label.strip().lower()
                if label in _ALLOWED_CONSTRAINT_TYPES:
                    return label, max(self._base_confidence, min(1.0, pred.confidence))

        chunk_type = getattr(getattr(chunk, "chunk_type", None), "value", "")
        mapped = _CHUNK_TYPE_TO_CONSTRAINT.get(str(chunk_type).lower())
        if mapped:
            return mapped, self._base_confidence
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
        """Return True when NEW supersedes OLD using modelpack or LLM only."""
        if old.constraint_type != new.constraint_type:
            return False
        if old.status != "active":
            return False

        modelpack = get_modelpack_runtime()
        if modelpack.available:
            sup_pred = modelpack.predict_pair("supersession", old.description, new.description)
            if sup_pred and sup_pred.confidence >= 0.55:
                return sup_pred.label == "supersedes"

            scope_pred = modelpack.predict_pair("scope_match", old.description, new.description)
            if scope_pred and scope_pred.label == "no_match" and scope_pred.confidence >= 0.8:
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

        Format: ``user:{type}:{scope_hash}``
        """
        scope_str = ",".join(sorted(constraint.scope)) if constraint.scope else "general"
        scope_hash = hashlib.sha256(scope_str.encode()).hexdigest()[:12]
        return f"user:{constraint.constraint_type}:{scope_hash}"

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _extract_scope(self, text: str, chunk_entities: list[str] | None = None) -> list[str]:
        out: list[str] = []

        if self._modelpack.available:
            pred = self._modelpack.predict_single("constraint_scope", text)
            if pred and pred.label:
                label = pred.label.strip()
                if label and label.lower() not in {"none", "other"}:
                    out.append(label)

        for ent in extract_entities(text, max_entities=12):
            value = ent.normalized.strip()
            if value:
                out.append(value)

        if chunk_entities:
            out.extend([str(e).strip() for e in chunk_entities if str(e).strip()])
        return list(dict.fromkeys(out))[:8]

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
