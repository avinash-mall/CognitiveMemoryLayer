"""Conflict detection between new information and existing memories."""

from dataclasses import dataclass
from enum import StrEnum

from ..core.schemas import MemoryRecord
from ..utils.llm import LLMClient
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime


class ConflictType(StrEnum):
    NONE = "none"
    TEMPORAL_CHANGE = "temporal_change"
    DIRECT_CONTRADICTION = "contradiction"
    REFINEMENT = "refinement"
    CORRECTION = "correction"
    AMBIGUITY = "ambiguity"


@dataclass
class ConflictResult:
    """Result of conflict detection."""

    conflict_type: ConflictType
    confidence: float
    old_statement: str
    new_statement: str
    conflicting_aspect: str | None = None
    suggested_resolution: str | None = None
    is_superseding: bool = False
    reasoning: str = ""


CONFLICT_DETECTION_PROMPT = """Compare these two statements and determine if they conflict.

EXISTING MEMORY:
{old_statement}

NEW INFORMATION:
{new_statement}

Determine:
1. conflict_type: one of "none", "temporal_change", "contradiction", "refinement", "correction", "ambiguity"
2. conflicting_aspect: what specific part conflicts (if any)
3. is_superseding: does new info replace old (true/false)
4. reasoning: brief explanation

Return only valid JSON, no other text:
{{
  "conflict_type": "none",
  "conflicting_aspect": null,
  "is_superseding": false,
  "confidence": 0.9,
  "reasoning": "The statements are about different topics"
}}"""


class ConflictDetector:
    """Detects conflicts between new information and existing memories."""

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        modelpack: ModelPackRuntime | None = None,
    ):
        self.llm = llm_client
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

    async def detect(
        self,
        old_memory: MemoryRecord,
        new_statement: str,
        context: str | None = None,
    ) -> ConflictResult:
        """Detect if new statement conflicts with existing memory."""
        from ..core.config import get_settings

        feat = get_settings().features
        model_result = self._modelpack_detect(old_memory.text, new_statement)

        if self.llm and feat.use_llm_enabled:
            if feat.use_llm_conflict_detection_only or model_result is None:
                return await self._llm_detect(old_memory.text, new_statement, context)

        if model_result is not None:
            return model_result

        return ConflictResult(
            conflict_type=ConflictType.NONE,
            confidence=0.0,
            old_statement=old_memory.text,
            new_statement=new_statement,
            reasoning="No conflict model available",
        )

    async def detect_batch(
        self,
        memories: list[MemoryRecord],
        new_statement: str,
    ) -> list[ConflictResult]:
        """Detect conflicts against multiple memories."""
        import asyncio

        return await asyncio.gather(*[self.detect(mem, new_statement) for mem in memories])

    def _modelpack_detect(
        self,
        old_statement: str,
        new_statement: str,
    ) -> ConflictResult | None:
        if not self.modelpack.available:
            return None

        pred = self.modelpack.predict_pair("conflict_detection", old_statement, new_statement)
        if pred is None or not pred.label:
            return None

        mapped = self._map_conflict_label(pred.label)
        if mapped is None:
            return None

        return ConflictResult(
            conflict_type=mapped,
            confidence=max(0.0, min(1.0, pred.confidence)),
            old_statement=old_statement,
            new_statement=new_statement,
            is_superseding=mapped in {ConflictType.CORRECTION, ConflictType.TEMPORAL_CHANGE},
            reasoning=f"Modelpack prediction: {pred.label}",
        )

    @staticmethod
    def _map_conflict_label(raw_label: str) -> ConflictType | None:
        label = raw_label.strip().lower()
        mapping = {
            "none": ConflictType.NONE,
            "no_conflict": ConflictType.NONE,
            "temporal_change": ConflictType.TEMPORAL_CHANGE,
            "change": ConflictType.TEMPORAL_CHANGE,
            "contradiction": ConflictType.DIRECT_CONTRADICTION,
            "direct_contradiction": ConflictType.DIRECT_CONTRADICTION,
            "refinement": ConflictType.REFINEMENT,
            "correction": ConflictType.CORRECTION,
            "supersedes": ConflictType.CORRECTION,
            "ambiguity": ConflictType.AMBIGUITY,
            "ambiguous": ConflictType.AMBIGUITY,
        }
        return mapping.get(label)

    async def _llm_detect(
        self,
        old_statement: str,
        new_statement: str,
        context: str | None = None,
    ) -> ConflictResult:
        """LLM-based conflict detection. Uses complete_json() for reliable parsing."""
        prompt = CONFLICT_DETECTION_PROMPT.format(
            old_statement=old_statement,
            new_statement=new_statement,
        )
        if context:
            prompt = f"CONTEXT:\n{context}\n\n{prompt}"
        try:
            if self.llm is None:
                return ConflictResult(
                    conflict_type=ConflictType.AMBIGUITY,
                    confidence=0.0,
                    old_statement=old_statement,
                    new_statement=new_statement,
                )
            data = await self.llm.complete_json(prompt, temperature=0.0)
            raw_type = str(data.get("conflict_type", "none")).lower()
            try:
                ctype = ConflictType(raw_type)
            except ValueError:
                ctype = ConflictType.AMBIGUITY
            return ConflictResult(
                conflict_type=ctype,
                confidence=float(data.get("confidence", 0.7)),
                old_statement=old_statement,
                new_statement=new_statement,
                conflicting_aspect=data.get("conflicting_aspect"),
                is_superseding=bool(data.get("is_superseding", False)),
                reasoning=str(data.get("reasoning", "")),
            )
        except Exception as e:
            return ConflictResult(
                conflict_type=ConflictType.AMBIGUITY,
                confidence=0.3,
                old_statement=old_statement,
                new_statement=new_statement,
                reasoning=f"LLM detection failed: {e}",
            )
