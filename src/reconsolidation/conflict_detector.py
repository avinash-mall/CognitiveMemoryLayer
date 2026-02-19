"""Conflict detection between new information and existing memories."""

from dataclasses import dataclass
from enum import StrEnum

from ..core.schemas import MemoryRecord
from ..utils.llm import LLMClient


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

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client

    async def detect(
        self,
        old_memory: MemoryRecord,
        new_statement: str,
        context: str | None = None,
    ) -> ConflictResult:
        """Detect if new statement conflicts with existing memory."""
        from ..core.config import get_settings

        fast_result: ConflictResult | None = None
        if not get_settings().features.use_llm_conflict_detection_only:
            fast_result = self._fast_detect(old_memory.text, new_statement)
        if fast_result and fast_result.confidence > 0.8:
            return fast_result
        if self.llm:
            return await self._llm_detect(old_memory.text, new_statement, context)
        return fast_result or ConflictResult(
            conflict_type=ConflictType.NONE,
            confidence=0.5,
            old_statement=old_memory.text,
            new_statement=new_statement,
            reasoning="No conflict detected (heuristic)",
        )

    async def detect_batch(
        self,
        memories: list[MemoryRecord],
        new_statement: str,
    ) -> list[ConflictResult]:
        """Detect conflicts against multiple memories."""
        import asyncio

        return await asyncio.gather(*[self.detect(mem, new_statement) for mem in memories])

    def _fast_detect(
        self,
        old_statement: str,
        new_statement: str,
    ) -> ConflictResult | None:
        """Fast heuristic conflict detection."""
        old_lower = old_statement.lower()
        new_lower = new_statement.lower()

        correction_markers = [
            "actually",
            "no,",
            "that's wrong",
            "i meant",
            "correction:",
            "not anymore",
            "changed",
        ]
        for marker in correction_markers:
            if marker in new_lower:
                return ConflictResult(
                    conflict_type=ConflictType.CORRECTION,
                    confidence=0.85,
                    old_statement=old_statement,
                    new_statement=new_statement,
                    is_superseding=True,
                    reasoning=f"Contains correction marker: '{marker}'",
                )

        negations = ["not", "don't", "doesn't", "no longer", "never"]
        for neg in negations:
            if neg in new_lower and neg not in old_lower:
                old_words = set(old_lower.replace(neg, "").split())
                new_words = set(new_lower.replace(neg, "").split())
                overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)
                if overlap > 0.5:
                    return ConflictResult(
                        conflict_type=ConflictType.DIRECT_CONTRADICTION,
                        confidence=0.75,
                        old_statement=old_statement,
                        new_statement=new_statement,
                        conflicting_aspect="Negation of similar content",
                        reasoning=f"High word overlap ({overlap:.0%}) with negation",
                    )

        preference_words = [
            "like",
            "prefer",
            "favorite",
            "enjoy",
            "love",
            "hate",
        ]
        old_has_pref = any(w in old_lower for w in preference_words)
        new_has_pref = any(w in new_lower for w in preference_words)
        if old_has_pref and new_has_pref:
            # Only classify as temporal change if the statements share topic
            # overlap, avoiding false positives on unrelated preferences (MED-27)
            old_words = (
                set(old_lower.split())
                - set(preference_words)
                - {"i", "my", "a", "the", "is", "are"}
            )
            new_words = (
                set(new_lower.split())
                - set(preference_words)
                - {"i", "my", "a", "the", "is", "are"}
            )
            if old_words and new_words:
                topic_overlap = len(old_words & new_words) / max(len(old_words | new_words), 1)
                if topic_overlap > 0.2:
                    return ConflictResult(
                        conflict_type=ConflictType.TEMPORAL_CHANGE,
                        confidence=0.6,
                        old_statement=old_statement,
                        new_statement=new_statement,
                        is_superseding=True,
                        reasoning=f"Both express preferences with topic overlap ({topic_overlap:.0%})",
                    )
        return None

    async def _llm_detect(
        self,
        old_statement: str,
        new_statement: str,
        context: str | None = None,
    ) -> ConflictResult:
        """LLM-based conflict detection. Uses complete_json() for reliable parsing (MED-45)."""
        prompt = CONFLICT_DETECTION_PROMPT.format(
            old_statement=old_statement,
            new_statement=new_statement,
        )
        if context:
            prompt = f"CONTEXT:\n{context}\n\n{prompt}"
        try:
            data = await self.llm.complete_json(prompt, temperature=0.0)
            raw_type = data.get("conflict_type", "none")
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
                is_superseding=data.get("is_superseding", False),
                reasoning=data.get("reasoning", ""),
            )
        except Exception as e:
            return ConflictResult(
                conflict_type=ConflictType.AMBIGUITY,
                confidence=0.3,
                old_statement=old_statement,
                new_statement=new_statement,
                reasoning=f"LLM detection failed: {e}",
            )
