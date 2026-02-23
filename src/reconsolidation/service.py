"""Reconsolidation orchestrator service."""

import logging
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..core.enums import MemoryType, OperationType
from ..core.schemas import MemoryRecord
from ..storage.postgres import PostgresMemoryStore
from .belief_revision import RevisionOperation, RevisionPlan
from .conflict_detector import ConflictDetector
from .labile_tracker import LabileStateTracker

logger = logging.getLogger(__name__)

try:
    from ..extraction.fact_extractor import FactExtractor
except ImportError:
    FactExtractor = None  # type: ignore

try:
    from ..utils.llm import LLMClient
except ImportError:
    LLMClient = None  # type: ignore


@dataclass
class ReconsolidationResult:
    """Result of reconsolidation process."""

    turn_id: str
    memories_processed: int
    operations_applied: list[dict[str, Any]]
    conflicts_found: int
    elapsed_ms: float


class ReconsolidationService:
    """
    Orchestrates the full reconsolidation process.

    Flow:
    1. Mark retrieved memories as labile
    2. Extract new facts from user message + response
    3. Detect conflicts between new facts and labile memories
    4. Plan and apply revisions
    5. Release memories from labile state
    """

    def __init__(
        self,
        memory_store: PostgresMemoryStore,
        llm_client: Any | None = None,
        fact_extractor: Any | None = None,
        redis_client: Any | None = None,
    ):
        self.store = memory_store
        self.labile_tracker = LabileStateTracker(redis_client=redis_client)
        self.conflict_detector = ConflictDetector(llm_client)
        from .belief_revision import BeliefRevisionEngine

        self.revision_engine = BeliefRevisionEngine()
        self.fact_extractor = fact_extractor

    # Maximum memories to run LLM conflict detection against per new fact.
    # Memories are ranked by word-overlap similarity to the new fact before
    # the cap is applied, so the most likely conflicts are always checked.
    _CONFLICT_TOP_K: int = 5

    @staticmethod
    def _word_overlap(a: str, b: str) -> float:
        """Jaccard word-overlap between two strings (case-insensitive)."""
        wa = set(a.lower().split())
        wb = set(b.lower().split())
        if not wa or not wb:
            return 0.0
        return len(wa & wb) / len(wa | wb)

    def _top_k_similar_memories(
        self,
        fact_text: str,
        memories: list[MemoryRecord],
        k: int,
    ) -> list[MemoryRecord]:
        """Return the k memories most similar (by word overlap) to fact_text."""
        if len(memories) <= k:
            return memories
        scored = sorted(
            memories,
            key=lambda m: self._word_overlap(fact_text, m.text),
            reverse=True,
        )
        return scored[:k]

    async def process_turn(
        self,
        tenant_id: str,
        scope_id: str,
        turn_id: str,
        user_message: str,
        assistant_response: str,
        retrieved_memories: list[MemoryRecord],
    ) -> ReconsolidationResult:
        """Process a conversation turn for reconsolidation.

        To avoid O(N_facts x N_memories) LLM calls, each new fact is compared
        only against the top-_CONFLICT_TOP_K most text-similar memories.
        """
        start = datetime.now(UTC)
        operations_applied: list[dict[str, Any]] = []
        conflicts_found = 0

        if retrieved_memories:
            await self.labile_tracker.mark_labile(
                tenant_id,
                scope_id,
                turn_id,
                memory_ids=[m.id for m in retrieved_memories],
                query=user_message,
                retrieved_texts=[m.text for m in retrieved_memories],
                relevance_scores=[m.metadata.get("_similarity", 0.5) for m in retrieved_memories],
                confidences=[m.confidence for m in retrieved_memories],
            )

        new_facts = await self._extract_new_facts(user_message, assistant_response)

        if not new_facts:
            await self.labile_tracker.release_labile(tenant_id, scope_id, turn_id)
            elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
            return ReconsolidationResult(
                turn_id=turn_id,
                memories_processed=len(retrieved_memories),
                operations_applied=[],
                conflicts_found=0,
                elapsed_ms=elapsed,
            )

        for new_fact in new_facts:
            # Only compare against the most similar memories to avoid an
            # O(N_facts x N_memories) explosion of LLM conflict-detection calls.
            candidate_memories = self._top_k_similar_memories(
                new_fact["text"], retrieved_memories, self._CONFLICT_TOP_K
            )
            for memory in candidate_memories:
                conflict = await self.conflict_detector.detect(memory, new_fact["text"])
                if conflict.conflict_type.value != "none":
                    conflicts_found += 1

                fact_type = new_fact.get("type", "episodic_event")
                try:
                    mem_type = MemoryType(fact_type)
                except ValueError:
                    mem_type = MemoryType.EPISODIC_EVENT

                plan: RevisionPlan = self.revision_engine.plan_revision(
                    conflict=conflict,
                    old_memory=memory,
                    new_info_type=mem_type,
                    tenant_id=tenant_id,
                    evidence_id=turn_id,
                )

                for op in plan.operations:
                    result = await self._apply_operation(op)
                    operations_applied.append(
                        {
                            "operation": op.op_type.value,
                            "target_id": str(op.target_id) if op.target_id else None,
                            "reason": op.reason,
                            "success": result,
                        }
                    )

        await self.labile_tracker.release_labile(tenant_id, scope_id, turn_id)
        elapsed = (datetime.now(UTC) - start).total_seconds() * 1000

        return ReconsolidationResult(
            turn_id=turn_id,
            memories_processed=len(retrieved_memories),
            operations_applied=operations_applied,
            conflicts_found=conflicts_found,
            elapsed_ms=elapsed,
        )

    async def _extract_new_facts(
        self,
        user_message: str,
        assistant_response: str,
    ) -> list[dict[str, Any]]:
        """Extract facts from conversation turn."""
        if self.fact_extractor:
            text = f"User: {user_message}\nAssistant: {assistant_response}"
            facts = await self.fact_extractor.extract(text)
            return [{"text": f.text, "type": getattr(f, "type", "semantic_fact")} for f in facts]

        # BUG-09: include assistant_response in fallback; split on . ! ? ;
        facts = []
        combined = f"{user_message} {assistant_response or ''}".strip()
        for sentence in re.split(r"[.!?;]+", combined):
            sentence = sentence.strip()
            if not sentence:
                continue
            lower = sentence.lower()
            if any(
                m in lower
                for m in [
                    "i am",
                    "i'm",
                    "my name",
                    "i live",
                    "i work",
                    "i like",
                    "i prefer",
                ]
            ):
                facts.append(
                    {
                        "text": sentence,
                        "type": (
                            "semantic_fact"
                            if "my name" in lower or "i live" in lower
                            else "preference"
                        ),
                    }
                )
        return facts

    async def _apply_operation(self, op: RevisionOperation) -> bool:
        """Apply a single revision operation."""
        try:
            if op.op_type == OperationType.ADD:
                if op.new_record:
                    await self.store.upsert(op.new_record)
                return True
            if op.op_type in (
                OperationType.UPDATE,
                OperationType.REINFORCE,
                OperationType.DECAY,
            ):
                if op.target_id and op.patch:
                    await self.store.update(op.target_id, op.patch)
                return True
            if op.op_type == OperationType.DELETE:
                if op.target_id:
                    await self.store.delete(op.target_id, hard=False)
                return True
            # BUG-14: unknown operation type — do not report success
            logger.warning("revision_unknown_operation_type op_type=%s", op.op_type.value)
            return False
        except Exception as e:
            logger.error(
                "revision_operation_failed",
                extra={"op_type": op.op_type.value, "error": str(e)},
            )
            return False
