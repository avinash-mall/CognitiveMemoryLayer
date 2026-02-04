"""Reconsolidation orchestrator service."""
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.enums import MemoryType, OperationType
from ..core.schemas import MemoryRecord
from ..storage.postgres import PostgresMemoryStore

from .belief_revision import RevisionOperation, RevisionPlan
from .conflict_detector import ConflictDetector
from .labile_tracker import LabileStateTracker

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
    operations_applied: List[Dict[str, Any]]
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
        llm_client: Optional[Any] = None,
        fact_extractor: Optional[Any] = None,
    ):
        self.store = memory_store
        self.labile_tracker = LabileStateTracker()
        self.conflict_detector = ConflictDetector(llm_client)
        from .belief_revision import BeliefRevisionEngine

        self.revision_engine = BeliefRevisionEngine()
        self.fact_extractor = fact_extractor

    async def process_turn(
        self,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        user_message: str,
        assistant_response: str,
        retrieved_memories: List[MemoryRecord],
    ) -> ReconsolidationResult:
        """Process a conversation turn for reconsolidation."""
        start = datetime.utcnow()
        operations_applied: List[Dict[str, Any]] = []
        conflicts_found = 0

        if retrieved_memories:
            await self.labile_tracker.mark_labile(
                tenant_id,
                user_id,
                turn_id,
                memory_ids=[m.id for m in retrieved_memories],
                query=user_message,
                retrieved_texts=[m.text for m in retrieved_memories],
                relevance_scores=[
                    m.metadata.get("_similarity", 0.5) for m in retrieved_memories
                ],
                confidences=[m.confidence for m in retrieved_memories],
            )

        new_facts = await self._extract_new_facts(
            user_message, assistant_response
        )

        if not new_facts:
            await self.labile_tracker.release_labile(
                tenant_id, user_id, turn_id
            )
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            return ReconsolidationResult(
                turn_id=turn_id,
                memories_processed=len(retrieved_memories),
                operations_applied=[],
                conflicts_found=0,
                elapsed_ms=elapsed,
            )

        for new_fact in new_facts:
            for memory in retrieved_memories:
                conflict = await self.conflict_detector.detect(
                    memory, new_fact["text"]
                )
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
                    user_id=user_id,
                    evidence_id=turn_id,
                )

                for op in plan.operations:
                    result = await self._apply_operation(op)
                    operations_applied.append({
                        "operation": op.op_type.value,
                        "target_id": str(op.target_id) if op.target_id else None,
                        "reason": op.reason,
                        "success": result,
                    })

        await self.labile_tracker.release_labile(tenant_id, user_id, turn_id)
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000

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
    ) -> List[Dict[str, Any]]:
        """Extract facts from conversation turn."""
        if self.fact_extractor:
            text = f"User: {user_message}\nAssistant: {assistant_response}"
            facts = await self.fact_extractor.extract(text)
            return [{"text": f.text, "type": getattr(f, "type", "semantic_fact")} for f in facts]

        facts = []
        for sentence in user_message.split("."):
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
                facts.append({
                    "text": sentence,
                    "type": "semantic_fact"
                    if "my name" in lower or "i live" in lower
                    else "preference",
                })
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
            return True
        except Exception:
            return False
