"""Belief revision strategies based on conflict detection."""
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional
from uuid import UUID

from ..core.enums import MemorySource, MemoryType, OperationType
from ..core.schemas import MemoryRecord, MemoryRecordCreate, Provenance

from .conflict_detector import ConflictResult, ConflictType


class RevisionStrategy(str, Enum):
    REINFORCE = "reinforce"
    TIME_SLICE = "time_slice"
    OVERWRITE = "overwrite"
    ADD_HYPOTHESIS = "add_hypothesis"
    MERGE = "merge"
    INVALIDATE = "invalidate"
    NOOP = "noop"


@dataclass
class RevisionOperation:
    """A single revision operation to apply."""

    op_type: OperationType
    target_id: Optional[UUID] = None
    new_record: Optional[MemoryRecordCreate] = None
    patch: Optional[Dict[str, Any]] = None
    reason: str = ""


@dataclass
class RevisionPlan:
    """Complete revision plan."""

    strategy: RevisionStrategy
    operations: List[RevisionOperation]
    confidence: float
    reasoning: str


class BeliefRevisionEngine:
    """
    Applies belief revision strategies based on detected conflicts.
    """

    def plan_revision(
        self,
        conflict: ConflictResult,
        old_memory: MemoryRecord,
        new_info_type: MemoryType,
        tenant_id: str,
        evidence_id: Optional[str] = None,
    ) -> RevisionPlan:
        """Create a revision plan based on conflict analysis. Holistic: uses old_memory context_tags/source_session_id."""
        if conflict.conflict_type == ConflictType.NONE:
            return self._plan_reinforcement(old_memory, conflict)
        if conflict.conflict_type == ConflictType.TEMPORAL_CHANGE:
            return self._plan_time_slice(
                old_memory, conflict, new_info_type, tenant_id, evidence_id
            )
        if conflict.conflict_type == ConflictType.CORRECTION:
            return self._plan_correction(
                old_memory, conflict, new_info_type, tenant_id, evidence_id
            )
        if conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION:
            return self._plan_contradiction_resolution(
                old_memory, conflict, new_info_type, tenant_id, evidence_id
            )
        if conflict.conflict_type == ConflictType.REFINEMENT:
            return self._plan_refinement(
                old_memory, conflict, new_info_type, tenant_id, evidence_id
            )
        return self._plan_hypothesis(old_memory, conflict, tenant_id, evidence_id)

    def _plan_reinforcement(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
    ) -> RevisionPlan:
        """Plan reinforcement when no conflict."""
        new_confidence = min(1.0, old_memory.confidence + 0.1)
        return RevisionPlan(
            strategy=RevisionStrategy.REINFORCE,
            operations=[
                RevisionOperation(
                    op_type=OperationType.REINFORCE,
                    target_id=old_memory.id,
                    patch={
                        "confidence": new_confidence,
                        "access_count": old_memory.access_count + 1,
                        "last_accessed_at": datetime.utcnow(),
                    },
                    reason="Consistent with new information - reinforcing",
                )
            ],
            confidence=conflict.confidence,
            reasoning="No conflict detected, reinforcing existing memory",
        )

    def _plan_time_slice(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        evidence_id: Optional[str],
    ) -> RevisionPlan:
        """Plan time-slice for temporal changes. Archive old record (valid_to, status=archived)."""
        from ..core.enums import MemoryStatus

        now = datetime.utcnow()
        meta = dict(old_memory.metadata)
        meta["superseded"] = True
        return RevisionPlan(
            strategy=RevisionStrategy.TIME_SLICE,
            operations=[
                RevisionOperation(
                    op_type=OperationType.UPDATE,
                    target_id=old_memory.id,
                    patch={
                        "valid_to": now,
                        "status": MemoryStatus.ARCHIVED.value,
                        "metadata": meta,
                    },
                    reason="Archiving as historical - superseded by newer information",
                ),
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        context_tags=old_memory.context_tags,
                        source_session_id=old_memory.source_session_id,
                        type=new_type,
                        text=conflict.new_statement,
                        key=old_memory.key,
                        confidence=conflict.confidence,
                        importance=old_memory.importance,
                        provenance=Provenance(
                            source=MemorySource.RECONSOLIDATION,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={
                            "supersedes": str(old_memory.id),
                            "revision_type": "time_slice",
                        },
                    ),
                    reason="Adding new value as current",
                ),
            ],
            confidence=conflict.confidence,
            reasoning=f"Temporal change detected: {conflict.reasoning}",
        )

    def _plan_correction(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        evidence_id: Optional[str],
    ) -> RevisionPlan:
        """Plan correction when user explicitly corrects. Archive old record instead of delete."""
        from ..core.enums import MemoryStatus

        now = datetime.utcnow()
        meta = dict(old_memory.metadata)
        meta["invalidated_by"] = evidence_id
        meta["invalidated_at"] = now.isoformat()
        return RevisionPlan(
            strategy=RevisionStrategy.TIME_SLICE,
            operations=[
                RevisionOperation(
                    op_type=OperationType.UPDATE,
                    target_id=old_memory.id,
                    patch={
                        "valid_to": now,
                        "status": MemoryStatus.ARCHIVED.value,
                        "confidence": 0.0,
                        "metadata": meta,
                    },
                    reason="User correction - archiving old memory",
                ),
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        context_tags=old_memory.context_tags,
                        source_session_id=old_memory.source_session_id,
                        type=new_type,
                        text=conflict.new_statement,
                        key=old_memory.key,
                        confidence=0.95,
                        importance=old_memory.importance,
                        provenance=Provenance(
                            source=MemorySource.USER_CONFIRMED,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={"corrects": str(old_memory.id)},
                    ),
                    reason="Adding user's corrected information",
                ),
            ],
            confidence=0.95,
            reasoning=f"User explicitly corrected: {conflict.reasoning}",
        )

    def _plan_contradiction_resolution(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        evidence_id: Optional[str],
    ) -> RevisionPlan:
        """Plan resolution for direct contradictions. Holistic: inherit context_tags/source_session_id."""
        old_is_user_confirmed = (
            old_memory.provenance.source == MemorySource.USER_CONFIRMED
        )
        new_confidence = conflict.confidence
        if old_is_user_confirmed and old_memory.confidence > new_confidence:
            return self._plan_hypothesis(old_memory, conflict, tenant_id, evidence_id)
        if conflict.is_superseding or new_confidence > old_memory.confidence:
            return self._plan_time_slice(
                old_memory, conflict, new_type, tenant_id, evidence_id
            )
        return RevisionPlan(
            strategy=RevisionStrategy.ADD_HYPOTHESIS,
            operations=[
                RevisionOperation(
                    op_type=OperationType.UPDATE,
                    target_id=old_memory.id,
                    patch={
                        "confidence": max(0.1, old_memory.confidence - 0.2),
                        "metadata": {
                            **old_memory.metadata,
                            "contested": True,
                            "contested_by": conflict.new_statement,
                        },
                    },
                    reason="Contradiction detected - reducing confidence",
                ),
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        context_tags=old_memory.context_tags,
                        source_session_id=old_memory.source_session_id,
                        type=MemoryType.HYPOTHESIS,
                        text=conflict.new_statement,
                        confidence=max(0.3, new_confidence - 0.2),
                        importance=0.5,
                        provenance=Provenance(
                            source=MemorySource.AGENT_INFERRED,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={
                            "contradicts": str(old_memory.id),
                            "needs_confirmation": True,
                        },
                    ),
                    reason="Adding contradicting info as hypothesis",
                ),
            ],
            confidence=0.5,
            reasoning=f"Contradiction with uncertainty - keeping both: {conflict.reasoning}",
        )

    def _plan_refinement(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        evidence_id: Optional[str],
    ) -> RevisionPlan:
        """Plan refinement when new info adds to existing. Holistic: inherit context_tags/source_session_id."""
        return RevisionPlan(
            strategy=RevisionStrategy.MERGE,
            operations=[
                RevisionOperation(
                    op_type=OperationType.REINFORCE,
                    target_id=old_memory.id,
                    patch={
                        "confidence": min(1.0, old_memory.confidence + 0.05),
                        "access_count": old_memory.access_count + 1,
                    },
                    reason="Related information found - reinforcing",
                ),
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        context_tags=old_memory.context_tags,
                        source_session_id=old_memory.source_session_id,
                        type=new_type,
                        text=conflict.new_statement,
                        confidence=conflict.confidence,
                        importance=old_memory.importance * 0.8,
                        provenance=Provenance(
                            source=MemorySource.AGENT_INFERRED,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={
                            "refines": str(old_memory.id),
                            "relationship": "adds_detail",
                        },
                    ),
                    reason="Adding refinement/detail",
                ),
            ],
            confidence=conflict.confidence,
            reasoning=f"Refinement detected - adding detail: {conflict.reasoning}",
        )

    def _plan_hypothesis(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        tenant_id: str,
        evidence_id: Optional[str],
    ) -> RevisionPlan:
        """Plan adding new info as hypothesis (uncertain). Holistic: inherit context_tags/source_session_id."""
        return RevisionPlan(
            strategy=RevisionStrategy.ADD_HYPOTHESIS,
            operations=[
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        context_tags=old_memory.context_tags,
                        source_session_id=old_memory.source_session_id,
                        type=MemoryType.HYPOTHESIS,
                        text=conflict.new_statement,
                        confidence=min(0.5, conflict.confidence),
                        importance=0.4,
                        provenance=Provenance(
                            source=MemorySource.AGENT_INFERRED,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={
                            "needs_confirmation": True,
                            "related_to": str(old_memory.id),
                        },
                    ),
                    reason="Adding as hypothesis pending confirmation",
                )
            ],
            confidence=conflict.confidence * 0.5,
            reasoning=f"Ambiguous - adding as hypothesis: {conflict.reasoning}",
        )
