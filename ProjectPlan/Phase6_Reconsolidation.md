# Phase 6: Reconsolidation & Belief Revision

## Overview
**Duration**: Week 6-7  
**Goal**: Implement memory updating after retrieval, conflict detection, and belief revision algorithms.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Retrieved                              │
│            (From retrieval system)                               │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Labile State Manager                            │
│   - Mark retrieved memories as "labile" (unstable)               │
│   - Track which memories were used in this turn                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  New Information Extractor                       │
│   - Extract facts from user message + assistant response         │
│   - Compare against retrieved memories                           │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Conflict Detector                               │
│   - Identify contradictions                                      │
│   - Classify conflict type (temporal change, correction, etc.)  │
└─────────────────────────────────────────────────────────────────┘
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
    ┌───────────────────┐      ┌───────────────────┐
    │   Consistent      │      │   Contradictory   │
    │   → Reinforce     │      │   → Revise        │
    └───────────────────┘      └───────────────────┘
                 │                         │
                 └────────────┬────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                  Belief Revision Engine                          │
│   - Apply revision strategy (time-slice, update, invalidate)    │
│   - Update confidence scores                                     │
│   - Maintain audit trail                                         │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 6.1: Labile State Management

### Description
Track memories that are in a "labile" (unstable) state after retrieval.

### Subtask 6.1.1: Labile State Tracker

```python
# src/reconsolidation/labile_tracker.py
from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional
from datetime import datetime, timedelta
from uuid import UUID
import asyncio

@dataclass
class LabileMemory:
    """A memory in labile state."""
    memory_id: UUID
    retrieved_at: datetime
    context: str                    # Query that triggered retrieval
    relevance_score: float
    original_confidence: float
    expires_at: datetime            # When labile state expires
    
@dataclass
class LabileSession:
    """Tracks labile memories for a user session."""
    tenant_id: str
    user_id: str
    turn_id: str
    
    memories: Dict[UUID, LabileMemory] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.utcnow)
    
    # Context from retrieval
    query: str = ""
    retrieved_texts: List[str] = field(default_factory=list)

class LabileStateTracker:
    """
    Tracks memories in labile (unstable) state.
    
    After retrieval, memories enter a labile state where they
    can be modified based on new information. This mimics
    biological reconsolidation.
    """
    
    def __init__(
        self,
        labile_duration_seconds: float = 300,  # 5 minutes
        max_sessions_per_user: int = 10
    ):
        self.labile_duration = timedelta(seconds=labile_duration_seconds)
        self.max_sessions = max_sessions_per_user
        
        # Sessions indexed by (tenant_id, user_id, turn_id)
        self._sessions: Dict[str, LabileSession] = {}
        self._user_sessions: Dict[str, List[str]] = {}  # user -> [session_keys]
        self._lock = asyncio.Lock()
    
    def _session_key(self, tenant_id: str, user_id: str, turn_id: str) -> str:
        return f"{tenant_id}:{user_id}:{turn_id}"
    
    def _user_key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}:{user_id}"
    
    async def mark_labile(
        self,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        memory_ids: List[UUID],
        query: str,
        retrieved_texts: List[str],
        relevance_scores: List[float],
        confidences: List[float]
    ) -> LabileSession:
        """
        Mark memories as labile after retrieval.
        """
        async with self._lock:
            session_key = self._session_key(tenant_id, user_id, turn_id)
            user_key = self._user_key(tenant_id, user_id)
            now = datetime.utcnow()
            expires = now + self.labile_duration
            
            # Create session
            session = LabileSession(
                tenant_id=tenant_id,
                user_id=user_id,
                turn_id=turn_id,
                query=query,
                retrieved_texts=retrieved_texts
            )
            
            # Add memories
            for mid, score, conf in zip(memory_ids, relevance_scores, confidences):
                session.memories[mid] = LabileMemory(
                    memory_id=mid,
                    retrieved_at=now,
                    context=query,
                    relevance_score=score,
                    original_confidence=conf,
                    expires_at=expires
                )
            
            # Store session
            self._sessions[session_key] = session
            
            # Track per user (for cleanup)
            if user_key not in self._user_sessions:
                self._user_sessions[user_key] = []
            self._user_sessions[user_key].append(session_key)
            
            # Enforce max sessions per user
            await self._cleanup_old_sessions(user_key)
            
            return session
    
    async def get_labile_memories(
        self,
        tenant_id: str,
        user_id: str,
        turn_id: Optional[str] = None
    ) -> List[LabileMemory]:
        """
        Get all currently labile memories for a user.
        """
        async with self._lock:
            user_key = self._user_key(tenant_id, user_id)
            now = datetime.utcnow()
            
            labile = []
            session_keys = self._user_sessions.get(user_key, [])
            
            for sk in session_keys:
                if turn_id and not sk.endswith(turn_id):
                    continue
                    
                session = self._sessions.get(sk)
                if not session:
                    continue
                
                for mem in session.memories.values():
                    if mem.expires_at > now:
                        labile.append(mem)
            
            return labile
    
    async def get_session(
        self,
        tenant_id: str,
        user_id: str,
        turn_id: str
    ) -> Optional[LabileSession]:
        """Get a specific session."""
        session_key = self._session_key(tenant_id, user_id, turn_id)
        return self._sessions.get(session_key)
    
    async def release_labile(
        self,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        memory_ids: Optional[List[UUID]] = None
    ):
        """
        Release memories from labile state.
        Called after reconsolidation is complete.
        """
        async with self._lock:
            session_key = self._session_key(tenant_id, user_id, turn_id)
            session = self._sessions.get(session_key)
            
            if not session:
                return
            
            if memory_ids:
                for mid in memory_ids:
                    session.memories.pop(mid, None)
            else:
                session.memories.clear()
            
            # Remove empty session
            if not session.memories:
                del self._sessions[session_key]
                user_key = self._user_key(tenant_id, user_id)
                if user_key in self._user_sessions:
                    self._user_sessions[user_key].remove(session_key)
    
    async def _cleanup_old_sessions(self, user_key: str):
        """Remove old sessions for a user."""
        sessions = self._user_sessions.get(user_key, [])
        
        if len(sessions) <= self.max_sessions:
            return
        
        # Remove oldest sessions
        now = datetime.utcnow()
        to_remove = []
        
        for sk in sessions:
            session = self._sessions.get(sk)
            if not session:
                to_remove.append(sk)
                continue
            
            # Check if all memories expired
            all_expired = all(
                m.expires_at <= now 
                for m in session.memories.values()
            )
            
            if all_expired:
                to_remove.append(sk)
        
        # Remove
        for sk in to_remove:
            self._sessions.pop(sk, None)
            sessions.remove(sk)
        
        # If still over limit, remove oldest
        while len(sessions) > self.max_sessions:
            oldest_key = sessions.pop(0)
            self._sessions.pop(oldest_key, None)
```

---

## Task 6.2: Conflict Detection

### Description
Detect contradictions between new information and existing memories.

### Subtask 6.2.1: Conflict Types and Detector

```python
# src/reconsolidation/conflict_detector.py
from dataclasses import dataclass, field
from typing import List, Optional, Tuple
from enum import Enum
import json
from ..core.schemas import MemoryRecord
from ..utils.llm import LLMClient

class ConflictType(str, Enum):
    NONE = "none"                          # No conflict
    TEMPORAL_CHANGE = "temporal_change"    # Value changed over time (preference)
    DIRECT_CONTRADICTION = "contradiction" # New info contradicts old
    REFINEMENT = "refinement"              # New info refines/adds to old
    CORRECTION = "correction"              # User explicitly correcting
    AMBIGUITY = "ambiguity"                # Unclear if conflict

@dataclass
class ConflictResult:
    """Result of conflict detection."""
    conflict_type: ConflictType
    confidence: float
    
    old_statement: str
    new_statement: str
    
    # Details
    conflicting_aspect: Optional[str] = None  # What specifically conflicts
    suggested_resolution: Optional[str] = None
    
    # For temporal changes
    is_superseding: bool = False  # Does new info replace old?
    
    # Evidence
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

Definitions:
- none: statements are compatible or about different things
- temporal_change: preference/fact that legitimately changed over time
- contradiction: direct factual contradiction
- refinement: new info adds detail to existing
- correction: user explicitly says previous info was wrong
- ambiguity: unclear if conflict exists

Return JSON:
{{
  "conflict_type": "none",
  "conflicting_aspect": null,
  "is_superseding": false,
  "confidence": 0.9,
  "reasoning": "The statements are about different topics"
}}"""

class ConflictDetector:
    """
    Detects conflicts between new information and existing memories.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
    
    async def detect(
        self,
        old_memory: MemoryRecord,
        new_statement: str,
        context: Optional[str] = None
    ) -> ConflictResult:
        """
        Detect if new statement conflicts with existing memory.
        """
        # Try fast heuristic detection first
        fast_result = self._fast_detect(old_memory.text, new_statement)
        if fast_result and fast_result.confidence > 0.8:
            return fast_result
        
        # Use LLM for nuanced detection
        if self.llm:
            return await self._llm_detect(old_memory.text, new_statement, context)
        
        return fast_result or ConflictResult(
            conflict_type=ConflictType.NONE,
            confidence=0.5,
            old_statement=old_memory.text,
            new_statement=new_statement,
            reasoning="No conflict detected (heuristic)"
        )
    
    async def detect_batch(
        self,
        memories: List[MemoryRecord],
        new_statement: str
    ) -> List[ConflictResult]:
        """Detect conflicts against multiple memories."""
        import asyncio
        return await asyncio.gather(*[
            self.detect(mem, new_statement)
            for mem in memories
        ])
    
    def _fast_detect(
        self,
        old_statement: str,
        new_statement: str
    ) -> Optional[ConflictResult]:
        """
        Fast heuristic conflict detection.
        """
        old_lower = old_statement.lower()
        new_lower = new_statement.lower()
        
        # Check for explicit corrections
        correction_markers = [
            "actually", "no,", "that's wrong", "i meant", 
            "correction:", "not anymore", "changed"
        ]
        for marker in correction_markers:
            if marker in new_lower:
                return ConflictResult(
                    conflict_type=ConflictType.CORRECTION,
                    confidence=0.85,
                    old_statement=old_statement,
                    new_statement=new_statement,
                    is_superseding=True,
                    reasoning=f"Contains correction marker: '{marker}'"
                )
        
        # Check for negation of similar content
        negations = ["not", "don't", "doesn't", "no longer", "never"]
        for neg in negations:
            if neg in new_lower and neg not in old_lower:
                # Check if core content overlaps
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
                        reasoning=f"High word overlap ({overlap:.0%}) with negation"
                    )
        
        # Check for identical topics with different values
        # (simplified - would use NER in production)
        preference_words = ["like", "prefer", "favorite", "enjoy", "love", "hate"]
        old_has_pref = any(w in old_lower for w in preference_words)
        new_has_pref = any(w in new_lower for w in preference_words)
        
        if old_has_pref and new_has_pref:
            # Both are preferences - potential temporal change
            return ConflictResult(
                conflict_type=ConflictType.TEMPORAL_CHANGE,
                confidence=0.6,
                old_statement=old_statement,
                new_statement=new_statement,
                is_superseding=True,
                reasoning="Both statements express preferences"
            )
        
        return None
    
    async def _llm_detect(
        self,
        old_statement: str,
        new_statement: str,
        context: Optional[str] = None
    ) -> ConflictResult:
        """
        LLM-based conflict detection.
        """
        prompt = CONFLICT_DETECTION_PROMPT.format(
            old_statement=old_statement,
            new_statement=new_statement
        )
        
        if context:
            prompt = f"CONTEXT:\n{context}\n\n{prompt}"
        
        try:
            response = await self.llm.complete(prompt, temperature=0.0)
            data = json.loads(response)
            
            return ConflictResult(
                conflict_type=ConflictType(data.get("conflict_type", "none")),
                confidence=float(data.get("confidence", 0.7)),
                old_statement=old_statement,
                new_statement=new_statement,
                conflicting_aspect=data.get("conflicting_aspect"),
                is_superseding=data.get("is_superseding", False),
                reasoning=data.get("reasoning", "")
            )
        except (json.JSONDecodeError, ValueError) as e:
            return ConflictResult(
                conflict_type=ConflictType.AMBIGUITY,
                confidence=0.3,
                old_statement=old_statement,
                new_statement=new_statement,
                reasoning=f"LLM detection failed: {e}"
            )
```

---

## Task 6.3: Belief Revision Engine

### Description
Apply belief revision strategies based on conflict type.

### Subtask 6.3.1: Revision Strategies

```python
# src/reconsolidation/belief_revision.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime
from enum import Enum
from uuid import UUID
from .conflict_detector import ConflictResult, ConflictType
from ..core.schemas import MemoryRecord, MemoryRecordCreate, Provenance
from ..core.enums import MemoryType, OperationType, MemorySource

class RevisionStrategy(str, Enum):
    REINFORCE = "reinforce"          # Increase confidence
    TIME_SLICE = "time_slice"        # Mark old as historical, add new
    OVERWRITE = "overwrite"          # Replace old with new
    ADD_HYPOTHESIS = "add_hypothesis" # Add new as hypothesis
    MERGE = "merge"                   # Combine information
    INVALIDATE = "invalidate"         # Mark old as invalid
    NOOP = "noop"                     # No change needed

@dataclass
class RevisionOperation:
    """A single revision operation to apply."""
    op_type: OperationType
    target_id: Optional[UUID] = None    # For UPDATE/DELETE
    new_record: Optional[MemoryRecordCreate] = None  # For ADD
    patch: Optional[Dict[str, Any]] = None  # For UPDATE
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
    
    Strategies by conflict type:
    - NONE: Reinforce (increase confidence if consistent)
    - TEMPORAL_CHANGE: Time-slice (keep old as historical)
    - DIRECT_CONTRADICTION: Depends on confidence/source
    - CORRECTION: Invalidate old, add new
    - REFINEMENT: Merge or add as new
    """
    
    def __init__(self):
        pass
    
    def plan_revision(
        self,
        conflict: ConflictResult,
        old_memory: MemoryRecord,
        new_info_type: MemoryType,
        tenant_id: str,
        user_id: str,
        evidence_id: Optional[str] = None
    ) -> RevisionPlan:
        """
        Create a revision plan based on conflict analysis.
        """
        if conflict.conflict_type == ConflictType.NONE:
            return self._plan_reinforcement(old_memory, conflict)
        
        elif conflict.conflict_type == ConflictType.TEMPORAL_CHANGE:
            return self._plan_time_slice(
                old_memory, conflict, new_info_type, tenant_id, user_id, evidence_id
            )
        
        elif conflict.conflict_type == ConflictType.CORRECTION:
            return self._plan_correction(
                old_memory, conflict, new_info_type, tenant_id, user_id, evidence_id
            )
        
        elif conflict.conflict_type == ConflictType.DIRECT_CONTRADICTION:
            return self._plan_contradiction_resolution(
                old_memory, conflict, new_info_type, tenant_id, user_id, evidence_id
            )
        
        elif conflict.conflict_type == ConflictType.REFINEMENT:
            return self._plan_refinement(
                old_memory, conflict, new_info_type, tenant_id, user_id, evidence_id
            )
        
        else:  # AMBIGUITY or unknown
            return self._plan_hypothesis(
                old_memory, conflict, tenant_id, user_id, evidence_id
            )
    
    def _plan_reinforcement(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult
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
                        "last_accessed_at": datetime.utcnow()
                    },
                    reason="Consistent with new information - reinforcing"
                )
            ],
            confidence=conflict.confidence,
            reasoning="No conflict detected, reinforcing existing memory"
        )
    
    def _plan_time_slice(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        user_id: str,
        evidence_id: Optional[str]
    ) -> RevisionPlan:
        """Plan time-slice for temporal changes."""
        now = datetime.utcnow()
        
        return RevisionPlan(
            strategy=RevisionStrategy.TIME_SLICE,
            operations=[
                # Mark old as historical
                RevisionOperation(
                    op_type=OperationType.UPDATE,
                    target_id=old_memory.id,
                    patch={
                        "valid_to": now,
                        "metadata": {**old_memory.metadata, "superseded": True}
                    },
                    reason="Marking as historical - superseded by newer information"
                ),
                # Add new as current
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        type=new_type,
                        text=conflict.new_statement,
                        key=old_memory.key,  # Same key for lookups
                        confidence=conflict.confidence,
                        importance=old_memory.importance,
                        provenance=Provenance(
                            source=MemorySource.RECONSOLIDATION,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={
                            "supersedes": str(old_memory.id),
                            "revision_type": "time_slice"
                        }
                    ),
                    reason="Adding new value as current"
                )
            ],
            confidence=conflict.confidence,
            reasoning=f"Temporal change detected: {conflict.reasoning}"
        )
    
    def _plan_correction(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        user_id: str,
        evidence_id: Optional[str]
    ) -> RevisionPlan:
        """Plan correction when user explicitly corrects."""
        return RevisionPlan(
            strategy=RevisionStrategy.OVERWRITE,
            operations=[
                # Invalidate old
                RevisionOperation(
                    op_type=OperationType.UPDATE,
                    target_id=old_memory.id,
                    patch={
                        "status": "invalid",
                        "confidence": 0.0,
                        "metadata": {
                            **old_memory.metadata,
                            "invalidated_by": evidence_id,
                            "invalidated_at": datetime.utcnow().isoformat()
                        }
                    },
                    reason="User correction - invalidating old memory"
                ),
                # Add corrected version
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        user_id=user_id,
                        type=new_type,
                        text=conflict.new_statement,
                        key=old_memory.key,
                        confidence=0.95,  # High confidence for explicit correction
                        importance=old_memory.importance,
                        provenance=Provenance(
                            source=MemorySource.USER_CONFIRMED,
                            evidence_refs=[evidence_id] if evidence_id else [],
                        ),
                        metadata={"corrects": str(old_memory.id)}
                    ),
                    reason="Adding user's corrected information"
                )
            ],
            confidence=0.95,
            reasoning=f"User explicitly corrected: {conflict.reasoning}"
        )
    
    def _plan_contradiction_resolution(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        user_id: str,
        evidence_id: Optional[str]
    ) -> RevisionPlan:
        """Plan resolution for direct contradictions."""
        # Decide based on confidence and source
        old_is_user_confirmed = old_memory.provenance.source == MemorySource.USER_CONFIRMED
        new_confidence = conflict.confidence
        
        if old_is_user_confirmed and old_memory.confidence > new_confidence:
            # Keep old, add new as hypothesis
            return self._plan_hypothesis(
                old_memory, conflict, tenant_id, user_id, evidence_id
            )
        
        elif conflict.is_superseding or new_confidence > old_memory.confidence:
            # New info wins - time slice
            return self._plan_time_slice(
                old_memory, conflict, new_type, tenant_id, user_id, evidence_id
            )
        
        else:
            # Uncertain - reduce both confidences, keep both
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
                                "contested_by": conflict.new_statement
                            }
                        },
                        reason="Contradiction detected - reducing confidence"
                    ),
                    RevisionOperation(
                        op_type=OperationType.ADD,
                        new_record=MemoryRecordCreate(
                            tenant_id=tenant_id,
                            user_id=user_id,
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
                                "needs_confirmation": True
                            }
                        ),
                        reason="Adding contradicting info as hypothesis"
                    )
                ],
                confidence=0.5,
                reasoning=f"Contradiction with uncertainty - keeping both: {conflict.reasoning}"
            )
    
    def _plan_refinement(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        new_type: MemoryType,
        tenant_id: str,
        user_id: str,
        evidence_id: Optional[str]
    ) -> RevisionPlan:
        """Plan refinement when new info adds to existing."""
        # Option 1: Update old with merged info
        # Option 2: Add new as separate memory
        
        # For now, add as separate (safer)
        return RevisionPlan(
            strategy=RevisionStrategy.MERGE,
            operations=[
                # Reinforce old
                RevisionOperation(
                    op_type=OperationType.REINFORCE,
                    target_id=old_memory.id,
                    patch={
                        "confidence": min(1.0, old_memory.confidence + 0.05),
                        "access_count": old_memory.access_count + 1
                    },
                    reason="Related information found - reinforcing"
                ),
                # Add new detail
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        user_id=user_id,
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
                            "relationship": "adds_detail"
                        }
                    ),
                    reason="Adding refinement/detail"
                )
            ],
            confidence=conflict.confidence,
            reasoning=f"Refinement detected - adding detail: {conflict.reasoning}"
        )
    
    def _plan_hypothesis(
        self,
        old_memory: MemoryRecord,
        conflict: ConflictResult,
        tenant_id: str,
        user_id: str,
        evidence_id: Optional[str]
    ) -> RevisionPlan:
        """Plan adding new info as hypothesis (uncertain)."""
        return RevisionPlan(
            strategy=RevisionStrategy.ADD_HYPOTHESIS,
            operations=[
                RevisionOperation(
                    op_type=OperationType.ADD,
                    new_record=MemoryRecordCreate(
                        tenant_id=tenant_id,
                        user_id=user_id,
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
                            "related_to": str(old_memory.id)
                        }
                    ),
                    reason="Adding as hypothesis pending confirmation"
                )
            ],
            confidence=conflict.confidence * 0.5,
            reasoning=f"Ambiguous - adding as hypothesis: {conflict.reasoning}"
        )
```

---

## Task 6.4: Reconsolidation Orchestrator

### Description
Coordinate the full reconsolidation process.

### Subtask 6.4.1: Reconsolidation Service

```python
# src/reconsolidation/service.py
from typing import List, Optional, Dict, Any
from datetime import datetime
from uuid import UUID
from dataclasses import dataclass
from .labile_tracker import LabileStateTracker, LabileSession
from .conflict_detector import ConflictDetector, ConflictResult
from .belief_revision import BeliefRevisionEngine, RevisionPlan, RevisionOperation
from ..core.schemas import MemoryRecord
from ..core.enums import OperationType, MemoryType
from ..storage.postgres import PostgresMemoryStore
from ..extraction.fact_extractor import FactExtractor, LLMFactExtractor
from ..utils.llm import LLMClient

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
    1. Extract new facts from user message + response
    2. Get labile memories from current session
    3. Detect conflicts between new facts and labile memories
    4. Plan and apply revisions
    5. Release memories from labile state
    """
    
    def __init__(
        self,
        memory_store: PostgresMemoryStore,
        llm_client: Optional[LLMClient] = None,
        fact_extractor: Optional[FactExtractor] = None
    ):
        self.store = memory_store
        self.labile_tracker = LabileStateTracker()
        self.conflict_detector = ConflictDetector(llm_client)
        self.revision_engine = BeliefRevisionEngine()
        # Orchestrator wires LLMFactExtractor(llm_client) by default for LLM-based fact extraction
        self.fact_extractor = fact_extractor
    
    async def process_turn(
        self,
        tenant_id: str,
        user_id: str,
        turn_id: str,
        user_message: str,
        assistant_response: str,
        retrieved_memories: List[MemoryRecord]
    ) -> ReconsolidationResult:
        """
        Process a conversation turn for reconsolidation.
        """
        start = datetime.utcnow()
        operations_applied = []
        conflicts_found = 0
        
        # 1. Mark retrieved memories as labile
        if retrieved_memories:
            await self.labile_tracker.mark_labile(
                tenant_id, user_id, turn_id,
                memory_ids=[m.id for m in retrieved_memories],
                query=user_message,
                retrieved_texts=[m.text for m in retrieved_memories],
                relevance_scores=[m.metadata.get("_similarity", 0.5) for m in retrieved_memories],
                confidences=[m.confidence for m in retrieved_memories]
            )
        
        # 2. Extract new facts from conversation
        new_facts = await self._extract_new_facts(
            user_message, assistant_response
        )
        
        if not new_facts:
            # Nothing to reconsolidate
            await self.labile_tracker.release_labile(tenant_id, user_id, turn_id)
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            return ReconsolidationResult(
                turn_id=turn_id,
                memories_processed=len(retrieved_memories),
                operations_applied=[],
                conflicts_found=0,
                elapsed_ms=elapsed
            )
        
        # 3. Compare new facts against labile memories
        for new_fact in new_facts:
            for memory in retrieved_memories:
                # Detect conflict
                conflict = await self.conflict_detector.detect(
                    memory, new_fact["text"]
                )
                
                if conflict.conflict_type.value != "none":
                    conflicts_found += 1
                
                # Plan revision
                plan = self.revision_engine.plan_revision(
                    conflict=conflict,
                    old_memory=memory,
                    new_info_type=MemoryType(new_fact.get("type", "episodic_event")),
                    tenant_id=tenant_id,
                    user_id=user_id,
                    evidence_id=turn_id
                )
                
                # Apply revision operations
                for op in plan.operations:
                    result = await self._apply_operation(op)
                    operations_applied.append({
                        "operation": op.op_type.value,
                        "target_id": str(op.target_id) if op.target_id else None,
                        "reason": op.reason,
                        "success": result
                    })
        
        # 4. Release from labile state
        await self.labile_tracker.release_labile(tenant_id, user_id, turn_id)
        
        elapsed = (datetime.utcnow() - start).total_seconds() * 1000
        
        return ReconsolidationResult(
            turn_id=turn_id,
            memories_processed=len(retrieved_memories),
            operations_applied=operations_applied,
            conflicts_found=conflicts_found,
            elapsed_ms=elapsed
        )
    
    async def _extract_new_facts(
        self,
        user_message: str,
        assistant_response: str
    ) -> List[Dict[str, Any]]:
        """Extract facts from conversation turn."""
        if self.fact_extractor:
            # Use dedicated extractor
            facts = await self.fact_extractor.extract(
                f"User: {user_message}\nAssistant: {assistant_response}"
            )
            return [{"text": f.text, "type": f.type} for f in facts]
        
        # Simple extraction: look for statements that look like facts
        facts = []
        
        # From user message
        for sentence in user_message.split("."):
            sentence = sentence.strip()
            if not sentence:
                continue
            
            # Check for fact-like patterns
            lower = sentence.lower()
            if any(m in lower for m in ["i am", "i'm", "my name", "i live", "i work", "i like", "i prefer"]):
                facts.append({
                    "text": sentence,
                    "type": "semantic_fact" if "my name" in lower or "i live" in lower else "preference"
                })
        
        return facts
    
    async def _apply_operation(self, op: RevisionOperation) -> bool:
        """Apply a single revision operation."""
        try:
            if op.op_type == OperationType.ADD:
                if op.new_record:
                    await self.store.upsert(op.new_record)
                return True
            
            elif op.op_type in [OperationType.UPDATE, OperationType.REINFORCE, OperationType.DECAY]:
                if op.target_id and op.patch:
                    await self.store.update(op.target_id, op.patch)
                return True
            
            elif op.op_type == OperationType.DELETE:
                if op.target_id:
                    await self.store.delete(op.target_id, hard=False)
                return True
            
            return True  # NOOP
            
        except Exception as e:
            print(f"Failed to apply operation: {e}")
            return False
```

---

## Deliverables Checklist

- [ ] LabileMemory and LabileSession models
- [ ] LabileStateTracker with session management
- [ ] ConflictType enum and ConflictResult model
- [ ] ConflictDetector with fast heuristics and LLM fallback
- [ ] RevisionStrategy enum and RevisionPlan model
- [ ] BeliefRevisionEngine with all strategies
- [ ] ReconsolidationService orchestrating the flow
- [ ] Unit tests for conflict detection
- [ ] Unit tests for revision planning
- [ ] Integration tests for full reconsolidation
