# Phase 8: Active Forgetting

## Overview
**Duration**: Week 8-9  
**Goal**: Implement intelligent forgetting mechanisms including decay, silencing, compression, and pruning.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Forgetting Triggers                           │
│   - Scheduled (daily/weekly)                                     │
│   - Quota exceeded (storage limit)                               │
│   - Performance degradation (retrieval latency)                  │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Relevance Scorer                              │
│   - Importance score                                             │
│   - Recency score                                                │
│   - Usage frequency score                                        │
│   - Confidence score                                             │
│   - Combined weighted score                                      │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Forgetting Policy Engine                      │
│   Score > 0.7  → Keep as-is                                     │
│   Score 0.4-0.7 → Decay confidence                              │
│   Score 0.2-0.4 → Silence (hard to retrieve)                    │
│   Score 0.1-0.2 → Compress (keep gist only)                     │
│   Score < 0.1  → Delete (if no dependencies)                    │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┴─────────────────┐
            ▼                                   ▼
    ┌───────────────┐                  ┌───────────────┐
    │  Soft Actions │                  │  Hard Actions │
    │  - Decay      │                  │  - Compress   │
    │  - Silence    │                  │  - Archive    │
    └───────────────┘                  │  - Delete     │
                                       └───────────────┘
```

---

## Task 8.1: Relevance Scoring

### Description
Calculate relevance scores for memories to determine forgetting priority.

### Subtask 8.1.1: Relevance Score Calculator

```python
# src/forgetting/scorer.py
from dataclasses import dataclass, field
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import math
from ..core.schemas import MemoryRecord
from ..core.enums import MemoryType, MemoryStatus

@dataclass
class RelevanceWeights:
    """Weights for relevance score components."""
    importance: float = 0.25
    recency: float = 0.20
    frequency: float = 0.20
    confidence: float = 0.15
    type_bonus: float = 0.10
    dependency: float = 0.10
    
    def validate(self):
        total = (self.importance + self.recency + self.frequency + 
                 self.confidence + self.type_bonus + self.dependency)
        assert abs(total - 1.0) < 0.01, f"Weights must sum to 1.0, got {total}"

@dataclass
class RelevanceScore:
    """Detailed relevance score breakdown."""
    memory_id: str
    total_score: float
    
    # Component scores (0-1)
    importance_score: float
    recency_score: float
    frequency_score: float
    confidence_score: float
    type_bonus_score: float
    dependency_score: float
    
    # Metadata
    computed_at: datetime = field(default_factory=datetime.utcnow)
    
    # Suggested action
    suggested_action: str = "keep"  # keep, decay, silence, compress, delete

@dataclass
class ScorerConfig:
    weights: RelevanceWeights = field(default_factory=RelevanceWeights)
    
    # Recency decay parameters
    recency_half_life_days: float = 30.0  # Score halves every 30 days
    
    # Frequency normalization
    frequency_log_base: float = 10.0
    
    # Type bonuses (some types should be harder to forget)
    type_bonuses: Dict[str, float] = field(default_factory=lambda: {
        MemoryType.CONSTRAINT.value: 1.0,      # Never forget constraints
        MemoryType.PREFERENCE.value: 0.8,      # Preferences are important
        MemoryType.SEMANTIC_FACT.value: 0.7,   # Facts should persist
        MemoryType.PROCEDURE.value: 0.6,       # Procedures are useful
        MemoryType.EPISODIC_EVENT.value: 0.3,  # Episodes can fade
        MemoryType.HYPOTHESIS.value: 0.2,      # Hypotheses are tentative
        MemoryType.TASK_STATE.value: 0.1,      # Task states are transient
    })
    
    # Thresholds for actions
    keep_threshold: float = 0.7
    decay_threshold: float = 0.4
    silence_threshold: float = 0.2
    compress_threshold: float = 0.1

class RelevanceScorer:
    """
    Calculates relevance scores for memories.
    
    Mimics biological forgetting where:
    - Important things are remembered
    - Frequently accessed things are remembered
    - Recent things are remembered
    - High-confidence things are remembered
    """
    
    def __init__(self, config: Optional[ScorerConfig] = None):
        self.config = config or ScorerConfig()
        self.config.weights.validate()
    
    def score(
        self,
        record: MemoryRecord,
        dependency_count: int = 0
    ) -> RelevanceScore:
        """
        Calculate relevance score for a memory.
        """
        # 1. Importance score (directly from record)
        importance = record.importance
        
        # 2. Recency score (exponential decay)
        age_days = (datetime.utcnow() - record.timestamp).total_seconds() / 86400
        recency = math.pow(0.5, age_days / self.config.recency_half_life_days)
        
        # 3. Frequency score (logarithmic)
        frequency = math.log(1 + record.access_count, self.config.frequency_log_base)
        frequency = min(frequency, 1.0)  # Cap at 1.0
        
        # 4. Confidence score (directly from record)
        confidence = record.confidence
        
        # 5. Type bonus
        record_type = record.type if isinstance(record.type, str) else record.type.value
        type_bonus = self.config.type_bonuses.get(record_type, 0.5)
        
        # 6. Dependency score (if other memories reference this)
        dependency = min(dependency_count / 10.0, 1.0)  # Normalize to 0-1
        
        # Weighted combination
        w = self.config.weights
        total = (
            w.importance * importance +
            w.recency * recency +
            w.frequency * frequency +
            w.confidence * confidence +
            w.type_bonus * type_bonus +
            w.dependency * dependency
        )
        
        # Determine suggested action
        suggested = self._suggest_action(total, record_type)
        
        return RelevanceScore(
            memory_id=str(record.id),
            total_score=total,
            importance_score=importance,
            recency_score=recency,
            frequency_score=frequency,
            confidence_score=confidence,
            type_bonus_score=type_bonus,
            dependency_score=dependency,
            suggested_action=suggested
        )
    
    def score_batch(
        self,
        records: List[MemoryRecord],
        dependency_counts: Optional[Dict[str, int]] = None
    ) -> List[RelevanceScore]:
        """Score multiple records."""
        dep_counts = dependency_counts or {}
        
        return [
            self.score(r, dep_counts.get(str(r.id), 0))
            for r in records
        ]
    
    def _suggest_action(self, score: float, memory_type: str) -> str:
        """Suggest forgetting action based on score."""
        # Never delete constraints
        if memory_type == MemoryType.CONSTRAINT.value:
            return "keep"
        
        if score >= self.config.keep_threshold:
            return "keep"
        elif score >= self.config.decay_threshold:
            return "decay"
        elif score >= self.config.silence_threshold:
            return "silence"
        elif score >= self.config.compress_threshold:
            return "compress"
        else:
            return "delete"
```

---

## Task 8.2: Forgetting Policy Engine

### Description
Apply forgetting policies based on relevance scores.

### Subtask 8.2.1: Forgetting Actions

```python
# src/forgetting/actions.py
from dataclasses import dataclass
from typing import Optional, List, Dict, Any
from datetime import datetime
from enum import Enum
from uuid import UUID
from ..core.enums import MemoryStatus

class ForgettingAction(str, Enum):
    KEEP = "keep"              # No change
    DECAY = "decay"            # Reduce confidence
    SILENCE = "silence"        # Make hard to retrieve
    COMPRESS = "compress"      # Keep summary only
    ARCHIVE = "archive"        # Move to cold storage
    DELETE = "delete"          # Remove from active store

@dataclass
class ForgettingOperation:
    """A forgetting operation to apply."""
    action: ForgettingAction
    memory_id: UUID
    
    # For DECAY
    new_confidence: Optional[float] = None
    
    # For COMPRESS
    compressed_text: Optional[str] = None
    
    # Metadata
    reason: str = ""
    relevance_score: float = 0.0
    
@dataclass
class ForgettingResult:
    """Result of applying forgetting operations."""
    operations_planned: int
    operations_applied: int
    
    kept: int = 0
    decayed: int = 0
    silenced: int = 0
    compressed: int = 0
    archived: int = 0
    deleted: int = 0
    
    errors: List[str] = None
    
    def __post_init__(self):
        if self.errors is None:
            self.errors = []

class ForgettingPolicyEngine:
    """
    Applies forgetting policies to memories.
    """
    
    def __init__(
        self,
        decay_rate: float = 0.1,       # Confidence reduction per decay
        min_confidence: float = 0.05,   # Don't decay below this
        compression_max_chars: int = 100
    ):
        self.decay_rate = decay_rate
        self.min_confidence = min_confidence
        self.compression_max_chars = compression_max_chars
    
    def plan_operations(
        self,
        scores: List["RelevanceScore"],
        max_operations: Optional[int] = None
    ) -> List[ForgettingOperation]:
        """
        Plan forgetting operations based on scores.
        """
        operations = []
        
        for score in scores:
            action = ForgettingAction(score.suggested_action)
            
            if action == ForgettingAction.KEEP:
                continue
            
            op = ForgettingOperation(
                action=action,
                memory_id=UUID(score.memory_id),
                relevance_score=score.total_score,
                reason=f"Score {score.total_score:.2f} below threshold"
            )
            
            if action == ForgettingAction.DECAY:
                # Calculate new confidence
                current_conf = score.confidence_score
                op.new_confidence = max(
                    self.min_confidence,
                    current_conf - self.decay_rate
                )
            
            operations.append(op)
            
            if max_operations and len(operations) >= max_operations:
                break
        
        return operations
    
    def create_compression(self, text: str) -> str:
        """Create compressed version of text."""
        if len(text) <= self.compression_max_chars:
            return text
        
        # Simple truncation with ellipsis
        # In production, use LLM summarization
        return text[:self.compression_max_chars - 3] + "..."
```

### Subtask 8.2.2: Policy Executor

```python
# src/forgetting/executor.py
from typing import List, Optional, Dict
from datetime import datetime
from uuid import UUID
from .actions import ForgettingOperation, ForgettingAction, ForgettingResult
from ..storage.postgres import PostgresMemoryStore
from ..core.enums import MemoryStatus

class ForgettingExecutor:
    """
    Executes forgetting operations on the memory store.
    """
    
    def __init__(
        self,
        store: PostgresMemoryStore,
        archive_store: Optional[PostgresMemoryStore] = None
    ):
        self.store = store
        self.archive_store = archive_store
    
    async def execute(
        self,
        operations: List[ForgettingOperation],
        dry_run: bool = False
    ) -> ForgettingResult:
        """
        Execute forgetting operations.
        
        Args:
            operations: Operations to execute
            dry_run: If True, don't actually modify anything
        """
        result = ForgettingResult(
            operations_planned=len(operations),
            operations_applied=0
        )
        
        for op in operations:
            try:
                if op.action == ForgettingAction.KEEP:
                    result.kept += 1
                    continue
                
                if dry_run:
                    # Just count
                    self._count_action(result, op.action)
                    result.operations_applied += 1
                    continue
                
                success = await self._execute_operation(op)
                
                if success:
                    self._count_action(result, op.action)
                    result.operations_applied += 1
                else:
                    result.errors.append(f"Failed to execute {op.action} on {op.memory_id}")
                    
            except Exception as e:
                result.errors.append(f"Error executing {op.action} on {op.memory_id}: {e}")
        
        return result
    
    async def _execute_operation(self, op: ForgettingOperation) -> bool:
        """Execute a single operation."""
        if op.action == ForgettingAction.DECAY:
            return await self._execute_decay(op)
        
        elif op.action == ForgettingAction.SILENCE:
            return await self._execute_silence(op)
        
        elif op.action == ForgettingAction.COMPRESS:
            return await self._execute_compress(op)
        
        elif op.action == ForgettingAction.ARCHIVE:
            return await self._execute_archive(op)
        
        elif op.action == ForgettingAction.DELETE:
            return await self._execute_delete(op)
        
        return False
    
    async def _execute_decay(self, op: ForgettingOperation) -> bool:
        """Reduce confidence of a memory."""
        patch = {
            "confidence": op.new_confidence,
            "metadata": {"last_decay": datetime.utcnow().isoformat()}
        }
        
        result = await self.store.update(op.memory_id, patch, increment_version=False)
        return result is not None
    
    async def _execute_silence(self, op: ForgettingOperation) -> bool:
        """Mark memory as silent (hard to retrieve)."""
        patch = {
            "status": MemoryStatus.SILENT.value,
            "metadata": {"silenced_at": datetime.utcnow().isoformat()}
        }
        
        result = await self.store.update(op.memory_id, patch)
        return result is not None
    
    async def _execute_compress(self, op: ForgettingOperation) -> bool:
        """Compress memory to gist only."""
        # Get current memory
        record = await self.store.get_by_id(op.memory_id)
        if not record:
            return False
        
        patch = {
            "text": op.compressed_text or record.text[:100],
            "status": MemoryStatus.COMPRESSED.value,
            "embedding": None,  # Remove embedding to save space
            "entities": [],
            "relations": [],
            "metadata": {
                **record.metadata,
                "compressed_at": datetime.utcnow().isoformat(),
                "original_length": len(record.text)
            }
        }
        
        result = await self.store.update(op.memory_id, patch)
        return result is not None
    
    async def _execute_archive(self, op: ForgettingOperation) -> bool:
        """Move to archive store."""
        if not self.archive_store:
            # No archive store - just mark as archived
            patch = {
                "status": MemoryStatus.ARCHIVED.value,
                "metadata": {"archived_at": datetime.utcnow().isoformat()}
            }
            result = await self.store.update(op.memory_id, patch)
            return result is not None
        
        # Get record
        record = await self.store.get_by_id(op.memory_id)
        if not record:
            return False
        
        # Copy to archive
        # (In production, convert to MemoryRecordCreate)
        # await self.archive_store.upsert(record)
        
        # Delete from main store
        await self.store.delete(op.memory_id, hard=True)
        
        return True
    
    async def _execute_delete(self, op: ForgettingOperation) -> bool:
        """Delete memory from store."""
        # Check for dependencies first
        # (In production, check if other memories reference this)
        
        return await self.store.delete(op.memory_id, hard=False)
    
    def _count_action(self, result: ForgettingResult, action: ForgettingAction):
        """Increment counter for action type."""
        if action == ForgettingAction.KEEP:
            result.kept += 1
        elif action == ForgettingAction.DECAY:
            result.decayed += 1
        elif action == ForgettingAction.SILENCE:
            result.silenced += 1
        elif action == ForgettingAction.COMPRESS:
            result.compressed += 1
        elif action == ForgettingAction.ARCHIVE:
            result.archived += 1
        elif action == ForgettingAction.DELETE:
            result.deleted += 1
```

---

## Task 8.3: Interference Management

### Description
Handle cases where new memories interfere with old ones.

### Subtask 8.3.1: Interference Detector

```python
# src/forgetting/interference.py
from dataclasses import dataclass
from typing import List, Optional, Tuple
from ..core.schemas import MemoryRecord
from ..utils.embeddings import EmbeddingClient
import numpy as np

@dataclass
class InterferenceResult:
    """Result of interference detection."""
    memory_id: str
    interfering_memory_id: str
    similarity: float
    interference_type: str  # "duplicate", "conflicting", "overlapping"
    recommendation: str     # "merge", "keep_newer", "keep_higher_confidence"

class InterferenceDetector:
    """
    Detects interference between memories.
    
    Interference occurs when:
    1. Memories are highly similar (duplicates)
    2. Memories conflict but cover same topic
    3. Memories overlap significantly in content
    """
    
    def __init__(
        self,
        embedding_client: Optional[EmbeddingClient] = None,
        similarity_threshold: float = 0.9,
        conflict_threshold: float = 0.7
    ):
        self.embeddings = embedding_client
        self.similarity_threshold = similarity_threshold
        self.conflict_threshold = conflict_threshold
    
    def detect_duplicates(
        self,
        records: List[MemoryRecord]
    ) -> List[InterferenceResult]:
        """
        Detect near-duplicate memories.
        """
        results = []
        
        # Get embeddings
        embeddings = {}
        for r in records:
            if r.embedding:
                embeddings[str(r.id)] = np.array(r.embedding)
        
        # Compare all pairs
        ids = list(embeddings.keys())
        for i, id1 in enumerate(ids):
            for id2 in ids[i+1:]:
                similarity = self._cosine_similarity(
                    embeddings[id1], embeddings[id2]
                )
                
                if similarity >= self.similarity_threshold:
                    # Find which record to keep
                    r1 = next(r for r in records if str(r.id) == id1)
                    r2 = next(r for r in records if str(r.id) == id2)
                    
                    results.append(InterferenceResult(
                        memory_id=id1,
                        interfering_memory_id=id2,
                        similarity=similarity,
                        interference_type="duplicate",
                        recommendation=self._recommend_resolution(r1, r2)
                    ))
        
        return results
    
    def detect_overlapping(
        self,
        records: List[MemoryRecord],
        text_overlap_threshold: float = 0.7
    ) -> List[InterferenceResult]:
        """
        Detect memories with significant text overlap.
        """
        results = []
        
        for i, r1 in enumerate(records):
            for r2 in records[i+1:]:
                overlap = self._text_overlap(r1.text, r2.text)
                
                if overlap >= text_overlap_threshold:
                    results.append(InterferenceResult(
                        memory_id=str(r1.id),
                        interfering_memory_id=str(r2.id),
                        similarity=overlap,
                        interference_type="overlapping",
                        recommendation=self._recommend_resolution(r1, r2)
                    ))
        
        return results
    
    def _cosine_similarity(self, v1: np.ndarray, v2: np.ndarray) -> float:
        """Calculate cosine similarity."""
        dot = np.dot(v1, v2)
        norm = np.linalg.norm(v1) * np.linalg.norm(v2)
        return dot / norm if norm > 0 else 0.0
    
    def _text_overlap(self, text1: str, text2: str) -> float:
        """Calculate word-level text overlap."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
    
    def _recommend_resolution(
        self,
        r1: MemoryRecord,
        r2: MemoryRecord
    ) -> str:
        """Recommend how to resolve interference."""
        # Keep the one with higher confidence
        if abs(r1.confidence - r2.confidence) > 0.2:
            return "keep_higher_confidence"
        
        # Keep the more recent one
        if r1.timestamp > r2.timestamp:
            return "keep_newer"
        elif r2.timestamp > r1.timestamp:
            return "keep_newer"
        
        # If similar, merge
        return "merge"
```

---

## Task 8.4: Forgetting Worker

### Description
Main forgetting service that orchestrates the process.

### Subtask 8.4.1: Forgetting Worker Service

```python
# src/forgetting/worker.py
from typing import Optional, List, Dict
from datetime import datetime, timedelta
from dataclasses import dataclass
import asyncio
from .scorer import RelevanceScorer, ScorerConfig, RelevanceScore
from .actions import ForgettingPolicyEngine, ForgettingResult
from .executor import ForgettingExecutor
from .interference import InterferenceDetector
from ..storage.postgres import PostgresMemoryStore
from ..core.enums import MemoryStatus

@dataclass
class ForgettingReport:
    """Report from a forgetting run."""
    tenant_id: str
    user_id: str
    started_at: datetime
    completed_at: datetime
    
    # Stats
    memories_scanned: int
    memories_scored: int
    result: ForgettingResult
    
    # Interference
    duplicates_found: int = 0
    duplicates_resolved: int = 0
    
    # Performance
    elapsed_seconds: float = 0.0

class ForgettingWorker:
    """
    Orchestrates the active forgetting process.
    """
    
    def __init__(
        self,
        store: PostgresMemoryStore,
        scorer_config: Optional[ScorerConfig] = None,
        archive_store: Optional[PostgresMemoryStore] = None
    ):
        self.store = store
        self.scorer = RelevanceScorer(scorer_config)
        self.policy = ForgettingPolicyEngine()
        self.executor = ForgettingExecutor(store, archive_store)
        self.interference = InterferenceDetector()
    
    async def run_forgetting(
        self,
        tenant_id: str,
        user_id: str,
        max_memories: int = 5000,
        dry_run: bool = False
    ) -> ForgettingReport:
        """
        Run forgetting process for a user.
        """
        started = datetime.utcnow()
        
        # 1. Scan memories
        memories = await self.store.scan(
            tenant_id, user_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=max_memories
        )
        
        if not memories:
            return ForgettingReport(
                tenant_id=tenant_id,
                user_id=user_id,
                started_at=started,
                completed_at=datetime.utcnow(),
                memories_scanned=0,
                memories_scored=0,
                result=ForgettingResult(0, 0)
            )
        
        # 2. Calculate dependency counts
        dep_counts = await self._get_dependency_counts(
            tenant_id, user_id, memories
        )
        
        # 3. Score memories
        scores = self.scorer.score_batch(memories, dep_counts)
        
        # 4. Plan operations
        operations = self.policy.plan_operations(scores)
        
        # 5. Detect and handle duplicates
        duplicates = self.interference.detect_duplicates(memories)
        dup_operations = self._plan_duplicate_resolution(duplicates)
        operations.extend(dup_operations)
        
        # 6. Execute
        result = await self.executor.execute(operations, dry_run=dry_run)
        
        completed = datetime.utcnow()
        
        return ForgettingReport(
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=started,
            completed_at=completed,
            memories_scanned=len(memories),
            memories_scored=len(scores),
            result=result,
            duplicates_found=len(duplicates),
            duplicates_resolved=len(dup_operations),
            elapsed_seconds=(completed - started).total_seconds()
        )
    
    async def _get_dependency_counts(
        self,
        tenant_id: str,
        user_id: str,
        memories: List
    ) -> Dict[str, int]:
        """
        Count how many other memories reference each memory.
        """
        counts = {}
        
        for mem in memories:
            mem_id = str(mem.id)
            counts[mem_id] = 0
            
            # Check if any memory references this in metadata
            for other in memories:
                if other.id == mem.id:
                    continue
                
                # Check supersedes_id
                if other.supersedes_id and str(other.supersedes_id) == mem_id:
                    counts[mem_id] += 1
                
                # Check evidence_refs in metadata
                refs = other.metadata.get("evidence_refs", [])
                if mem_id in refs:
                    counts[mem_id] += 1
        
        return counts
    
    def _plan_duplicate_resolution(
        self,
        duplicates: List
    ) -> List:
        """Plan operations to resolve duplicates."""
        from .actions import ForgettingOperation, ForgettingAction
        from uuid import UUID
        
        operations = []
        resolved_ids = set()
        
        for dup in duplicates:
            # Don't process if already resolved
            if dup.memory_id in resolved_ids or dup.interfering_memory_id in resolved_ids:
                continue
            
            # Keep one, delete the other
            if dup.recommendation == "keep_newer":
                to_delete = dup.memory_id  # Assuming first is older
            else:
                to_delete = dup.interfering_memory_id
            
            operations.append(ForgettingOperation(
                action=ForgettingAction.DELETE,
                memory_id=UUID(to_delete),
                reason=f"Duplicate of {dup.interfering_memory_id if to_delete == dup.memory_id else dup.memory_id}"
            ))
            
            resolved_ids.add(to_delete)
        
        return operations


class ForgettingScheduler:
    """
    Schedules and manages forgetting runs.
    """
    
    def __init__(
        self,
        worker: ForgettingWorker,
        interval_hours: float = 24.0
    ):
        self.worker = worker
        self.interval = timedelta(hours=interval_hours)
        self._running = False
        self._task: Optional[asyncio.Task] = None
        self._user_last_run: Dict[str, datetime] = {}
    
    async def start(self):
        """Start the scheduler."""
        self._running = True
        self._task = asyncio.create_task(self._scheduler_loop())
    
    async def stop(self):
        """Stop the scheduler."""
        self._running = False
        if self._task:
            self._task.cancel()
    
    async def schedule_user(
        self,
        tenant_id: str,
        user_id: str,
        force: bool = False
    ):
        """Schedule forgetting for a user."""
        key = f"{tenant_id}:{user_id}"
        now = datetime.utcnow()
        
        last_run = self._user_last_run.get(key)
        
        if force or not last_run or (now - last_run) >= self.interval:
            report = await self.worker.run_forgetting(tenant_id, user_id)
            self._user_last_run[key] = now
            return report
        
        return None
    
    async def _scheduler_loop(self):
        """Background scheduler loop."""
        while self._running:
            # In production, iterate through all users
            # For now, just sleep
            await asyncio.sleep(self.interval.total_seconds())
```

---

## Task 8.5: LLM-Based Compression (Optional)

### Description
Use an LLM to summarize long memory text when compressing (instead of truncation). Supports vLLM with Llama 3.2-1B in Docker for local inference.

### Implementation

- **`src/forgetting/compression.py`**: `summarize_for_compression(text, max_chars, llm_client)` — when `llm_client` is provided and text exceeds `max_chars`, calls LLM for one-sentence gist; otherwise truncates.
- **`src/utils/llm.py`**: `VLLMClient` — OpenAI-compatible client for vLLM (e.g. `http://vllm:8000/v1`). Config: `LLM__PROVIDER=vllm`, `LLM__VLLM_BASE_URL`, `LLM__VLLM_MODEL`.
- **`ForgettingExecutor`**: Accepts `compression_llm_client` and `compression_max_chars`; in `_execute_compress` uses `summarize_for_compression` when client is set.
- **Docker**: Optional service `vllm` (profile `vllm`) in `docker/docker-compose.yml` — image `vllm/vllm-openai`, model e.g. `unslop/Llama-3.2-1B-Instruct`. Run with `docker compose --profile vllm up`.

---

## Task 8.6: Dependency Check Before Delete

### Description
Block soft-delete of a memory when other memories reference it (via `supersedes_id` or `metadata.evidence_refs`).

### Implementation

- **`PostgresMemoryStore.count_references_to(record_id)`**: Returns the number of other records in the same tenant/user that reference this ID (`supersedes_id` or `evidence_refs`). Used before allowing delete.
- **`ForgettingExecutor._execute_delete`**: Calls `count_references_to`; if count > 0, skips delete and appends a clear error to `ForgettingResult.errors` (e.g. "Skipped delete &lt;id&gt;: N dependency(ies)").

---

## Task 8.7: Celery / Background Task

### Description
Run active forgetting as a Celery task so it can be triggered from the API or on a schedule via Celery Beat.

### Implementation

- **`src/celery_app.py`**: Celery app with broker/backend from `DATABASE__REDIS_URL`. Task `run_forgetting_task(tenant_id, user_id, dry_run=False, max_memories=5000)` runs `ForgettingWorker.run_forgetting` via `asyncio.run` and returns a JSON-serializable report dict.
- **Beat schedule**: `forgetting-daily` — runs `run_forgetting_task` every 24 hours (86400 s). In production, call the task with specific tenant/user or iterate over registered users.
- **Queue**: Task routed to queue `forgetting`. Run worker with: `celery -A src.celery_app worker -Q forgetting` and beat with: `celery -A src.celery_app beat`.

---

## Deliverables Checklist

- [x] RelevanceWeights and ScorerConfig models
- [x] RelevanceScorer with multi-factor scoring
- [x] ForgettingAction enum and operation models
- [x] ForgettingPolicyEngine with action thresholds
- [x] ForgettingExecutor for all action types
- [x] InterferenceDetector for duplicates
- [x] ForgettingWorker orchestrating the flow
- [x] ForgettingScheduler for background runs
- [x] ForgettingReport for audit
- [x] Unit tests for scoring
- [x] Unit tests for policy decisions
- [x] Integration tests for full forgetting flow
- [x] LLM-based compression (summarize_for_compression, VLLMClient, vLLM in Docker)
- [x] Dependency check before delete (count_references_to, skip delete with error)
- [x] Celery task and beat schedule for forgetting
