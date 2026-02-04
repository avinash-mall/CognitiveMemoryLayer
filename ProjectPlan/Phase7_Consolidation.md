# Phase 7: Consolidation Engine ("Sleep" Cycle)

## Overview

**Duration**: Week 7-8
**Goal**: Implement offline consolidation that transfers knowledge from episodic to semantic memory, extracts patterns, and compresses episodes.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                    Consolidation Trigger                         │
│   - Scheduled (every N hours)                                    │
│   - Quota-based (episodic store > threshold)                    │
│   - Event-based (task completed, contradiction resolved)        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Episode Sampler                               │
│   - Select recent high-importance episodes                       │
│   - Prioritize by access count, confidence, recency             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Pattern Clusterer                             │
│   - Cluster episodes by semantic similarity                      │
│   - Identify recurring themes/topics                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Gist Extractor                                │
│   - Summarize clusters into semantic facts                       │
│   - Extract generalizable knowledge                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Schema Aligner                                │
│   - Match gists to existing semantic schemas                     │
│   - Rapid integration if schema exists                           │
│   - Create new schema if novel pattern                          │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Migrator                                      │
│   - Write to neocortical (semantic) store                       │
│   - Mark episodes as consolidated                                │
│   - Optionally compress or archive episodes                     │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 7.1: Consolidation Triggers and Scheduler

### Description

Manage when consolidation runs based on various triggers.

### Subtask 7.1.1: Consolidation Trigger System

```python
# src/consolidation/triggers.py
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Callable, Any
from datetime import datetime, timedelta
from enum import Enum
import asyncio

class TriggerType(str, Enum):
    SCHEDULED = "scheduled"           # Time-based
    QUOTA = "quota"                   # Memory size threshold
    EVENT = "event"                   # Specific event occurred
    MANUAL = "manual"                 # Explicitly triggered

@dataclass
class TriggerCondition:
    """Condition that can trigger consolidation."""
    trigger_type: TriggerType
  
    # For SCHEDULED
    interval_hours: Optional[float] = None
  
    # For QUOTA
    min_episodes: Optional[int] = None
    max_memory_mb: Optional[float] = None
  
    # For EVENT
    event_types: List[str] = field(default_factory=list)
  
    # Metadata
    last_triggered: Optional[datetime] = None
    trigger_count: int = 0

@dataclass
class ConsolidationTask:
    """A scheduled consolidation task."""
    tenant_id: str
    user_id: str
    trigger_type: TriggerType
    trigger_reason: str
    priority: int = 0
    created_at: datetime = field(default_factory=datetime.utcnow)
  
    # Scope
    episode_limit: int = 200
    time_window_days: int = 7

class ConsolidationScheduler:
    """
    Manages consolidation scheduling and triggers.
    """
  
    def __init__(
        self,
        default_interval_hours: float = 6.0,
        quota_threshold_episodes: int = 500,
        quota_threshold_mb: float = 100.0
    ):
        self.default_interval = timedelta(hours=default_interval_hours)
        self.quota_episodes = quota_threshold_episodes
        self.quota_mb = quota_threshold_mb
      
        self._conditions: Dict[str, List[TriggerCondition]] = {}
        self._task_queue: asyncio.Queue[ConsolidationTask] = asyncio.Queue()
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
  
    def _user_key(self, tenant_id: str, user_id: str) -> str:
        return f"{tenant_id}:{user_id}"
  
    def register_user(
        self,
        tenant_id: str,
        user_id: str,
        conditions: Optional[List[TriggerCondition]] = None
    ):
        """Register a user with their trigger conditions."""
        key = self._user_key(tenant_id, user_id)
      
        if conditions:
            self._conditions[key] = conditions
        else:
            # Default conditions
            self._conditions[key] = [
                TriggerCondition(
                    trigger_type=TriggerType.SCHEDULED,
                    interval_hours=self.default_interval.total_seconds() / 3600
                ),
                TriggerCondition(
                    trigger_type=TriggerType.QUOTA,
                    min_episodes=self.quota_episodes
                )
            ]
  
    async def check_triggers(
        self,
        tenant_id: str,
        user_id: str,
        episode_count: int,
        memory_size_mb: float,
        event: Optional[str] = None
    ) -> bool:
        """
        Check if any trigger conditions are met.
        Returns True if consolidation should run.
        """
        key = self._user_key(tenant_id, user_id)
        conditions = self._conditions.get(key, [])
      
        now = datetime.utcnow()
        triggered = False
        trigger_reason = ""
      
        for condition in conditions:
            should_trigger = False
          
            if condition.trigger_type == TriggerType.SCHEDULED:
                if condition.interval_hours and condition.last_triggered:
                    elapsed = (now - condition.last_triggered).total_seconds() / 3600
                    should_trigger = elapsed >= condition.interval_hours
                    trigger_reason = f"Scheduled: {elapsed:.1f}h since last run"
                elif condition.interval_hours:
                    should_trigger = True
                    trigger_reason = "Scheduled: first run"
          
            elif condition.trigger_type == TriggerType.QUOTA:
                if condition.min_episodes and episode_count >= condition.min_episodes:
                    should_trigger = True
                    trigger_reason = f"Quota: {episode_count} episodes"
                elif condition.max_memory_mb and memory_size_mb >= condition.max_memory_mb:
                    should_trigger = True
                    trigger_reason = f"Quota: {memory_size_mb:.1f}MB"
          
            elif condition.trigger_type == TriggerType.EVENT:
                if event and event in condition.event_types:
                    should_trigger = True
                    trigger_reason = f"Event: {event}"
          
            if should_trigger:
                condition.last_triggered = now
                condition.trigger_count += 1
                triggered = True
              
                # Queue task
                await self._task_queue.put(ConsolidationTask(
                    tenant_id=tenant_id,
                    user_id=user_id,
                    trigger_type=condition.trigger_type,
                    trigger_reason=trigger_reason
                ))
                break
      
        return triggered
  
    async def trigger_manual(
        self,
        tenant_id: str,
        user_id: str,
        reason: str = "Manual trigger",
        priority: int = 10
    ):
        """Manually trigger consolidation."""
        await self._task_queue.put(ConsolidationTask(
            tenant_id=tenant_id,
            user_id=user_id,
            trigger_type=TriggerType.MANUAL,
            trigger_reason=reason,
            priority=priority
        ))
  
    async def get_next_task(self) -> Optional[ConsolidationTask]:
        """Get next consolidation task from queue."""
        try:
            return await asyncio.wait_for(
                self._task_queue.get(),
                timeout=1.0
            )
        except asyncio.TimeoutError:
            return None
  
    def has_pending_tasks(self) -> bool:
        return not self._task_queue.empty()
```

---

## Task 7.2: Episode Sampling and Clustering

### Description

Select and cluster episodes for consolidation.

### Subtask 7.2.1: Episode Sampler

```python
# src/consolidation/sampler.py
from typing import List, Optional, Dict, Any
from datetime import datetime, timedelta
from dataclasses import dataclass
from ..core.schemas import MemoryRecord
from ..core.enums import MemoryStatus, MemoryType
from ..storage.postgres import PostgresMemoryStore

@dataclass
class SamplingConfig:
    max_episodes: int = 200
    time_window_days: int = 7
    min_importance: float = 0.3
    min_confidence: float = 0.3
  
    # Prioritization weights
    importance_weight: float = 0.4
    access_count_weight: float = 0.3
    recency_weight: float = 0.3

class EpisodeSampler:
    """
    Samples episodes for consolidation.
    Prioritizes by importance, access frequency, and recency.
    """
  
    def __init__(
        self,
        store: PostgresMemoryStore,
        config: Optional[SamplingConfig] = None
    ):
        self.store = store
        self.config = config or SamplingConfig()
  
    async def sample(
        self,
        tenant_id: str,
        user_id: str,
        max_episodes: Optional[int] = None,
        exclude_consolidated: bool = True
    ) -> List[MemoryRecord]:
        """
        Sample episodes for consolidation.
        """
        max_eps = max_episodes or self.config.max_episodes
      
        # Build filters
        filters = {
            "status": MemoryStatus.ACTIVE.value,
            "type": [
                MemoryType.EPISODIC_EVENT.value,
                MemoryType.PREFERENCE.value,
                MemoryType.HYPOTHESIS.value
            ]
        }
      
        if exclude_consolidated:
            filters["not_metadata_key"] = "consolidated"
      
        # Calculate time window
        since = datetime.utcnow() - timedelta(days=self.config.time_window_days)
        filters["since"] = since
      
        # Get candidates
        candidates = await self.store.scan(
            tenant_id, user_id,
            filters=filters,
            limit=max_eps * 3  # Get more than needed for filtering
        )
      
        # Filter by thresholds
        candidates = [
            c for c in candidates
            if c.importance >= self.config.min_importance
            and c.confidence >= self.config.min_confidence
        ]
      
        # Score and rank
        scored = [(self._score(c), c) for c in candidates]
        scored.sort(key=lambda x: x[0], reverse=True)
      
        # Return top N
        return [c for _, c in scored[:max_eps]]
  
    def _score(self, record: MemoryRecord) -> float:
        """Calculate priority score for a record."""
        # Importance component
        importance_score = record.importance
      
        # Access count component (logarithmic)
        import math
        access_score = math.log1p(record.access_count) / 5.0  # Normalize
        access_score = min(access_score, 1.0)
      
        # Recency component
        age_days = (datetime.utcnow() - record.timestamp).days
        recency_score = 1.0 / (1.0 + age_days * 0.1)
      
        # Weighted sum
        return (
            self.config.importance_weight * importance_score +
            self.config.access_count_weight * access_score +
            self.config.recency_weight * recency_score
        )
```

### Subtask 7.2.2: Semantic Clusterer

```python
# src/consolidation/clusterer.py
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
import numpy as np
from ..core.schemas import MemoryRecord

@dataclass
class EpisodeCluster:
    """A cluster of related episodes."""
    cluster_id: int
    episodes: List[MemoryRecord]
    centroid: Optional[List[float]] = None
  
    # Extracted metadata
    common_entities: List[str] = None
    dominant_type: str = None
    avg_confidence: float = 0.0
  
    def __post_init__(self):
        if self.common_entities is None:
            self.common_entities = []

class SemanticClusterer:
    """
    Clusters episodes by semantic similarity.
    Uses embeddings for similarity and agglomerative clustering.
    """
  
    def __init__(
        self,
        min_cluster_size: int = 2,
        max_clusters: int = 20,
        similarity_threshold: float = 0.7
    ):
        self.min_cluster_size = min_cluster_size
        self.max_clusters = max_clusters
        self.similarity_threshold = similarity_threshold
  
    def cluster(
        self,
        episodes: List[MemoryRecord]
    ) -> List[EpisodeCluster]:
        """
        Cluster episodes by semantic similarity.
        """
        if not episodes:
            return []
      
        # Extract embeddings
        embeddings = []
        valid_episodes = []
      
        for ep in episodes:
            if ep.embedding:
                embeddings.append(ep.embedding)
                valid_episodes.append(ep)
      
        if not embeddings:
            # No embeddings - return single cluster
            return [EpisodeCluster(
                cluster_id=0,
                episodes=episodes,
                avg_confidence=sum(e.confidence for e in episodes) / len(episodes)
            )]
      
        embeddings = np.array(embeddings)
      
        # Use agglomerative clustering
        from sklearn.cluster import AgglomerativeClustering
      
        n_clusters = min(
            self.max_clusters,
            max(1, len(valid_episodes) // self.min_cluster_size)
        )
      
        if len(valid_episodes) <= n_clusters:
            # Too few episodes - each is its own cluster
            clusters = []
            for i, ep in enumerate(valid_episodes):
                clusters.append(EpisodeCluster(
                    cluster_id=i,
                    episodes=[ep],
                    centroid=ep.embedding,
                    avg_confidence=ep.confidence
                ))
            return clusters
      
        clustering = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='cosine',
            linkage='average'
        )
      
        labels = clustering.fit_predict(embeddings)
      
        # Group episodes by cluster
        cluster_map: Dict[int, List[MemoryRecord]] = {}
        for label, episode in zip(labels, valid_episodes):
            if label not in cluster_map:
                cluster_map[label] = []
            cluster_map[label].append(episode)
      
        # Build cluster objects
        clusters = []
        for cluster_id, eps in cluster_map.items():
            if len(eps) >= self.min_cluster_size:
                cluster = self._build_cluster(cluster_id, eps, embeddings, labels)
                clusters.append(cluster)
      
        # Handle unclustered episodes (merge into nearest cluster)
        unclustered = [
            ep for label, ep in zip(labels, valid_episodes)
            if len(cluster_map.get(label, [])) < self.min_cluster_size
        ]
      
        if unclustered and clusters:
            # Add to nearest cluster
            for ep in unclustered:
                nearest = self._find_nearest_cluster(ep, clusters)
                if nearest:
                    nearest.episodes.append(ep)
      
        return clusters
  
    def _build_cluster(
        self,
        cluster_id: int,
        episodes: List[MemoryRecord],
        all_embeddings: np.ndarray,
        labels: np.ndarray
    ) -> EpisodeCluster:
        """Build a cluster object with metadata."""
        # Calculate centroid
        cluster_embeddings = [
            all_embeddings[i] for i, label in enumerate(labels)
            if label == cluster_id
        ]
        centroid = np.mean(cluster_embeddings, axis=0).tolist()
      
        # Find common entities
        entity_counts: Dict[str, int] = {}
        for ep in episodes:
            for ent in ep.entities:
                key = ent.normalized if hasattr(ent, 'normalized') else str(ent)
                entity_counts[key] = entity_counts.get(key, 0) + 1
      
        # Entities appearing in > 50% of episodes
        threshold = len(episodes) * 0.5
        common_entities = [
            ent for ent, count in entity_counts.items()
            if count >= threshold
        ]
      
        # Dominant type
        type_counts: Dict[str, int] = {}
        for ep in episodes:
            t = ep.type if isinstance(ep.type, str) else ep.type.value
            type_counts[t] = type_counts.get(t, 0) + 1
      
        dominant_type = max(type_counts, key=type_counts.get) if type_counts else "unknown"
      
        return EpisodeCluster(
            cluster_id=cluster_id,
            episodes=episodes,
            centroid=centroid,
            common_entities=common_entities,
            dominant_type=dominant_type,
            avg_confidence=sum(e.confidence for e in episodes) / len(episodes)
        )
  
    def _find_nearest_cluster(
        self,
        episode: MemoryRecord,
        clusters: List[EpisodeCluster]
    ) -> Optional[EpisodeCluster]:
        """Find nearest cluster for an episode."""
        if not episode.embedding or not clusters:
            return clusters[0] if clusters else None
      
        best_cluster = None
        best_similarity = -1
      
        ep_embedding = np.array(episode.embedding)
      
        for cluster in clusters:
            if cluster.centroid:
                centroid = np.array(cluster.centroid)
                similarity = np.dot(ep_embedding, centroid) / (
                    np.linalg.norm(ep_embedding) * np.linalg.norm(centroid)
                )
              
                if similarity > best_similarity:
                    best_similarity = similarity
                    best_cluster = cluster
      
        return best_cluster
```

---

## Task 7.3: Gist Extraction and Summarization

### Description

Extract semantic gist from episode clusters.

### Subtask 7.3.1: Gist Extractor

```python
# src/consolidation/summarizer.py
from typing import List, Optional, Dict, Any
from dataclasses import dataclass
import json
from .clusterer import EpisodeCluster
from ..core.schemas import MemoryRecord
from ..utils.llm import LLMClient

@dataclass
class ExtractedGist:
    """Extracted semantic gist from a cluster."""
    text: str
    gist_type: str  # "fact", "preference", "pattern", "summary"
    confidence: float
    supporting_episode_ids: List[str]
  
    # Structured extraction
    key: Optional[str] = None       # For fact store key
    subject: Optional[str] = None
    predicate: Optional[str] = None
    value: Optional[Any] = None

GIST_EXTRACTION_PROMPT = """Analyze these related memories and extract the key semantic information.

MEMORIES (from a conversation with a user):
{memories}

COMMON THEMES: {themes}

Extract:
1. The main fact or pattern these memories represent
2. The confidence level (how consistent/certain the info is)
3. The type: "fact" (definite info), "preference" (user likes/dislikes), "pattern" (behavioral tendency), or "summary" (general synopsis)
4. A structured representation if possible (subject, predicate, value)

Return JSON:
{{
  "gist": "User prefers vegetarian food",
  "type": "preference",
  "confidence": 0.9,
  "subject": "user",
  "predicate": "food_preference",
  "value": "vegetarian",
  "key": "user:preference:food"
}}

Rules:
- Combine information across memories to get the core meaning
- Don't include episodic details (times, specific conversations)
- Focus on durable, generalizable information
- Higher confidence if multiple memories support the same conclusion"""

class GistExtractor:
    """
    Extracts semantic gist from episode clusters.
    Converts multiple episodic memories into consolidated facts.
    """
  
    def __init__(self, llm_client: LLMClient):
        self.llm = llm_client
  
    async def extract_gist(
        self,
        cluster: EpisodeCluster
    ) -> List[ExtractedGist]:
        """
        Extract gists from a single cluster.
        May return multiple gists if cluster contains multiple distinct facts.
        """
        if not cluster.episodes:
            return []
      
        # Format memories for prompt
        memory_texts = []
        for i, ep in enumerate(cluster.episodes[:10], 1):  # Limit for prompt size
            memory_texts.append(f"{i}. [{ep.type}] {ep.text}")
      
        memories_str = "\n".join(memory_texts)
        themes_str = ", ".join(cluster.common_entities) if cluster.common_entities else "none identified"
      
        prompt = GIST_EXTRACTION_PROMPT.format(
            memories=memories_str,
            themes=themes_str
        )
      
        try:
            response = await self.llm.complete(prompt, temperature=0.0)
            data = json.loads(response)
          
            # Handle single gist or list
            if isinstance(data, list):
                gists_data = data
            else:
                gists_data = [data]
          
            gists = []
            for gd in gists_data:
                gists.append(ExtractedGist(
                    text=gd.get("gist", ""),
                    gist_type=gd.get("type", "summary"),
                    confidence=float(gd.get("confidence", 0.7)) * cluster.avg_confidence,
                    supporting_episode_ids=[str(ep.id) for ep in cluster.episodes],
                    key=gd.get("key"),
                    subject=gd.get("subject"),
                    predicate=gd.get("predicate"),
                    value=gd.get("value")
                ))
          
            return gists
          
        except (json.JSONDecodeError, KeyError) as e:
            # Fallback: create simple summary gist
            return [ExtractedGist(
                text=self._simple_summary(cluster),
                gist_type="summary",
                confidence=cluster.avg_confidence * 0.5,
                supporting_episode_ids=[str(ep.id) for ep in cluster.episodes]
            )]
  
    async def extract_from_clusters(
        self,
        clusters: List[EpisodeCluster]
    ) -> List[ExtractedGist]:
        """Extract gists from all clusters."""
        import asyncio
      
        all_gists = []
        results = await asyncio.gather(*[
            self.extract_gist(cluster)
            for cluster in clusters
        ])
      
        for gist_list in results:
            all_gists.extend(gist_list)
      
        return all_gists
  
    def _simple_summary(self, cluster: EpisodeCluster) -> str:
        """Create simple summary without LLM."""
        if cluster.common_entities:
            return f"User discussed: {', '.join(cluster.common_entities[:3])}"
      
        # Take first episode as representative
        if cluster.episodes:
            return cluster.episodes[0].text[:100]
      
        return "Cluster summary"
```

---

## Task 7.4: Schema Alignment and Migration

### Description

Align gists with existing schemas and migrate to semantic store.

### Subtask 7.4.1: Schema Aligner

```python
# src/consolidation/schema_aligner.py
from typing import List, Optional, Dict, Any, Tuple
from dataclasses import dataclass
from .summarizer import ExtractedGist
from ..memory.neocortical.schemas import SemanticFact, FactCategory, DEFAULT_FACT_SCHEMAS
from ..memory.neocortical.fact_store import SemanticFactStore

@dataclass
class AlignmentResult:
    """Result of schema alignment."""
    gist: ExtractedGist
  
    # Alignment info
    matched_schema: Optional[str] = None
    schema_similarity: float = 0.0
  
    # For rapid integration
    can_integrate_rapidly: bool = False
    integration_key: Optional[str] = None
  
    # For new schema creation
    suggested_schema: Optional[Dict] = None

class SchemaAligner:
    """
    Aligns extracted gists with existing semantic schemas.
    Determines if gist can be rapidly integrated or needs new schema.
    """
  
    def __init__(
        self,
        fact_store: SemanticFactStore,
        rapid_integration_threshold: float = 0.7
    ):
        self.fact_store = fact_store
        self.threshold = rapid_integration_threshold
  
    async def align(
        self,
        tenant_id: str,
        user_id: str,
        gist: ExtractedGist
    ) -> AlignmentResult:
        """
        Align a gist with existing schemas.
        """
        # Try to match by key if provided
        if gist.key:
            existing = await self.fact_store.get_fact(tenant_id, user_id, gist.key)
          
            if existing:
                return AlignmentResult(
                    gist=gist,
                    matched_schema=gist.key,
                    schema_similarity=0.9,
                    can_integrate_rapidly=True,
                    integration_key=gist.key
                )
      
        # Try to match by gist type
        if gist.gist_type == "preference" and gist.predicate:
            key = f"user:preference:{gist.predicate}"
            existing = await self.fact_store.get_fact(tenant_id, user_id, key)
          
            if existing:
                return AlignmentResult(
                    gist=gist,
                    matched_schema=key,
                    schema_similarity=0.8,
                    can_integrate_rapidly=True,
                    integration_key=key
                )
      
        # Try to find similar facts by text search
        similar_facts = await self.fact_store.search_facts(
            tenant_id, user_id, gist.text, limit=5
        )
      
        if similar_facts:
            # Calculate similarity to best match
            best_match = similar_facts[0]
            similarity = self._calculate_similarity(gist.text, best_match.value)
          
            if similarity >= self.threshold:
                return AlignmentResult(
                    gist=gist,
                    matched_schema=best_match.key,
                    schema_similarity=similarity,
                    can_integrate_rapidly=True,
                    integration_key=best_match.key
                )
      
        # No match - suggest new schema
        suggested = self._suggest_schema(gist)
      
        return AlignmentResult(
            gist=gist,
            schema_similarity=0.0,
            can_integrate_rapidly=False,
            suggested_schema=suggested
        )
  
    async def align_batch(
        self,
        tenant_id: str,
        user_id: str,
        gists: List[ExtractedGist]
    ) -> List[AlignmentResult]:
        """Align multiple gists."""
        import asyncio
        return await asyncio.gather(*[
            self.align(tenant_id, user_id, gist)
            for gist in gists
        ])
  
    def _calculate_similarity(self, text1: str, text2: Any) -> float:
        """Calculate text similarity."""
        text2_str = str(text2)
      
        words1 = set(text1.lower().split())
        words2 = set(text2_str.lower().split())
      
        if not words1 or not words2:
            return 0.0
      
        intersection = len(words1 & words2)
        union = len(words1 | words2)
      
        return intersection / union if union > 0 else 0.0
  
    def _suggest_schema(self, gist: ExtractedGist) -> Dict:
        """Suggest a new schema for the gist."""
        # Infer category
        if gist.gist_type == "preference":
            category = FactCategory.PREFERENCE
        elif gist.gist_type == "fact":
            category = FactCategory.ATTRIBUTE
        else:
            category = FactCategory.CUSTOM
      
        # Generate key
        key = gist.key or f"user:{category.value}:{gist.predicate or 'unknown'}"
      
        return {
            "category": category.value,
            "key": key,
            "value_type": type(gist.value).__name__ if gist.value else "string",
            "source": "consolidation"
        }
```

### Subtask 7.4.2: Consolidation Migrator

```python
# src/consolidation/migrator.py
from typing import List, Optional, Dict, Any
from datetime import datetime
from dataclasses import dataclass
from uuid import UUID
from .summarizer import ExtractedGist
from .schema_aligner import AlignmentResult
from ..memory.neocortical.store import NeocorticalStore
from ..storage.postgres import PostgresMemoryStore
from ..core.enums import MemoryStatus

@dataclass
class MigrationResult:
    """Result of migrating consolidated knowledge."""
    gists_processed: int
    facts_created: int
    facts_updated: int
    episodes_marked: int
    errors: List[str]

class ConsolidationMigrator:
    """
    Migrates consolidated gists to semantic store.
    Marks source episodes as consolidated.
    """
  
    def __init__(
        self,
        neocortical: NeocorticalStore,
        episodic_store: PostgresMemoryStore
    ):
        self.semantic = neocortical
        self.episodic = episodic_store
  
    async def migrate(
        self,
        tenant_id: str,
        user_id: str,
        alignments: List[AlignmentResult],
        mark_episodes_consolidated: bool = True,
        compress_episodes: bool = False
    ) -> MigrationResult:
        """
        Migrate aligned gists to semantic store.
        """
        result = MigrationResult(
            gists_processed=0,
            facts_created=0,
            facts_updated=0,
            episodes_marked=0,
            errors=[]
        )
      
        for alignment in alignments:
            try:
                gist = alignment.gist
              
                if alignment.can_integrate_rapidly and alignment.integration_key:
                    # Update existing fact
                    await self._update_existing_fact(
                        tenant_id, user_id, alignment
                    )
                    result.facts_updated += 1
                else:
                    # Create new fact
                    await self._create_new_fact(
                        tenant_id, user_id, alignment
                    )
                    result.facts_created += 1
              
                result.gists_processed += 1
              
                # Mark source episodes
                if mark_episodes_consolidated:
                    marked = await self._mark_episodes_consolidated(
                        gist.supporting_episode_ids,
                        compress_episodes
                    )
                    result.episodes_marked += marked
                  
            except Exception as e:
                result.errors.append(f"Failed to migrate gist '{alignment.gist.text[:50]}': {e}")
      
        return result
  
    async def _update_existing_fact(
        self,
        tenant_id: str,
        user_id: str,
        alignment: AlignmentResult
    ):
        """Update an existing fact with new evidence."""
        gist = alignment.gist
      
        await self.semantic.store_fact(
            tenant_id=tenant_id,
            user_id=user_id,
            key=alignment.integration_key,
            value=gist.value or gist.text,
            confidence=gist.confidence,
            evidence_ids=gist.supporting_episode_ids
        )
  
    async def _create_new_fact(
        self,
        tenant_id: str,
        user_id: str,
        alignment: AlignmentResult
    ):
        """Create a new fact in semantic store."""
        gist = alignment.gist
        schema = alignment.suggested_schema or {}
      
        key = schema.get("key") or gist.key or f"user:custom:{hash(gist.text) % 10000}"
      
        await self.semantic.store_fact(
            tenant_id=tenant_id,
            user_id=user_id,
            key=key,
            value=gist.value or gist.text,
            confidence=gist.confidence,
            evidence_ids=gist.supporting_episode_ids
        )
  
    async def _mark_episodes_consolidated(
        self,
        episode_ids: List[str],
        compress: bool = False
    ) -> int:
        """Mark episodes as consolidated."""
        marked = 0
      
        for ep_id in episode_ids:
            try:
                ep_uuid = UUID(ep_id)
              
                patch = {
                    "metadata": {"consolidated": True, "consolidated_at": datetime.utcnow().isoformat()}
                }
              
                if compress:
                    patch["status"] = MemoryStatus.COMPRESSED.value
              
                await self.episodic.update(ep_uuid, patch, increment_version=False)
                marked += 1
              
            except (ValueError, Exception):
                continue
      
        return marked
```

---

## Task 7.5: Main Consolidation Worker

### Subtask 7.5.1: Consolidation Worker Service

```python
# src/consolidation/worker.py
from typing import Optional
from datetime import datetime
from dataclasses import dataclass
import asyncio
from .triggers import ConsolidationScheduler, ConsolidationTask
from .sampler import EpisodeSampler
from .clusterer import SemanticClusterer
from .summarizer import GistExtractor
from .schema_aligner import SchemaAligner
from .migrator import ConsolidationMigrator, MigrationResult
from ..storage.postgres import PostgresMemoryStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.llm import LLMClient

@dataclass
class ConsolidationReport:
    """Report from a consolidation run."""
    tenant_id: str
    user_id: str
    started_at: datetime
    completed_at: datetime
  
    # Stats
    episodes_sampled: int
    clusters_formed: int
    gists_extracted: int
    migration: MigrationResult
  
    # Performance
    elapsed_seconds: float
  
    @property
    def success(self) -> bool:
        return len(self.migration.errors) == 0

class ConsolidationWorker:
    """
    Main consolidation worker that orchestrates the full process.
    """
  
    def __init__(
        self,
        episodic_store: PostgresMemoryStore,
        neocortical_store: NeocorticalStore,
        llm_client: LLMClient,
        scheduler: Optional[ConsolidationScheduler] = None
    ):
        self.sampler = EpisodeSampler(episodic_store)
        self.clusterer = SemanticClusterer()
        self.extractor = GistExtractor(llm_client)
        self.aligner = SchemaAligner(neocortical_store.facts)
        self.migrator = ConsolidationMigrator(neocortical_store, episodic_store)
      
        self.scheduler = scheduler or ConsolidationScheduler()
      
        self._running = False
        self._worker_task: Optional[asyncio.Task] = None
  
    async def consolidate(
        self,
        tenant_id: str,
        user_id: str,
        task: Optional[ConsolidationTask] = None
    ) -> ConsolidationReport:
        """
        Run full consolidation for a user.
        """
        started = datetime.utcnow()
      
        # 1. Sample episodes
        episodes = await self.sampler.sample(
            tenant_id, user_id,
            max_episodes=task.episode_limit if task else 200
        )
      
        if not episodes:
            return ConsolidationReport(
                tenant_id=tenant_id,
                user_id=user_id,
                started_at=started,
                completed_at=datetime.utcnow(),
                episodes_sampled=0,
                clusters_formed=0,
                gists_extracted=0,
                migration=MigrationResult(0, 0, 0, 0, []),
                elapsed_seconds=0.0
            )
      
        # 2. Cluster episodes
        clusters = self.clusterer.cluster(episodes)
      
        # 3. Extract gists
        gists = await self.extractor.extract_from_clusters(clusters)
      
        # 4. Align with schemas
        alignments = await self.aligner.align_batch(tenant_id, user_id, gists)
      
        # 5. Migrate to semantic store
        migration = await self.migrator.migrate(
            tenant_id, user_id, alignments,
            mark_episodes_consolidated=True,
            compress_episodes=False
        )
      
        completed = datetime.utcnow()
      
        return ConsolidationReport(
            tenant_id=tenant_id,
            user_id=user_id,
            started_at=started,
            completed_at=completed,
            episodes_sampled=len(episodes),
            clusters_formed=len(clusters),
            gists_extracted=len(gists),
            migration=migration,
            elapsed_seconds=(completed - started).total_seconds()
        )
  
    async def start_background_worker(self):
        """Start background consolidation worker."""
        self._running = True
        self._worker_task = asyncio.create_task(self._worker_loop())
  
    async def stop_background_worker(self):
        """Stop background worker."""
        self._running = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
  
    async def _worker_loop(self):
        """Background worker loop."""
        while self._running:
            task = await self.scheduler.get_next_task()
          
            if task:
                try:
                    report = await self.consolidate(
                        task.tenant_id,
                        task.user_id,
                        task
                    )
                    # Log report
                    print(f"Consolidation complete: {report.gists_extracted} gists extracted")
                except Exception as e:
                    print(f"Consolidation failed: {e}")
            else:
                await asyncio.sleep(1)
```

---

## Deliverables Checklist

**Status:** ✅ Complete

| #  | Deliverable                                        | Status |
| -- | -------------------------------------------------- | ------ |
| 1  | TriggerCondition and ConsolidationTask models      | ✅     |
| 2  | ConsolidationScheduler with multiple trigger types | ✅     |
| 3  | EpisodeSampler with priority scoring               | ✅     |
| 4  | SemanticClusterer using embeddings                 | ✅     |
| 5  | GistExtractor with LLM summarization               | ✅     |
| 6  | SchemaAligner for rapid integration                | ✅     |
| 7  | ConsolidationMigrator for semantic store           | ✅     |
| 8  | ConsolidationWorker orchestrating full flow        | ✅     |
| 9  | ConsolidationReport for audit                      | ✅     |
| 10 | Background worker with task queue                  | ✅     |
| 11 | Unit tests for clustering and triggers             | ✅     |
| 12 | Integration tests for full consolidation           | ✅     |
