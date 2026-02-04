# Phase 5: Retrieval System

## Overview
**Duration**: Week 5-6  
**Goal**: Implement hybrid retrieval with query classification, multi-source search, and memory packet construction.

## Architecture Context

```
┌─────────────────────────────────────────────────────────────────┐
│                      Incoming Query                              │
│              "What's my favorite cuisine?"                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Query Classifier                             │
│   Intent: preference_lookup                                      │
│   Entities: ["cuisine", "favorite"]                              │
│   Time scope: current                                            │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                     Retrieval Planner                            │
│   Plan: [                                                        │
│     { source: "semantic_facts", key: "user:preference:cuisine" } │
│     { source: "vector", query: "favorite cuisine", top_k: 5 }   │
│   ]                                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
            ┌─────────────────┼─────────────────┐
            ▼                 ▼                 ▼
    ┌───────────────┐ ┌───────────────┐ ┌───────────────┐
    │   KV Lookup   │ │ Vector Search │ │ Graph PPR     │
    │   (Fast)      │ │ (Medium)      │ │ (Multi-hop)   │
    └───────────────┘ └───────────────┘ └───────────────┘
            │                 │                 │
            └─────────────────┼─────────────────┘
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                       Reranker                                   │
│   - Deduplicate                                                  │
│   - Score by relevance + recency + confidence                   │
│   - Apply diversity                                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    Memory Packet Builder                         │
│   - Categorize results                                           │
│   - Format for LLM consumption                                   │
│   - Include provenance                                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Task 5.1: Query Classification

### Description
Classify incoming queries to determine optimal retrieval strategy.

### Subtask 5.1.1: Query Intent Types

```python
# src/retrieval/query_types.py
from enum import Enum
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from datetime import datetime

class QueryIntent(str, Enum):
    # Fast path intents (keyed lookup)
    PREFERENCE_LOOKUP = "preference_lookup"      # "What do I like?"
    IDENTITY_LOOKUP = "identity_lookup"          # "What's my name?"
    TASK_STATUS = "task_status"                  # "Where am I in the process?"
    
    # Medium path intents (vector search)
    EPISODIC_RECALL = "episodic_recall"          # "What did I say about X?"
    GENERAL_QUESTION = "general_question"        # "Tell me about X"
    
    # Deep path intents (graph + vector)
    MULTI_HOP = "multi_hop"                      # "How is X related to Y?"
    TEMPORAL_QUERY = "temporal_query"            # "What happened last week?"
    PROCEDURAL = "procedural"                    # "How do I do X?"
    
    # Special
    CONSTRAINT_CHECK = "constraint_check"        # Check against policies
    UNKNOWN = "unknown"

@dataclass
class QueryAnalysis:
    """Analysis result of a query."""
    original_query: str
    intent: QueryIntent
    confidence: float
    
    # Extracted elements
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    
    # Temporal scope
    time_reference: Optional[str] = None  # "recent", "last week", "always"
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    
    # Suggested retrieval params
    suggested_sources: List[str] = field(default_factory=list)
    suggested_top_k: int = 10
    
    # Additional context
    metadata: Dict[str, Any] = field(default_factory=dict)
```

### Subtask 5.1.2: Query Classifier Implementation

```python
# src/retrieval/classifier.py
from typing import Optional, List
import json
import re
from .query_types import QueryIntent, QueryAnalysis
from ..utils.llm import LLMClient

# Patterns for fast classification (no LLM needed)
FAST_PATTERNS = {
    QueryIntent.PREFERENCE_LOOKUP: [
        r"what (do|does) (i|my) (like|prefer|want|enjoy)",
        r"(my|i) (favorite|preferred)",
        r"do i (like|prefer|enjoy)",
    ],
    QueryIntent.IDENTITY_LOOKUP: [
        r"what('s| is) my (name|email|phone|address|job|title)",
        r"who am i",
        r"my (name|email|phone)",
    ],
    QueryIntent.TASK_STATUS: [
        r"where (am i|are we) (in|on|at)",
        r"what('s| is) (the|my) (status|progress)",
        r"(current|next) (step|task)",
    ],
    QueryIntent.TEMPORAL_QUERY: [
        r"(last|past) (week|month|day|year)",
        r"(yesterday|today|recently)",
        r"when did (i|we)",
        r"what happened",
    ],
    QueryIntent.PROCEDURAL: [
        r"how (do|can|should) (i|we)",
        r"what('s| are) the steps",
        r"(procedure|process) for",
    ],
}

CLASSIFICATION_PROMPT = """Classify this query for a memory retrieval system.

Query: {query}

Determine:
1. Intent (one of: preference_lookup, identity_lookup, task_status, episodic_recall, general_question, multi_hop, temporal_query, procedural, constraint_check, unknown)
2. Key entities mentioned
3. Time reference if any (recent, specific date, always, etc.)
4. Confidence (0.0-1.0)

Return JSON:
{{
  "intent": "preference_lookup",
  "entities": ["cuisine", "food"],
  "time_reference": null,
  "confidence": 0.9
}}

Rules:
- preference_lookup: asking about likes/dislikes/preferences
- identity_lookup: asking about personal info (name, email, etc.)
- task_status: asking about current progress on something
- episodic_recall: asking about past conversations/events
- general_question: broad questions about topics
- multi_hop: questions requiring connecting multiple pieces of info
- temporal_query: questions with specific time references
- procedural: how-to questions
- constraint_check: checking rules/policies"""

class QueryClassifier:
    """
    Classifies queries to determine retrieval strategy.
    Uses fast patterns first, falls back to LLM.
    """
    
    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in FAST_PATTERNS.items()
        }
    
    async def classify(self, query: str) -> QueryAnalysis:
        """
        Classify a query and extract relevant information.
        """
        # Try fast pattern matching first
        fast_result = self._fast_classify(query)
        if fast_result and fast_result.confidence > 0.8:
            return fast_result
        
        # Fall back to LLM classification
        if self.llm:
            return await self._llm_classify(query)
        
        # Default if no LLM
        return fast_result or QueryAnalysis(
            original_query=query,
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.5,
            suggested_sources=["vector", "facts"],
            suggested_top_k=10
        )
    
    def _fast_classify(self, query: str) -> Optional[QueryAnalysis]:
        """
        Fast pattern-based classification.
        """
        query_lower = query.lower()
        
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    return QueryAnalysis(
                        original_query=query,
                        intent=intent,
                        confidence=0.85,
                        entities=self._extract_entities_simple(query),
                        suggested_sources=self._get_sources_for_intent(intent),
                        suggested_top_k=self._get_top_k_for_intent(intent)
                    )
        
        return None
    
    async def _llm_classify(self, query: str) -> QueryAnalysis:
        """
        LLM-based classification for complex queries.
        """
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        
        try:
            response = await self.llm.complete(prompt, temperature=0.0)
            data = json.loads(response)
            
            intent = QueryIntent(data.get("intent", "unknown"))
            
            return QueryAnalysis(
                original_query=query,
                intent=intent,
                confidence=float(data.get("confidence", 0.7)),
                entities=data.get("entities", []),
                time_reference=data.get("time_reference"),
                suggested_sources=self._get_sources_for_intent(intent),
                suggested_top_k=self._get_top_k_for_intent(intent)
            )
        except (json.JSONDecodeError, ValueError):
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10
            )
    
    def _extract_entities_simple(self, query: str) -> List[str]:
        """Simple entity extraction using capitalization."""
        words = query.split()
        entities = []
        
        for word in words:
            # Capitalized words (excluding sentence start)
            if word[0].isupper() and len(word) > 1:
                entities.append(word.strip("?.,!"))
        
        return entities
    
    def _get_sources_for_intent(self, intent: QueryIntent) -> List[str]:
        """Map intent to retrieval sources."""
        mapping = {
            QueryIntent.PREFERENCE_LOOKUP: ["facts"],
            QueryIntent.IDENTITY_LOOKUP: ["facts"],
            QueryIntent.TASK_STATUS: ["facts", "vector"],
            QueryIntent.EPISODIC_RECALL: ["vector"],
            QueryIntent.GENERAL_QUESTION: ["vector", "facts"],
            QueryIntent.MULTI_HOP: ["graph", "vector"],
            QueryIntent.TEMPORAL_QUERY: ["vector"],
            QueryIntent.PROCEDURAL: ["facts", "vector"],
            QueryIntent.CONSTRAINT_CHECK: ["facts"],
            QueryIntent.UNKNOWN: ["vector", "facts", "graph"],
        }
        return mapping.get(intent, ["vector"])
    
    def _get_top_k_for_intent(self, intent: QueryIntent) -> int:
        """Map intent to suggested top_k."""
        mapping = {
            QueryIntent.PREFERENCE_LOOKUP: 3,
            QueryIntent.IDENTITY_LOOKUP: 1,
            QueryIntent.TASK_STATUS: 5,
            QueryIntent.EPISODIC_RECALL: 10,
            QueryIntent.GENERAL_QUESTION: 10,
            QueryIntent.MULTI_HOP: 15,
            QueryIntent.TEMPORAL_QUERY: 15,
            QueryIntent.PROCEDURAL: 5,
            QueryIntent.CONSTRAINT_CHECK: 5,
        }
        return mapping.get(intent, 10)
```

---

## Task 5.2: Retrieval Planner

### Description
Generate retrieval plans based on query analysis.

### Subtask 5.2.1: Retrieval Plan Model

```python
# src/retrieval/planner.py
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Any
from enum import Enum
from .query_types import QueryAnalysis, QueryIntent

class RetrievalSource(str, Enum):
    FACTS = "facts"           # Semantic fact store (fast KV)
    VECTOR = "vector"         # Vector similarity search
    GRAPH = "graph"           # Knowledge graph PPR
    LEXICAL = "lexical"       # Full-text search
    CACHE = "cache"           # Hot memory cache

@dataclass
class RetrievalStep:
    """Single step in a retrieval plan."""
    source: RetrievalSource
    priority: int = 0            # Higher = more important
    
    # Source-specific params
    key: Optional[str] = None    # For FACTS
    query: Optional[str] = None  # For VECTOR, LEXICAL
    seeds: List[str] = field(default_factory=list)  # For GRAPH
    
    # Filters
    memory_types: List[str] = field(default_factory=list)
    time_filter: Optional[Dict[str, Any]] = None
    min_confidence: float = 0.0
    
    # Limits
    top_k: int = 10
    timeout_ms: int = 100        # Max time for this step
    
    # Whether to skip if earlier step found results
    skip_if_found: bool = False

@dataclass
class RetrievalPlan:
    """Complete retrieval plan."""
    query: str
    analysis: QueryAnalysis
    steps: List[RetrievalStep]
    
    # Global settings
    total_timeout_ms: int = 500
    max_results: int = 20
    
    # Execution hints
    parallel_steps: List[List[int]] = field(default_factory=list)  # Step indices to run in parallel

class RetrievalPlanner:
    """
    Generates retrieval plans based on query analysis.
    """
    
    def __init__(self):
        pass
    
    def plan(self, analysis: QueryAnalysis) -> RetrievalPlan:
        """
        Generate a retrieval plan for the analyzed query.
        """
        steps = []
        parallel_groups = []
        
        # Fast path: keyed lookup intents
        if analysis.intent in [
            QueryIntent.PREFERENCE_LOOKUP,
            QueryIntent.IDENTITY_LOOKUP,
            QueryIntent.TASK_STATUS
        ]:
            steps.append(self._create_fact_lookup_step(analysis))
            # Also add vector search as fallback
            steps.append(RetrievalStep(
                source=RetrievalSource.VECTOR,
                query=analysis.original_query,
                top_k=5,
                priority=1,
                skip_if_found=True
            ))
            parallel_groups = [[0], [1]]  # Sequential
        
        # Multi-hop: graph first, then vector
        elif analysis.intent == QueryIntent.MULTI_HOP:
            if analysis.entities:
                steps.append(RetrievalStep(
                    source=RetrievalSource.GRAPH,
                    seeds=analysis.entities,
                    top_k=15,
                    priority=2
                ))
            steps.append(RetrievalStep(
                source=RetrievalSource.VECTOR,
                query=analysis.original_query,
                top_k=10,
                priority=1
            ))
            steps.append(RetrievalStep(
                source=RetrievalSource.FACTS,
                query=analysis.original_query,
                top_k=5,
                priority=0
            ))
            parallel_groups = [[0, 1, 2]]  # All in parallel
        
        # Temporal: vector with time filter
        elif analysis.intent == QueryIntent.TEMPORAL_QUERY:
            time_filter = self._build_time_filter(analysis)
            steps.append(RetrievalStep(
                source=RetrievalSource.VECTOR,
                query=analysis.original_query,
                time_filter=time_filter,
                top_k=15,
                priority=2
            ))
            parallel_groups = [[0]]
        
        # General/episodic: hybrid search
        else:
            # Run vector and facts in parallel
            steps.append(RetrievalStep(
                source=RetrievalSource.VECTOR,
                query=analysis.original_query,
                top_k=analysis.suggested_top_k,
                priority=2
            ))
            steps.append(RetrievalStep(
                source=RetrievalSource.FACTS,
                query=analysis.original_query,
                top_k=5,
                priority=1
            ))
            # Add graph if entities found
            if analysis.entities:
                steps.append(RetrievalStep(
                    source=RetrievalSource.GRAPH,
                    seeds=analysis.entities,
                    top_k=10,
                    priority=1
                ))
            parallel_groups = [[0, 1, 2] if len(steps) == 3 else [0, 1]]
        
        return RetrievalPlan(
            query=analysis.original_query,
            analysis=analysis,
            steps=steps,
            parallel_steps=parallel_groups,
            total_timeout_ms=self._calculate_timeout(steps),
            max_results=analysis.suggested_top_k
        )
    
    def _create_fact_lookup_step(self, analysis: QueryAnalysis) -> RetrievalStep:
        """Create a fast fact lookup step."""
        # Infer key from intent
        key_prefix = {
            QueryIntent.PREFERENCE_LOOKUP: "user:preference:",
            QueryIntent.IDENTITY_LOOKUP: "user:identity:",
            QueryIntent.TASK_STATUS: "user:task:",
        }.get(analysis.intent, "user:")
        
        # Try to determine specific key from entities
        key = None
        if analysis.entities:
            key = f"{key_prefix}{analysis.entities[0].lower()}"
        
        return RetrievalStep(
            source=RetrievalSource.FACTS,
            key=key,
            query=analysis.original_query if not key else None,
            top_k=3,
            priority=3,
            timeout_ms=50
        )
    
    def _build_time_filter(self, analysis: QueryAnalysis) -> Optional[Dict]:
        """Build time filter from analysis."""
        from datetime import datetime, timedelta
        
        if not analysis.time_reference:
            return None
        
        now = datetime.utcnow()
        ref = analysis.time_reference.lower()
        
        if "today" in ref:
            return {"since": now.replace(hour=0, minute=0, second=0)}
        elif "yesterday" in ref:
            yesterday = now - timedelta(days=1)
            return {
                "since": yesterday.replace(hour=0, minute=0, second=0),
                "until": yesterday.replace(hour=23, minute=59, second=59)
            }
        elif "week" in ref:
            return {"since": now - timedelta(days=7)}
        elif "month" in ref:
            return {"since": now - timedelta(days=30)}
        elif "recent" in ref:
            return {"since": now - timedelta(days=3)}
        
        return None
    
    def _calculate_timeout(self, steps: List[RetrievalStep]) -> int:
        """Calculate total timeout based on steps."""
        # Assume parallel execution within groups
        return sum(s.timeout_ms for s in steps) // 2 + 100
```

---

## Task 5.3: Hybrid Retriever

### Description
Execute retrieval plans across multiple sources.

### Subtask 5.3.1: Retriever Implementation

```python
# src/retrieval/retriever.py
from typing import List, Dict, Any, Optional
from dataclasses import dataclass
import asyncio
from datetime import datetime
from .planner import RetrievalPlan, RetrievalStep, RetrievalSource
from .query_types import QueryAnalysis
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..core.schemas import MemoryRecord, MemoryPacket, RetrievedMemory
from ..core.enums import MemoryType

@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""
    source: RetrievalSource
    items: List[Dict[str, Any]]
    elapsed_ms: float
    success: bool
    error: Optional[str] = None

class HybridRetriever:
    """
    Executes retrieval plans across multiple memory sources.
    """
    
    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        cache: Optional[Any] = None  # Redis cache
    ):
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.cache = cache
    
    async def retrieve(
        self,
        tenant_id: str,
        user_id: str,
        plan: RetrievalPlan
    ) -> List[RetrievedMemory]:
        """
        Execute a retrieval plan and return results.
        """
        all_results = []
        
        # Execute parallel groups
        for group_indices in plan.parallel_steps:
            group_steps = [plan.steps[i] for i in group_indices if i < len(plan.steps)]
            
            # Run steps in parallel within group
            group_results = await asyncio.gather(*[
                self._execute_step(tenant_id, user_id, step)
                for step in group_steps
            ], return_exceptions=True)
            
            # Process results
            for step, result in zip(group_steps, group_results):
                if isinstance(result, Exception):
                    continue
                
                if result.success and result.items:
                    all_results.extend(result.items)
                    
                    # Check skip_if_found
                    if step.skip_if_found and result.items:
                        break
        
        # Convert to RetrievedMemory objects
        retrieved = self._to_retrieved_memories(all_results, plan.analysis)
        
        return retrieved
    
    async def _execute_step(
        self,
        tenant_id: str,
        user_id: str,
        step: RetrievalStep
    ) -> RetrievalResult:
        """Execute a single retrieval step."""
        start = datetime.utcnow()
        
        try:
            if step.source == RetrievalSource.FACTS:
                items = await self._retrieve_facts(tenant_id, user_id, step)
            elif step.source == RetrievalSource.VECTOR:
                items = await self._retrieve_vector(tenant_id, user_id, step)
            elif step.source == RetrievalSource.GRAPH:
                items = await self._retrieve_graph(tenant_id, user_id, step)
            elif step.source == RetrievalSource.CACHE:
                items = await self._retrieve_cache(tenant_id, user_id, step)
            else:
                items = []
            
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            
            return RetrievalResult(
                source=step.source,
                items=items,
                elapsed_ms=elapsed,
                success=True
            )
        
        except Exception as e:
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=elapsed,
                success=False,
                error=str(e)
            )
    
    async def _retrieve_facts(
        self,
        tenant_id: str,
        user_id: str,
        step: RetrievalStep
    ) -> List[Dict]:
        """Retrieve from semantic fact store."""
        results = []
        
        if step.key:
            # Direct key lookup
            fact = await self.neocortical.get_fact(tenant_id, user_id, step.key)
            if fact:
                results.append({
                    "type": "fact",
                    "source": "facts",
                    "key": fact.key,
                    "text": f"{fact.predicate}: {fact.value}",
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "relevance": 1.0,
                    "record": fact
                })
        
        if step.query:
            # Text search
            facts = await self.neocortical.text_search(
                tenant_id, user_id, step.query, limit=step.top_k
            )
            for f in facts:
                results.append({
                    "type": "fact",
                    "source": "facts",
                    "key": f.get("key"),
                    "text": f"{f.get('key', '')}: {f.get('value', '')}",
                    "value": f.get("value"),
                    "confidence": f.get("confidence", 0.5),
                    "relevance": 0.8,
                    "record": f
                })
        
        return results[:step.top_k]
    
    async def _retrieve_vector(
        self,
        tenant_id: str,
        user_id: str,
        step: RetrievalStep
    ) -> List[Dict]:
        """Retrieve via vector similarity search."""
        filters = {}
        
        if step.time_filter:
            filters.update(step.time_filter)
        if step.memory_types:
            filters["type"] = step.memory_types
        if step.min_confidence > 0:
            filters["min_confidence"] = step.min_confidence
        
        records = await self.hippocampal.search(
            tenant_id, user_id,
            query=step.query,
            top_k=step.top_k,
            filters=filters if filters else None
        )
        
        return [
            {
                "type": "episode",
                "source": "vector",
                "text": r.text,
                "confidence": r.confidence,
                "relevance": r.metadata.get("_similarity", 0.5),
                "timestamp": r.timestamp,
                "record": r
            }
            for r in records
        ]
    
    async def _retrieve_graph(
        self,
        tenant_id: str,
        user_id: str,
        step: RetrievalStep
    ) -> List[Dict]:
        """Retrieve via knowledge graph PPR."""
        if not step.seeds:
            return []
        
        results = await self.neocortical.multi_hop_query(
            tenant_id, user_id,
            seed_entities=step.seeds,
            max_hops=3
        )
        
        return [
            {
                "type": "graph",
                "source": "graph",
                "entity": r.get("entity"),
                "text": self._format_entity_info(r),
                "relevance": r.get("relevance_score", 0.5),
                "record": r
            }
            for r in results
        ]
    
    async def _retrieve_cache(
        self,
        tenant_id: str,
        user_id: str,
        step: RetrievalStep
    ) -> List[Dict]:
        """Retrieve from hot cache."""
        if not self.cache:
            return []
        
        # Implement cache lookup
        cache_key = f"hot:{tenant_id}:{user_id}"
        cached = await self.cache.get(cache_key)
        
        if cached:
            import json
            return json.loads(cached)
        
        return []
    
    def _to_retrieved_memories(
        self,
        items: List[Dict],
        analysis: QueryAnalysis
    ) -> List[RetrievedMemory]:
        """Convert raw items to RetrievedMemory objects."""
        seen_texts = set()
        retrieved = []
        
        for item in items:
            # Deduplicate by text
            text = item.get("text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)
            
            # Create MemoryRecord if we have one
            record = item.get("record")
            if isinstance(record, MemoryRecord):
                mem_record = record
            else:
                # Create minimal record from dict
                mem_record = self._dict_to_record(item)
            
            retrieved.append(RetrievedMemory(
                record=mem_record,
                relevance_score=item.get("relevance", 0.5),
                retrieval_source=item.get("source", "unknown")
            ))
        
        # Sort by relevance
        retrieved.sort(key=lambda x: x.relevance_score, reverse=True)
        
        return retrieved
    
    def _format_entity_info(self, entity_data: Dict) -> str:
        """Format entity data as readable text."""
        lines = [f"Entity: {entity_data.get('entity', 'Unknown')}"]
        
        for rel in entity_data.get("relations", [])[:5]:
            lines.append(f"  - {rel.get('predicate', '')}: {rel.get('related_entity', '')}")
        
        return "\n".join(lines)
    
    def _dict_to_record(self, item: Dict) -> MemoryRecord:
        """Create minimal MemoryRecord from dict."""
        from uuid import uuid4
        from ..core.schemas import Provenance
        from ..core.enums import MemorySource
        
        return MemoryRecord(
            id=uuid4(),
            tenant_id="",
            user_id="",
            type=MemoryType(item.get("type", "episodic_event")) if item.get("type") in [t.value for t in MemoryType] else MemoryType.EPISODIC_EVENT,
            text=item.get("text", ""),
            confidence=item.get("confidence", 0.5),
            importance=0.5,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=item.get("timestamp", datetime.utcnow()),
            written_at=datetime.utcnow()
        )
```

---

## Task 5.4: Reranker

### Description
Rerank and diversify retrieval results.

### Subtask 5.4.1: Reranker Implementation

```python
# src/retrieval/reranker.py
from typing import List, Optional
from dataclasses import dataclass
from ..core.schemas import RetrievedMemory

@dataclass
class RerankerConfig:
    # Weights for scoring
    relevance_weight: float = 0.5
    recency_weight: float = 0.2
    confidence_weight: float = 0.2
    diversity_weight: float = 0.1
    
    # Diversity settings
    diversity_threshold: float = 0.8  # Similarity threshold for diversity penalty
    
    # Limits
    max_results: int = 20

class MemoryReranker:
    """
    Reranks retrieved memories based on multiple factors.
    """
    
    def __init__(self, config: Optional[RerankerConfig] = None):
        self.config = config or RerankerConfig()
    
    def rerank(
        self,
        memories: List[RetrievedMemory],
        query: str,
        max_results: Optional[int] = None
    ) -> List[RetrievedMemory]:
        """
        Rerank memories by combined score.
        """
        if not memories:
            return []
        
        max_results = max_results or self.config.max_results
        
        # Calculate scores
        scored = []
        for mem in memories:
            score = self._calculate_score(mem, memories)
            scored.append((score, mem))
        
        # Sort by score
        scored.sort(key=lambda x: x[0], reverse=True)
        
        # Apply diversity filter
        diverse = self._apply_diversity(scored, max_results)
        
        return [mem for _, mem in diverse]
    
    def _calculate_score(
        self,
        memory: RetrievedMemory,
        all_memories: List[RetrievedMemory]
    ) -> float:
        """Calculate combined score for a memory."""
        from datetime import datetime
        
        # Relevance score (from retrieval)
        relevance = memory.relevance_score
        
        # Recency score (exponential decay)
        age_days = (datetime.utcnow() - memory.record.timestamp).days
        recency = 1.0 / (1.0 + age_days * 0.1)
        
        # Confidence score
        confidence = memory.record.confidence
        
        # Diversity score (penalize if too similar to higher-ranked items)
        diversity = 1.0  # Will be adjusted later
        
        # Combine
        score = (
            self.config.relevance_weight * relevance +
            self.config.recency_weight * recency +
            self.config.confidence_weight * confidence +
            self.config.diversity_weight * diversity
        )
        
        return score
    
    def _apply_diversity(
        self,
        scored: List[tuple],
        max_results: int
    ) -> List[tuple]:
        """
        Apply diversity filter using MMR-style selection.
        """
        if len(scored) <= max_results:
            return scored
        
        selected = []
        candidates = list(scored)
        
        while len(selected) < max_results and candidates:
            if not selected:
                # First item: highest score
                selected.append(candidates.pop(0))
            else:
                # Find item that maximizes score while being diverse
                best_idx = 0
                best_mmr = -float('inf')
                
                for i, (score, mem) in enumerate(candidates):
                    # Calculate max similarity to already selected
                    max_sim = max(
                        self._text_similarity(mem.record.text, s[1].record.text)
                        for s in selected
                    )
                    
                    # MMR score
                    mmr = score - self.config.diversity_threshold * max_sim
                    
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                
                selected.append(candidates.pop(best_idx))
        
        return selected
    
    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        
        return intersection / union if union > 0 else 0.0
```

---

## Task 5.5: Memory Packet Builder

### Description
Build structured memory packets for LLM consumption.

### Subtask 5.5.1: Packet Builder Implementation

```python
# src/retrieval/packet_builder.py
from typing import List, Optional, Dict
from datetime import datetime
from ..core.schemas import MemoryPacket, RetrievedMemory
from ..core.enums import MemoryType

class MemoryPacketBuilder:
    """
    Builds structured memory packets from retrieved memories.
    """
    
    def __init__(self):
        pass
    
    def build(
        self,
        memories: List[RetrievedMemory],
        query: str,
        include_provenance: bool = True
    ) -> MemoryPacket:
        """
        Build a memory packet from retrieved memories.
        """
        packet = MemoryPacket(query=query)
        
        for mem in memories:
            retrieved = mem
            mem_type = mem.record.type
            
            # Categorize by type
            if mem_type == MemoryType.SEMANTIC_FACT.value:
                packet.facts.append(retrieved)
            elif mem_type == MemoryType.PREFERENCE.value:
                packet.preferences.append(retrieved)
            elif mem_type == MemoryType.PROCEDURE.value:
                packet.procedures.append(retrieved)
            elif mem_type == MemoryType.CONSTRAINT.value:
                packet.constraints.append(retrieved)
            else:
                # Episodic and others
                packet.recent_episodes.append(retrieved)
        
        # Check for conflicts
        conflicts = self._detect_conflicts(packet)
        if conflicts:
            packet.warnings.extend(conflicts)
        
        # Add open questions for low-confidence items
        for mem in memories:
            if mem.record.confidence < 0.5:
                packet.open_questions.append(
                    f"Uncertain: {mem.record.text} (confidence: {mem.record.confidence:.2f})"
                )
        
        return packet
    
    def _detect_conflicts(self, packet: MemoryPacket) -> List[str]:
        """Detect potential conflicts in retrieved memories."""
        conflicts = []
        
        # Check for contradicting preferences
        preference_values = {}
        for pref in packet.preferences:
            key = pref.record.key
            if key in preference_values:
                if preference_values[key] != pref.record.text:
                    conflicts.append(
                        f"Conflicting preferences for {key}: "
                        f"'{preference_values[key]}' vs '{pref.record.text}'"
                    )
            else:
                preference_values[key] = pref.record.text
        
        # Check for contradicting facts
        fact_values = {}
        for fact in packet.facts:
            key = fact.record.key
            if key and key in fact_values:
                if fact_values[key] != fact.record.text:
                    conflicts.append(
                        f"Conflicting facts for {key}"
                    )
            elif key:
                fact_values[key] = fact.record.text
        
        return conflicts
    
    def to_llm_context(
        self,
        packet: MemoryPacket,
        max_tokens: int = 2000,
        format: str = "markdown"
    ) -> str:
        """
        Format packet for LLM context injection.
        """
        if format == "markdown":
            return self._format_markdown(packet, max_tokens)
        elif format == "json":
            return self._format_json(packet, max_tokens)
        else:
            return packet.to_context_string(max_tokens)
    
    def _format_markdown(self, packet: MemoryPacket, max_tokens: int) -> str:
        """Format as markdown."""
        lines = ["# Retrieved Memory Context\n"]
        
        if packet.constraints:
            lines.append("## Constraints (Must Follow)")
            for c in packet.constraints[:3]:
                lines.append(f"- **{c.record.text}**")
            lines.append("")
        
        if packet.facts:
            lines.append("## Known Facts")
            for f in packet.facts[:5]:
                conf = f"[{f.record.confidence:.0%}]" if f.record.confidence < 1.0 else ""
                lines.append(f"- {f.record.text} {conf}")
            lines.append("")
        
        if packet.preferences:
            lines.append("## User Preferences")
            for p in packet.preferences[:5]:
                lines.append(f"- {p.record.text}")
            lines.append("")
        
        if packet.recent_episodes:
            lines.append("## Recent Context")
            for e in packet.recent_episodes[:5]:
                timestamp = e.record.timestamp.strftime("%Y-%m-%d")
                lines.append(f"- [{timestamp}] {e.record.text}")
            lines.append("")
        
        if packet.warnings:
            lines.append("## Warnings")
            for w in packet.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")
        
        result = "\n".join(lines)
        
        # Truncate if too long (rough estimate: 4 chars per token)
        max_chars = max_tokens * 4
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        
        return result
    
    def _format_json(self, packet: MemoryPacket, max_tokens: int) -> str:
        """Format as JSON string."""
        import json
        
        data = {
            "facts": [
                {"text": f.record.text, "confidence": f.record.confidence}
                for f in packet.facts[:5]
            ],
            "preferences": [
                {"text": p.record.text}
                for p in packet.preferences[:5]
            ],
            "recent": [
                {"text": e.record.text, "date": e.record.timestamp.isoformat()}
                for e in packet.recent_episodes[:5]
            ],
            "constraints": [c.record.text for c in packet.constraints],
            "warnings": packet.warnings
        }
        
        return json.dumps(data, indent=2)
```

---

## Task 5.6: Main Retriever Facade

### Subtask 5.6.1: Unified Retrieval Interface

```python
# src/retrieval/memory_retriever.py
from typing import Optional, List
from .classifier import QueryClassifier
from .planner import RetrievalPlanner
from .retriever import HybridRetriever
from .reranker import MemoryReranker, RerankerConfig
from .packet_builder import MemoryPacketBuilder
from ..core.schemas import MemoryPacket, RetrievedMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.llm import LLMClient

class MemoryRetriever:
    """
    Main entry point for memory retrieval.
    Coordinates classification, planning, retrieval, and formatting.
    """
    
    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        llm_client: Optional[LLMClient] = None,
        cache: Optional[object] = None
    ):
        self.classifier = QueryClassifier(llm_client)
        self.planner = RetrievalPlanner()
        self.retriever = HybridRetriever(hippocampal, neocortical, cache)
        self.reranker = MemoryReranker()
        self.packet_builder = MemoryPacketBuilder()
    
    async def retrieve(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        max_results: int = 20,
        return_packet: bool = True
    ) -> MemoryPacket:
        """
        Retrieve relevant memories for a query.
        
        Args:
            tenant_id: Tenant identifier
            user_id: User identifier
            query: The query string
            max_results: Maximum memories to return
            return_packet: If True, return structured packet
        
        Returns:
            MemoryPacket with categorized memories
        """
        # 1. Classify query
        analysis = await self.classifier.classify(query)
        
        # 2. Generate retrieval plan
        plan = self.planner.plan(analysis)
        
        # 3. Execute retrieval
        raw_results = await self.retriever.retrieve(
            tenant_id, user_id, plan
        )
        
        # 4. Rerank results
        reranked = self.reranker.rerank(
            raw_results, query, max_results=max_results
        )
        
        # 5. Build packet
        if return_packet:
            return self.packet_builder.build(reranked, query)
        
        # Return raw list
        return MemoryPacket(
            query=query,
            recent_episodes=reranked
        )
    
    async def retrieve_for_llm(
        self,
        tenant_id: str,
        user_id: str,
        query: str,
        max_tokens: int = 2000,
        format: str = "markdown"
    ) -> str:
        """
        Retrieve and format memories for LLM context.
        """
        packet = await self.retrieve(tenant_id, user_id, query)
        return self.packet_builder.to_llm_context(packet, max_tokens, format)
```

---

## Deliverables Checklist

- [ ] QueryIntent enum and QueryAnalysis model
- [ ] QueryClassifier with fast patterns and LLM fallback
- [ ] RetrievalPlan and RetrievalStep models
- [ ] RetrievalPlanner generating plans from analysis
- [ ] HybridRetriever executing plans
- [ ] Individual source retrievers (facts, vector, graph)
- [ ] MemoryReranker with MMR diversity
- [ ] MemoryPacketBuilder with multiple formats
- [ ] MemoryRetriever facade
- [ ] Unit tests for classification
- [ ] Unit tests for planning
- [ ] Integration tests for full retrieval flow
