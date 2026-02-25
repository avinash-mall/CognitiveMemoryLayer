# CML Codebase Audit Report

## 1. Architecture Map

### 1.1 Data Flow Diagram

```
WRITE PATH:
  Client → POST /api/v1/memory/write (routes.py:54)
    → MemoryOrchestrator.write() (orchestrator.py:200)
      → ShortTermMemory.ingest()
        → SensoryBuffer.encode() (sensory/buffer.py)
        → WorkingMemoryManager.process() (working/manager.py)
          → SemchunkChunker.chunk() (working/chunker.py)
      → For each SemanticChunk:
        → WriteGate.evaluate() (hippocampal/write_gate.py:75)
        → PIIRedactor.redact() (hippocampal/redactor.py)
        → [if LLM enabled] UnifiedWritePathExtractor.extract() → 1 LLM call
          → entities, relations, constraints, facts, salience, importance, memory_type
        → [else] rule-based ConstraintExtractor.extract() (constraint_extractor.py)
        → EmbeddingClient.embed_batch() → 1 embedding call per batch
        → PostgresMemoryStore.upsert() → pgvector INSERT/UPDATE
        → [if write_time_facts] NeocorticalStore.store_facts() → Postgres semantic_facts
        → [if entities] Neo4jGraphStore.store_entities/relations() → Neo4j

READ PATH:
  Client → POST /api/v1/memory/read (routes.py:146)
    → MemoryOrchestrator.read() (orchestrator.py:303)
      → EmbeddingClient.embed() → 1 embedding call for query
      → QueryClassifier.classify() (classifier.py:130)
        → _fast_classify() regex patterns, OR 1 LLM call
      → RetrievalPlanner.plan() (planner.py:58)
        → Creates RetrievalPlan with parallel step groups
      → HybridRetriever.retrieve() (retriever.py:49)
        → For each step group (parallel):
          → FACTS: SemanticFactStore.get_facts_by_category() → Postgres
          → VECTOR: PostgresMemoryStore.vector_search() → pgvector cosine
          → GRAPH: Neo4jGraphStore.personalized_pagerank() → Neo4j Cypher
          → CONSTRAINTS: vector_search(type=constraint) + fact_lookup(cognitive)
        → [if associative_expansion] scan for co-session constraints
      → MemoryReranker.rerank() (reranker.py:34)
        → Score: 0.5*relevance + 0.2*confidence + 0.1*diversity + recency_weight*recency
        → [if LLM reranker] _score_constraints_batch() → 1 LLM call
        → MMR-style diversity selection
      → MemoryPacketBuilder.build() (packet_builder.py:18)
        → Sorts into: facts, preferences, procedures, constraints, episodes
        → to_llm_context(): constraints-first markdown with reserved budget

CONSOLIDATION (background):
  ConsolidationWorker.run() (consolidation/worker.py)
    → EpisodeSampler.sample() — 7d episodes, 90d constraints
    → SemanticClusterer.cluster() — embedding-based grouping
    → GistSummarizer.summarize() — 1 LLM call per cluster
    → SchemaAligner.align() — maps gist type to FactCategory
    → ConsolidationMigrator.migrate() — upserts into semantic_facts

FORGETTING (background):
  ForgettingWorker.run() (forgetting/worker.py)
    → RelevanceScorer.score_all() — per-record composite score
    → PolicyEngine → Keep (>0.7) / Decay (>0.4) / Silence (>0.2) / Compress (>0.1) / Delete (<0.1)
    → [Compress] LLMCompression → 1 LLM call per compressed memory
```

### 1.2 Key Modules → Responsibilities

| Module | Responsibility | Key Files |
|--------|---------------|-----------|
| `src/api/` | HTTP endpoints, auth, rate limiting | `routes.py`, `auth.py`, `schemas.py` |
| `src/memory/sensory/` | Raw input tokenisation | `buffer.py` |
| `src/memory/working/` | Semantic chunking, working memory | `manager.py`, `chunker.py` |
| `src/memory/hippocampal/` | Episodic encoding, write gate, PII | `store.py`, `write_gate.py`, `redactor.py` |
| `src/memory/neocortical/` | Semantic facts, schema management | `store.py`, `fact_store.py`, `schemas.py` |
| `src/extraction/` | Entity/relation/constraint/fact extraction | `unified_write_extractor.py`, `constraint_extractor.py` |
| `src/retrieval/` | Query classification, planning, retrieval, reranking | `classifier.py`, `planner.py`, `retriever.py`, `reranker.py` |
| `src/consolidation/` | Episode-to-fact migration | `sampler.py`, `clusterer.py`, `summarizer.py`, `migrator.py` |
| `src/forgetting/` | Active forgetting with relevance scoring | `scorer.py`, `actions.py`, `compression.py` |
| `src/reconsolidation/` | Belief revision, conflict detection | `conflict_detector.py`, `belief_revision.py`, `labile_tracker.py` |
| `src/storage/` | Postgres+pgvector, Neo4j, Redis | `postgres.py`, `neo4j.py`, `redis.py` |

---

## 2. Issue Register

### 2.1 Conceptual / Architectural Issues

#### C-01: Semantic Disconnect Failure — Constraints Retrieved Only by Trigger-Query Similarity (CRITICAL)

**Evidence:** `src/retrieval/retriever.py:381–407` — `_retrieve_constraints()` performs vector search using the query embedding against constraint records. If the user asks "Recommend a restaurant" (trigger), the vector similarity to "I'm allergic to shellfish" (cue constraint) will be low because the embedding spaces of "recommend restaurant" and "allergic to shellfish" are distant.

**How it fails:** The constraint vector search at line 388 uses `self.hippocampal.search(tenant_id, query=step.query, ...)` with cosine similarity. The query "recommend a restaurant" will rank "I like Italian food" highly but "I'm allergic to shellfish" low, because vector similarity does not capture the *logical relevance* of a safety constraint to a recommendation action.

**Mitigation that partially exists:** Lines 411–451 do a secondary fact lookup across all cognitive categories (GOAL, STATE, VALUE, CAUSAL, POLICY), returning *all* active facts regardless of query similarity. However, this returns **all** constraints for the tenant with a flat `relevance: 0.75`, causing constraint dilution (see C-02).

**Fix plan:**
1. Add a constraint-type-aware retrieval layer that indexes constraints by *domain scope* (e.g., "food", "travel", "health") rather than only by embedding similarity.
2. At write time, extract scope tags from constraints (already partially done — `ConstraintObject.scope` exists at `constraint_extractor.py:33` but is always empty `[]` in rule-based extraction).
3. At read time, classify the query domain and retrieve constraints matching that domain, regardless of embedding distance.

**Test plan:** Write a constraint "I'm allergic to shellfish", then query "Recommend a restaurant". Assert the allergy constraint appears in retrieved constraints. Currently this works only by the flat fact-lookup path, not by vector search.

---

#### C-02: Constraint Dilution — All Active Constraints Returned Regardless of Relevance (HIGH)

**Evidence:** `src/retrieval/retriever.py:421–428` — when `step.constraint_categories` is empty (which is the default for non-CONSTRAINT_CHECK intents per `planner.py:162–165`), the code iterates over ALL five cognitive categories and returns ALL active facts:

```python
cognitive_categories = [
    FactCategory.GOAL, FactCategory.STATE, FactCategory.VALUE,
    FactCategory.CAUSAL, FactCategory.POLICY,
]
```

Each fact gets a flat `relevance: 0.75` (line 447). If a user has 20 active constraints, all 20 are retrieved with equal priority. The reranker (`reranker.py:47–54`) boosts constraints by `boost * 2.0`, but the boost is computed by `_score_constraints_batch()` which — when LLM reranking is disabled (default) — falls back to word-overlap Jaccard similarity (`reranker.py:156–164`). Word-overlap between "Recommend a restaurant" and "I'm allergic to shellfish" is low.

**Impact:** In the `packet_builder.py:139`, constraints are capped at 6 (`packet.constraints[:6]`) and formatted with `[!IMPORTANT]` markers. If the 6 slots are filled with irrelevant constraints, the relevant one gets excluded.

**Fix plan:**
1. Score constraints by *domain relevance* to the query using scope-tag matching before the reranker.
2. In `_retrieve_constraints()`, filter cognitive facts by whether their `scope` or `key` intersects with the query's detected entities/topics.
3. Reduce the flat `relevance: 0.75` to `0.5` as a base, and boost based on scope match.

---

#### C-03: Wrong Constraint Type Retrieved — No Type-Aware Prioritisation (MEDIUM)

**Evidence:** The retrieval planner at `planner.py:121–155` creates a CONSTRAINTS step for `CONSTRAINT_CHECK` intent with `constraint_categories=analysis.constraint_dimensions`. However, `QueryAnalysis.constraint_dimensions` is only populated by the LLM classifier (`classifier.py:240–245`), and the fast-path regex classifier never sets it. With `FEATURES__USE_LLM_ENABLED=false` (default), `constraint_dimensions` is always `None`.

This means: when the user asks "Should I eat sushi?" (a policy/value question), and the system uses fast-path classification, the constraint retrieval gets `constraint_categories=None` and falls through to retrieving ALL categories. A `STATE` constraint ("I'm stressed about work") gets returned alongside the relevant `POLICY` constraint ("I never eat raw fish").

**Fix plan:** Extend `_fast_classify()` to heuristically detect constraint dimensions from query keywords (e.g., "should" → policy, "is it aligned with my goals" → goal).

---

#### C-04: Temporal/Recency Bias Mistakes — Recency Formula Penalises Old Active Constraints (MEDIUM)

**Evidence:** `src/retrieval/reranker.py:98` — `recency = 1.0 / (1.0 + age_days * 0.1)`. A 30-day-old constraint gets `recency = 1/(1+3.0) = 0.25`. The recency weight for generic constraints is 0.10 (line 81), so the penalty is `0.10 * 0.25 = 0.025` vs a 1-day-old episodic event getting `0.10 * 0.91 = 0.091`.

The mitigation at lines 62–82 (`_get_recency_weight()`) correctly reduces recency weight for stable constraints (value/policy → 0.0), but only when the constraint metadata contains a `constraint_type` field. Constraints extracted by the **rule-based** extractor at `constraint_extractor.py:148–165` DO populate `ConstraintObject.constraint_type`, and these are stored in `MemoryRecord.metadata["constraints"]`. So this mitigation works **when the constraint was correctly extracted and typed at write time**. If the constraint was written as a plain `episodic_event` or `preference` (e.g., LLM off and no pattern match), it gets full recency penalty.

**Fix plan:** Add a fallback: if a memory's text contains constraint-signal words (already defined in `summarizer.py:52–73`), reduce recency weight even if no structured constraint metadata exists.

---

#### C-05: Stale Constraint Application — Supersession Logic Only Active at Write Time (HIGH)

**Evidence:** `src/extraction/constraint_extractor.py:140–164` — `detect_supersession()` checks if a new constraint supersedes an existing one by matching `constraint_type` + overlapping `scope`. However:

1. Supersession is only checked **at write time** (`hippocampal/store.py:170–200`), not during consolidation or retrieval.
2. Scope tags are always empty lists for rule-based extraction (`constraint_extractor.py:143` — `c.scope` defaults to `[]`), so scope-based matching never triggers.
3. If a user says "I'm vegetarian" and later "Actually, I eat fish now", the reconsolidation path (`conflict_detector.py`) may detect a contradiction, but only if the new statement is processed within the 5-minute labile window (`labile_tracker.py`).

**Impact:** Old constraints persist as `status=active` even after being contradicted, because:
- Rule-based constraint extraction doesn't populate `scope`, so `detect_supersession()` scope intersection check (`constraint_extractor.py:157`) always returns `False`.
- The labile window is too short (5 minutes) to catch delayed corrections.

**Fix plan:**
1. Populate `scope` during rule-based extraction by extracting topic keywords from the chunk.
2. During consolidation, run supersession detection on clustered constraints.
3. At retrieval time, detect conflicting constraints in the packet and surface the most recent.

---

#### C-06: Inconsistent Consolidation — Gist Extraction Can Lose Constraint Semantics (HIGH)

**Evidence:** `src/consolidation/summarizer.py:10–48` — the `GIST_EXTRACTION_PROMPT` instructs the LLM to "Combine information across memories to get the core meaning" and "Don't include episodic details". This can cause:

1. A cluster of memories like ["I'm allergic to shellfish", "I went to a seafood restaurant last week and had to be careful", "My doctor confirmed the shellfish allergy"] gets consolidated into a gist "User has shellfish allergy" — losing the *constraint framing* ("I can't eat shellfish" vs "I have an allergy"). The gist is stored as `FactCategory.PREFERENCE` instead of `POLICY`.

2. The constraint signal words check at `summarizer.py:52–73` (`_CONSTRAINT_SIGNAL_WORDS`) only checks the gist *text*, not the source memories. If the LLM paraphrases "I never eat shellfish" to "User avoids shellfish", the word "never" is lost, and the constraint may be re-categorised as a preference.

**Fix plan:**
1. When the source memories are all `MemoryType.CONSTRAINT`, force the gist type to remain a constraint type.
2. Add a post-gist validation: if any source memory contains constraint-signal words, ensure the gist preserves the constraint category.

---

### 2.2 Efficiency Issues

#### E-01: O(n²) Text Similarity in Reranker Diversity Scoring (MEDIUM)

**Evidence:** `src/retrieval/reranker.py:101–117` — `_calculate_score()` computes diversity by iterating over `all_memories[:20]` for each memory, computing word-overlap similarity. For 20 memories this is O(20²) = 400 comparisons, each involving set operations. Then the MMR diversity pass at lines 129–153 does another O(n*selected) loop with the same text similarity.

The `_text_similarity()` method (lines 156–164) uses Jaccard word overlap, which is O(|words1| + |words2|) per call. With average text of 50 words, each comparison is ~100 operations, giving ~40,000 operations for 20 memories in the scoring phase alone.

**Fix plan:** Pre-compute word sets once per memory, pass them as a lookup dict rather than recomputing on each call.

---

#### E-02: Redundant Embedding Call for Query (LOW)

**Evidence:** `src/memory/orchestrator.py:310–340` — `read()` calls `self.hippocampal.search()` which internally computes the query embedding. But the orchestrator also passes the embedding to the retriever. The embedding is cached in `CachedEmbeddings` (`utils/embeddings.py`) if Redis is available, but without Redis, the same query may be embedded multiple times across retrieval steps.

Specifically, `_retrieve_constraints()` at `retriever.py:388` calls `self.hippocampal.search(query=step.query, query_embedding=query_embedding)` — passing the pre-computed embedding. But `_retrieve_vector()` also calls `self.hippocampal.search()` with the same embedding. The hippocampal search re-uses the passed embedding, so no duplicate call occurs **when the embedding is passed**. The risk is when `query_embedding` is `None` (not pre-computed), causing each step to embed independently.

**Fix plan:** Ensure the orchestrator always pre-computes and passes the query embedding to the retriever (already mostly done, but verify all code paths).

---

#### E-03: Unbounded Constraint Fact Scan (MEDIUM)

**Evidence:** `src/retrieval/retriever.py:431–449` — for each of 5 cognitive categories, `get_facts_by_category()` issues a separate SQL query. With many tenants and many constraints, this is 5 sequential DB round-trips per read request.

**Fix plan:** Add a single `get_facts_by_categories(tenant_id, categories, current_only=True)` method that uses `WHERE category IN (...)` in one query.

---

#### E-04: LLM Call Per Cluster During Consolidation (LOW)

**Evidence:** `src/consolidation/summarizer.py:90–130` — `GistSummarizer.summarize()` makes one LLM call per episode cluster. If a tenant has 50 clusters, this is 50 LLM calls. The batch method (`summarize_batch`) at lines 132+ processes all clusters, but it's called sequentially.

**Fix plan:** Batch multiple cluster summaries into a single LLM call (similar to the batch extraction in `unified_write_extractor.py:194`).

---

### 2.3 Logical Issues

#### L-01: _build_api_keys LRU Cache Pollution Between Tests (FIXED)

**Evidence:** `src/api/auth.py:29` — `@lru_cache` on `_build_api_keys()`. Previously, test `test_invalid_api_key_returns_401` set `AUTH__API_KEY="valid-key"` and triggered `_build_api_keys()`, caching the key. Subsequent tests that sent a different API key received 401. **Fixed in this PR** by adding `_build_api_keys.cache_clear()` to the autouse fixture.

---

#### L-02: Race Condition in Labile State Tracker (MEDIUM)

**Evidence:** `src/reconsolidation/labile_tracker.py` — the labile state is stored in Redis with a TTL. If two concurrent write operations both trigger reconsolidation for the same memory, they could both read the labile state, both attempt belief revision, and produce conflicting updates. There is no locking mechanism.

**Fix plan:** Use Redis `SETNX` or Lua scripting for atomic labile state transitions.

---

#### L-03: Silent Failure in Neo4j Graph Store (MEDIUM)

**Evidence:** `src/storage/neo4j.py` — all graph operations are wrapped in try/except that silently swallow failures. If Neo4j is down, all graph-based retrieval (PPR, entity search) silently returns empty results. The dashboard routes also silently return empty data without warning (`src/api/dashboard/graph_routes.py`).

**Impact:** Users may not realise that their knowledge graph is not being populated or queried, leading to degraded retrieval quality without any visible error.

**Fix plan:** Log at WARNING level when Neo4j operations fail. Add a `graph_available` field to the health endpoint response.

---

#### L-04: Constraint Extractor Scope Always Empty (HIGH)

**Evidence:** `src/extraction/constraint_extractor.py:126–134` — `extract()` creates `ConstraintObject` with `scope=[]` always. The `detect_supersession()` method at line 157 checks `if set(c1.scope) & set(c2.scope)`, which is always `set() & set()` = `False` for rule-based extraction. This means supersession never triggers.

```python
# Line 126-134
return ConstraintObject(
    constraint_type=best_type,
    subject="user",
    description=chunk.text,
    scope=[],           # <-- Always empty
    ...
)
```

**Fix plan:** Extract scope from the chunk text using entity extraction or keyword matching. For example, "I never eat shellfish" → `scope=["food", "diet"]`.

---

#### L-05: Associative Expansion O(n) Session Scans (LOW)

**Evidence:** `src/retrieval/retriever.py:120–163` — the associative expansion loop iterates over all results to collect `source_session_id` values, then for each session, issues a `scan()` call with `type=constraint, source_session_id=sid`. With many sessions, this is O(sessions) DB queries.

**Fix plan:** Batch session IDs into a single query using `source_session_id IN (...)`.

---

#### L-06: WriteGate min_importance Threshold Silently Drops Constraints (MEDIUM)

**Evidence:** `src/memory/hippocampal/write_gate.py:92–140` — `_compute_importance()` assigns importance based on chunk type and keyword matching. A casual statement like "I prefer tea" gets `importance = 0.4`. The threshold `min_importance = 0.3` lets it through. But a statement like "I take medication at 9am" might only match `_STATE_PATTERNS` with a low confidence boost, yielding importance below 0.3 and being silently dropped.

However, at line 110, the code adds a constraint boost: `if chunk.chunk_type == ChunkType.CONSTRAINT: importance += 0.2`. And at line 136–138, constraint chunks get `min(importance + 0.2, 1.0)`. So constraints get a double-boost. The risk is for constraints that the chunker classifies as a non-constraint `ChunkType` — they'd miss the boost.

**Fix plan:** In the write gate, if `unified_result.constraints` is non-empty (from the LLM extractor), apply the constraint boost regardless of `ChunkType`.

---

### 2.4 Documentation Issues

#### D-01: README Version Badge Stale (FIXED)

**Evidence:** `README.md:15` showed version 1.3.4, but `VERSION` file contains 1.3.6. **Fixed in this PR.**

#### D-02: tenant_flags.py Docstring References Non-Existent Function (FIXED)

**Evidence:** `src/core/tenant_flags.py:10` referenced `get_tenant_features()` but only `get_tenant_overrides()` exists. **Fixed in this PR.**

---

## 3. Failure-Mode Analysis

### 3.1 Semantic Disconnect Failure

**Where it occurs:** `retriever.py:388` — vector search for constraints uses query embedding similarity.

**Example:** User stores "I'm deathly allergic to peanuts." Query: "What should I order at the Thai restaurant?" Vector cosine similarity between these is low (~0.15–0.25 depending on model).

**Current mitigation:** The flat fact-lookup path (`retriever.py:430–449`) retrieves ALL active facts across all cognitive categories, giving the peanut allergy constraint a fighting chance. But with a flat `relevance: 0.75`, it competes equally with unrelated constraints like "I want to save money" or "I value punctuality".

**Proposed fix:** Add a **query-domain classifier** that maps the query to one or more domains (food, health, finance, etc.), then filters constraints by matching domain scope. This requires populating `ConstraintObject.scope` at write time (currently always empty — see L-04).

**Proposed test:**
```python
async def test_semantic_disconnect_retrieval():
    """Constraint retrieved even when query is semantically distant."""
    await orchestrator.write(tenant_id, "I'm deathly allergic to peanuts.")
    result = await orchestrator.read(tenant_id, "What should I order at the Thai restaurant?")
    constraint_texts = [c.record.text for c in result.constraints]
    assert any("peanut" in t.lower() or "allergic" in t.lower() for t in constraint_texts)
```

### 3.2 Constraint Dilution

**Where it occurs:** `retriever.py:421–449` returns all cognitive facts; `packet_builder.py:139` caps at 6.

**Example:** User has 10 active constraints. Query touches only 1. All 10 are retrieved with `relevance: 0.75`. The reranker's word-overlap scoring is too weak to distinguish. The relevant constraint may be at position 7+ and get truncated.

**Proposed fix:** In `_retrieve_constraints()`, after collecting facts, apply a lightweight relevance filter — e.g., check if any of the query's entities or keywords appear in the constraint text. Only return constraints with entity-overlap or scope-match above a threshold.

**Proposed test:**
```python
async def test_constraint_dilution():
    """Relevant constraint not buried under irrelevant ones."""
    for topic in ["finance", "health", "travel", "work", "social", "diet"]:
        await orchestrator.write(tenant_id, f"My {topic} policy is to be careful.")
    await orchestrator.write(tenant_id, "I never eat shellfish due to allergy.")
    result = await orchestrator.read(tenant_id, "Should I try the lobster?")
    # The shellfish constraint should be in top 3
    top_3_texts = [c.record.text for c in result.constraints[:3]]
    assert any("shellfish" in t.lower() for t in top_3_texts)
```

### 3.3 Wrong Constraint Type

**Where it occurs:** `planner.py:162–165` — `constraint_categories` is `None` when fast-path classification is used (no LLM), so all categories are retrieved.

**Proposed fix:** Add keyword-based constraint dimension detection in `_fast_classify()`:
- "should I" / "can I" / "is it ok" → POLICY
- "aligned with my goals" → GOAL
- "consistent with my values" → VALUE
- "given my situation" → STATE

**Proposed test:** Classify "Should I eat this?" and verify `constraint_dimensions` includes `"policy"`.

### 3.4 Temporal/Recency Bias

**Where it occurs:** `reranker.py:98` — age-based decay.

**Example:** A 90-day-old constraint "I never eat pork" (still active) gets `recency = 1/(1+9) = 0.1`. A 1-day-old episodic event "I had pizza yesterday" gets `recency = 0.91`. With recency weight 0.1, the constraint loses 0.081 score points relative to the episode. This can push the constraint below the top-K threshold.

**Current mitigation:** `_get_recency_weight()` (lines 62–82) returns 0.0 for stable constraints (value/policy), effectively disabling recency penalty. This works when constraint metadata is correctly populated.

**Residual risk:** Constraints written without structured metadata (e.g., ingested as plain text without constraint extraction) don't benefit from this mitigation.

**Proposed fix:** Add a text-based fallback: if `memory.record.text` contains constraint-signal words and `memory.record.type` is not `CONSTRAINT`, reduce recency weight.

### 3.5 Stale Constraint Application

**Where it occurs:** `constraint_extractor.py:140–164` — `detect_supersession()` relies on scope intersection, which is always empty.

**Example:** User says "I'm vegetarian." Later: "Actually I eat fish now." The new statement creates a new constraint but doesn't supersede the old one because `scope=[]` for both. Both "I'm vegetarian" and "I eat fish now" remain active simultaneously.

**Proposed fix:**
1. Populate scope at extraction time.
2. During consolidation, run pairwise conflict detection on active constraints of the same type.
3. Add a `superseded_by` field to `MemoryRecord` so old constraints can be explicitly marked.

**Proposed test:**
```python
async def test_constraint_supersession():
    """Newer constraint supersedes older one of same type/scope."""
    await orchestrator.write(tenant_id, "I'm vegetarian.")
    await orchestrator.write(tenant_id, "Actually I eat fish now.")
    result = await orchestrator.read(tenant_id, "What should I eat?")
    texts = [c.record.text for c in result.constraints]
    assert any("fish" in t.lower() for t in texts)
    assert not any("vegetarian" in t.lower() for t in texts)
```

### 3.6 Inconsistent Consolidation

**Where it occurs:** `summarizer.py:10–48` — gist extraction prompt.

**Example:** Memories ["I never eat gluten", "Went to a bakery and couldn't eat anything", "Doctor confirmed celiac"] → gist "User has celiac disease" (type: "fact"). The constraint framing ("never eat gluten") is lost. The gist is stored as `FactCategory.ATTRIBUTE` instead of `POLICY`.

**Proposed fix:**
1. In `SchemaAligner` (`schema_aligner.py`), if any source memory is `MemoryType.CONSTRAINT`, force the alignment to a cognitive category.
2. In `GistSummarizer`, include the source memory types in the prompt context so the LLM preserves constraint framing.

**Proposed test:**
```python
async def test_consolidation_preserves_constraint_type():
    """Consolidation gist preserves constraint category."""
    # Store multiple constraint memories
    for _ in range(3):
        await orchestrator.write(tenant_id, "I never eat gluten because I have celiac disease.")
    # Run consolidation
    await consolidation_worker.run(tenant_id)
    # Check that resulting facts include POLICY category
    facts = await fact_store.get_facts_by_category(tenant_id, FactCategory.POLICY)
    assert any("gluten" in f.value.lower() for f in facts)
```

---

## 4. Prioritised Roadmap

### Quick Wins (1–2 days each)

| ID | Fix | Impact |
|----|-----|--------|
| L-04 | Populate `ConstraintObject.scope` from entity extraction at write time | Enables supersession detection |
| C-03 | Add constraint dimension heuristics to `_fast_classify()` | Better constraint type filtering without LLM |
| E-01 | Pre-compute word sets in reranker to avoid O(n²) recomputation | ~4x speedup on reranking |
| E-03 | Batch cognitive fact queries into single SQL call | 5 DB round-trips → 1 |
| C-04 | Text-based fallback for recency weight reduction | Protects untyped constraints from recency bias |

### Medium (3–5 days each)

| ID | Fix | Impact |
|----|-----|--------|
| C-02 | Add query-domain matching to constraint filtering | Reduces dilution; relevant constraints surface |
| C-05 | Run supersession during consolidation, not just write time | Catches delayed corrections |
| C-06 | Preserve constraint type through consolidation gist | Prevents gist from losing constraint semantics |
| L-02 | Add Redis locking to labile state tracker | Prevents concurrent reconsolidation race conditions |

### Larger Refactors (1–2 weeks)

| ID | Fix | Impact |
|----|-----|--------|
| C-01 | Domain-scoped constraint index + query-domain classifier | Addresses root cause of semantic disconnect |
| NEW | Constraint-first context assembly with evidence provenance | Improves judge-consistent behavior |
| NEW | Structured constraint KB separate from episodic vector store | Clean separation of stable constraints from episodic noise |

---

## 5. Patch Plan

### PR #1: Constraint Extraction Scope Population (Quick Win)
- `src/extraction/constraint_extractor.py`: Extract topic keywords into `scope` field using entity overlap with chunk text.
- `tests/unit/test_constraint_layer.py`: Add test for non-empty scope in extracted constraints.
- Commit 1: Add `_extract_scope()` helper using keyword/entity extraction.
- Commit 2: Update `extract()` to call `_extract_scope()`.
- Commit 3: Add tests.

### PR #2: Fast-Path Constraint Dimension Detection
- `src/retrieval/classifier.py`: Add keyword rules for `constraint_dimensions` in `_fast_classify()`.
- `src/retrieval/planner.py`: Use dimensions to filter constraint categories.
- Commit 1: Add dimension detection patterns.
- Commit 2: Add tests for dimension detection.

### PR #3: Reranker Efficiency + Recency Bias Fix
- `src/retrieval/reranker.py`: Pre-compute word sets; add text-based recency fallback.
- Commit 1: Optimise `_text_similarity()` with pre-computed sets.
- Commit 2: Add constraint-signal-word fallback for `_get_recency_weight()`.
- Commit 3: Add tests.

### PR #4: Batch Cognitive Fact Query
- `src/memory/neocortical/fact_store.py`: Add `get_facts_by_categories()` method.
- `src/retrieval/retriever.py`: Use batch method in `_retrieve_constraints()`.
- Commit 1: Add batch query method.
- Commit 2: Wire into retriever.
- Commit 3: Add integration test.

### PR #5: Constraint Dilution Reduction
- `src/retrieval/retriever.py`: Add entity/keyword overlap filter before returning constraints.
- `src/retrieval/packet_builder.py`: Sort constraints by domain-relevance score before capping at 6.
- Commit 1: Add `_filter_relevant_constraints()`.
- Commit 2: Update packet builder sorting.
- Commit 3: Add end-to-end test for semantic disconnect scenario.

### PR #6: Consolidation Constraint Preservation
- `src/consolidation/summarizer.py`: Add source memory types to gist extraction prompt.
- `src/consolidation/schema_aligner.py`: Force constraint category when source memories are constraints.
- Commit 1: Update prompt template.
- Commit 2: Add alignment override logic.
- Commit 3: Add test for constraint preservation through consolidation.

---

## 6. New/Updated Tests

| Test | File | What it covers |
|------|------|----------------|
| `test_semantic_disconnect_retrieval` | `tests/integration/test_retrieval_flow.py` | Constraint retrieved when query is semantically distant |
| `test_constraint_dilution` | `tests/integration/test_retrieval_flow.py` | Relevant constraint surfaces when many are active |
| `test_constraint_supersession` | `tests/integration/test_hippocampal_encode_flow.py` | Newer constraint supersedes older one |
| `test_consolidation_preserves_constraint_type` | `tests/integration/test_consolidation_flow.py` | Gist preserves constraint category |
| `test_constraint_scope_populated` | `tests/unit/test_constraint_layer.py` | Rule-based extraction produces non-empty scope |
| `test_fast_classify_constraint_dimensions` | `tests/unit/test_retrieval_classifier_planner_reranker.py` | Fast classifier detects constraint dimensions |
| `test_reranker_no_recency_penalty_for_old_constraints` | `tests/unit/test_retrieval_classifier_planner_reranker.py` | Old but active constraints not penalised |
| `test_batch_cognitive_fact_query` | `tests/integration/test_fact_store_integration.py` | Batch query returns same results as N individual queries |

---

## 7. Tradeoffs

| Decision | Tradeoff | Justification |
|----------|----------|---------------|
| Scope extraction at write time | Slightly slower writes (~1ms) | Enables domain-aware retrieval without LLM at read time |
| Constraint type filtering | May miss cross-domain constraints | Reduces dilution; can be tuned with broader scope tags |
| Batch SQL queries | Slightly more complex SQL | 5x fewer DB round-trips on the hot read path |
| Preserve constraint type through consolidation | May reduce generalisation of gists | Prevents safety-critical constraints from being diluted into preferences |
| Not changing: embedding model | Model change is high-risk | Current model (nomic-embed-text-v2-moe) is reasonable; improvements come from indexing/retrieval logic, not embeddings |
| Not changing: storage schema | Schema migration is disruptive | All proposed changes work within existing MemoryRecord and SemanticFact schemas |
| Not changing: LLM prompts (mostly) | Prompt engineering is brittle | Focus on algorithmic improvements that are testable and measurable |

---

## 8. Quality Bar

Every proposed change must satisfy:

1. **Measurable:** Each fix has a corresponding test that fails before the fix and passes after.
2. **Surgical:** Changes are localized to specific functions/methods with clear evidence trails.
3. **Constraint-consistent:** Improvements are evaluated against constraint retrieval under semantic disconnect, not just classic RAG recall.
4. **Backward-compatible:** No schema migrations, no API contract changes, no config key removals.
5. **Evidence-driven:** Every claim cites specific file paths, function names, and line numbers from the actual codebase.

---

## 9. Context Assembly for Judge-Consistent Behaviour

The current context assembly (`packet_builder.py:105–180`) uses a **constraints-first markdown format** with reserved token budget, `[!IMPORTANT]` markers, and confidence annotations. This is well-designed for judge-consistent behavior. Key improvements:

1. **Present constraints as natural reminders:** Instead of `- [!IMPORTANT] **I'm allergic to shellfish**`, use `- Earlier you mentioned: "I'm allergic to shellfish" — this is an active health constraint that should inform recommendations.`

2. **Add provenance context:** The `_constraint_provenance()` method (line 86–103) extracts type and source turn ID, but could include the original statement timestamp to help the LLM judge temporal validity.

3. **Separate "must-follow" from "consider" constraints:** Value/policy constraints should be `## Must Follow`, while state/goal constraints should be `## Consider`. This maps to the cognitive categories already defined in `FactCategory`.
