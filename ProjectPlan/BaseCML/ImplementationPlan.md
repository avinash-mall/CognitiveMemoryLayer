# CognitiveMemoryLayer — Deep Research Implementation Plan

> **Implementation status (Feb 2025):** All 6 phases are **implemented**. Feature flags and retrieval settings are in `src/core/config.py`; see [UsageDocumentation.md § Configuration Reference](../UsageDocumentation.md#configuration-reference) and [.env.example](../../.env.example). Unit tests: `tests/unit/test_deep_research_improvements.py` (38 tests). Summary: [CHANGELOG.md](../../CHANGELOG.md) — "Deep-research implementation (BaseCML plan)".
>
> **Cognitive Constraint Layer (Feb 2025):** A follow-on implementation based on the [deep-research-report](../../evaluation/deep-research-report.md) adds **Level-2 Cognitive Memory** for LoCoMo-Plus. Constraint extraction (`ConstraintExtractor`), constraint-aware retrieval (`CONSTRAINT_CHECK` intent, `RetrievalSource.CONSTRAINTS`), cognitive `FactCategory` values (GOAL, STATE, VALUE, CAUSAL, POLICY), supersession, and 90-day consolidation window. Feature flag: `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` (default true). See [CHANGELOG.md](../../CHANGELOG.md) — "Cognitive Constraint Layer".

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Deep Analysis: Design & Correctness Issues](#2-deep-analysis-design--correctness-issues)
   - 2.1 [Semantic-Store vs Episodic-Store Mismatch](#21-semantic-store-vs-episodic-store-mismatch)
   - 2.2 [Key Identity & Deduplication are Unsafe](#22-key-identity--deduplication-are-unsafe)
   - 2.3 [Tenant vs Session Scoping Inconsistency](#23-tenant-vs-session-scoping-inconsistency)
   - 2.4 [Temporal Query Semantics Under-specified](#24-temporal-query-semantics-under-specified)
   - 2.5 [Retrieval Timeouts Exist on Paper Only](#25-retrieval-timeouts-exist-on-paper-only)
   - 2.6 [Concurrency Safety & Multi-Instance Correctness](#26-concurrency-safety--multi-instance-correctness)
3. [Deep Analysis: Efficiency & Scalability Issues](#3-deep-analysis-efficiency--scalability-issues)
   - 3.1 [End-to-End Turn Cost is Too High](#31-end-to-end-turn-cost-is-too-high)
   - 3.2 [Embedding Calls Not Batched](#32-embedding-calls-not-batched)
   - 3.3 [Sensory Buffer Tokenisation is Expensive](#33-sensory-buffer-tokenisation-is-expensive)
   - 3.4 [Graph Multi-hop N+1 Query Pattern](#34-graph-multi-hop-n1-query-pattern)
   - 3.5 [Neo4j GDS PageRank is Expensive Per Query](#35-neo4j-gds-pagerank-is-expensive-per-query)
   - 3.6 [O(n²) Dependency Counting in Forgetting](#36-on²-dependency-counting-in-forgetting)
   - 3.7 [pgvector HNSW Parameters Not Tuned](#37-pgvector-hnsw-parameters-not-tuned)
4. [Architecture Changes](#4-architecture-changes)
   - 4.1 [Current vs Proposed Architecture](#41-current-vs-proposed-architecture)
   - 4.2 [New Component: WriteTimeFactExtractor](#42-new-component-writetimefactextractor)
   - 4.3 [Modified Component: AsyncStoragePipeline](#43-modified-component-asyncstoragepipeline)
   - 4.4 [Modified Component: TimeoutAwareRetriever](#44-modified-component-timeoutawareretriever)
   - 4.5 [New Component: BoundedStateManager](#45-new-component-boundedstatemanager)
5. [Implementation Plan](#5-implementation-plan)
   - Phase 1: [Critical Correctness Fixes (P0)](#phase-1-critical-correctness-fixes-p0)
   - Phase 2: [Hot-Path Performance (P0)](#phase-2-hot-path-performance-p0)
   - Phase 3: [Retrieval Reliability (P1)](#phase-3-retrieval-reliability-p1)
   - Phase 4: [Background Work Scalability (P1)](#phase-4-background-work-scalability-p1)
   - Phase 5: [Operational Stability (P2)](#phase-5-operational-stability-p2)
   - Phase 6: [Observability & Tuning (P2)](#phase-6-observability--tuning-p2)
6. [Testing Strategy](#6-testing-strategy)
7. [Migration & Rollback Plan](#7-migration--rollback-plan)
8. [Acceptance Criteria & SLOs](#8-acceptance-criteria--slos)

---

## 1. Executive Summary

The deep research report identifies **13 concrete issues** across design correctness and performance, grouped into 4 themes:

| Theme | Issue Count | Severity | Estimated Effort |
|-------|-------------|----------|-----------------|
| Semantic pipeline misalignment | 2 | Critical | 2-3 weeks |
| Key identity & deduplication | 2 | Critical | 1-2 weeks |
| Hot-path inefficiencies | 5 | High | 2-3 weeks |
| Operational boundaries | 4 | Medium-High | 2-3 weeks |

This plan addresses each issue with a deep benefit/risk analysis, then provides a phased implementation plan with architecture diagrams, pseudo-code, test strategies, and migration paths. Total estimated implementation time: **8-10 weeks** across 6 phases.

### Priority Matrix

```
Impact ▲
       │
  HIGH │  [P0-1] Stable Keys    [P0-3] Batch Embeddings
       │  [P0-2] Write-Time     [P0-4] Async Store
       │         Facts
       │
  MED  │  [P1-1] Timeouts       [P1-3] O(n²) Forgetting
       │  [P1-2] Skip-If-Found  [P1-4] Embedding Cache
       │
  LOW  │  [P2-1] Sensory Tokens [P2-3] HNSW Tuning
       │  [P2-2] Bounded State  [P2-4] Session Scoping
       │
       └────────────────────────────────────────► Risk
            LOW          MEDIUM          HIGH
```

---

## 2. Deep Analysis: Design & Correctness Issues

### 2.1 Semantic-Store vs Episodic-Store Mismatch

**Current Behaviour:**
The retrieval planner (`src/retrieval/planner.py`) prioritises the semantic fact store for intents like `PREFERENCE_LOOKUP` and `IDENTITY_LOOKUP`, constructing key prefixes like `user:preference:` and `user:identity:`. However, the main write path (`MemoryOrchestrator.write` → `ShortTermMemory.ingest_turn` → `HippocampalStore.encode_batch`) stores chunks exclusively into the episodic store (`memory_records` table) and does **not** populate `semantic_facts`.

The only path that writes to `semantic_facts` is the consolidation pipeline (`ConsolidationMigrator` → `NeocorticalStore.store_fact`), which runs as a background job.

**Affected Files:**
- `src/retrieval/planner.py` — Plans fact-first retrieval for preference/identity
- `src/retrieval/retriever.py` — Executes fact lookup, gets empty results
- `src/memory/hippocampal/store.py` — Writes only to episodic store
- `src/memory/neocortical/fact_store.py` — Empty until consolidation runs
- `src/memory/neocortical/schemas.py` — Defines key patterns like `user:preference:cuisine`

**Root Cause:**
The write path and read path were designed with different assumptions about where facts live. The retrieval planner assumes facts are immediately available in the semantic store, but the write path treats consolidation as the exclusive migration mechanism.

**Benefits of Fixing:**
1. **Immediate fact availability** — Preference/identity queries return structured facts from the first conversation turn, not after consolidation delay
2. **Lower read latency** — Fact-store lookups are O(1) by key vs O(log n) vector search; eliminating vector fallback saves 50-200ms per retrieval
3. **Higher retrieval quality** — Structured facts (`user:preference:cuisine = "vegetarian"`) are more precise than raw episode fragments (`"I mentioned I don't eat meat"`)
4. **Better LoCoMo benchmark scores** — LoCoMo explicitly tests profile-fact queries; fast structured retrieval directly improves accuracy
5. **Reduced embedding costs** — Fewer vector search fallbacks means fewer embedding API calls for query vectors

**Risks of Fixing:**
1. **Dual-write complexity** — Writing to both episodic and semantic stores on the hot path introduces consistency concerns if one write fails
2. **LLM dependency on write path** — Extracting structured facts at write-time may require an LLM call, increasing write latency if done synchronously
3. **Schema rigidity** — Only facts matching `DEFAULT_FACT_SCHEMAS` patterns can be extracted; novel fact types will still require consolidation
4. **Over-eager extraction** — Rule-based extraction may create false-positive facts (e.g., "I used to live in Paris" → `user:location:current_city = "Paris"` is wrong)
5. **Migration of existing data** — Tenants with existing episodic-only data need a backfill job

**Risk Mitigation:**
- Use rule-based extraction first (no LLM on hot path); LLM extraction only in background consolidation
- Treat write-time facts as `confidence=0.6` (lower than consolidation-derived facts at `confidence=0.8`)
- Fire semantic fact write as a fire-and-forget async task; episodic write is the source of truth
- Add a feature flag `WRITE_TIME_FACTS_ENABLED` for gradual rollout

---

### 2.2 Key Identity & Deduplication are Unsafe

**Current Behaviour:**
Two separate key-generation bugs create data corruption:

**Bug A — Hippocampal `_generate_key()` (store.py:231-239):**
```python
def _generate_key(self, chunk: SemanticChunk, memory_type: MemoryType) -> str | None:
    if memory_type not in (MemoryType.PREFERENCE, MemoryType.SEMANTIC_FACT):
        return None
    if chunk.entities:
        return f"{memory_type.value}:{chunk.entities[0].lower()}"
    return None
```
This uses only the **first entity** as the key. If a user says "I love Italian food" and later "I love Italian music", both map to key `preference:italian`, and the second **silently overwrites** the first because `PostgresMemoryStore.upsert()` updates in-place when `(tenant_id, key)` matches.

**Bug B — Consolidation `_create_new_fact()` (migrator.py:104):**
```python
key = schema.get("key") or gist.key or f"user:custom:{hash(gist.text) % 10000}"
```
Python's `hash()` is salted per-process by default (`PYTHONHASHSEED`), producing **different keys for the same text** across process restarts, deployments, or multiple workers. This creates uncontrolled duplicate semantic facts.

**Affected Files:**
- `src/memory/hippocampal/store.py` — `_generate_key()` method
- `src/consolidation/migrator.py` — `_create_new_fact()` method
- `src/storage/postgres.py` — `upsert()` uses `(tenant_id, key)` for dedup

**Benefits of Fixing:**
1. **Prevents data loss** — Distinct preferences/facts with shared entities won't silently overwrite each other
2. **Deterministic deduplication** — Same content always produces the same key, regardless of process/worker
3. **Controlled semantic store growth** — No more unbounded duplicate facts from consolidation
4. **Reliable versioning** — `SemanticFactStore.upsert_fact()` superseding logic works correctly when keys are stable
5. **Correct retrieval** — Fact-first lookups return the right fact, not an overwritten one

**Risks of Fixing:**
1. **Key format change breaks existing data** — Old keys (`preference:italian`) differ from new keys (`preference:sha256_of_subject_predicate`); upsert dedup will fail to find existing records
2. **Backward compatibility** — Any external system reading keys will see format changes
3. **Hash collision (theoretical)** — SHA256-truncated keys have vanishingly small collision probability but should be monitored
4. **Migration complexity** — Need to handle both old and new key formats during transition

**Risk Mitigation:**
- New key format includes subject+predicate+category: `user:preference:{sha256(category:predicate:subject)[:16]}`
- Write a one-time migration script that re-keys existing records
- During transition, check both old and new key formats in upsert lookups
- Add a unique constraint on `(tenant_id, key, is_current)` to catch duplicates at the DB level

---

### 2.3 Tenant vs Session Scoping Inconsistency

**Current Behaviour:**
The codebase comments say "Holistic: tenant-only", but the API exposes session constructs (`/session/create`, `/session/{session_id}/context`). Writes accept `session_id` and store it as `source_session_id`. However, reads generally do **not** scope to session — vector search, fact lookup, and graph queries all operate over the entire tenant partition.

**Affected Files:**
- `src/retrieval/retriever.py` — All `_retrieve_*` methods are tenant-only
- `src/memory/orchestrator.py` — `read()` passes `context_filter` but not session filter
- `src/api/routes.py` — Session endpoints exist but retrieval ignores session
- `src/storage/postgres.py` — `vector_search()` filters by `tenant_id` only

**Benefits of Fixing:**
1. **Privacy correctness** — Sessions from different contexts (e.g., different app features) don't leak into each other
2. **Reduced retrieval noise** — Session-scoped queries search a smaller partition, improving relevance
3. **Lower latency** — Smaller search space means faster vector search and lower resource usage
4. **Multi-tenant safety** — Clear session boundaries prevent accidental cross-context exposure

**Risks of Fixing:**
1. **Major architectural change** — Adding session scoping to all read paths requires changes across retriever, planner, store, and API layers
2. **Regression risk** — Existing users relying on tenant-global retrieval will get different (fewer) results
3. **Complexity of "expand to tenant"** — Need logic to decide when session results are insufficient and tenant-global fallback is needed
4. **Index performance** — Adding `source_session_id` to vector search filters may require composite indexes

**Risk Mitigation:**
- Implement as opt-in: `session_scope: "session_first" | "tenant_only" | "session_only"` parameter
- Default to `tenant_only` for backward compatibility
- Add composite index `(tenant_id, source_session_id)` on `memory_records`
- Defer to Phase 5 (P2) — not on the critical path

**Decision: DEFER to P2.** The current tenant-only behaviour is explicitly documented. Session scoping is an enhancement, not a correctness fix. Implement after critical P0/P1 items.

---

### 2.4 Temporal Query Semantics Under-specified

**Current Behaviour:**
The planner constructs time filters using `datetime.now(UTC)` for references like "yesterday", "last week". This is server-time, not user-local-time. Timestamps are stored with mixed timezone awareness (some naive, some aware) via `storage/utils.py` `naive_utc()`.

**Affected Files:**
- `src/retrieval/planner.py` — `_build_time_filter()` uses server UTC
- `src/storage/utils.py` — `naive_utc()` strips timezone info
- `src/core/schemas.py` — `MemoryRecord.timestamp` definition

**Benefits of Fixing:**
1. **Correct temporal queries** — "What did I say yesterday?" returns the right results for the user's timezone
2. **Better LoCoMo scores** — Temporal queries are a major evaluation axis
3. **Consistent storage** — All timestamps stored as UTC with explicit timezone info
4. **Debuggability** — Clear timezone semantics make investigation easier

**Risks of Fixing:**
1. **API change** — Need to accept `user_timezone` parameter
2. **Storage migration** — Existing naive timestamps may need annotation
3. **Complexity** — Timezone conversion in queries adds logic to planner and storage layers

**Risk Mitigation:**
- Accept `timezone` as optional API parameter, default to UTC
- All storage remains UTC; conversion only at query-time in planner
- Defer full migration; new records are timezone-aware, old records assumed UTC

**Decision: PARTIAL FIX in P1.** Add timezone parameter to planner, keep storage UTC, defer migration.

---

### 2.5 Retrieval Timeouts Exist on Paper Only

**Current Behaviour:**
`RetrievalStep.timeout_ms` (default 100ms) and `RetrievalPlan.total_timeout_ms` (default 500ms) are defined but never enforced. `HybridRetriever.retrieve()` uses `asyncio.gather()` without `asyncio.wait_for()`, so a slow graph query (Neo4j GDS PageRank can take seconds) will block the entire retrieval pipeline.

**Affected Files:**
- `src/retrieval/retriever.py` — `retrieve()` and `_execute_step()` methods
- `src/retrieval/planner.py` — Defines timeouts that are ignored

**Current Code (retriever.py:42-63):**
```python
async def retrieve(self, tenant_id, plan, context_filter):
    all_results = []
    for group_indices in plan.parallel_steps:
        group_steps = [plan.steps[i] for i in group_indices]
        group_results = await asyncio.gather(          # ← No timeout!
            *[self._execute_step(...) for step in group_steps],
            return_exceptions=True,
        )
        # ...
```

**Benefits of Fixing:**
1. **Predictable tail latency** — p95/p99 latency becomes bounded; SLOs become enforceable
2. **Graceful degradation** — If Neo4j is slow, vector results still return within budget
3. **Resource protection** — Timed-out connections are released back to the pool
4. **Operational confidence** — Can set SLOs that the system actually respects

**Risks of Fixing:**
1. **Partial results** — Timed-out steps return no results, potentially reducing retrieval quality
2. **False timeouts** — If timeout values are too aggressive, good results get dropped
3. **Cancellation complexity** — `asyncio.wait_for` raises `TimeoutError` but doesn't cancel the underlying DB query

**Risk Mitigation:**
- Start with generous timeouts (step: 500ms, total: 2000ms), tune down with data
- Log timeout events with step metadata for tuning
- On timeout, return partial results with a `timed_out: true` flag
- Use `asyncio.wait_for` + explicit task cancellation for clean cleanup

---

### 2.6 Concurrency Safety & Multi-Instance Correctness

**Current Behaviour:**
Short-term state is held in process-local Python dicts:
- `WorkingMemoryManager._states: dict[str, WorkingMemoryState]`
- `SensoryBufferManager._buffers: dict[str, SensoryBuffer]`

These are unbounded (no TTL, no LRU eviction) and protected by global `asyncio.Lock` objects that serialize all operations across all tenants.

**Affected Files:**
- `src/memory/working/manager.py` — `_states` dict with global lock
- `src/memory/sensory/buffer.py` — Per-buffer lock, but manager has global scope

**Benefits of Fixing:**
1. **Prevents memory leaks** — Long-running servers won't accumulate unbounded state
2. **Reduces lock contention** — Per-scope locks instead of global locks
3. **Multi-worker correctness** — Redis-backed state works across Uvicorn workers
4. **Predictable resource usage** — LRU bounds keep memory footprint stable

**Risks of Fixing:**
1. **Redis dependency** — Multi-worker state requires Redis (already optional dependency)
2. **Serialization overhead** — State must be serializable to move to Redis
3. **Latency increase** — Redis round-trip vs in-process dict access
4. **Complexity** — LRU + TTL logic to implement and test

**Risk Mitigation:**
- In-process LRU with TTL as default (no new dependency)
- Redis-backed state as opt-in for multi-worker deployments
- Bounded LRU (1000 entries default) with 30-minute TTL
- Per-scope locking via `asyncio.Lock` per key (not global)

---

## 3. Deep Analysis: Efficiency & Scalability Issues

### 3.1 End-to-End Turn Cost is Too High

**Current Behaviour:**
The `/memory/turn` flow in `SeamlessMemoryProvider.process_turn()` executes:
1. `orchestrator.read()` — Embedding + vector search + graph/fact search
2. `orchestrator.write(user_message)` — Chunking + embedding + DB upsert
3. `orchestrator.write(assistant_response)` — Same as above
4. `reconsolidation.process_turn()` — Conflict detection + belief revision

All steps are **synchronous** (awaited sequentially). Even with async I/O, the end-to-end latency is the **sum** of all steps.

**Affected Files:**
- `src/memory/seamless_provider.py` — `process_turn()` orchestrates all steps
- `src/memory/hippocampal/store.py` — `encode_batch()` is called twice (user + assistant)

**Current Flow Latency Estimate (per turn):**
| Step | Network Calls | Estimated Latency |
|------|--------------|-------------------|
| Read (embed query) | 1 embedding API | 50-100ms |
| Read (vector search) | 1 Postgres query | 10-30ms |
| Read (fact lookup) | 1 Postgres query | 5-10ms |
| Read (graph) | 1-3 Neo4j queries | 50-200ms |
| Write user (chunk) | 0-1 LLM call | 0-500ms |
| Write user (embed) | N embedding APIs | 50-300ms |
| Write user (upsert) | N Postgres inserts | 10-50ms |
| Write assistant (same) | same | 50-500ms |
| Reconsolidation | 1-F LLM calls | 0-2000ms |
| **Total** | | **225-3690ms** |

**Benefits of Fixing:**
1. **50-80% latency reduction** — Async storage moves writes off the critical path
2. **User-perceived latency = retrieval only** — Read result returns immediately; writes complete in background
3. **Resilience** — Write failures don't block the response
4. **Higher throughput** — Write pipeline can be batched and rate-limited independently

**Risks of Fixing:**
1. **Eventual consistency** — Writes may not be visible in the next read (acceptable for memory systems)
2. **Message loss on crash** — If the process dies before async write completes, data is lost
3. **Queue complexity** — Needs a reliable queue (Redis/Celery) or at-least-once semantics
4. **Debugging difficulty** — Async errors are harder to trace

**Risk Mitigation:**
- Use Redis queue for async writes (already a dependency)
- Idempotency keys prevent double-writes on retry
- Synchronous fallback via `STORE_ASYNC=false` feature flag (exists in codebase as placeholder)
- Structured logging with correlation IDs for traceability

---

### 3.2 Embedding Calls Not Batched

**Current Behaviour:**
`HippocampalStore.encode_batch()` (store.py:129-171) iterates chunks and calls `encode_chunk()` per chunk. Each `encode_chunk()` calls `await self.embeddings.embed(text)` — a separate network API call per chunk.

The `EmbeddingClient` interface already supports `embed_batch(texts)` which makes **one** API call for all texts.

**Latency Impact:**
- 5 chunks per turn → 5 sequential API calls → 5 × 80ms = **400ms**
- Batched: 1 API call → **80ms** (5x improvement)

**Benefits of Fixing:**
1. **5-10x latency reduction on writes** — Single network round-trip for all chunks
2. **Lower API costs** — Batch calls often have lower per-token pricing
3. **Reduced rate-limit pressure** — Fewer requests to embedding provider
4. **Better throughput** — Fewer connections in use at any time

**Risks of Fixing:**
1. **All-or-nothing failure** — If batch embed fails, all chunks fail (vs partial success today)
2. **Ordering guarantee** — Must map batch results back to correct chunks
3. **Max batch size** — Some providers limit batch size; need chunking of batches
4. **Write-gate interaction** — Must evaluate write gate and redaction before batching embeds

**Risk Mitigation:**
- Implement 3-stage pipeline: (a) gate + redact all chunks, (b) batch embed surviving chunks, (c) batch upsert
- Preserve chunk-to-embedding mapping via index alignment
- Add `max_batch_size` config (default 100) with automatic sub-batching
- On partial failure, retry individual chunks as fallback

---

### 3.3 Sensory Buffer Tokenisation is Expensive

**Current Behaviour:**
`SensoryBuffer._tokenize()` (buffer.py:134-143):
```python
def _tokenize(self, text: str) -> list[str]:
    enc = tiktoken.get_encoding("cl100k_base")
    tokens = enc.encode(text)
    return [enc.decode([t]) for t in tokens]  # ← Per-token decode!
```
For each token, `enc.decode([t])` is called, creating a new Python string object. For a 500-token message, this creates 500 string objects with 500 decode calls.

**Benefits of Fixing:**
1. **~10x reduction in tokenization CPU** — Store token IDs (ints), decode only when needed
2. **Lower GC pressure** — 500 ints vs 500 string objects
3. **Lower memory footprint** — int is 28 bytes, average string is 50+ bytes
4. **Faster cleanup** — Integer comparison vs string comparison in decay logic

**Risks of Fixing:**
1. **API change** — `BufferedToken.token` changes from `str` to `int`; callers of `get_recent()` need updating
2. **Encoding state** — Must keep a reference to the tiktoken encoder for later decoding
3. **Minor** — Risk is very low; this is an internal implementation detail

**Risk Mitigation:**
- Rename field to `token_id: int`, add `decode()` method
- Cache tiktoken encoder instance (already expensive to instantiate)
- `get_text()` does batch decode when called

---

### 3.4 Graph Multi-hop N+1 Query Pattern

**Current Behaviour:**
`NeocorticalStore.multi_hop_query()` runs PageRank/PPR to get top entities, then for **each** entity calls:
1. `graph.get_entity_facts(entity)` — 1 Neo4j query
2. `fact_store.search_facts(entity)` — 1 Postgres query

With 10 entities, this is **20 sequential queries** (N+1 pattern).

**Benefits of Fixing:**
1. **10x latency reduction for graph retrieval** — Batch queries instead of sequential
2. **Lower connection pool pressure** — Fewer concurrent DB connections
3. **Simpler error handling** — Single batch query vs 20 individual error paths

**Risks of Fixing:**
1. **Query complexity** — Batch Cypher/SQL is harder to write and debug
2. **Result size** — Batched results may be large; need to bound carefully
3. **Partial failure** — One entity failure shouldn't block others

**Risk Mitigation:**
- Use `UNWIND` in Cypher for batch entity lookup
- Use `WHERE key IN (...)` in Postgres for batch fact search
- Limit result set with `LIMIT` per entity

---

### 3.5 Neo4j GDS PageRank is Expensive Per Query

**Current Behaviour:**
The Neo4j store builds and streams PageRank on the fly per request using `gds.pageRank.stream` with dynamically supplied node/relationship queries. This creates a temporary in-memory graph projection for each request.

**Benefits of Fixing:**
1. **Order-of-magnitude query speedup** — Pre-projected named graphs avoid rebuild
2. **Lower Neo4j memory usage** — Shared projection vs per-query projection
3. **Predictable performance** — Named graph has known size and characteristics

**Risks of Fixing:**
1. **Stale graph** — Named projections need periodic refresh when data changes
2. **Memory usage** — Named projections consume Neo4j memory permanently
3. **Graph management** — Need lifecycle management (create/refresh/drop)

**Risk Mitigation:**
- Refresh named graph on consolidation runs (already a background job)
- Fall back to dynamic projection if named graph is stale (>1 hour)
- Monitor Neo4j heap usage and auto-drop if memory pressure

**Decision: DEFER to P2.** The existing fallback path (non-GDS PPR) works. Optimise only after verifying Neo4j is in the critical path via profiling.

---

### 3.6 O(n²) Dependency Counting in Forgetting

**Current Behaviour:**
`ForgettingWorker._get_dependency_counts()` (worker.py:116-135):
```python
for mem in memories:          # O(n)
    for other in memories:    # O(n) → total O(n²)
        if other.supersedes_id == mem_id:
            counts[mem_id] += 1
        if mem_id in refs:    # O(k) per check
            counts[mem_id] += 1
```
With `max_memories=5000`, this is 25 million iterations **in pure Python**.

**Benefits of Fixing:**
1. **100-1000x speedup for forgetting runs** — DB-side aggregation vs Python loops
2. **Enables higher memory limits** — Can process 50k+ memories without timeout
3. **Reduces CPU usage** — Offloads work to Postgres (optimised for set operations)
4. **Faster maintenance cycles** — Forgetting can run more frequently without disrupting service

**Risks of Fixing:**
1. **DB query complexity** — Aggregation query with JSON field traversal needs careful testing
2. **JSON field indexing** — `metadata.evidence_refs` in JSON needs GIN index for performance
3. **DB load** — Aggregation query runs over entire tenant partition

**Risk Mitigation:**
- Add GIN index on `metadata` JSON field
- Use `jsonb_array_elements_text` for Postgres JSON traversal
- Test with 10k, 50k, 100k records to verify performance
- Keep Python fallback for compatibility

---

### 3.7 pgvector HNSW Parameters Not Tuned

**Current Behaviour:**
HNSW index created with `m=16, ef_construction=64` (reasonable defaults). Query-time `hnsw.ef_search` is never set, defaulting to pgvector's built-in default (typically 40). The system doesn't tune `ef_search` based on `top_k` or query criticality.

**Benefits of Fixing:**
1. **Higher recall for important queries** — Increasing `ef_search` improves recall at cost of latency
2. **Lower latency for simple queries** — Decreasing `ef_search` speeds up low-importance queries
3. **Measurable tuning** — Expose `ef_search` as a metric for recall vs latency trade-off analysis
4. **Production readiness** — Explicit parameter management vs relying on defaults

**Risks of Fixing:**
1. **Session-level setting** — `SET LOCAL hnsw.ef_search` requires running inside a transaction
2. **Over-tuning** — Too-high `ef_search` defeats the purpose of ANN indexing
3. **Complexity** — Adaptive ef_search based on query type adds decision logic

**Risk Mitigation:**
- Start with static config: `hnsw_ef_search: int = 64`
- Set via `SET LOCAL` at start of vector_search transaction
- Log actual ef_search value alongside query latency for later tuning
- Defer adaptive tuning to a later iteration

---

## 4. Architecture Changes

### 4.1 Current vs Proposed Architecture

**Current Write Flow:**
```
User Message
    │
    ▼
ShortTermMemory.ingest_turn()
    │
    ├─► SensoryBuffer.ingest()          (per-token decode)
    ├─► WorkingMemory.process_input()   (LLM chunking)
    │
    ▼
HippocampalStore.encode_batch()
    │
    ├─► for chunk in chunks:            (sequential loop)
    │       WriteGate.evaluate()
    │       PIIRedactor.redact()
    │       EmbeddingClient.embed()     (per-chunk API call!)
    │       EntityExtractor.extract()
    │       RelationExtractor.extract()
    │       PostgresStore.upsert()
    │
    ▼
[No semantic fact write]
```

**Proposed Write Flow:**
```
User Message
    │
    ▼
ShortTermMemory.ingest_turn()
    │
    ├─► SensoryBuffer.ingest()               (token ID storage)
    ├─► WorkingMemory.process_input()        (LLM chunking)
    │
    ▼
HippocampalStore.encode_batch()  [REFACTORED]
    │
    ├─► Phase 1: Gate + Redact ALL chunks    (no network calls)
    │       for chunk in chunks:
    │           WriteGate.evaluate(chunk)
    │           PIIRedactor.redact(chunk)
    │
    ├─► Phase 2: BATCH embed surviving texts  (ONE API call)
    │       texts = [c.text for c in surviving_chunks]
    │       embeddings = EmbeddingClient.embed_batch(texts)
    │
    ├─► Phase 3: PARALLEL extract + upsert    (bounded concurrency)
    │       async with Semaphore(3):
    │           extract_entities(chunk)
    │           extract_relations(chunk)
    │       PostgresStore.upsert_batch(records)
    │
    ▼
WriteTimeFactExtractor  [NEW]
    │
    ├─► Classify chunks → preference/fact/other
    ├─► Rule-based key derivation (no LLM)
    ├─► NeocorticalStore.store_fact()  (async fire-and-forget)
    │
    ▼
AsyncStoragePipeline  [NEW — when STORE_ASYNC=true]
    │
    ├─► Enqueue write job to Redis
    ├─► Return immediately to caller
    ├─► Background worker processes queue
```

### 4.2 New Component: WriteTimeFactExtractor

**Purpose:** Extract structured semantic facts from chunks at write-time, populating the semantic fact store immediately instead of waiting for consolidation.

**Location:** `src/extraction/write_time_facts.py`

**Architecture:**
```
                    SemanticChunk
                        │
                        ▼
            ┌─────────────────────┐
            │ WriteTimeFactExtract│
            │                     │
            │  1. Type filter     │ ── Only PREFERENCE, SEMANTIC_FACT, IDENTITY chunks
            │  2. Pattern match   │ ── Regex/rule-based key extraction
            │  3. Key derivation  │ ── user:{category}:{predicate} format
            │  4. Confidence adj  │ ── Lower confidence (0.6) than consolidation (0.8)
            │                     │
            └────────┬────────────┘
                     │
                     ▼
            NeocorticalStore.store_fact()
```

**Pseudo-code:**
```python
import hashlib
import re
from dataclasses import dataclass
from typing import Optional

from ..core.enums import MemoryType
from ..memory.neocortical.schemas import FactCategory
from ..memory.working.models import SemanticChunk


@dataclass
class ExtractedFact:
    key: str
    category: FactCategory
    predicate: str
    value: str
    confidence: float


# --- Rule-based patterns for common fact types ---
PREFERENCE_PATTERNS = [
    # "I prefer X", "I like X", "I love X", "I enjoy X"
    (r"(?:i|my)\s+(?:prefer|like|love|enjoy|hate|dislike)\s+(.+)",
     "preference", 0.7),
    # "My favorite X is Y"
    (r"(?:my|the)\s+(?:favorite|favourite)\s+(\w+)\s+(?:is|are)\s+(.+)",
     "preference", 0.75),
]

IDENTITY_PATTERNS = [
    # "My name is X", "I'm X", "Call me X"
    (r"(?:my name is|i'?m|call me|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
     "identity:name", 0.85),
    # "I live in X", "I'm from X", "I moved to X"
    (r"(?:i live in|i'?m from|i moved to|i'm based in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)",
     "location:current_city", 0.7),
    # "I work as X", "I'm a X", "My job is X"
    (r"(?:i work as|i'?m a|my job is|my occupation is)\s+(.+)",
     "occupation:role", 0.7),
]


class WriteTimeFactExtractor:
    """Extract structured facts from chunks at write-time.
    
    Uses rule-based patterns (no LLM) to keep latency minimal.
    Only extracts high-confidence, well-structured facts.
    Lower confidence (0.6) than consolidation-derived facts (0.8).
    """

    WRITE_TIME_CONFIDENCE_BASE = 0.6

    def extract(self, chunk: SemanticChunk) -> list[ExtractedFact]:
        """Extract facts from a single chunk.
        
        Returns empty list for chunk types that don't contain facts.
        """
        # Only process fact-bearing chunk types
        memory_type = self._classify_memory_type(chunk)
        if memory_type not in (
            MemoryType.PREFERENCE,
            MemoryType.SEMANTIC_FACT,
        ):
            return []
        
        facts: list[ExtractedFact] = []
        text = chunk.text.strip()
        
        # Try preference patterns
        for pattern, predicate_prefix, confidence_boost in PREFERENCE_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip().rstrip(".")
                predicate = self._derive_predicate(predicate_prefix, value)
                key = f"user:{predicate_prefix}:{predicate}"
                facts.append(ExtractedFact(
                    key=key,
                    category=FactCategory.PREFERENCE,
                    predicate=predicate,
                    value=value,
                    confidence=self.WRITE_TIME_CONFIDENCE_BASE * confidence_boost,
                ))
        
        # Try identity patterns
        for pattern, key_suffix, confidence_boost in IDENTITY_PATTERNS:
            match = re.search(pattern, text, re.IGNORECASE)
            if match:
                value = match.group(1).strip()
                key = f"user:{key_suffix}"
                category_str = key_suffix.split(":")[0]
                try:
                    category = FactCategory(category_str)
                except ValueError:
                    category = FactCategory.CUSTOM
                facts.append(ExtractedFact(
                    key=key,
                    category=category,
                    predicate=key_suffix.split(":")[-1],
                    value=value,
                    confidence=self.WRITE_TIME_CONFIDENCE_BASE * confidence_boost,
                ))
        
        return facts

    def _classify_memory_type(self, chunk: SemanticChunk) -> MemoryType:
        """Classify chunk into memory type based on chunk_type."""
        from ..memory.working.models import ChunkType
        type_map = {
            ChunkType.PREFERENCE: MemoryType.PREFERENCE,
            ChunkType.FACT: MemoryType.SEMANTIC_FACT,
            ChunkType.IDENTITY: MemoryType.SEMANTIC_FACT,
        }
        return type_map.get(chunk.chunk_type, MemoryType.EPISODIC_EVENT)

    def _derive_predicate(self, prefix: str, value: str) -> str:
        """Derive a stable predicate from prefix and value.
        
        For generic preferences, uses the value category.
        For specific schemas, uses the schema-defined predicate.
        """
        # Check against known schema predicates
        known_predicates = {
            "cuisine": ["food", "restaurant", "eat", "cook", "meal"],
            "music": ["music", "song", "band", "listen", "genre"],
            "color": ["color", "colour"],
            "language": ["language", "speak"],
        }
        value_lower = value.lower()
        for predicate, keywords in known_predicates.items():
            if any(kw in value_lower for kw in keywords):
                return predicate
        # Fallback: use stable hash of value for unique predicate
        return hashlib.sha256(value_lower.encode()).hexdigest()[:12]
```

### 4.3 Modified Component: AsyncStoragePipeline

**Purpose:** Decouple storage from the hot path by enqueuing writes to a background processor.

**Location:** Modified `src/memory/seamless_provider.py`, new `src/storage/async_pipeline.py`

**Architecture:**
```
SeamlessMemoryProvider.process_turn()
    │
    ├─► orchestrator.read()              [SYNC — on hot path]
    │       returns MemoryPacket
    │
    ├─► AsyncStoragePipeline.enqueue()   [ASYNC — off hot path]
    │       ├─► Redis RPUSH job
    │       └─► return immediately
    │
    └─► return SeamlessTurnResult        [Fast response]

Background Worker (Celery or asyncio):
    │
    ├─► Redis BLPOP job
    ├─► orchestrator.write(user_message)
    ├─► orchestrator.write(assistant_response)
    └─► reconsolidation.process_turn()
```

**Pseudo-code:**
```python
import json
import uuid
from dataclasses import dataclass, asdict
from datetime import datetime
from typing import Optional


@dataclass
class StorageJob:
    """A queued storage job."""
    job_id: str
    tenant_id: str
    user_message: str
    assistant_response: Optional[str]
    session_id: Optional[str]
    turn_id: Optional[str]
    timestamp: Optional[str]  # ISO format
    idempotency_key: str      # Prevents double-writes


class AsyncStoragePipeline:
    """Enqueues storage operations for background processing.
    
    When STORE_ASYNC is enabled, writes are queued to Redis
    and processed by a background worker, removing storage
    latency from the user-facing response path.
    """

    QUEUE_KEY = "cml:storage:queue"
    PROCESSED_KEY_PREFIX = "cml:storage:processed:"
    PROCESSED_TTL = 3600  # 1 hour dedup window

    def __init__(self, redis_client, enabled: bool = False):
        self.redis = redis_client
        self.enabled = enabled

    async def enqueue(
        self,
        tenant_id: str,
        user_message: str,
        assistant_response: str | None = None,
        session_id: str | None = None,
        turn_id: str | None = None,
        timestamp: datetime | None = None,
    ) -> str:
        """Enqueue a storage job. Returns job_id."""
        # Generate idempotency key from content
        import hashlib
        content_hash = hashlib.sha256(
            f"{tenant_id}:{user_message}:{assistant_response}:{turn_id}".encode()
        ).hexdigest()[:32]
        
        # Check for duplicate
        dedup_key = f"{self.PROCESSED_KEY_PREFIX}{content_hash}"
        if await self.redis.exists(dedup_key):
            return f"dedup:{content_hash}"
        
        job = StorageJob(
            job_id=str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_message=user_message,
            assistant_response=assistant_response,
            session_id=session_id,
            turn_id=turn_id,
            timestamp=timestamp.isoformat() if timestamp else None,
            idempotency_key=content_hash,
        )
        
        await self.redis.rpush(self.QUEUE_KEY, json.dumps(asdict(job)))
        return job.job_id

    async def process_next(self, orchestrator) -> bool:
        """Process next job from queue. Returns True if a job was processed."""
        raw = await self.redis.blpop(self.QUEUE_KEY, timeout=5)
        if not raw:
            return False
        
        job_data = json.loads(raw[1])
        job = StorageJob(**job_data)
        
        # Idempotency check
        dedup_key = f"{self.PROCESSED_KEY_PREFIX}{job.idempotency_key}"
        if await self.redis.exists(dedup_key):
            return True  # Already processed
        
        try:
            # Write user message
            await orchestrator.write(
                tenant_id=job.tenant_id,
                content=job.user_message,
                session_id=job.session_id,
                context_tags=["conversation", "user_input"],
            )
            
            # Write assistant response
            if job.assistant_response:
                await orchestrator.write(
                    tenant_id=job.tenant_id,
                    content=job.assistant_response,
                    session_id=job.session_id,
                    context_tags=["conversation", "assistant_response"],
                )
            
            # Mark as processed (with TTL for dedup window)
            await self.redis.setex(dedup_key, self.PROCESSED_TTL, "1")
            
        except Exception:
            # Re-queue on failure (with retry limit)
            retry_count = job_data.get("_retry", 0) + 1
            if retry_count < 3:
                job_data["_retry"] = retry_count
                await self.redis.rpush(self.QUEUE_KEY, json.dumps(job_data))
            # else: dead-letter (log and discard)
        
        return True
```

### 4.4 Modified Component: TimeoutAwareRetriever

**Purpose:** Enforce per-step and total timeouts during retrieval.

**Location:** Modified `src/retrieval/retriever.py`

**Pseudo-code (key changes):**
```python
import asyncio
from datetime import UTC, datetime


class HybridRetriever:
    """Executes retrieval plans with enforced timeouts."""

    async def retrieve(self, tenant_id, plan, context_filter=None):
        """Execute plan with total timeout enforcement."""
        all_results = []
        plan_start = datetime.now(UTC)
        plan_budget_s = plan.total_timeout_ms / 1000.0
        
        for group_indices in plan.parallel_steps:
            # Check remaining budget
            elapsed = (datetime.now(UTC) - plan_start).total_seconds()
            remaining = plan_budget_s - elapsed
            if remaining <= 0:
                # Log: "retrieval_plan_timeout_exceeded"
                break
            
            group_steps = [plan.steps[i] for i in group_indices if i < len(plan.steps)]
            
            # Execute group with timeout
            try:
                group_results = await asyncio.wait_for(
                    self._execute_group(tenant_id, group_steps, context_filter),
                    timeout=remaining,
                )
            except asyncio.TimeoutError:
                # Log: "retrieval_group_timeout", include which steps were pending
                break
            
            # Process results + skip_if_found logic
            found_high_confidence = False
            for step, result in zip(group_steps, group_results, strict=False):
                if isinstance(result, Exception):
                    continue
                if result.success and result.items:
                    all_results.extend(result.items)
                    if step.skip_if_found:
                        found_high_confidence = True
            
            if found_high_confidence:
                break  # Skip remaining groups
        
        return self._to_retrieved_memories(all_results, plan.analysis)

    async def _execute_group(self, tenant_id, steps, context_filter):
        """Execute a group of steps in parallel with per-step timeouts."""
        tasks = []
        for step in steps:
            coro = self._execute_step_with_timeout(tenant_id, step, context_filter)
            tasks.append(coro)
        return await asyncio.gather(*tasks, return_exceptions=True)

    async def _execute_step_with_timeout(self, tenant_id, step, context_filter):
        """Execute a single step with its own timeout."""
        timeout_s = step.timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                self._execute_step(tenant_id, step, context_filter),
                timeout=timeout_s,
            )
        except asyncio.TimeoutError:
            # Return empty result with timeout flag
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=step.timeout_ms,
                success=False,
                error=f"Timeout after {step.timeout_ms}ms",
            )
```

### 4.5 New Component: BoundedStateManager

**Purpose:** Replace unbounded `_states` and `_buffers` dicts with LRU-bounded, TTL-aware state management.

**Location:** `src/utils/bounded_state.py`

**Pseudo-code:**
```python
import asyncio
import time
from collections import OrderedDict
from typing import TypeVar, Generic

T = TypeVar("T")


class BoundedStateMap(Generic[T]):
    """LRU + TTL bounded state map.
    
    Replaces unbounded dict[str, T] in WorkingMemoryManager
    and SensoryBufferManager. Uses per-key locking instead
    of global lock.
    
    Thread-safe for asyncio concurrent access.
    """

    def __init__(
        self,
        max_size: int = 1000,
        ttl_seconds: float = 1800.0,  # 30 minutes
    ):
        self._max_size = max_size
        self._ttl = ttl_seconds
        self._data: OrderedDict[str, tuple[T, float]] = OrderedDict()
        self._locks: dict[str, asyncio.Lock] = {}
        self._global_lock = asyncio.Lock()  # Only for structural changes

    async def get(self, key: str) -> T | None:
        """Get value, returns None if expired or missing."""
        async with self._global_lock:
            if key not in self._data:
                return None
            value, created_at = self._data[key]
            if time.time() - created_at > self._ttl:
                del self._data[key]
                return None
            # Move to end (most recently used)
            self._data.move_to_end(key)
            return value

    async def get_or_create(self, key: str, factory) -> T:
        """Get existing or create new. Factory is called once."""
        existing = await self.get(key)
        if existing is not None:
            return existing
        
        async with self._global_lock:
            # Double-check after acquiring lock
            if key in self._data:
                value, created_at = self._data[key]
                if time.time() - created_at <= self._ttl:
                    self._data.move_to_end(key)
                    return value
            
            # Create new
            value = factory() if not asyncio.iscoroutinefunction(factory) else await factory()
            self._data[key] = (value, time.time())
            
            # Evict oldest if over capacity
            while len(self._data) > self._max_size:
                self._data.popitem(last=False)
            
            return value

    async def delete(self, key: str) -> bool:
        """Delete a key. Returns True if existed."""
        async with self._global_lock:
            if key in self._data:
                del self._data[key]
                return True
            return False

    async def cleanup_expired(self) -> int:
        """Remove all expired entries. Returns count removed."""
        now = time.time()
        to_remove = []
        async with self._global_lock:
            for key, (_, created_at) in self._data.items():
                if now - created_at > self._ttl:
                    to_remove.append(key)
            for key in to_remove:
                del self._data[key]
        return len(to_remove)

    @property
    def size(self) -> int:
        return len(self._data)
```

---

## 5. Implementation Plan

### Phase 1: Critical Correctness Fixes (P0)

**Duration:** 1.5 weeks  
**Risk Level:** Medium  
**Dependencies:** None

#### Task 1.1: Fix Non-deterministic Consolidation Fact Keys

**File:** `src/consolidation/migrator.py`

**Change:** Replace `hash(gist.text) % 10000` with stable SHA256-based key.

**Pseudo-code:**
```python
import hashlib


def _stable_fact_key(prefix: str, text: str) -> str:
    """Generate a stable, deterministic key for a semantic fact.
    
    Uses SHA256 (not Python hash()) so the key is identical across:
    - Different Python processes
    - Different workers/deployments
    - Process restarts
    
    Format: user:custom:{sha256_hex[:16]}
    Collision probability: ~1 in 2^64 (negligible)
    """
    h = hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]
    return f"{prefix}:{h}"


# In ConsolidationMigrator._create_new_fact():
# BEFORE:
#   key = schema.get("key") or gist.key or f"user:custom:{hash(gist.text) % 10000}"
# AFTER:
key = (
    schema.get("key")
    or gist.key
    or _stable_fact_key("user:custom", gist.text)
)
```

**Tests to add:**
```python
def test_stable_fact_key_deterministic():
    """Same input always produces same key, across calls."""
    key1 = _stable_fact_key("user:custom", "User likes hiking")
    key2 = _stable_fact_key("user:custom", "User likes hiking")
    assert key1 == key2

def test_stable_fact_key_different_inputs():
    """Different inputs produce different keys."""
    key1 = _stable_fact_key("user:custom", "User likes hiking")
    key2 = _stable_fact_key("user:custom", "User likes swimming")
    assert key1 != key2

def test_stable_fact_key_format():
    """Key follows expected format."""
    key = _stable_fact_key("user:custom", "any text")
    assert key.startswith("user:custom:")
    assert len(key.split(":")[-1]) == 16  # 16 hex chars
```

#### Task 1.2: Fix Hippocampal Key Generation

**File:** `src/memory/hippocampal/store.py`

**Change:** Replace first-entity-only key with subject+predicate composite key.

**Pseudo-code:**
```python
def _generate_key(self, chunk: SemanticChunk, memory_type: MemoryType) -> str | None:
    """Generate a stable key for deduplication.
    
    BEFORE: f"{memory_type.value}:{chunk.entities[0].lower()}"
      → "preference:italian" for BOTH "Italian food" and "Italian music"
      → silent overwrite!
    
    AFTER: f"{memory_type.value}:{sha256(text_normalized)[:16]}"
      → unique per distinct content
      → same content always produces same key
    """
    if memory_type not in (
        MemoryType.PREFERENCE,
        MemoryType.SEMANTIC_FACT,
    ):
        return None
    
    # Use content-based hash for stable, unique key
    import hashlib
    text_normalized = chunk.text.strip().lower()
    content_hash = hashlib.sha256(text_normalized.encode()).hexdigest()[:16]
    
    # Include first entity for human readability, but hash for uniqueness
    entity_prefix = ""
    if chunk.entities:
        entity_prefix = chunk.entities[0].lower().replace(" ", "_") + ":"
    
    return f"{memory_type.value}:{entity_prefix}{content_hash}"
```

**Tests to add:**
```python
def test_generate_key_distinct_for_different_facts():
    """Two facts sharing an entity get different keys."""
    chunk1 = SemanticChunk(text="I love Italian food", entities=["Italian"])
    chunk2 = SemanticChunk(text="I love Italian music", entities=["Italian"])
    key1 = store._generate_key(chunk1, MemoryType.PREFERENCE)
    key2 = store._generate_key(chunk2, MemoryType.PREFERENCE)
    assert key1 != key2

def test_generate_key_stable_for_same_fact():
    """Same fact always gets the same key."""
    chunk = SemanticChunk(text="I love Italian food", entities=["Italian"])
    key1 = store._generate_key(chunk, MemoryType.PREFERENCE)
    key2 = store._generate_key(chunk, MemoryType.PREFERENCE)
    assert key1 == key2
```

#### Task 1.3: Align Write-Time Facts with Retrieval Planner Keys

**File:** New `src/extraction/write_time_facts.py` + modified `src/memory/hippocampal/store.py`

**Change:** After encoding chunks into episodic store, extract structured facts and write them to the semantic store with keys matching `DEFAULT_FACT_SCHEMAS`.

**Integration Point (hippocampal store.encode_batch):**
```python
async def encode_batch(self, ...):
    # ... existing chunk processing ...
    
    # NEW: Write-time fact extraction
    if self.fact_extractor:
        for chunk, record in zip(chunks, results):
            if record:  # Only for successfully stored chunks
                extracted_facts = self.fact_extractor.extract(chunk)
                for fact in extracted_facts:
                    try:
                        await self.neocortical.store_fact(
                            tenant_id=tenant_id,
                            key=fact.key,
                            value=fact.value,
                            confidence=fact.confidence,
                            evidence_ids=[str(record.id)],
                        )
                    except Exception:
                        pass  # Fire-and-forget; episodic is source of truth
    
    return results
```

**Tests to add:**
```python
async def test_write_time_fact_populates_semantic_store():
    """Writing a preference chunk also creates a semantic fact."""
    await orchestrator.write(
        tenant_id="t1",
        content="I love Italian food",
    )
    fact = await fact_store.get_fact("t1", "user:preference:cuisine")
    assert fact is not None
    assert "Italian" in str(fact.value)

async def test_write_time_fact_key_matches_planner():
    """Extracted fact key matches what the retrieval planner looks up."""
    # Write
    await orchestrator.write(tenant_id="t1", content="My name is Alice")
    # Read — planner will look up user:identity:name
    packet = await orchestrator.read(tenant_id="t1", query="What is my name?")
    assert any("Alice" in m.record.text for m in packet.facts)
```

---

### Phase 2: Hot-Path Performance (P0)

**Duration:** 2 weeks  
**Risk Level:** Low-Medium  
**Dependencies:** Phase 1 (for write-time facts integration)

#### Task 2.1: Batch Embeddings in Hippocampal Store

**File:** `src/memory/hippocampal/store.py`

**Change:** Refactor `encode_batch()` from sequential per-chunk embedding to a 3-phase pipeline.

**Pseudo-code:**
```python
async def encode_batch(
    self,
    tenant_id: str,
    chunks: list[SemanticChunk],
    context_tags: list[str] | None = None,
    source_session_id: str | None = None,
    agent_id: str | None = None,
    namespace: str | None = None,
    timestamp: datetime | None = None,
    request_metadata: dict[str, Any] | None = None,
    memory_type_override: MemoryType | None = None,
    return_gate_results: bool = False,
):
    """Encode chunks using batched embedding calls.
    
    3-phase pipeline:
    1. Gate + redact all chunks (CPU only, no network)
    2. Batch embed surviving texts (ONE API call)
    3. Build records + batch upsert
    """
    # Fetch existing for write-gate context
    existing = await self.store.scan(
        tenant_id,
        filters={"status": MemoryStatus.ACTIVE.value},
        limit=50,
        order_by="-timestamp",
    )
    existing_dicts = [{"text": m.text} for m in existing]
    
    # ---- Phase 1: Gate + Redact (no network calls) ----
    surviving_chunks: list[tuple[SemanticChunk, WriteGateResult, str]] = []
    gate_results_list: list[dict] = []
    
    for chunk in chunks:
        gate_result = self.write_gate.evaluate(chunk, existing_memories=existing_dicts)
        if return_gate_results:
            gate_results_list.append(_gate_result_to_dict(gate_result))
        
        if gate_result.decision == WriteDecision.SKIP:
            continue
        
        text = chunk.text
        if gate_result.redaction_required:
            redaction_result = self.redactor.redact(text)
            text = redaction_result.redacted_text
        
        surviving_chunks.append((chunk, gate_result, text))
    
    if not surviving_chunks:
        return ([], gate_results_list) if return_gate_results else []
    
    # ---- Phase 2: Batch embed (ONE API call) ----
    texts_to_embed = [text for _, _, text in surviving_chunks]
    embedding_results = await self.embeddings.embed_batch(texts_to_embed)
    
    # ---- Phase 3: Build records + extract + upsert ----
    import asyncio
    
    records: list[MemoryRecord] = []
    semaphore = asyncio.Semaphore(3)  # Limit concurrent extractors
    
    async def process_chunk(idx: int):
        chunk, gate_result, text = surviving_chunks[idx]
        embedding_result = embedding_results[idx]
        
        # Entity extraction (bounded concurrency)
        entities: list[EntityMention] = []
        async with semaphore:
            if self.entity_extractor:
                entities = await self.entity_extractor.extract(text)
            elif chunk.entities:
                entities = [
                    EntityMention(text=e, normalized=e, entity_type="CONCEPT")
                    for e in chunk.entities
                ]
        
        # Relation extraction
        relations: list[Relation] = []
        if self.relation_extractor:
            async with semaphore:
                entity_texts = [e.normalized for e in entities]
                relations = await self.relation_extractor.extract(text, entities=entity_texts)
        
        # Build record
        memory_type = memory_type_override or (
            gate_result.memory_types[0] if gate_result.memory_types else MemoryType.EPISODIC_EVENT
        )
        key = self._generate_key(chunk, memory_type)
        
        system_metadata = {
            "chunk_type": chunk.chunk_type.value,
            "source_turn_id": chunk.source_turn_id,
            "source_role": chunk.source_role,
        }
        merged_metadata = {**system_metadata, **(request_metadata or {})}
        
        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=context_tags or [],
            source_session_id=source_session_id,
            agent_id=agent_id,
            namespace=namespace,
            type=memory_type,
            text=text,
            key=key,
            embedding=embedding_result.embedding,
            entities=entities,
            relations=relations,
            metadata=merged_metadata,
            timestamp=timestamp or chunk.timestamp,
            confidence=chunk.confidence,
            importance=gate_result.importance,
            provenance=Provenance(
                source=MemorySource.AGENT_INFERRED,
                evidence_refs=[chunk.source_turn_id] if chunk.source_turn_id else [],
                model_version=embedding_result.model,
            ),
        )
        stored = await self.store.upsert(record)
        if stored:
            records.append(stored)
            existing_dicts.append({"text": stored.text})
    
    # Process all chunks (extractors are bounded by semaphore)
    await asyncio.gather(*[process_chunk(i) for i in range(len(surviving_chunks))])
    
    if return_gate_results:
        return records, gate_results_list
    return records
```

**Expected Impact:**
- 5 chunks × 80ms embedding = 400ms → 1 × 80ms batch = **80ms** (5x improvement)
- Total write path: ~500ms → ~200ms

#### Task 2.2: Implement Async Storage Pipeline

**Files:** New `src/storage/async_pipeline.py`, modified `src/memory/seamless_provider.py`

**Change:** Make the seamless turn flow non-blocking on writes when `STORE_ASYNC=true`.

**Modified SeamlessMemoryProvider.process_turn():**
```python
async def process_turn(self, tenant_id, user_message, assistant_response=None, ...):
    """Process turn: read is sync (fast), write is async (background)."""
    
    # Step 1: Retrieve (SYNC — on critical path)
    memory_context, injected_memories = await self._retrieve_context(
        tenant_id, user_message
    )
    
    stored_count = 0
    reconsolidation_applied = False
    
    # Step 2+3: Store (ASYNC or SYNC based on config)
    if self.auto_store:
        if self.async_pipeline and self.async_pipeline.enabled:
            # Non-blocking: enqueue and return immediately
            await self.async_pipeline.enqueue(
                tenant_id=tenant_id,
                user_message=user_message,
                assistant_response=assistant_response,
                session_id=session_id,
                turn_id=turn_id,
                timestamp=timestamp,
            )
            stored_count = -1  # Indicates "queued"
        else:
            # Existing synchronous path (unchanged)
            write_result = await self.orchestrator.write(...)
            stored_count += write_result.get("chunks_created", 0)
            # ... existing response processing ...
    
    return SeamlessTurnResult(
        memory_context=memory_context,
        injected_memories=injected_memories,
        stored_count=stored_count,
        reconsolidation_applied=reconsolidation_applied,
    )
```

**Expected Impact:**
- Turn latency drops from sum(read + write + recon) to just read latency
- p95: ~800ms → ~200ms

#### Task 2.3: Activate CachedEmbeddings in Orchestrator

**File:** `src/memory/orchestrator.py`

**Change:** Wrap the embedding client with `CachedEmbeddings` when Redis is available.

**Pseudo-code:**
```python
# In MemoryOrchestrator.create():
embedding_client = get_embedding_client()

# NEW: Wrap with cache if Redis is available
if redis_client:
    from ..utils.embeddings import CachedEmbeddings
    embedding_client = CachedEmbeddings(
        client=embedding_client,
        redis_client=redis_client,
        ttl_seconds=86400,  # 24 hours
    )
```

**Also fix CachedEmbeddings.embed_batch() to use MGET:**
```python
async def embed_batch(self, texts: list[str]) -> list[EmbeddingResult]:
    """Batch embed with Redis MGET for cache lookup."""
    import json
    
    cache_keys = [self._cache_key(text) for text in texts]
    
    # Batch cache lookup (MGET instead of N×GET)
    cached_values = await self.redis.mget(*cache_keys)
    
    results: list[tuple[int, EmbeddingResult | None]] = []
    uncached_texts: list[str] = []
    uncached_indices: list[int] = []
    
    for i, (text, cached) in enumerate(zip(texts, cached_values)):
        if cached:
            results.append((i, EmbeddingResult(**json.loads(cached))))
        else:
            uncached_texts.append(text)
            uncached_indices.append(i)
            results.append((i, None))
    
    # Batch compute uncached
    if uncached_texts:
        computed = await self.client.embed_batch(uncached_texts)
        
        # Batch cache store (pipeline instead of N×SETEX)
        pipe = self.redis.pipeline()
        for idx, result in zip(uncached_indices, computed):
            results[idx] = (idx, result)
            pipe.setex(
                cache_keys[idx],
                self.ttl,
                json.dumps({
                    "embedding": result.embedding,
                    "model": result.model,
                    "dimensions": result.dimensions,
                    "tokens_used": result.tokens_used,
                }),
            )
        await pipe.execute()
    
    return [r for _, r in sorted(results, key=lambda x: x[0]) if r is not None]
```

---

### Phase 3: Retrieval Reliability (P1)

**Duration:** 1.5 weeks  
**Risk Level:** Low  
**Dependencies:** None (can run parallel to Phase 2)

#### Task 3.1: Enforce Per-Step and Total Retrieval Timeouts

**File:** `src/retrieval/retriever.py`

**Change:** Wrap `_execute_step()` with `asyncio.wait_for()` and track total budget.

See full pseudo-code in [Section 4.4](#44-modified-component-timeoutawareretriever).

**Configuration:**
```python
# src/core/config.py — add to RetrievalConfig
class RetrievalConfig(BaseModel):
    default_step_timeout_ms: int = 500    # Per-step timeout
    total_timeout_ms: int = 2000          # Total retrieval budget
    graph_timeout_ms: int = 1000          # Graph steps get more time
    fact_timeout_ms: int = 200            # Fact lookups are fast
```

**Tests to add:**
```python
async def test_retrieval_respects_step_timeout():
    """A slow step times out without blocking others."""
    # Mock graph retrieval to take 5 seconds
    # Assert: total retrieval completes within 2 seconds
    # Assert: vector results are still returned

async def test_retrieval_respects_total_timeout():
    """Total retrieval budget is enforced."""
    # Mock all steps to take 1 second each
    # Assert: retrieval completes within total_timeout_ms
    # Assert: later groups are skipped
```

#### Task 3.2: Implement Effective skip_if_found Behaviour

**File:** `src/retrieval/retriever.py`

**Change:** When a fact-lookup step in group 0 returns high-confidence results, skip the vector-search step in group 1.

**Current Issue:**
The planner creates `parallel_groups = [[0], [1]]` for preference/identity queries, where step 0 is fact-lookup and step 1 is vector-search with `skip_if_found=True`. But the current code only checks `skip_if_found` within a group, not across groups.

**Fix:**
```python
async def retrieve(self, tenant_id, plan, context_filter=None):
    all_results = []
    skip_remaining = False
    
    for group_indices in plan.parallel_steps:
        if skip_remaining:
            break
        
        group_steps = [plan.steps[i] for i in group_indices if i < len(plan.steps)]
        group_results = await asyncio.gather(
            *[self._execute_step_with_timeout(tenant_id, step, context_filter)
              for step in group_steps],
            return_exceptions=True,
        )
        
        for step, result in zip(group_steps, group_results, strict=False):
            if isinstance(result, Exception):
                continue
            if result.success and result.items:
                all_results.extend(result.items)
                if step.skip_if_found and result.items:
                    skip_remaining = True  # Skip ALL remaining groups
    
    return self._to_retrieved_memories(all_results, plan.analysis)
```

#### Task 3.3: Add Timezone-Aware Temporal Queries

**File:** `src/retrieval/planner.py`

**Change:** Accept user timezone in `QueryAnalysis` and convert time filters accordingly.

**Pseudo-code:**
```python
from zoneinfo import ZoneInfo

def _build_time_filter(self, analysis: QueryAnalysis) -> dict | None:
    if not analysis.time_reference:
        return None
    
    # Use user timezone if provided, else UTC
    tz = ZoneInfo(analysis.user_timezone) if analysis.user_timezone else UTC
    user_now = datetime.now(tz)
    
    ref = (analysis.time_reference or "").lower()
    if "today" in ref:
        start = user_now.replace(hour=0, minute=0, second=0, microsecond=0)
        return {"since": start.astimezone(UTC)}  # Convert to UTC for DB query
    if "yesterday" in ref:
        yesterday = user_now - timedelta(days=1)
        start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
        end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
        return {"since": start.astimezone(UTC), "until": end.astimezone(UTC)}
    # ... similar for week, month, recent
```

---

### Phase 4: Background Work Scalability (P1)

**Duration:** 1 week  
**Risk Level:** Low  
**Dependencies:** None

#### Task 4.1: Replace O(n²) Forgetting Dependency Counts with DB Aggregation

**File:** `src/forgetting/worker.py`, `src/storage/postgres.py`

**Change:** Move dependency counting from Python to a single Postgres query.

**New Postgres method:**
```python
# src/storage/postgres.py
async def bulk_dependency_counts(
    self,
    tenant_id: str,
    memory_ids: list[str],
) -> dict[str, int]:
    """Count references to each memory ID in a single DB query.
    
    Counts two reference types:
    1. supersedes_id: Direct version chains
    2. metadata.evidence_refs: JSON array references
    
    Returns: {memory_id: reference_count}
    """
    if not memory_ids:
        return {}
    
    query = """
    WITH target_ids AS (
        SELECT unnest(:ids::text[]) AS target_id
    ),
    supersede_counts AS (
        SELECT 
            CAST(supersedes_id AS text) AS target_id,
            COUNT(*) AS cnt
        FROM memory_records
        WHERE tenant_id = :tenant_id
          AND supersedes_id IS NOT NULL
          AND CAST(supersedes_id AS text) = ANY(:ids)
        GROUP BY supersedes_id
    ),
    evidence_counts AS (
        SELECT 
            ref.value::text AS target_id,
            COUNT(*) AS cnt
        FROM memory_records,
             jsonb_array_elements_text(
                 COALESCE(metadata->'evidence_refs', '[]'::jsonb)
             ) AS ref(value)
        WHERE tenant_id = :tenant_id
          AND ref.value::text = ANY(:ids)
        GROUP BY ref.value
    )
    SELECT 
        t.target_id,
        COALESCE(s.cnt, 0) + COALESCE(e.cnt, 0) AS total_refs
    FROM target_ids t
    LEFT JOIN supersede_counts s ON t.target_id = s.target_id
    LEFT JOIN evidence_counts e ON t.target_id = e.target_id
    """
    
    async with self.session_factory() as session:
        result = await session.execute(
            text(query),
            {"tenant_id": tenant_id, "ids": memory_ids},
        )
        return {row.target_id: row.total_refs for row in result}
```

**Modified ForgettingWorker:**
```python
async def _get_dependency_counts(self, tenant_id, user_id, memories):
    """Count references using DB aggregation instead of O(n²) Python loop."""
    memory_ids = [str(m.id) for m in memories]
    
    # Try DB-side aggregation first
    if hasattr(self.store, "bulk_dependency_counts"):
        return await self.store.bulk_dependency_counts(tenant_id, memory_ids)
    
    # Fallback to original O(n²) for compatibility
    # ... existing code ...
```

**Expected Impact:**
- 5000 memories: 25M iterations in Python → 1 SQL query
- Runtime: ~30 seconds → ~200ms

#### Task 4.2: Batch Graph Entity Queries

**File:** `src/memory/neocortical/store.py`

**Change:** Replace N+1 `query_entity()` calls with batch Cypher query.

**Pseudo-code:**
```python
async def multi_hop_query_batched(
    self,
    tenant_id: str,
    seed_entities: list[str],
    max_hops: int = 3,
) -> list[dict]:
    """Multi-hop query with batched entity lookup."""
    
    # Step 1: Get ranked entities (existing PPR/PageRank logic)
    ranked_entities = await self._get_ranked_entities(tenant_id, seed_entities)
    entity_names = [e["entity"] for e in ranked_entities[:10]]
    
    if not entity_names:
        return []
    
    # Step 2: Batch fetch all entity data in ONE query
    # Instead of: for entity in entities: await query_entity(entity)
    batch_cypher = """
    UNWIND $entities AS entity_name
    MATCH (e {name: entity_name, tenant_id: $tenant_id})
    OPTIONAL MATCH (e)-[r]->(related)
    RETURN entity_name,
           collect({
               predicate: type(r),
               related_entity: related.name,
               properties: properties(r)
           }) AS relations
    """
    entity_data = await self.graph.execute_query(
        batch_cypher,
        {"entities": entity_names, "tenant_id": tenant_id},
    )
    
    # Step 3: Batch fetch facts from semantic store
    facts_by_entity = {}
    if entity_names:
        # Single SQL query with IN clause
        all_facts = await self.fact_store.search_facts_batch(
            tenant_id, entity_names, limit_per_entity=5
        )
        for fact in all_facts:
            key = fact.subject or "user"
            facts_by_entity.setdefault(key, []).append(fact)
    
    # Step 4: Combine results
    results = []
    for entity_row in entity_data:
        name = entity_row["entity_name"]
        results.append({
            "entity": name,
            "relations": entity_row["relations"],
            "facts": facts_by_entity.get(name, []),
            "relevance_score": next(
                (e["score"] for e in ranked_entities if e["entity"] == name), 0.5
            ),
        })
    
    return results
```

---

### Phase 5: Operational Stability (P2)

**Duration:** 1.5 weeks  
**Risk Level:** Low  
**Dependencies:** None

#### Task 5.1: Optimize Sensory Token Storage

**File:** `src/memory/sensory/buffer.py`

**Change:** Store token IDs (ints) instead of decoded strings. Decode only on `get_text()`.

**Pseudo-code:**
```python
import tiktoken
from dataclasses import dataclass


@dataclass
class BufferedToken:
    """A token with its ingestion timestamp."""
    token_id: int             # Token ID (int) instead of decoded string
    timestamp: float
    turn_id: str | None = None
    role: str | None = None


class SensoryBuffer:
    def __init__(self, config=None):
        self.config = config or SensoryBufferConfig()
        self._tokens: deque[BufferedToken] = deque()
        self._lock = asyncio.Lock()
        self._encoder = None  # Lazy init
    
    def _get_encoder(self):
        """Lazy-init tiktoken encoder (reuse across calls)."""
        if self._encoder is None:
            try:
                import tiktoken
                self._encoder = tiktoken.get_encoding("cl100k_base")
            except ImportError:
                self._encoder = None
        return self._encoder
    
    def _tokenize(self, text: str) -> list[int]:
        """Tokenize text, returning token IDs (not strings).
        
        BEFORE: enc.encode(text) → [enc.decode([t]) for t in tokens]  # N decode calls!
        AFTER:  enc.encode(text) → token_ids                          # 0 decode calls
        """
        enc = self._get_encoder()
        if enc:
            return enc.encode(text)
        # Fallback: hash whitespace tokens to pseudo-IDs
        return [hash(w) for w in text.split()]
    
    async def get_text(self, max_tokens=None, role_filter=None):
        """Decode token IDs to text in a single batch."""
        tokens = await self.get_recent(max_tokens=max_tokens, role_filter=role_filter)
        if not tokens:
            return ""
        
        enc = self._get_encoder()
        if enc:
            token_ids = [bt.token_id for bt in tokens]
            return enc.decode(token_ids)  # ONE decode call for all tokens
        
        # Fallback: tokens are pseudo-IDs, can't decode
        return ""
```

#### Task 5.2: Add Bounded LRU + TTL to Working Memory and Sensory Buffer

**File:** `src/memory/working/manager.py`, `src/memory/sensory/manager.py`

**Change:** Replace `self._states: dict` with `BoundedStateMap`.

**Pseudo-code:**
```python
# src/memory/working/manager.py
from ...utils.bounded_state import BoundedStateMap

class WorkingMemoryManager:
    def __init__(self, llm_client=None, max_chunks_per_user=10, ...):
        # BEFORE: self._states: dict[str, WorkingMemoryState] = {}
        # AFTER:
        self._states = BoundedStateMap[WorkingMemoryState](
            max_size=1000,           # Max 1000 concurrent scopes
            ttl_seconds=1800.0,      # 30 min TTL
        )
        self._lock = asyncio.Lock()  # Keep for backward compat, but less contended
        # ...
    
    async def get_state(self, tenant_id, scope_id):
        key = self._get_key(tenant_id, scope_id)
        return await self._states.get_or_create(
            key,
            factory=lambda: WorkingMemoryState(
                tenant_id=tenant_id,
                user_id=scope_id,
                max_chunks=self.max_chunks,
            ),
        )
```

---

### Phase 6: Observability & Tuning (P2)

**Duration:** 1 week  
**Risk Level:** Low  
**Dependencies:** Phases 1-4 for meaningful metrics

#### Task 6.1: Add HNSW ef_search Tuning

**File:** `src/storage/postgres.py`

**Change:** Set `hnsw.ef_search` session variable before vector search queries.

**Pseudo-code:**
```python
# src/core/config.py
class DatabaseConfig(BaseModel):
    hnsw_ef_search: int = 64  # Default: higher than pgvector default of 40

# src/storage/postgres.py
async def vector_search(self, tenant_id, embedding, top_k=10, ...):
    async with self.session_factory() as session:
        # Set ef_search for this query's transaction
        ef_search = max(self.config.hnsw_ef_search, top_k)
        await session.execute(
            text(f"SET LOCAL hnsw.ef_search = {ef_search}")
        )
        
        # ... existing vector search query ...
```

#### Task 6.2: Add Retrieval Step Metrics

**File:** `src/retrieval/retriever.py`

**Change:** Emit Prometheus histograms per retrieval source.

**Pseudo-code:**
```python
from ..utils.metrics import (
    RETRIEVAL_STEP_DURATION,
    RETRIEVAL_STEP_RESULTS,
    RETRIEVAL_TIMEOUT_COUNT,
)

async def _execute_step_with_timeout(self, tenant_id, step, context_filter):
    start = time.perf_counter()
    try:
        result = await asyncio.wait_for(
            self._execute_step(tenant_id, step, context_filter),
            timeout=step.timeout_ms / 1000.0,
        )
        elapsed_ms = (time.perf_counter() - start) * 1000
        RETRIEVAL_STEP_DURATION.labels(source=step.source.value).observe(elapsed_ms)
        RETRIEVAL_STEP_RESULTS.labels(source=step.source.value).observe(len(result.items))
        return result
    except asyncio.TimeoutError:
        elapsed_ms = (time.perf_counter() - start) * 1000
        RETRIEVAL_TIMEOUT_COUNT.labels(source=step.source.value).inc()
        return RetrievalResult(
            source=step.source, items=[], elapsed_ms=elapsed_ms,
            success=False, error="timeout",
        )
```

#### Task 6.3: Add Fact Hit Rate Tracking

**File:** `src/retrieval/retriever.py`

**Change:** Track whether preference/identity queries are answered from facts vs vector.

**Pseudo-code:**
```python
FACT_HIT_RATE = Counter(
    "cml_retrieval_fact_hit_total",
    "Queries answered from semantic fact store",
    ["intent", "hit"],
)

async def retrieve(self, tenant_id, plan, context_filter=None):
    # ... existing logic ...
    
    # Track fact hit rate
    if plan.analysis.intent in (QueryIntent.PREFERENCE_LOOKUP, QueryIntent.IDENTITY_LOOKUP):
        has_fact_results = any(
            r.get("source") == "facts" for r in all_results
        )
        FACT_HIT_RATE.labels(
            intent=plan.analysis.intent.value,
            hit="true" if has_fact_results else "false",
        ).inc()
```

---

## 6. Testing Strategy

### Unit Tests (per phase)

| Phase | Test Category | Count | Key Scenarios |
|-------|--------------|-------|---------------|
| 1 | Key determinism | 6 | Stable keys, no overwrites, schema alignment |
| 1 | Write-time facts | 8 | Pattern matching, key format, confidence, edge cases |
| 2 | Batch embeddings | 5 | Batch call, error handling, mapping correctness |
| 2 | Async pipeline | 6 | Enqueue, dedup, retry, dead-letter |
| 3 | Timeout enforcement | 4 | Step timeout, total timeout, partial results |
| 3 | Skip-if-found | 3 | Cross-group skip, no-skip when empty |
| 4 | DB dependency counts | 3 | Correctness, edge cases, fallback |
| 5 | Bounded state | 5 | LRU eviction, TTL expiry, concurrent access |
| 5 | Token storage | 3 | ID storage, batch decode, encoder reuse |
| 6 | Metrics | 3 | Histogram labels, timeout counter, fact hit rate |

### Integration Tests

```python
# test_write_read_consistency.py
async def test_preference_write_then_read():
    """End-to-end: write preference → immediate fact retrieval."""
    await orchestrator.write(tenant_id="t1", content="I love sushi")
    packet = await orchestrator.read(tenant_id="t1", query="What food do I like?")
    assert any("sushi" in m.record.text for m in packet.all_memories)

async def test_batch_embedding_correctness():
    """Batch embeddings produce same results as individual calls."""
    text1, text2 = "Hello world", "Goodbye world"
    single_1 = await client.embed(text1)
    single_2 = await client.embed(text2)
    batch = await client.embed_batch([text1, text2])
    assert batch[0].embedding == single_1.embedding
    assert batch[1].embedding == single_2.embedding

async def test_retrieval_timeout_graceful():
    """Slow graph doesn't block fast vector results."""
    # Insert test data
    # Mock neo4j to be slow
    # Assert: retrieval completes within 2s with vector results
```

### Performance Regression Tests

```python
# test_performance.py (pytest-benchmark)
def test_encode_batch_latency(benchmark, mock_embeddings):
    """encode_batch should complete within 200ms for 10 chunks."""
    chunks = [make_chunk(f"text {i}") for i in range(10)]
    result = benchmark.pedantic(
        lambda: asyncio.run(store.encode_batch("t1", chunks)),
        iterations=10,
    )
    assert benchmark.stats["mean"] < 0.2  # 200ms

def test_forgetting_dependency_counts(benchmark, populated_store):
    """Dependency counting should complete within 500ms for 5000 memories."""
    memories = asyncio.run(populated_store.scan("t1", limit=5000))
    result = benchmark.pedantic(
        lambda: asyncio.run(worker._get_dependency_counts("t1", "u1", memories)),
        iterations=5,
    )
    assert benchmark.stats["mean"] < 0.5  # 500ms
```

---

## 7. Migration & Rollback Plan

### Phase 1 Migrations

#### Fact Key Migration Script

```python
"""One-time migration: re-key semantic facts with stable hashes."""

import hashlib
from sqlalchemy import select, update

async def migrate_fact_keys(session):
    """Migrate user:custom:{hash()%10000} keys to user:custom:{sha256[:16]}."""
    
    # Find all custom-keyed facts
    facts = await session.execute(
        select(SemanticFactModel).where(
            SemanticFactModel.key.like("user:custom:%")
        )
    )
    
    migrated = 0
    for fact in facts.scalars():
        old_key = fact.key
        # Derive new stable key from the fact's value text
        value_text = str(fact.value) if fact.value else ""
        new_key = f"user:custom:{hashlib.sha256(value_text.encode()).hexdigest()[:16]}"
        
        if old_key != new_key:
            # Check if new key already exists (merge if so)
            existing = await session.execute(
                select(SemanticFactModel).where(
                    SemanticFactModel.tenant_id == fact.tenant_id,
                    SemanticFactModel.key == new_key,
                    SemanticFactModel.is_current == True,
                )
            )
            existing_fact = existing.scalar_one_or_none()
            
            if existing_fact:
                # Merge: keep higher-confidence version
                if fact.confidence > existing_fact.confidence:
                    existing_fact.is_current = False
                    fact.key = new_key
                else:
                    fact.is_current = False
            else:
                fact.key = new_key
            
            migrated += 1
    
    await session.commit()
    return migrated
```

#### Hippocampal Key Migration

```python
"""Re-key episodic records with stable content-based keys."""

async def migrate_episodic_keys(session, tenant_id: str):
    """Migrate preference:entity keys to preference:entity:sha256."""
    
    records = await session.execute(
        select(MemoryRecordModel).where(
            MemoryRecordModel.tenant_id == tenant_id,
            MemoryRecordModel.key.isnot(None),
            MemoryRecordModel.type.in_(["preference", "semantic_fact"]),
        )
    )
    
    migrated = 0
    for record in records.scalars():
        old_key = record.key
        text_normalized = record.text.strip().lower()
        content_hash = hashlib.sha256(text_normalized.encode()).hexdigest()[:16]
        
        # Preserve entity prefix for readability
        parts = old_key.split(":", 1)
        if len(parts) == 2:
            mem_type = parts[0]
            entity = parts[1]
            new_key = f"{mem_type}:{entity}:{content_hash}"
        else:
            new_key = f"{old_key}:{content_hash}"
        
        record.key = new_key
        migrated += 1
    
    await session.commit()
    return migrated
```

### Rollback Strategy

Each phase has an independent rollback path:

| Phase | Rollback Mechanism | Data Impact |
|-------|-------------------|-------------|
| 1 (Keys) | Feature flag `STABLE_KEYS_ENABLED=false` reverts to old key generation | Old keys remain readable; new keys orphaned but harmless |
| 1 (Write-time facts) | Feature flag `WRITE_TIME_FACTS_ENABLED=false` | Semantic store stops getting new writes; existing facts remain valid |
| 2 (Batch embed) | Revert `encode_batch()` to sequential loop | No data impact; only performance regression |
| 2 (Async storage) | `STORE_ASYNC=false` (existing placeholder) | Reverts to synchronous writes |
| 3 (Timeouts) | Set timeout values very high (e.g., 999999ms) | Effectively disables timeouts |
| 4 (DB dep counts) | Method falls back to Python loop if DB method unavailable | Performance regression only |
| 5 (Bounded state) | Set `max_size=999999, ttl_seconds=999999` | Effectively unbounded (same as before) |

### Feature Flags Summary

```python
# src/core/config.py — new feature flags
class FeatureFlags(BaseModel):
    stable_keys_enabled: bool = True          # Phase 1.1-1.2
    write_time_facts_enabled: bool = True     # Phase 1.3
    batch_embeddings_enabled: bool = True     # Phase 2.1
    store_async: bool = False                 # Phase 2.2 (opt-in)
    cached_embeddings_enabled: bool = True    # Phase 2.3
    retrieval_timeouts_enabled: bool = True   # Phase 3.1
    skip_if_found_cross_group: bool = True    # Phase 3.2
    db_dependency_counts: bool = True         # Phase 4.1
    bounded_state_enabled: bool = True        # Phase 5.2
    hnsw_ef_search_tuning: bool = True        # Phase 6.1
```

---

## 8. Acceptance Criteria & SLOs

### Functional Acceptance

| Criterion | Test Method | Target |
|-----------|-------------|--------|
| Fact key determinism | Unit test across 100 runs | 100% identical keys |
| Write-time fact availability | Integration test: write → immediate read | Facts appear in <100ms |
| No accidental overwrites | Unit test with shared-entity facts | 0 overwrites |
| Timeout enforcement | Integration test with slow mock | p99 < total_timeout_ms |
| Batch embedding correctness | Comparison test batch vs sequential | Identical embeddings |

### Performance SLOs

| Endpoint | Metric | Current (est.) | Target | Method |
|----------|--------|---------------|--------|--------|
| `/memory/read` | p95 latency | 300-800ms | <200ms | Benchmark |
| `/memory/write` | p95 latency | 400-1000ms | <150ms | Benchmark |
| `/memory/turn` | p95 latency | 800-3000ms | <300ms (async) | Benchmark |
| Forgetting (5k memories) | Total runtime | ~30s | <1s | Benchmark |
| Fact hit rate (pref/identity) | % from semantic store | ~0% (cold) | >80% | Metric |

### Quality Gates

| Gate | Measurement | Threshold |
|------|-------------|-----------|
| Duplicate semantic facts | Count per tenant per day | <5 |
| Key collisions | SHA256 collision rate | 0 (theoretical) |
| Retrieval timeouts | % of steps that timeout | <5% |
| Embedding cache hit rate | Redis cache hits / total | >60% after warmup |
| Working memory RSS | Process RSS growth over 1hr | <50MB |

---

## Appendix: File Change Summary

| File | Phase | Change Type | Description |
|------|-------|-------------|-------------|
| `src/consolidation/migrator.py` | 1.1 | Modify | Stable fact key hashing |
| `src/memory/hippocampal/store.py` | 1.2, 2.1 | Modify | Key fix + batch embeddings |
| `src/extraction/write_time_facts.py` | 1.3 | **New** | Write-time fact extraction |
| `src/memory/seamless_provider.py` | 2.2 | Modify | Async storage integration |
| `src/storage/async_pipeline.py` | 2.2 | **New** | Async storage queue |
| `src/memory/orchestrator.py` | 2.3 | Modify | CachedEmbeddings activation |
| `src/utils/embeddings.py` | 2.3 | Modify | MGET batch cache lookup |
| `src/retrieval/retriever.py` | 3.1, 3.2, 6.2, 6.3 | Modify | Timeouts + skip + metrics |
| `src/retrieval/planner.py` | 3.3 | Modify | Timezone-aware time filters |
| `src/forgetting/worker.py` | 4.1 | Modify | DB dependency counts |
| `src/storage/postgres.py` | 4.1, 6.1 | Modify | Bulk counts + ef_search |
| `src/memory/neocortical/store.py` | 4.2 | Modify | Batch graph queries |
| `src/memory/sensory/buffer.py` | 5.1 | Modify | Token ID storage |
| `src/memory/working/manager.py` | 5.2 | Modify | BoundedStateMap |
| `src/utils/bounded_state.py` | 5.2 | **New** | LRU + TTL state manager |
| `src/core/config.py` | All | Modify | Feature flags + config |
| `src/utils/metrics.py` | 6.2, 6.3 | Modify | New Prometheus metrics |

### New Files (4)
1. `src/extraction/write_time_facts.py` — Write-time fact extractor
2. `src/storage/async_pipeline.py` — Async storage queue
3. `src/utils/bounded_state.py` — LRU + TTL state manager
4. `migrations/versions/xxx_add_gin_index_metadata.py` — GIN index for JSON evidence_refs

### Modified Files (14)
All existing files preserve backward compatibility via feature flags.

---

## Appendix: Phase Dependency Graph

```
Phase 1 (Correctness) ──────────────────► Phase 2 (Hot Path)
   │                                         │
   │  P0: Keys, Write-time facts             │  P0: Batch embed, Async store, Cache
   │                                         │
   └──────────► Phase 3 (Retrieval) ◄────────┘
                   │
                   │  P1: Timeouts, Skip-if-found, Timezone
                   │
                   ├──► Phase 4 (Background)
                   │       │
                   │       │  P1: DB dep counts, Batch graph
                   │       │
                   └──► Phase 5 (Stability)
                           │
                           │  P2: Tokens, Bounded state
                           │
                           └──► Phase 6 (Observability)
                                   │
                                   │  P2: HNSW tuning, Metrics
                                   │
                                   └──► [DONE]
```

**Estimated Total Timeline: 8-10 weeks**
- Phase 1: Weeks 1-2
- Phase 2: Weeks 2-4 (overlaps with Phase 1 end)
- Phase 3: Weeks 3-5 (can start parallel to Phase 2)
- Phase 4: Weeks 5-6
- Phase 5: Weeks 6-7
- Phase 6: Weeks 7-8
- Buffer + integration testing: Weeks 8-10
