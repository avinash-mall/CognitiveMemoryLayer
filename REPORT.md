# CML Codebase Audit Report

> **Status**: Updated after code fixes. Issues marked **(FIXED IN THIS PR)** have been addressed.
> Issues marked **(ALREADY IN CODEBASE)** were found to be handled by the modelpack/NER architecture.

---

## 1. Architecture Map

### 1.1 Data Flow

```
WRITE PATH:
  POST /api/v1/memory/write → MemoryOrchestrator.write()
    → ShortTermMemory.ingest() → SensoryBuffer → SemchunkChunker
    → WriteGate.evaluate() → PIIRedactor.redact()
    → [modelpack/NER or LLM] extraction → entities, relations, constraints, facts
    → EmbeddingClient.embed_batch() → PostgresMemoryStore.upsert() + Neo4j + Redis

READ PATH:
  POST /api/v1/memory/read → MemoryOrchestrator.read()
    → EmbeddingClient.embed() (query embedding)
    → QueryClassifier.classify() (modelpack or LLM)
    → RetrievalPlanner.plan() → parallel step groups
    → HybridRetriever.retrieve()
      → VECTOR: pgvector cosine search
      → FACTS: semantic fact store lookup
      → CONSTRAINTS: vector search (type=constraint) + cognitive fact lookup + domain rescoring
      → GRAPH: Neo4j PPR
      → Associative expansion (co-session constraints)
    → MemoryReranker.rerank() (pre-computed word sets, MMR diversity)
    → MemoryPacketBuilder (constraints-first, natural reminder framing)
```

### 1.2 Key Modules

| Module | Responsibility |
|--------|---------------|
| `src/api/` | HTTP endpoints, auth, rate limiting |
| `src/memory/hippocampal/` | Episodic encoding, write gate, PII |
| `src/memory/neocortical/` | Semantic facts, schema management |
| `src/extraction/` | Entity/relation/constraint extraction (modelpack + NER + LLM) |
| `src/retrieval/` | Classification, planning, hybrid retrieval, reranking, packet assembly |
| `src/consolidation/` | Episode-to-fact migration with constraint preservation |
| `src/forgetting/` | Active forgetting with relevance scoring |
| `src/reconsolidation/` | Belief revision, conflict detection, supersession |

---

## 2. Issue Register

### 2.1 Conceptual / Architectural

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| C-01 | CRITICAL | Semantic disconnect: vector search misses constraints when query is semantically distant | **(ALREADY IN CODEBASE)** — `_retrieve_constraints()` does flat cognitive fact lookup + `_rescore_constraints()` with modelpack domain bonus |
| C-02 | HIGH | Constraint dilution: all constraints returned with flat relevance | **(ALREADY IN CODEBASE)** — `_rescore_constraints()` applies domain bonus + modelpack `constraint_rerank` and `scope_match` signals |
| C-03 | MEDIUM | No constraint dimension detection in fast-path classifier | **(ALREADY IN CODEBASE)** — `_enrich_with_modelpack()` detects `constraint_dimension` |
| C-04 | MEDIUM | Recency bias penalises old active constraints | **(ALREADY IN CODEBASE)** — `_get_recency_weight()` uses modelpack `constraint_stability` signal |
| C-05 | HIGH | Stale constraint: supersession only at write time | **(ALREADY IN CODEBASE)** — `_resolve_constraint_conflicts()` in packet builder uses modelpack `supersession` at retrieval time |
| C-06 | HIGH | Consolidation gist loses constraint semantics | **(FIXED IN THIS PR)** — Added `source_types` to gist prompt + schema aligner already forces constraint type |

### 2.2 Efficiency

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| E-01 | MEDIUM | O(n²) text similarity in reranker | **(FIXED IN THIS PR)** — Pre-computed `frozenset` word sets passed through scoring pipeline |
| E-03 | MEDIUM | 5 sequential DB queries for cognitive fact categories | **(ALREADY IN CODEBASE)** — `get_facts_by_categories()` batch method with fallback |

### 2.3 Logical

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| L-04 | HIGH | Constraint scope always empty | **(ALREADY IN CODEBASE)** — `_extract_scope()` uses modelpack + NER |

### 2.4 Context Assembly

| ID | Severity | Issue | Status |
|----|----------|-------|--------|
| CA-01 | MEDIUM | Context uses `[!IMPORTANT] **...**` instead of natural reminders | **(FIXED IN THIS PR)** — Split into "Must Follow" (value/policy) vs "Consider" (state/goal) with `Earlier you said: "..."` framing |

---

## 3. Failure-Mode Analysis Summary

| Failure Class | Root Cause | Mitigation |
|---------------|-----------|------------|
| Semantic disconnect | Vector similarity misses logically relevant constraints | Flat cognitive fact lookup + modelpack domain rescoring |
| Constraint dilution | Too many constraints with equal relevance | Modelpack `constraint_rerank` + `scope_match` scoring |
| Wrong constraint type | No type-aware filtering | Modelpack `constraint_dimension` detection in classifier |
| Temporal/recency bias | Age-based decay penalises old constraints | Modelpack `constraint_stability` signal reduces recency weight |
| Stale constraints | Old preferences persist after update | Retrieval-time `_resolve_constraint_conflicts()` with modelpack `supersession` |
| Inconsistent consolidation | Gist loses constraint framing | **(FIXED)** Source types in prompt + schema aligner override |

---

## 4. Changes Made in This PR

### Code Fixes
1. **Reranker efficiency** (`src/retrieval/reranker.py`): Pre-compute `frozenset` word sets once per memory, pass through `_calculate_score()` and `_apply_diversity()` to avoid O(n²) recomputation.
2. **Context assembly** (`src/retrieval/packet_builder.py`): Split constraints into "Constraints (Must Follow)" for value/policy types and "Other Constraints to Consider" for state/goal/causal, using natural reminder framing (`Earlier you said: "..."` / `You also mentioned: "..."`).
3. **Consolidation preservation** (`src/consolidation/summarizer.py`): Added `{source_types}` placeholder to gist extraction prompt so LLM preserves constraint semantics when source memories include constraint types.

### Test Updates
- Updated 5 packet builder test assertions to match new context assembly format.
- All 569 unit + integration tests pass (3 skipped for env-specific reasons).
