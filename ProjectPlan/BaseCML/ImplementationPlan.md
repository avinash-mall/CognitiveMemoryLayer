# Implementation Plan: Deep Research Report Fixes

**Scope:** Resolve all issues identified in `ProjectPlan/BaseCML/deep-research-report.md`.  
**Constraints:** No fallbacks or compatibility with older code. LLM calls use internal LLM only (see § LLM Configuration).  
**Basis:** Codebase review (`src/`, `tests/`) — not documentation alone.

---

## 1. LLM Configuration

All internal LLM usage (QueryClassifier, Entity/Relation extraction, GistExtractor, Consolidation, Reconsolidation, Forgetting) **must** use the dedicated internal LLM. No fallback to main `LLM__*` when internal is configured.

**Required .env (from `.env` lines 22–24):**

```env
LLM_INTERNAL__PROVIDER=ollama
LLM_INTERNAL__MODEL=llama3.2:3b
LLM_INTERNAL__BASE_URL=http://host.docker.internal:11434/v1
LLM_INTERNAL__API_KEY=
```

**Code touchpoints:**

- `src/utils/llm.py`: `get_internal_llm_client()` — when `LLM_INTERNAL__MODEL` is set, use only `LLM_INTERNAL__*`; do not fall back to `get_llm_client()` for any internal task.
- `src/memory/orchestrator.py`: Uses `get_internal_llm_client()` for ShortTermMemory, HippocampalStore (entity/relation), MemoryRetriever (classifier), ReconsolidationService, ConsolidationWorker, ForgettingWorker. No code path may use the primary LLM for these.

**Validation:** Unit test that, when `LLM_INTERNAL__*` is set, `get_internal_llm_client()` returns a client whose `base_url` and `model` match env (no fallback to LLM__*).

---

## 2. Architecture Overview

### 2.1 Current data flow (from codebase)

- **Write:** `FastAPI` → `MemoryOrchestrator.write()` → `ShortTermMemory.ingest_turn()` → chunks → `HippocampalStore.encode_batch()` → Postgres (pgvector) + Neo4j sync + `SemanticFactStore` (write-time facts + constraints via `ConstraintExtractor.constraint_fact_key()`).
- **Retrieval:** `MemoryOrchestrator.read()` → `MemoryRetriever.retrieve()` → `QueryClassifier.classify()` → `RetrievalPlanner.plan()` → `HybridRetriever.retrieve()` (steps: FACTS, VECTOR, GRAPH, CONSTRAINTS, CACHE) → `MemoryReranker.rerank()` → `MemoryPacketBuilder.build()` → `MemoryPacket`.
- **Constraint retrieval:** `HybridRetriever._retrieve_constraints()` does (1) vector search with `type=CONSTRAINT`, (2) fact lookup for all cognitive categories (GOAL, VALUE, STATE, CAUSAL, POLICY) unfiltered by query intent.
- **Consolidation:** `ConsolidationWorker.consolidate()` → `EpisodeSampler.sample()` → `SemanticClusterer.cluster()` → `GistExtractor.extract_from_clusters()` → `SchemaAligner.align_batch()` → `ConsolidationMigrator.migrate()`.

### 2.2 Target architecture (after fixes)

- **Constraint retrieval:** Always consider constraints for decision-like queries; constraint steps tagged by **constraint type** (goal/value/state/etc.); fact lookup filtered by `FactCategory` matching classified intent; episodic CONSTRAINT records carry `metadata.constraint_type` and are superseded when a new constraint of same key is stored.
- **Context assembly:** Active constraints first, hard budget favouring constraints; episodes filtered by relevance threshold; no “dump” of up-to-5/6 items regardless of relevance.
- **Stale constraints:** On storing a new constraint fact (orchestrator write path), deactivate older hippocampal CONSTRAINT records with same logical key (or scope); no legacy episodic “I like tea” surfacing after “I like coffee”.
- **Efficiency:** Bulk `increment_access_counts` only (no per-record update fallback); configurable reranker weights; consolidation re-extracts constraints from gists and writes them to fact store.

---

## 3. Issue-by-issue design and pseudo-code

### C1 – Semantic disconnect (constraints missed)

**Problem:** Queries that don’t trigger CONSTRAINT_CHECK or constraint step never run `_retrieve_constraints`. Pure vector similarity can miss latent constraints (e.g. “save money” vs “budget-friendly vacation”).

**Fix:**

1. **Always include a constraint step when the query is decision-like or has constraint dimensions.**  
   - In `RetrievalPlanner.plan()`: if `analysis.is_decision_query` or `analysis.constraint_dimensions` or `analysis.intent == QueryIntent.CONSTRAINT_CHECK`, the plan must include a `RetrievalStep(source=CONSTRAINTS, ...)`.  
   - Do not rely only on intent enum; use `constraint_dimensions` and `is_decision_query` from `QueryAnalysis` (already set by `QueryClassifier._enrich_constraint_dimensions()`).
2. **Constraint step carries desired category filter.**  
   - Add to `RetrievalStep`: `constraint_categories: list[str] | None` (e.g. `["goal", "value"]`).  
   - Planner sets it from `analysis.constraint_dimensions` when present; otherwise leave `None` (retrieve all constraint types).
3. **Associative expansion (optional but recommended):** After initial vector + constraint retrieval, optionally run a second step: for entities in the query, fetch facts/constraints by entity (e.g. `get_facts_by_category` + filter by subject/scope). Implement as an optional “constraint_by_entity” step or extend CONSTRAINTS step to accept `entity_seeds` and merge results from fact store by subject.

**Pseudo-code (planner):**

```text
function plan(analysis):
  steps = []
  if analysis.intent in (PREFERENCE_LOOKUP, IDENTITY_LOOKUP, TASK_STATUS):
    steps.append(create_fact_lookup_step(analysis))
    steps.append(RetrievalStep(VECTOR, query=analysis.original_query, top_k=5, skip_if_found=True))
    parallel_groups = [[0],[1]]
  elif analysis.intent == MULTI_HOP:
    ...
  elif analysis.intent == CONSTRAINT_CHECK or analysis.is_decision_query or analysis.constraint_dimensions:
    # Constraints-first; pass categories so retriever can filter
    steps.append(RetrievalStep(
      source=CONSTRAINTS,
      query=analysis.original_query,
      constraint_categories=analysis.constraint_dimensions if analysis.constraint_dimensions else None,
      memory_types=["constraint"], top_k=10, priority=0, timeout_ms=200))
    steps.append(RetrievalStep(VECTOR, query=analysis.original_query, top_k=analysis.suggested_top_k, priority=1))
    steps.append(RetrievalStep(FACTS, query=analysis.original_query, top_k=5, priority=1))
    parallel_groups = [[0,1,2]]
  else:
    # General path: still add a lightweight constraint step so latent constraints are not missed
    steps.append(RetrievalStep(CONSTRAINTS, query=analysis.original_query, constraint_categories=None, top_k=5, priority=1))
    steps.append(RetrievalStep(VECTOR, ...))
    steps.append(RetrievalStep(FACTS, ...))
    ...
  return RetrievalPlan(..., steps=steps, parallel_steps=parallel_groups)
```

**Pseudo-code (classifier):**  
- Keep existing `_enrich_constraint_dimensions` and decision-query upgrade.  
- Ensure vague queries that get `effective_query = recent_context + "User now asks: " + query` still run through constraint-dimension detection so that `constraint_dimensions` is set when the context implies goals/values.

**Files:** `src/retrieval/planner.py`, `src/retrieval/query_types.py` (add `constraint_categories` to step if not already on QueryAnalysis), `src/retrieval/retriever.py` (`_retrieve_constraints` to use step’s constraint_categories).

---

### C2 – Constraint dilution (irrelevant context)

**Problem:** `MemoryPacketBuilder._format_markdown` lists up to 5 facts, 5 episodes, 6 constraints without relevance filtering; key constraint can be buried.

**Fix:**

1. **Hard budget: constraints first.**  
   - In `MemoryPacketBuilder.build()`: sort/partition so that constraints are always placed in `packet.constraints` and appear first in the formatted output.  
   - In `_format_markdown`: emit “Active Constraints” first; cap episodes by a **relevance threshold** (e.g. only include episodes with `relevance_score >= min_episode_relevance`, default 0.4).
2. **Configurable limits.**  
   - Add to `MemoryPacketBuilder` (or a small config dataclass): `max_constraints=6`, `max_episodes=5`, `min_episode_relevance=0.4`. When building, filter `recent_episodes` by `min_episode_relevance` before slicing to `max_episodes`.
3. **Always include top-1 constraint if any.**  
   - In formatting, if `packet.constraints` is non-empty, always include at least the top constraint (by relevance/confidence) even if over a token budget elsewhere; truncate episodes more aggressively.

**Pseudo-code:**

```text
function build(memories, query):
  packet = MemoryPacket(query=query)
  for mem in memories:
    if mem.record.type == CONSTRAINT: packet.constraints.append(mem)
    elif ...: packet.facts.append(mem)
    ...
  # Enforce relevance threshold for episodes
  packet.recent_episodes = [e for e in packet.recent_episodes if e.relevance_score >= config.min_episode_relevance]
  packet.recent_episodes = packet.recent_episodes[:config.max_episodes]
  packet.constraints = packet.constraints[:config.max_constraints]
  ...

function _format_markdown(packet, max_tokens):
  lines = ["# Retrieved Memory Context\n"]
  if packet.constraints:
    lines.append("## Active Constraints (Must Follow)")
    for c in packet.constraints[:max_constraints]:
      lines.append("- **" + c.record.text + "** " + _constraint_provenance(c))
  # Then facts, preferences, then episodes (already filtered by relevance)
  ...
```

**Files:** `src/retrieval/packet_builder.py` (add config, filter episodes by relevance, keep constraints first).

---

### C3 – Wrong constraint type (goal vs value vs fact)

**Problem:** `_retrieve_constraints` returns all cognitive categories; no filtering by query intent, so “goal” queries can get “value” or generic facts.

**Fix:**

1. **RetrievalStep carries `constraint_categories`.**  
   - As in C1: `RetrievalStep.constraint_categories: list[str] | None` (e.g. `["goal"]` or `["value","policy"]`).  
   - If present, `_retrieve_constraints` restricts semantic fact lookup to those `FactCategory` values only (goal, value, state, causal, policy map 1:1 to constraint_type).
2. **Episodic constraint records have `constraint_type` in metadata.**  
   - Already done in `HippocampalStore.encode_chunk` / `encode_batch`: `metadata["constraints"]` contains list of constraint dicts with `constraint_type`.  
   - When building constraint items from vector search, set `type` or a field so that packet_builder/reranker can treat them as constraint and optionally filter by type.  
   - In `_retrieve_constraints`, when building results from vector search, attach `constraint_type` from `record.metadata["constraints"][0]["constraint_type"]` if present.  
   - In `HybridRetriever._retrieve_constraints`: when `step.constraint_categories` is set, filter both (1) vector results by `metadata.constraints[].constraint_type in step.constraint_categories`, and (2) fact lookup: only call `get_facts_by_category(tenant_id, category)` for categories in `step.constraint_categories` (map string to FactCategory).
3. **Formatting:** In `_constraint_provenance` and markdown, show label `[Goal]` / `[Value]` from `constraint_type` so the LLM sees the correct type.

**Pseudo-code (_retrieve_constraints):**

```text
function _retrieve_constraints(tenant_id, step, context_filter):
  results = []
  categories = step.constraint_categories  # e.g. ["goal","value"] or None
  if categories is None: categories = ["goal","value","state","causal","policy"]

  # 1. Vector search for episodic CONSTRAINT records
  constraint_filters = { type: [CONSTRAINT], status: "active" }
  records = await hippocampal.search(tenant_id, query=step.query, top_k=step.top_k, filters=constraint_filters)
  for r in records:
    meta = r.metadata.get("constraints") or []
    ctype = meta[0]["constraint_type"] if meta else ""
    if categories and ctype not in categories: continue
    results.append({ type: CONSTRAINT, text: r.text, record: r, relevance: r.metadata.get("_similarity",0.7), constraint_type: ctype })

  # 2. Fact store by category (only requested categories)
  for cat in categories:
    fact_cat = FactCategory(cat)  # string to enum
    facts = await neocortical.facts.get_facts_by_category(tenant_id, fact_cat, current_only=True)
    for fact in facts:
    results.append({ type: CONSTRAINT, text: "["+cat+"] "+fact.value, record: fact, relevance: 0.75, constraint_type: cat })
  return results[:step.top_k]
```

**Files:** `src/retrieval/retriever.py`, `src/retrieval/planner.py`, `src/retrieval/query_types.py` (or planner dataclass for step).

---

### L1 – Stale constraints (old episodic constraints not superseded)

**Problem:** When user says “I like coffee” after “I like tea”, the semantic fact store supersedes (existing `_update_fact` deactivates old row), but episodic CONSTRAINT records in Postgres remain active and can still appear in vector search.

**Fix:**

1. **On write path (orchestrator):** After storing new constraint facts (loop over `constraint_extractor.extract()` and `neocortical.store_fact(..., key=ConstraintExtractor.constraint_fact_key(constraint))`), for each stored constraint fact key, find active hippocampal records that (a) have `type=CONSTRAINT` and (b) have the same logical key. Logical key for an episodic record: from `record.key` if set, or derive from `metadata.constraints[0]` using `ConstraintExtractor.constraint_fact_key(ConstraintObject(...))` if we can reconstruct it; otherwise use a stable key from constraint type + scope hash stored in metadata.
2. **Deactivate old episodic constraints.**  
   - Add to `PostgresMemoryStore`: `async def deactivate_constraints_by_key(tenant_id: str, constraint_fact_key: str) -> int` which updates `status=SILENT` (or soft-delete) for records where `type=CONSTRAINT` and `key == constraint_fact_key` (and optionally `status=ACTIVE`).  
   - Orchestrator after storing a constraint fact: call `hippocampal.store.deactivate_constraints_by_key(tenant_id, fact_key)` (need to expose store from orchestrator’s hippocampal; currently `self.hippocampal.store` is the Postgres store).  
   - No fallback: if the store does not implement this method, require it for the full write path (or add to base interface as optional and only call when available). Per “no fallbacks”, we add the method to PostgresMemoryStore and call it from orchestrator; other stores (e.g. NoOp) can implement no-op.
3. **Optional: store constraint key on episodic record.**  
   - When encoding a constraint chunk, set `record.key = ConstraintExtractor.constraint_fact_key(constraint)` so that deactivation can match by key. Already partially there: `_generate_key` returns a key for CONSTRAINT type. Ensure the key written to the record matches what the fact store uses (orchestrator uses `constraint_fact_key(constraint)` for the fact key). Align key generation in `HippocampalStore._generate_key` for CONSTRAINT with `ConstraintExtractor.constraint_fact_key` so that the same key is used in both places (currently _generate_key uses `memory_type.value` + entity_prefix + content_hash; constraint_fact_key uses `user:{constraint_type}:{scope_hash}`. We need episodic constraint records to store the same key as the fact — so when encoding a constraint chunk, pass the fact key into the record. So: in `encode_chunk`/`encode_batch`, when memory_type is CONSTRAINT and we have extracted_constraints, set key = ConstraintExtractor.constraint_fact_key(extracted_constraints[0]) (or first high-confidence one). Then deactivate_constraints_by_key(tenant_id, key) will match.

**Pseudo-code (orchestrator after constraint store):**

```text
for chunk in chunks_for_encoding:
  for constraint in constraint_extractor.extract(chunk):
    fact_key = ConstraintExtractor.constraint_fact_key(constraint)
    await neocortical.store_fact(tenant_id, key=fact_key, value=constraint.description, ...)
    # Deactivate any prior episodic constraint with same key
    if hasattr(hippocampal.store, "deactivate_constraints_by_key"):
      await hippocampal.store.deactivate_constraints_by_key(tenant_id, fact_key)
```

**Pseudo-code (PostgresMemoryStore):**

```text
async def deactivate_constraints_by_key(tenant_id: str, constraint_fact_key: str) -> int:
  stmt = update(MemoryRecordModel).where(
    and_(
      MemoryRecordModel.tenant_id == tenant_id,
      MemoryRecordModel.type == MemoryType.CONSTRAINT.value,
      MemoryRecordModel.key == constraint_fact_key,
      MemoryRecordModel.status == MemoryStatus.ACTIVE.value,
    )
  ).values(status=MemoryStatus.SILENT.value)
  r = await session.execute(stmt)
  await session.commit()
  return r.rowcount
```

**Files:** `src/storage/postgres.py`, `src/memory/orchestrator.py`, `src/memory/hippocampal/store.py` (ensure key set from constraint_fact_key when type=CONSTRAINT).

---

### L2 – Query classifier coverage

**Problem:** If classifier misclassifies or is missing, constraint step may be omitted. Report asks to validate all intents and ensure CONSTRAINT_CHECK leads to constraint retrieval.

**Fix:**

1. **Tests:** Add unit tests for `QueryClassifier`: for each `QueryIntent` (including CONSTRAINT_CHECK), provide a representative query and assert (a) intent (or fallback to GENERAL_QUESTION/UNKNOWN), (b) for CONSTRAINT_CHECK and decision patterns, `suggested_sources` includes `"constraints"`.  
2. **Planner tests:** For `QueryAnalysis(intent=CONSTRAINT_CHECK, ...)` and for `QueryAnalysis(intent=GENERAL_QUESTION, is_decision_query=True)`, assert that the plan contains a step with `source=CONSTRAINTS`.  
3. **No fallback to “no classifier”:** Internal LLM is required for classification when query is not matched by fast patterns. Ensure `get_internal_llm_client()` is used and that when LLM fails, the code does not silently skip constraint step — e.g. on LLM failure, classifier returns a safe default that includes constraints for safety (e.g. GENERAL_QUESTION with suggested_sources including "constraints"). Per “no fallbacks”, we can either require LLM and fail loud, or define a single safe default (include constraints in suggested_sources) and document it.

**Files:** `tests/unit/test_classifier.py` (or new), `tests/unit/test_planner.py` (or new), `src/retrieval/classifier.py` (optional safe default for LLM failure).

---

### L3 – Recency weight tuning

**Problem:** Episodes get recency_weight=0.2; older stable constraints can be outranked by recent chitchat.

**Fix:**

1. **Expose reranker weights in config.**  
   - Add `RerankerSettings` (or reuse a nested structure) in `src/core/config.py`: `relevance_weight`, `recency_weight`, `confidence_weight`, `diversity_weight`, `recency_weight_constraint_stable`, `recency_weight_constraint_semi_stable` (defaults: 0.1, 0.05 for stable/semi-stable).  
   - Load from env e.g. `RETRIEVAL__RERANKER_RECENCY_WEIGHT=0.1`.  
   - `MemoryReranker` takes config from settings and uses it in `_calculate_score` and `_get_recency_weight`.  
2. **Lower default recency for episodes.**  
   - Default `recency_weight=0.1` (was 0.2). Keep constraint-specific weights (0.05 stable, 0.15 semi-stable) configurable.  
3. **Optional: time-window filter for episodes.**  
   - In planner, for non-temporal queries, add a default `time_filter` for VECTOR step (e.g. last 7 days) so very old episodes don’t dominate; apply only to episode type, not to constraints (constraint step already filters by type). This can be a separate step or a filter in `HippocampalStore.search` when step has `memory_types=episodic_event`.

**Pseudo-code (config):**

```text
class RerankerSettings:
  relevance_weight: float = 0.5
  recency_weight: float = 0.1
  confidence_weight: float = 0.2
  diversity_weight: float = 0.1
  recency_weight_constraint_stable: float = 0.05
  recency_weight_constraint_semi_stable: float = 0.15
```

**Files:** `src/core/config.py`, `src/retrieval/reranker.py`.

---

### L4 – Inconsistent consolidation (constraints lost in gist)

**Problem:** GistSummarizer may paraphrase constraints weakly; constraint phrases can vanish.

**Fix:**

1. **After gist extraction, re-extract constraints and ensure they are in the fact store.**  
   - In `ConsolidationWorker.consolidate()`, after `extractor.extract_from_clusters(clusters)` → gists, run a constraint-extraction pass over the gist texts (or over the concatenated cluster summaries). Use `ConstraintExtractor.extract()` on each gist text (treat as a single “chunk” with text = gist.text).  
   - For each extracted constraint, call `neocortical.store_fact(tenant_id, key=ConstraintExtractor.constraint_fact_key(constraint), value=constraint.description, confidence=..., evidence_ids=gist.supporting_episode_ids)`.  
   - This ensures that if the LLM gist says “User prefers vegetarian food”, the constraint extractor can map it to a structured constraint and store it with the same key scheme as write-time, so retrieval and supersession stay consistent.  
2. **GistExtractor prompt:** Already mentions goal/value/state/causal/policy in the prompt; no change required for “no fallbacks”, but ensure the returned type is one of fact, preference, goal, value, state, causal, policy so that SchemaAligner and fact store category align.

**Pseudo-code (ConsolidationWorker.consolidate):**

```text
gists = await self.extractor.extract_from_clusters(clusters)
# Re-extract constraints from gist text and persist to fact store
constraint_extractor = ConstraintExtractor()
for gist in gists:
  chunk = SemanticChunk(text=gist.text, ...)  # minimal chunk for extraction
  for constraint in constraint_extractor.extract(chunk):
    key = ConstraintExtractor.constraint_fact_key(constraint)
    await neocortical_store.store_fact(tenant_id, key=key, value=constraint.description, confidence=gist.confidence, evidence_ids=gist.supporting_episode_ids)
alignments = await self.aligner.align_batch(...)
migration = await self.migrator.migrate(...)
```

**Files:** `src/consolidation/worker.py`, `src/extraction/constraint_extractor.py` (already has extract on chunk; ensure SemanticChunk can be built from gist text).

---

### E1 – Retrieval performance (bulk access count, no N+1)

**Problem:** `HippocampalStore.search()` updates access counts: if the store has `increment_access_counts`, it uses it; else it does a per-record `store.update()` in a loop (N round-trips).

**Fix:**

1. **Require bulk increment.**  
   - Add `increment_access_counts(record_ids, last_accessed_at)` to `MemoryStoreBase` as a non-abstract method with default implementation that does nothing (or make it abstract and implement in Postgres and NoOp).  
   - In `HippocampalStore.search`, **always** call `self.store.increment_access_counts([r.id for r in results], now)` and **remove** the else branch that does `asyncio.gather(*[self.store.update(record.id, {...}) for record in results])`. So: only bulk path; if a store does not support it, it must implement a no-op or a bulk implementation.  
   - `PostgresMemoryStore` already has `increment_access_counts` (single SQL update). NoOp store: add no-op `increment_access_counts`.

**Pseudo-code (HippocampalStore.search):**

```text
async def search(...):
  results = await self.store.vector_search(...)
  if results:
    now = datetime.now(UTC)
    for r in results: r.access_count += 1; r.last_accessed_at = now
    await self.store.increment_access_counts([r.id for r in results], now)
  return results
```

Remove the `if hasattr(... increment_access_counts)` and the `else: asyncio.gather(update...)` block.

**Files:** `src/memory/hippocampal/store.py`, `src/storage/base.py` (optional default impl), `src/storage/noop_stores.py` (implement increment_access_counts as no-op).

---

### Error handling and edge cases (report §7)

- **Graph sync / broad except:** Replace `except Exception: logger.warning(...)` with explicit exception types where possible; at minimum log with `exc_info=True` and re-raise in critical paths or document that sync is best-effort and must not fail the write. No “swallow and continue” without logging.  
- **Cache (_retrieve_cache):** Ensure cache key format and TTL are documented; add a unit test that sets a value in Redis with key `hot:{tenant_id}` and asserts retrieval returns it (or mock Redis).  
- **Determinism:** Tests that depend on reranker or classifier should use fixed seeds or mock LLM (MockLLMClient) so that results are deterministic.

---

## 4. Implementation order (phases)

| Phase | Content | Deps |
|-------|---------|------|
| 1 | LLM config: no fallback for internal LLM; tests | - |
| 2 | E1: Bulk increment_access_counts only; add to base/noop | - |
| 3 | C3 + C1: RetrievalStep.constraint_categories; planner adds CONSTRAINT step for decision/constraint_dimensions; _retrieve_constraints filters by category; include constraint step in default path | 2 |
| 4 | L1: deactivate_constraints_by_key; orchestrator calls it; align episodic constraint key with constraint_fact_key | - |
| 5 | C2: Packet builder relevance threshold and constraints-first formatting; config | 3 |
| 6 | L3: Reranker config from settings; default recency 0.1 | - |
| 7 | L4: Consolidation constraint re-extraction after gists | - |
| 8 | L2: Classifier and planner tests; optional safe default for classifier on LLM failure | 3 |

---

## 5. Test plan

- **C1:** Synthetic dialogue: store “I want to save money”; later query “Which vacation is budget-friendly?”. Assert retrieval returns the constraint (constraint step included and returns the goal). Use mock classifier returning CONSTRAINT_CHECK or constraint_dimensions.  
- **C2:** Store many episodes and one constraint; retrieve; assert formatted context has constraint in “Active Constraints” and episode count is bounded and filtered by relevance.  
- **C3:** Store goal and value facts; query with intent goal; assert only goal-type constraint/fact in results (mock classifier with constraint_dimensions=["goal"]).  
- **L1:** Write “I like tea”, then “I like coffee”; retrieve for preference; assert only “coffee” (or latest) appears; assert old episodic record is SILENT.  
- **E1:** Unit test: HippocampalStore.search with a store that has increment_access_counts; verify it is called once with list of IDs (no per-record update).  
- **L4:** Unit test: Consolidation with a cluster whose gist text contains a constraint phrase; assert fact store has a fact with matching key/category after migrate.  
- **LLM internal:** Test that with LLM_INTERNAL_* set, get_internal_llm_client() returns client with correct base_url/model; no fallback to LLM__*.

---

## 6. Non-goals and constraints

- **No backward compatibility:** Do not add feature flags or “legacy mode” for old retrieval or old constraint behaviour. Remove the per-record access-count update path entirely.  
- **No fallbacks:** Internal tasks use internal LLM only when configured; do not fall back to main LLM. Classifier either uses internal LLM or a single documented default (e.g. include constraints in plan when LLM fails).  
- **LLM:** All internal LLM use (classifier, extraction, consolidation, reconsolidation, forgetting) uses `get_internal_llm_client()` configured from `.env` as in §1.

---

## 7. File change summary

| File | Changes |
|------|---------|
| `src/core/config.py` | Add RerankerSettings / reranker weights. |
| `src/utils/llm.py` | Ensure internal client uses only LLM_INTERNAL_* when set; no fallback. |
| `src/retrieval/query_types.py` | Optional: add constraint_categories on step (or in planner only). |
| `src/retrieval/planner.py` | Add CONSTRAINT step for decision/constraint_dimensions; add constraint step in default path; set constraint_categories from analysis. |
| `src/retrieval/retriever.py` | _retrieve_constraints: filter by step.constraint_categories (vector + fact); attach constraint_type. |
| `src/retrieval/packet_builder.py` | min_episode_relevance; max_constraints/max_episodes; constraints first in markdown. |
| `src/retrieval/reranker.py` | Use config from settings; configurable recency weights. |
| `src/memory/orchestrator.py` | After storing constraint fact, call deactivate_constraints_by_key. |
| `src/memory/hippocampal/store.py` | Ensure CONSTRAINT record key = ConstraintExtractor.constraint_fact_key; search: only bulk increment_access_counts. |
| `src/storage/postgres.py` | Add deactivate_constraints_by_key; already has increment_access_counts. |
| `src/storage/base.py` | Optional: default impl for increment_access_counts (no-op). |
| `src/storage/noop_stores.py` | Implement increment_access_counts (no-op). |
| `src/consolidation/worker.py` | After gists, run ConstraintExtractor on gist text and store_fact for each constraint. |
| `tests/unit/` | New or extended: test_classifier, test_planner, test_retriever_constraints, test_packet_builder, test_reranker_config, test_orchestrator_supersession, test_consolidation_constraint_recovery, test_llm_internal. |

This plan is complete from architecture down to pseudo-code and file-level changes, with no fallbacks or compatibility layers for older behaviour.
