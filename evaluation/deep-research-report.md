# CognitiveMemoryLayer repo audit and LoCoMo-Plus optimisation report

## Executive summary

- LoCoMo-Plus explicitly targets **Level-2 Cognitive Memory** where the “right” response depends on **latent constraints** (state/goal/value/causal) and must remain **behaviourally consistent** even under **cue–trigger semantic disconnect** (later trigger query is not semantically similar to the earlier cue). citeturn9view0
- In the current CognitiveMemoryLayer design, the bottleneck for LoCoMo-Plus Cognitive is not “more vector recall”, but **missing constraint extraction + missing constraint-first retrieval routing**, which the LoCoMo-Plus paper calls out as the central failure mode for existing systems. citeturn9view0

- Top conceptual issue: **Constraint objects are not first-class** in the write pipeline. The system has `MemoryType.CONSTRAINT`, but the working chunker doesn’t produce a “constraint” chunk type, and the write gate never maps a chunk to `MemoryType.CONSTRAINT`; so “latent constraints” are overwhelmingly stored as either `episodic_event`, `task_state`, or `hypothesis` and are then only recoverable via semantic similarity. (Evidence: `src/core/enums.py`, `src/memory/working/models.py`, `src/memory/hippocampal/write_gate.py`.)
- Top conceptual issue: Retrieval is “hybrid” but still mostly **trigger-query-conditioned**: the main plan still relies on vector similarity + (limited) fact key/text search + graph PPR that only runs when entities are extracted from the trigger query. That structure fails the paper’s cue–trigger disconnect setting by design. (Evidence: `src/retrieval/planner.py`, `src/retrieval/retriever.py`, `src/retrieval/classifier.py`.) citeturn9view0
- Top conceptual issue: Consolidation focuses on “fact/preference/pattern/summary” gists, not the LoCoMo-Plus cognitive constraint decomposition (goal/state/value/causal), so consolidation can **dilute or omit** the governing constraint. (Evidence: `src/consolidation/summarizer.py`, `src/consolidation/schema_aligner.py`.) citeturn9view0

- Top efficiency issue: The episodic encode path has good batching for embeddings (`encode_batch` does gate+redact → single `embed_batch` → bounded extract/upsert), but constraint-aware expansion and reranking are missing; current reranking uses an **O(n²)** word-overlap diversity calculation and recency weighting that can amplify temporal bias errors in LoCoMo-Plus Cognitive. (Evidence: `src/memory/hippocampal/store.py`, `src/retrieval/reranker.py`.)
- Top efficiency issue: Consolidation sampler defaults to a **7-day window** and samples only certain memory types; stable values/goals older than 7 days risk not being consolidated, which increases recall pressure on the vector store and worsens both latency and correctness in long-horizon evaluations. (Evidence: `src/consolidation/sampler.py`.) citeturn9view0
- Top efficiency issue (evaluation): the LoCoMo-Plus ingestion script sleeps per write and ignores provided DATE timestamps, producing “all memories are now” timestamps—this causes avoidable retrieval noise and wrong recency signals. (Evidence: `evaluation/scripts/eval_locomo_plus.py`.)

- Top correctness issue: LoCoMo-Plus evaluation ingestion parses `DATE:` but does **not pass timestamps** into CML writes, so temporal ordering and time-gap semantics are broken; this can depress LoCoMo temporal scores and distort cognitive constraint persistence. (Evidence: `evaluation/scripts/eval_locomo_plus.py`.)
- Top correctness issue: The LoCoMo-Plus Cognitive QA prompt in the repo explicitly says “This is a memory-aware dialogue setting” and instructs the model to show memory awareness, which the LoCoMo-Plus paper warns can be **systematically misleading** (task-disclosed prompting conflates memory with prompt adaptation). (Evidence: `evaluation/scripts/eval_locomo_plus.py` and LoCoMo-Plus paper’s critique.) citeturn9view0
- Top correctness issue: The write pipeline’s short-term encoder drops low-salience chunks (`min_salience_for_encoding=0.4` by default), which can silently skip subtle/implicit constraints (especially values/causal context) that LoCoMo-Plus is designed around. (Evidence: `src/memory/short_term.py`, `src/memory/working/chunker.py`.) citeturn9view0

- Expected LoCoMo-Plus impact (highest leverage): Implementing **constraint-aware memory representation + constraint-first retrieval routing + conflict/supersession rules** should improve Cognitive category significantly because the benchmark’s scoring is based on constraint-consistency, not lexical overlap. citeturn9view0
- Expected LoCoMo-Plus impact (secondary): Fixing evaluation ingestion timestamps and removing “memory-test disclosure” prompts will make scores more faithful and usually improves robustness by preventing overfitting to the judge prompt style. citeturn9view0
- The repository already contains a **Locomo-Plus evaluation harness** under `evaluation/` and a vendored `evaluation/locomo_plus/` pipeline; the required adapter deliverable exists but needs corrections to align with the paper’s intent and to properly stress constraint retrieval. (Evidence: `evaluation/README.md`, `evaluation/scripts/eval_locomo_plus.py`.) citeturn9view0

## Architecture map

### High-level module responsibilities

- **API layer**: FastAPI app constructs `MemoryOrchestrator` and exposes `/api/v1/memory/write` and `/api/v1/memory/read`. (Evidence: `src/api/app.py`, `src/api/routes.py`, `src/api/schemas.py`.)
- **Short-term memory**: `ShortTermMemory` buffers text and chunks into `SemanticChunk`s via either LLM-based `SemanticChunker` or `RuleBasedChunker`, then filters by salience for encoding. (Evidence: `src/memory/short_term.py`, `src/memory/working/manager.py`, `src/memory/working/chunker.py`.)
- **HippocampalStore (episodic)**: write gate + optional PII redaction + embedding + entity/relation extraction + Postgres upsert + (optional) write-time fact extraction into semantic store + graph sync. (Evidence: `src/memory/orchestrator.py`, `src/memory/hippocampal/store.py`, `src/memory/hippocampal/write_gate.py`, `src/extraction/write_time_facts.py`.)
- **NeocorticalStore (semantic)**: semantic facts with versioning + optional graph relations + multi-hop graph query/pagerank. (Evidence: `src/memory/neocortical/store.py`, `src/memory/neocortical/fact_store.py`, `src/storage/neo4j.py`.)
- **Retrieval stack**: classify query → plan steps → execute vector/fact/graph/cache steps → rerank by relevance/recency/confidence/diversity → build `MemoryPacket` and format for LLM injection. (Evidence: `src/retrieval/memory_retriever.py`, `src/retrieval/classifier.py`, `src/retrieval/planner.py`, `src/retrieval/retriever.py`, `src/retrieval/reranker.py`, `src/retrieval/packet_builder.py`.)
- **Lifecycle**: consolidation (sample→cluster→gist→schema align→migrate) + reconsolidation (mark labile → extract new facts → detect conflicts → belief revision) + forgetting (score→policy→execute). (Evidence: `src/consolidation/worker.py`, `src/consolidation/sampler.py`, `src/consolidation/summarizer.py`, `src/consolidation/schema_aligner.py`, `src/consolidation/migrator.py`, `src/reconsolidation/service.py`, `src/reconsolidation/belief_revision.py`, `src/forgetting/worker.py`.)

### Data flow diagram

Ingestion → storage → retrieval → rerank → context assembly → generation

- **Ingestion**
  - `/api/v1/memory/write` → `MemoryOrchestrator.write()`
  - `ShortTermMemory.ingest_turn()` → `WorkingMemoryManager.process_input()` → chunker → `chunks_for_encoding` (salience filter)
  - `HippocampalStore.encode_batch()` → write-gate + redact → embed_batch → entity/relation extraction → `PostgresMemoryStore.upsert()`
  - optional: `_sync_to_graph()` + write-time rule-based semantic facts → `NeocorticalStore.store_fact()`

- **Retrieval**
  - `/api/v1/memory/read` → `MemoryOrchestrator.read()` → `MemoryRetriever.retrieve()`
  - `QueryClassifier.classify()` → `RetrievalPlanner.plan()`
  - `HybridRetriever.retrieve()` executes steps: facts (`NeocorticalStore`) + vector (`HippocampalStore.search()` → `PostgresMemoryStore.vector_search()`) + graph (`NeocorticalStore.multi_hop_query()`)
  - `MemoryReranker.rerank()` → `MemoryPacketBuilder.build()` → `MemoryPacketBuilder.to_llm_context()` → returned to caller (or consumed by an external QA model)

- **Generation**
  - In normal usage: upstream agent injects `llm_context`
  - In repo’s Locomo-Plus eval: retrieved context → Ollama QA model prompt → prediction → judge

## Issue register

Below is a “table-like” register: each item has ID, severity, category, evidence (file/function), explanation, repro (where possible), fix plan, test plan, expected LoCoMo-Plus impact. Claims about LoCoMo-Plus objectives are grounded in the paper’s problem definition (Level‑2 constraints + semantic disconnect + constraint-consistency evaluation). citeturn9view0

- **ISS-01 — Critical — Conceptual**
  - **Evidence**: `src/core/enums.py` defines `MemoryType.CONSTRAINT`; `src/memory/working/models.py` defines `ChunkType` without any constraint type; `src/memory/hippocampal/write_gate.py::_determine_memory_types()` maps `ChunkType` to non-constraint types.
  - **Explanation**: The system has a “constraint slot” at the enum/packet layer (packets have `constraints` and the packet builder formats “Constraints (Must Follow)”), but the write path almost never labels anything as `constraint`. This means constraint retrieval is mostly accidental, reliant on vector similarity.
  - **LoCoMo-Plus failure class**: semantic disconnect failure + wrong constraint type. citeturn9view0
  - **Repro**: Ingest “I’m preparing for an important exam and want to minimise distractions.” Later ask “Should I start watching that new TV series?” In absence of explicit overlap, vector retrieval will often miss the earlier goal constraint. (This is the paper’s canonical example pattern.) citeturn9view0
  - **Fix plan**: Introduce a first-class **ConstraintChunk/ConstraintObject extraction step** at write time; upgrade chunk taxonomy and write gate mapping to produce/store constraint memories.
  - **Test plan**: Add a unit test that writes an implicit goal/value cue and asserts the stored record includes `MemoryType.CONSTRAINT` and a structured constraint payload.
  - **Expected impact**: Large uplift on LoCoMo-Plus Cognitive, because the benchmark rewards constraint-consistent behaviour under semantic disconnect. citeturn9view0

- **ISS-02 — Critical — Conceptual**
  - **Evidence**: `src/retrieval/retriever.py::_retrieve_vector()` calls `HippocampalStore.search()` which embeds the trigger query and does cosine similarity; graph step requires `step.seeds` (entities), which come from classifier entity extraction. `src/retrieval/planner.py` only adds graph steps if `analysis.entities` exists. `src/retrieval/classifier.py::_extract_entities_simple()` uses capitalisation heuristics.
  - **Explanation**: In cue–trigger semantic disconnect, trigger text often lacks shared entities; thus graph multi-hop is skipped. Retrieval becomes primarily semantic similarity over the trigger embedding—exactly what LoCoMo-Plus is designed to break. citeturn9view0
  - **LoCoMo-Plus failure class**: semantic disconnect failure. citeturn9view0
  - **Fix plan**: Add **constraint-aware query routing**: classify trigger into constraint dimensions (goal/state/value/causal) and add a plan branch that searches the constraint store using (a) metadata filters and (b) query expansion to “latent constraint queries”.
  - **Test plan**: A LoCoMo-Plus-style unit test where trigger has no lexical/semantic overlap with cue, but retrieval still returns the right constraint object with high rank.
  - **Expected impact**: Large uplift on Cognitive and also improves robustness when factual questions are cloaked or paraphrased.

- **ISS-03 — High — Conceptual**
  - **Evidence**: `src/extraction/write_time_facts.py` only extracts preference + identity patterns, restricted to chunk types `{PREFERENCE, FACT}`; `src/memory/neocortical/schemas.py` default schemas cover identity/location/preference/relationship; `src/consolidation/summarizer.py` gist types are {fact, preference, pattern, summary}.
  - **Explanation**: LoCoMo-Plus Cognitive decomposes constraints into (at least) **goal, state, value, causal** (paper Fig.2 and problem definition). None of these are schema-first objects today; they are treated as generic text. citeturn9view0
  - **LoCoMo-Plus failure class**: wrong constraint type + inconsistent consolidation. citeturn9view0
  - **Fix plan**: Expand schema to include cognitive categories, and extend extractors to populate them with provenance and validity windows.
  - **Test plan**: Structured extraction tests: given cue dialogues, verify constraint object fields (type/subject/scope/activation/expiry) and supersession rules.
  - **Expected impact**: High on Cognitive; moderate on factual categories via better routing.

- **ISS-04 — High — Logical/correctness**
  - **Evidence**: `evaluation/scripts/eval_locomo_plus.py::_parse_input_prompt_into_turns()` captures `DATE:` lines into `current_date`, but `phase_a_ingestion()` ignores `_date` and `_cml_write()` does not pass `timestamp`.
  - **Explanation**: The evaluation harness collapses all events to “now”, corrupting temporal structure and weakening any recency/validity logic (and any temporal QA). This can also break cognitive supersession (newer constraints should override older ones).
  - **Fix plan**: Parse the DATE into a real timestamp (UTC) and pass `timestamp` to `/memory/write` (the API and orchestrator already accept it).
  - **Test plan**: Golden test of ingestion: given a minimal `input_prompt` with two dates, verify stored records have increasing timestamps and correct day boundaries.
  - **Expected impact**: High on temporal category; moderate on Cognitive because stable vs volatile constraints must respect time. citeturn9view0

- **ISS-05 — High — Conceptual + evaluation fidelity**
  - **Evidence**: `evaluation/scripts/eval_locomo_plus.py` uses `COGNITIVE_PROMPT` starting with “This is a memory-aware dialogue setting…” and explicitly instructs memory awareness; the paper argues explicit task-type prompting can be misleading and misaligned for evaluating cognitive memory. citeturn9view0
  - **Explanation**: This prompt risks inflating “show memory” behaviour and hides whether the system truly retrieved latent constraints “naturally” (the benchmark’s intent). citeturn9view0
  - **Fix plan**: Replace Cognitive QA prompt with a normal conversational continuation prompt, without “memory-aware” disclosure; measure constraint-consistency using judge only.
  - **Test plan**: A/B evaluation: same retrieval backend, compare scores with and without disclosure to ensure we are optimising the right behaviour.
  - **Expected impact**: Improves validity; may reduce headline score short-term, but increases real LoCoMo-Plus alignment.

- **ISS-06 — Medium — Conceptual**
  - **Evidence**: `src/memory/short_term.py` filters `chunks_for_encoding` by `min_salience_for_encoding=0.4`; `src/memory/working/chunker.py` salience rubric focuses on preferences/personal facts/task details; subtle values/causal cues can be scored low.
  - **Explanation**: Latent constraints are often implicit and may not trip “importance” heuristics, leading to missing constraints at write time.
  - **Fix plan**: Add a targeted “constraint cue detector” that boosts salience for phrases indicating goals/values/constraints (“I’m trying to…”, “I don’t want…”, “It’s important that…”, “I’m anxious about… because…”).
  - **Test plan**: Unit tests for salience boosting; ensure such constraints survive the encoding threshold.
  - **Expected impact**: Medium to high on Cognitive.

- **ISS-07 — Medium — Efficiency + correctness**
  - **Evidence**: `src/retrieval/reranker.py` computes diversity via average text overlap against all other memories (quadratic) and mixes recency into the final score; `src/retrieval/retriever.py` already sorts by relevance and dedupes by exact text.
  - **Explanation**: The reranker can over-privilege recency and penalise older but still-active constraints; it also adds avoidable CPU cost in hot paths.
  - **Fix plan**: Separate “constraint ranking” from “episode ranking” and make recency weight constraint-type dependent (values stable vs moods volatile). Introduce a fast approximate diversity or cap pairwise comparisons.
  - **Test plan**: Regression test: older stable value should outrank newer irrelevant episode even when recency differs.
  - **Expected impact**: Medium (Cognitive + temporal robustness).

- **ISS-08 — Medium — Conceptual**
  - **Evidence**: `src/consolidation/sampler.py` uses `time_window_days=7` and filters types to episodic_event/preference/hypothesis.
  - **Explanation**: Stable constraints older than a week might never consolidate into semantic store; retrieval then depends on vector recall from distant episodes, which LoCoMo-Plus stresses heavily. citeturn9view0
  - **Fix plan**: Separate consolidation for stable constraint types with longer windows; always include constraint objects regardless of age until superseded.
  - **Test plan**: Integration test: insert a constraint 30 days ago; consolidation still picks it up and converts to structured semantic constraint.
  - **Expected impact**: Medium.

- **ISS-09 — Medium — Logical**
  - **Evidence**: `src/memory/orchestrator.py` write-time fact extraction wraps exceptions with `pass` (“Fire-and-forget; episodic is source of truth”), which can silently swallow systemic extraction failures.
  - **Explanation**: Silent failures make evaluation debugging hard; LoCoMo-Plus optimisation needs observability to know if constraint extraction is actually running.
  - **Fix plan**: Log structured counters/metrics for extraction success/failure and extracted-count per turn; keep behaviour best-effort but observable.
  - **Test plan**: Inject extractor failure, assert warning metric increments.
  - **Expected impact**: Indirect but important for iteration speed.

## LoCoMo-Plus optimisation plan

### What LoCoMo-Plus rewards and how the repo must adapt

LoCoMo-Plus defines Level‑2 Cognitive Memory as retaining/applying **latent constraints** (state/goal/value/causal) under **cue–trigger semantic disconnect**, and evaluates via an LLM judge with a constraint-consistency framing rather than surface overlap. citeturn9view0

Therefore, improvements must prioritise:

- **Constraint capture** at write time (structured representation + provenance + validity).
- **Constraint-aware retrieval** that is not purely conditioned on trigger embedding similarity.
- **Conflict/supersession rules** so stale constraints aren’t applied after they are superseded.
- **Context assembly** that foregrounds “active constraints” succinctly (so the generator actually uses them).
- **Evaluation fidelity** that does not “teach to the test” via explicit memory-test disclosure. citeturn9view0

### Proposed design changes aligned to the repo’s current architecture

#### Constraint-aware memory representation

Concrete proposal (minimally invasive to existing schema):

- Introduce a **ConstraintObject** structure stored in `MemoryRecord.metadata` (episodic store) and optionally mirrored into `semantic_facts` with new categories/keys.
- Fields (the ones you requested, tuned to LoCoMo-Plus):
  - `constraint_type`: `{state, goal, value, causal, preference, policy, identity}`
  - `subject`: speaker identifier (“user” default; optionally extracted from “Alice: …” lines in Locomo ingestion)
  - `scope`: domain/topic tag(s)
  - `activation`: conditions or triggers (free text + optional tags)
  - `status`: `{active, superseded, expired}`
  - `confidence`, `priority`, `created_at`, `valid_from`, `valid_to`, `decay_policy`
  - `provenance`: evidence turn IDs (`source_turn_id` list is already present in metadata at chunk creation)

Why it fits this repo: You already store rich metadata in `MemoryRecordModel.meta`, and you already have `valid_to`, `supersedes_id`, and versioning in both memory records and semantic facts. (Evidence: `src/core/schemas.py`, `src/storage/models.py`, `src/memory/neocortical/fact_store.py`.)

#### Hybrid retrieval & routing to beat semantic disconnect

- Add a retrieval “source” concept for constraints without breaking the existing interface:
  - Keep `HybridRetriever` but add a step type such as `RetrievalSource.CONSTRAINTS` (or reuse FACTS but with dedicated filtering keys).
  - Implement retrieval by:
    1) metadata filter: `type == constraint` AND `constraint.status == active`
    2) hybrid scoring: lexical match + embedding match on a **constraint canonical form** (not raw text)
    3) associative expansion: if a constraint mentions entities/goals, expand candidates (graph or self-links)
- Add a **Query→Constraint router**:
  - Extend `QueryIntent` or add a secondary classifier output: `constraint_dimensions = {goal/state/value/causal}`.
  - Even if the trigger doesn’t match lexically, route “temptation / recommendation / decision” queries to a constraint search plan.

Why it fits this repo: `MemoryRetriever.retrieve()` already does `analysis → plan → execute → rerank → packet`; you can add a constraints branch without replacing the whole system. (Evidence: `src/retrieval/memory_retriever.py`, `src/retrieval/planner.py`.)

#### Context assembly optimised for judge-consistent behaviour

- Change the packet builder formatting so the model sees:
  - “Active constraints” (1–6 items) with a short natural phrasing and provenance
  - Minimal evidence snippets (1 line each) to avoid burying the constraint (constraint dilution failure)
- Avoid the “memory dump” style. The repo currently prints a markdown header “Retrieved Memory Context” and then multiple sections, but it truncates by raw chars. That can cut constraint evidence mid-structure. (Evidence: `src/retrieval/packet_builder.py`.)

#### Consolidation & overwrite rules

- Extend consolidation gists to output constraint objects (goal/state/value/causal) rather than only “fact/preference/pattern/summary”.
- Implement explicit supersession:
  - If a newer constraint conflicts with an older one, mark the old as `superseded` (set `valid_to`) and ensure retrieval only returns “current” active constraints.
- Current `SemanticFactStore` already has a “one current per key” behaviour when value changes (`is_current` flips, version increments). That’s a good existing mechanism—use it for constraints by defining stable keys like `user:goal:<domain>` or `user:value:<domain>`. (Evidence: `src/memory/neocortical/fact_store.py`.)

### Prioritised roadmap

#### Quick wins (one to two days)

- Fix Locomo-Plus ingestion timestamps:
  - Parse `DATE:` and pass `timestamp` into `/memory/write` (the API supports it).
  - Also pass speaker role into metadata (e.g., `{speaker: "Alice"}`), without changing public API.
- Remove “memory-aware dialogue setting” disclosure from Cognitive QA prompt in evaluation scripts; use a normal continuation prompt and rely on the judge for constraint consistency. citeturn9view0
- Add constraint cue salience boost so implicit constraints survive the `min_salience_for_encoding` threshold.
- Add a dedicated retrieval filter option for constraints (even if stored as episodic initially): query memory types `[constraint, task_state, plan]` for Cognitive category runs.

#### Medium scope (one to two weeks)

- Implement write-time **constraint extraction**:
  - Rule-based extractor for obvious patterns (goal/value/state/causal)
  - LLM extractor (optional) for ambiguous cases (feature-flagged)
  - Store canonical constraint objects in `MemoryRecord.metadata` and optionally as semantic facts
- Implement **constraint-aware retrieval plan**:
  - Add query router dimension classification and add a constraints-first step
  - Add associative expansion from retrieved constraints (follow shared entities / subjects)
- Improve conflict handling:
  - Extend reconsolidation `_extract_new_facts` and conflict detector to understand constraints (not only preference/fact markers).

#### Larger refactors (multi-week)

- Introduce a dedicated constraint index (still inside Postgres):
  - A new table containing constraint objects with their own embeddings + typed columns for fast filtering.
  - Add a lightweight cross-encoder or LLM reranker that explicitly scores “constraint relevance” for the top-K constraint candidates (feature-flagged and cached).
- Introduce text/BM25 search inside Postgres for episodic + constraint canonical forms to improve recall when embeddings fail.

## Patch plan

This is structured as PRs with commit-level goals. It is designed to be **testable** and to keep backwards compatibility unless stated.

### PR one: Evaluation harness correctness and fidelity

- **Commit**: `eval: preserve DATE timestamps in CML ingestion`
  - Modify `evaluation/scripts/eval_locomo_plus.py`:
    - Convert `current_date` into a `timestamp` and include it in `_cml_write()` payload (API already accepts `timestamp`).
    - Optional: store `{speaker, date_str, session_idx}` in metadata for debugging.
- **Commit**: `eval: remove cognitive task-disclosure prompt`
  - Replace `COGNITIVE_PROMPT` with a neutral prompt like “Continue the conversation naturally…” (no “memory-aware” framing), per LoCoMo-Plus paper concerns about task prompting. citeturn9view0
- **Tests**:
  - Add a small unit test for `_parse_input_prompt_into_turns()` and date parsing.
  - Add a smoke test that produces a deterministic ingestion payload for a 2-day conversation.

### PR two: Constraint extraction and storage (no new DB tables yet)

- **Commit**: `memory: add constraint cue detector + chunk salience boost`
  - Extend chunking salience scoring for constraint cues (goal/value/state/causal markers).
- **Commit**: `memory: introduce ConstraintObject in metadata`
  - Add `src/extraction/constraint_extractor.py` (rule-based first) producing structured constraint objects.
  - In `HippocampalStore.encode_chunk/encode_batch`, attach extracted constraints into `metadata["constraints"]`.
- **Commit**: `memory: map constraints to MemoryType.CONSTRAINT where confident`
  - Extend write gate mapping or add a post-gate override: if constraint extractor returns `constraint_type` with high confidence, set `memory_type_override=MemoryType.CONSTRAINT`.
- **Tests**:
  - Unit tests around “goal/value/state/causal” extraction.
  - Regression: ensure existing preference/fact flows unchanged.

### PR three: Constraint-aware retrieval routing and context assembly

- **Commit**: `retrieval: extend QueryAnalysis with constraint_dimensions`
  - Add optional fields to `QueryAnalysis` (goal/state/value/causal likelihoods).
- **Commit**: `retrieval: add constraints-first plan branch`
  - Update `RetrievalPlanner.plan()` to include a first step that targets constraints (by memory type filter) when the trigger suggests a decision/behaviour question.
- **Commit**: `retrieval: packet builder outputs “Active constraints” with provenance`
  - Update `MemoryPacketBuilder._format_markdown()` (and JSON) to prioritise constraints and include compact provenance (turn IDs already exist).
- **Tests**:
  - A synthetic LoCoMo-Plus-style test: cue constraint, distant trigger, assert retrieved packet contains the constraint in top positions and formatted context includes it.

### PR four: Supersession and stale-constraint prevention

- **Commit**: `facts: add cognitive categories and stable keys`
  - Extend `FactCategory`/schemas to add cognitive keys like `user:goal:<domain>`, `user:value:<domain>`, `user:state:<domain>`, `user:causal:<domain>`.
- **Commit**: `reconsolidation: detect and supersede constraints`
  - Extend conflict detection strategies to treat constraints as supersedable or stable.
- **Tests**:
  - “Old goal superseded” test: verify retrieval returns newest active goal only.

### PR five: Performance instrumentation for LoCoMo-Plus iteration

- **Commit**: `metrics: add constraint extraction + retrieval counters`
  - Emit extraction counts, constraint hit rate, and retrieval latency per source.
- **Commit**: `eval: add per-category retrieval diagnostics`
  - In eval script, log how many constraints vs episodes were retrieved for Cognitive items.

## Risks and non-goals

- **Latency vs accuracy trade-off**: Constraint extraction with an LLM can be slow and expensive. Mitigation: rule-based first, LLM second, feature-flagged, and cache embeddings (the repo already supports `CachedEmbeddings`). (Evidence: `src/utils/embeddings.py`, `src/core/config.py`.)
- **Compatibility risk**: Adding new `QueryAnalysis` fields and new retrieval sources must not break existing API responses. Keep the external API stable; implement new behaviours behind feature flags (the repo already has a feature flag pattern). (Evidence: `src/core/config.py`.)
- **Evaluation score fluctuations**: Removing cognitive task-disclosure prompts can reduce apparent scores but increases alignment with the benchmark’s stated intent and prevents “teaching to the prompt”. citeturn9view0
- **Non-goal**: Replacing Postgres/pgvector/Neo4j with a different storage backend. The plan targets improvements within the current architecture.
- **Non-goal**: Fully reproducing LoCoMo-Plus external repo scripts. This repository already vendors `evaluation/locomo_plus/` and provides a `run_full_eval.py` pipeline; the focus is on correctness and alignment, not re-implementation.