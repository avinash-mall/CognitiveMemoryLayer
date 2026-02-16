# Architecture Overview

**CognitiveMemoryLayer** implements a dual-store memory inspired by neuroscience: a **hippocampal (episodic)** store for short-term, high-fidelity memory and a **neocortical (semantic)** store for long-term structured memory. The key components (per [README](https://github.com/avinash-mall/CognitiveMemoryLayer#hybrid-retrieval-ecphory) and code) are:

- **Entrypoint/API**: A FastAPI server (`src/api/app.py`) initializes `MemoryOrchestrator` on startup【19†L523-L532】. The orchestrator drives all memory operations.
- **Write Path**: Input text flows through **Short-Term Memory** (chunking and gating) into the **HippocampalStore**. In `MemoryOrchestrator.write()`, text is chunked (`ShortTermMemory.ingest_turn`), then passed to `HippocampalStore.encode_batch()`【18†L274-L283】. The _write gate_ (salience scoring, novelty) filters out unimportant chunks. Kept chunks are redacted of PII (`PIIRedactor`), embedded, and optionally annotated with entities/relations. A `MemoryRecord` is created for each chunk and upserted into Postgres/pgvector via `HippocampalStore.encode_batch()`【4†L236-L248】【18†L290-L299】.  Constraints (goals/values/states) are extracted by `ConstraintExtractor` (if enabled) and stored as semantic facts immediately【18†L327-L336】【19†L377-L386】.
- **Storage Backends**: Episodic chunks use **Postgres with pgvector** (`PostgresMemoryStore` via `HippocampalStore`), while semantic memories use a **Neo4j graph** (for entities/relations) and a **Postgres table** (`SemanticFactStore`) for facts【17†L91-L100】【17†L122-L131】. A Redis cache may be used for embeddings and hot items.
- **Retrieval Path**: On query, the `MemoryRetriever` orchestrates a **Classifier → Planner → HybridRetriever → Reranker → PacketBuilder** pipeline【81†L45-L54】【23†L21-L28】. A QueryClassifier (LLM-based) tags the query (e.g. as constraint-check, personal-history, etc.), and a `RetrievalPlanner` builds a multi-step plan of sources (vector, graph, facts, constraints)【104†L85-L93】.  The `HybridRetriever` executes each plan step in parallel with timeouts【33†L47-L57】: for example, `_retrieve_vector` does a semantic vector search in hippocampal store【34†L259-L268】; `_retrieve_graph` does Neo4j graph walk on entities【34†L277-L286】; `_retrieve_facts` searches semantic fact table by key or text【34†L203-L213】【34†L219-L228】; `_retrieve_constraints` does a two-pronged search (vector search on episodes labeled `MemoryType.CONSTRAINT` plus lookup of semantic facts in cognitive categories)【34†L309-L318】【34†L337-L346】. The raw results are merged, deduped, and sorted by `MemoryReranker` (combining relevance, recency, confidence, diversity)【51†L42-L51】【51†L73-L82】. Finally a `MemoryPacketBuilder` splits memories into **constraints, facts, preferences, episodes** and formats context for the LLM【55†L11-L20】【55†L78-L88】.  

- **Consolidation**: A background `ConsolidationWorker` asynchronously clusters related episodic memories (e.g. weekly episodes vs quarterly constraints) and extracts gists and schemas, migrating persistent knowledge into the semantic store【101†L103-L107】.  
- **Reconsolidation**: New memory writes can trigger belief revision (e.g. conflicting facts or constraints) via `ReconsolidationService`, updating or superseding old facts【19†L411-L420】【35†L443-L452】.  
- **Forgetting/Compression**: A `ForgettingWorker` decays or archives old memories (e.g. compressing repetitive content) over time. 

In summary, data flows as: **Text input → Short-term chunks → Write Gate → Hippocampal store (+ Neo4j sync, fact/constraint extraction) → Retrieval plan → Hybrid search (vector+graph+fact+constraint) → Rerank → Packet assembly for LLM**. (See README mermaid diagrams【104†L19-L27】【104†L71-L80】.)

# Identified Issues and Failure Modes

Below we detail conceptual, efficiency, and logical weaknesses tied to code, organized by failure class, with evidence and proposed fixes/tests.

1. **Semantic Disconnect** (Missed Constraints). The current retrieval relies on semantic similarity of the query to memory text【34†L259-L268】. If a latent constraint (e.g. a user preference) was expressed in very different language from the trigger query, vector search may never retrieve it. For example, if the user said “I want to save money” earlier, and later asks a loosely related question (“Which vacation is budget-friendly?”), a pure semantic match might fail. In our code, `MemoryRetriever._retrieve_vector` uses only the query embedding to find similar episode text【34†L259-L268】. Similarly, `HybridRetriever._retrieve_constraints` only runs if the plan includes a constraint step; it does **not** automatically fetch constraints for all queries. Thus **(i)** queries without explicit constraint-check classification skip constraints entirely (no `_retrieve_constraints` call), and even if run, the vector step will only find constraints if the query text is similar. **Fix**: Route queries through a constraint-aware path. For example, ensure the `QueryClassifier` identifies queries where latent constraints might apply (e.g. queries tagged as `CONSTRAINT_CHECK`)【104†L85-L93】. Always include a constraint-retrieval step for such queries. Another improvement is **associative expansion**: after initial retrieval, search for related constraints via shared entities or topics. Implement keyword/metadata filters (e.g. filter by topic or speaker) to bias constraint retrieval even when lexical overlap is low. **Test**: Create synthetic dialogues where a goal or value is mentioned early, then the user asks a semantically distant question. Verify the improved system retrieves the original constraint. Use a dummy `QueryClassifier` override or force a constraint step to test this behavior; ensure prior to fix the constraint is missing, and after fix it appears in results.

2. **Constraint Dilution** (Irrelevant Context). The context assembly may include too many irrelevant details, burying the key constraint. For instance, `MemoryPacketBuilder._format_markdown` naively lists up to 5 facts and 5 episodes, and up to 6 constraints【55†L99-L108】【55†L113-L122】, without strong filtering. If many loosely relevant episodes are retrieved, the single latent constraint can be obscured. In code, after reranking, all memories are placed into sections (facts, preferences, episodes, constraints)【55†L11-L20】; episodes not specifically identified as constraints will go into “Recent Context” even if unrelated. **Fix**: Tighten context: only include episodes highly relevant to the query or containing the same domain. For example, incorporate topic filters or thresholded relevance when assembling episodes. Always highlight active constraints separately. We could also enforce a *hard budget* favoring constraints: e.g. always include top-1 constraint (if any) and trim episodes aggressively. **Test**: Query with multiple background memories where only one is an active constraint. Confirm that before fix, context includes many irrelevant episodes (dilution), whereas after fix, irrelevant episodes are dropped. Automate by tagging certain memories as relevant/irrelevant and checking context length and content.

3. **Wrong Constraint Type**. The system mixes up constraint categories (state, goal, value, etc.) or retrieves generic facts instead of the correct constraint. For example, a query about *goal* may retrieve a related fact ("User is going on vacation") instead of the latent goal. In code, *constraint retrieval* fetches both episodic constraint chunks and semantic facts of all cognitive categories【34†L309-L318】【34†L337-L346】, but **no step disambiguates which constraint type the query needs**. Moreover, the `QueryClassifier` has 10 intents including a `CONSTRAINT_CHECK` intent【104†L85-L93】, but if the classifier misfires (or is missing), the plan may not include a constraint step. Meanwhile, `MemoryRetriever._retrieve_facts` handles non-constraint facts (predicate/value pairs)【34†L203-L212】, which could clutter answers. **Fix**: Enhance query classification and planning: explicitly tag queries for goal/value/state intent, and in the planner generate targeted constraint lookups (e.g. retrieval steps filtered to `FactCategory.GOAL` etc.). In retrieval, instead of blind fact lookup, filter facts by category matching predicted intent. For example, if classified as a *goal* query, retrieve only `FactCategory.GOAL`【34†L337-L346】. This avoids mixing value vs goal. Also ensure episodes tagged as `MemoryType.CONSTRAINT` carry metadata of their type (currently missing) so they can be identified and formatted properly. **Test**: Simulate queries of different intent (goal vs value). After fix, verify that the constraint returned (via `_retrieve_constraints`) has the correct category label (e.g. "[Goal] ..." prefix) and unrelated categories are suppressed.

4. **Temporal/Recency Bias**. Recent memories tend to dominate over older but still-active constraints. In `MemoryReranker`, the **recency score** is `1/(1+days*0.1)` weighted by `recency_weight` (default 0.2)【51†L73-L82】. All non-constraint memories use the full recency weight (0.2), while constraints have reduced weight (0.05–0.15)【51†L50-L59】. Despite this, if a user set a long-term value (e.g. “My favorite color is blue” last year), many newer episodic memories could outrank it, even though that value remains active. The code’s attempt to de-emphasize stable constraints may not fully mitigate this: episodes (type=`episodic_event`) still get recency weight 0.2, so recent chit-chat can drown out older facts. **Fix**: Further reduce recency weight for constraints (or increase for episodes) to ensure stable facts persist at top. For example, set `RerankerConfig.recency_weight=0.1` globally and/or lower stable constraint weights to near zero. Alternatively, apply a **time-window** policy (as in `EpisodeSampler`: 7d for episodes vs 90d for constraints【104†L101-L104】) by filtering out episodes older than, say, 7 days by default. **Test**: Create memories spanning different dates. Check ranking scores: after fix, verify that an older active constraint still appears above irrelevant new episodes. Compare reranker scores (in code or by unit test) before and after weight change to ensure constraint age has less impact.

5. **Stale Constraint Application**. The system lacks rules to **supersede or expire** old constraints. Currently, writing a new constraint fact does mark old facts invalid in `SemanticFactStore._update_fact`【15†L251-L260】 (deactivating the old fact row). But episodic constraint records remain in PostgreSQL and could still surface in vector search. Also, `MemoryOrchestrator.update` handles feedback like "outdated" by setting a `valid_to` or deleting, but only on individual memory records【19†L510-L518】. There’s no automatic linking between a new constraint and old episodic records. For example, if the user first says “I like tea” and later “Now I like coffee”, both constraint records exist in hippocampus. Retrieval may still return the outdated “I like tea” if it happens to have higher similarity. **Fix**: Implement explicit supersession: when a new constraint of the same type/scope is stored, mark any existing episodic constraint memories as archived or deleted (e.g. update their `status`)【15†L252-L260】. We can mimic the fact store logic: on insert, search for an existing active constraint key (e.g. using `ConstraintExtractor.constraint_fact_key` as in orchestrator) and deactivate older hippocampal records. Also add an “expiration” timestamp or TTL on constraint chunks (via metadata) so old constraints naturally drop out of search after a time. **Test**: Write two conflicting constraints (e.g. “like tea” then “like coffee”) and query for preference. Before fix, both might appear; after fix, only the latest should be returned. Implement a unit test that calls `MemoryOrchestrator.write` twice with contradictory constraints and then `retrieve`; verify only the new one appears (or the old is marked `status=SILENT`).

6. **Inconsistent Consolidation**. The summarization/clustering may lose or distort constraints. The `ConsolidationWorker` clusters episodes into semantic gists【104†L101-L107】, but if the gist extractor overlooks a subtle constraint (“I’m vegetarian”) and paraphrases it weakly, the underlying rule may vanish. We can’t inspect consolidation code here, but this risk suggests adding checks: ensure constraints are carried forward or re-extracted during consolidation. For instance, after summarizing a cluster, run `ConstraintExtractor` on the summary and reconcile with earlier constraints. **Fix**: Audit the summarizer (`GistSummarizer`) to verify it includes explicit facts (like constraints). If missing, integrate a pass that extracts constraint phrases from gists and stores them in the fact store as needed. **Test**: Cluster synthetic episodes containing a hidden constraint. Check that the final semantic facts still reflect that constraint.

7. **Logical Bugs and Edge Cases**. In code review we noted a few specifics:
   - *HippocampalStore.search* increments `access_count` one-by-one if no batch method【4†L236-L248】. If many results, this is an N× loop of DB updates (inefficient). Fix by using a bulk SQL update (patch `vector_store.increment_access_counts`).
   - *Cache utilization*: The memory cache step in `_retrieve_cache`【34†L378-L388】 is decoupled; ensure cache keys (hot items) match usage. We should test that cache (Redis) retrieval doesn’t error unexpectedly.
   - *Error Handling*: Many `except Exception: logger.warning(...)` blocks (e.g. graph sync【19†L415-L423】) swallow errors. Add tests to ensure failures are logged but do not corrupt state.

# Retrieval Improvements

Building on the above fixes, we propose these design enhancements:

- **Structured Constraints**: Extend memory schema to store **rich constraint objects**. For each extracted constraint, record: `type` (goal/state/value/etc), `subject` (e.g. user or agent), `scope` (topic), and `activation conditions`. Modify `ConstraintExtractor` to output these fields. Store the canonical form (e.g. normalized key) alongside raw text. Update `SemanticFactStore` to use these structured keys (e.g. `goal:vacation:meaning`), linking to evidence turns【15†L252-L260】. Keep pointers from fact back to original episodic record. This lets the retriever filter by constraint type and link easily to episodes. *This aligns with README’s vision of constraints as first-class objects【104†L113-L121】.* 

- **Hybrid Retrieval & Routing**: Improve the *RetrievalPlanner* (currently invisible in code) to classify query type more robustly. For example, use few-shot LLM prompts or a small classifier model to tag queries as *Preference, Goal-check, Procedural*, etc. Routes:
  - **Vector + Keyword Hybrid**: Instead of pure embedding similarity, do an auxiliary keyword search. E.g. if query contains named entities, boost items with those entities (via Neo4j or simple text filters).
  - **Metadata Filtering**: Inject filters into `HippocampalStore.search` queries (it already supports `time_filter`, `memory_types`, etc. [34†L243-L252]). We should use them: e.g. if query mentions “last week” set `since`, or if asking about personal history, restrict to user’s own session ID.
  - **Cross-Encoder Reranker**: After retrieving top-K, use a lightweight LLM or cross-encoder to re-score specifically for *constraint relevance*. For example, feed (query, memory_text) pairs to a small model trained to score whether the memory satisfies a constraint relevant to the query. This complements the current heuristic (text overlap) diversity score【51†L137-L146】.
  - **Association Expansion**: After initial top-K retrieval, find neighbors via graph edges or entity co-occurrence. E.g. retrieve episodes containing the same entity or relation as a top result. This can catch semantically “distant” memories.

- **Context Assembly**: Rather than dumping a memory list, assemble **active constraints first**. Always format active constraints as natural language reminders. Limit overall token budget: e.g. take only top-1 or 2 constraints (with evidence refs), then a handful of most relevant episodes. Show provenance (turn IDs or timestamps) for constraints【55†L98-L106】. For example: “You earlier said *I am vegetarian* (March 3rd).” This is in line with README’s suggested format. In code, modify `MemoryPacketBuilder._format_markdown` to emphasize constraints (maybe move them to top and use **bold** highlighting, as begun), and to truncate or collapse repetitive episodes into a concise summary instead of bulleting each. This avoids context “dump”.

- **Consolidation & Overwrite Rules**: Implement explicit *supersession logic*. For cognitive constraints and preferences, ensure only the latest holds. We can augment `SemanticFactStore._update_fact` to set `supersedes_id` (already does), and likewise update episodic store: maybe adding a `supersedes_id` field to `MemoryRecord` so we can deactivate old episodes upon writing a new constraint. Additionally, differentiate **stable vs volatile** memory: e.g. treat `policy/value` as stable (persist indefinitely) vs `state/mood` as volatile (auto-expire after short window). This can be encoded as metadata on MemoryRecord. Add “valid_until” logic for states (like forgetting). Tests: Write multiple constraints of different types and ensure older volatile ones expire or stop being returned after their window.

# Efficiency and Reliability Improvements

- **Profiling & Hot Paths**: The likely hot path is `HippocampalStore.encode_batch()`, which calls LLM embed/LLM extract on each chunk【4†L236-L248】. To optimize:
  - **Batch Embedding**: Already done if `chunks_for_encoding` >1【4†L352-L362】, but ensure batching when many small chunks.
  - **Cache Embeddings**: We wrap with `CachedEmbeddings` if Redis is enabled【17†L82-L91】. Verify cache hits by unit test simulating duplicate texts.
  - **Avoid Duplicate Extract Calls**: In `encode_batch`, entity/relation extractors are run per chunk. If chunks share text, consider deduplicating extraction or caching results.
- **I/O Bottlenecks**: `HippocampalStore.search` does a `vector_search` call then an update loop incrementing access counts【4†L358-L364】. This N+1 update is expensive. We should use a single SQL `UPDATE` on matching IDs instead. Similarly, `scan()` in `delete_all`/forget scans potentially large tables in batches【20†L678-L686】; consider adding bulk-delete methods or partitioning by tenant for efficiency.
- **Index Usage**: Ensure the PostgreSQL vector column has an index (pgvector index) to prevent full table scans. We cannot confirm here, but doc suggests this. Add tests measuring query time on simulated data.
- **Metrics & Logging**: Embed structured logs around retrieval steps (e.g. times already sent to Prometheus in `HybridRetriever._record_step_metrics`【33†L183-L190】). Add counters for hit/miss on constraint steps and retrieval times. Ensure logs include identifiers (tenant, step) for debugging.
- **Concurrency/Race**: The orchestrator uses async tasks (Consolidation/Forgetting as Celery workers). Potential race: if a new memory is written while consolidation reads the store. Recommend serializable isolation or simple locks on tenant consolidation to avoid missing data. Add a test: concurrently write and consolidate and check consistency.
- **Determinism**: Non-determinism can arise from random cluster seeds or LLM outputs. For testability, use fixed seeds or mock LLM clients.

# Issues and Roadmap

Below is a summary of identified issues, ordered by priority.

- **ID: C1 – Semantic Disconnect (Conceptual)**  
  **Severity**: High. Without targeting latent constraints, the agent may ignore key user preferences.  
  **Evidence**: `HippocampalStore.search` only uses semantic similarity of `query`【34†L259-L268】; no fallback to constraints unless query classified.  
  **Fix**: Ensure `QueryClassifier` plans a constraint lookup (via `RetrievalStep` with source=CONSTRAINT) when appropriate【104†L85-L93】. Include keyword/entity filtering.  
  **Test**: A query semantically unrelated to an existing constraint should retrieve it after fix.

- **ID: C2 – Constraint Dilution (Logical)**  
  **Severity**: Medium. Irrelevant episodes can bury constraints.  
  **Evidence**: `MemoryPacketBuilder` lists up to 5 episodes by recency【55†L119-L128】, regardless of their pertinence.  
  **Fix**: Tighten assembly: include only highly relevant episodes (use a relevance threshold) and always prioritize constraints (even if relevance is lower).  
  **Test**: Given many retrieved episodes and one constraint, verify context after fix contains the constraint prominently and fewer extraneous episodes.

- **ID: C3 – Wrong Constraint Type (Conceptual)**  
  **Severity**: Medium. Returns wrong category (e.g. facts instead of goals).  
  **Evidence**: All cognitive facts are returned unfiltered in `_retrieve_constraints`【34†L337-L346】, and classifier intent not enforced in code.  
  **Fix**: Augment `RetrievalPlanner` to tag steps by desired category. Filter fact lookups by that category. Ensure episodic `MemoryType.CONSTRAINT` records have metadata `"constraint_type"` so we can identify them in reranking and formatting.  
  **Test**: Force retrieve for a specific type (goal vs value) and check only that type appears.

- **ID: E1 – Retrieval Performance (Efficiency)**  
  **Severity**: Medium. Embedding and DB operations may be slow under load.  
  **Evidence**: `encode_batch` may call the embedder many times【4†L236-L248】; `increment_access_counts` loops one-by-one【4†L358-L364】.  
  **Fix**: Batch or cache all embed calls. Optimize DB updates (bulk `UPDATE`).  
  **Test**: Benchmark embedding latency on N chunks with/without caching; measure DB update times before/after fix with many records.

- **ID: L1 – Stale Constraints (Logical)**  
  **Severity**: Medium. Old constraints persist after being overridden.  
  **Evidence**: No code links episodic constraint records to semantic fact versions. The `update` logic only covers facts【15†L243-L252】.  
  **Fix**: On new constraint, mark old episodes as outdated (update status or delete). Consider adding `valid_to` to hippocampal records.  
  **Test**: After writing a new constraint, assert older constraint record’s status != ACTIVE or is not returned by search.

- **ID: L2 – Missing Query Classifier (Logical)**  
  **Severity**: Low. If `QueryClassifier` isn’t implemented or misclassifies, plans fail.  
  **Evidence**: Code references it【81†L45-L54】 but no source visible; no tests found.  
  **Fix**: Implement or test `QueryClassifier`. Validate that all 10 intents (including CONSTRAINT_CHECK) are handled.  
  **Test**: Provide queries for each intent; ensure planner steps align (e.g. constraint intent → include constraint retrieval).

- **ID: L3 – Recency Weight Tuning (Efficiency/Behavioral)**  
  **Severity**: Low/Medium. Current recency weighting may not align with use case.  
  **Evidence**: `RerankerConfig.recency_weight=0.2` for episodes【51†L50-L59】; stable constraints get only 0.05–0.10.  
  **Fix**: Expose these weights in settings; tune via experiments. Possibly set `recency_weight=0.1`.  
  **Test**: Parameter sweep in unit test (simulate memories of different ages) to validate that older constraints still score above irrelevant new ones.

- **ID: L4 – Inconsistent Consolidation (Logical)**  
  **Severity**: Low. Constraints may get paraphrased away.  
  **Evidence**: Summarization logic (not shown) may not preserve exact constraint phrases.  
  **Fix**: Add check: after summarizer runs, re-run constraint extraction on summary and reintegrate into semantic store if needed.  
  **Test**: Summarize a cluster with a clear goal; ensure `SemanticFactStore` still has a fact for it.

# Proposed PRs and Roadmap

- **PR#1**: *Constraint Retrieval Fix* – Enhance `MemoryRetriever._retrieve_constraints` to require query tagging. E.g. add a boolean flag or query analysis hint to include constraints. Update `QueryClassifier` to include a `CONSTRAINT_CHECK` category. Add unit tests simulating constraint queries.  
- **PR#2**: *Context Assembly Enhancement* – Modify `MemoryPacketBuilder` to limit episodes by relevance and highlight constraints. Add tests for context formatting (ensuring constraint appears in markdown).  
- **PR#3**: *Supersession Logic* – Update `HippocampalStore.encode_chunk` to mark old constraint memories as `status=SILENT` when a new constraint of same type is stored (using `constraint_fact_key` as in orchestrator)【19†L327-L336】. Add tests for conflicting writes.  
- **PR#4**: *Efficiency Improvements* – Batch DB updates for `increment_access_counts`. Wrap multiple embed calls in fewer API requests. Add metrics (if not already). Include benchmarks.  
- **PR#5**: *Weight Tuning Configuration* – Expose reranker weights in config, lower default recency weight. Write regression tests on `MemoryReranker._calculate_score`.  
- **PR#6**: *Tests for Edge Cases* – Add tests for query classification (using a mock `QueryClassifier`), for retrieval planner behavior, and for consolidation correctness.  

# Risks & Tradeoffs

- **Latency vs. Accuracy**: Adding steps (constraint retrieval, reranking, associative expansion) may increase response time. We must balance improved recall of constraints with acceptable latency. Using timeouts and caching can mitigate this.  
- **Backward Compatibility**: Changes to how memories are retrieved/flagged (e.g. filtering by type/time) may alter existing behavior. We should consider a feature flag or migration plan (e.g. old semantic-only mode vs new constraint-aware mode).  
- **LLM Dependence**: Some fixes rely on better query classification or summarization, which may need LLMs. Without keys, we cannot test fully – will require user’s API keys for deep testing. We assume availability or use small local models for test harness.  
- **Complexity**: Introduced schema fields (constraint objects) make the data model more complex. Need to ensure migration scripts (if any) update old memories gracefully.  

**Non-goals**: We will *not* overhaul the entire retrieval engine (e.g. replacing the hybrid plan structure), only augment it. We won’t drop vector search, but complement it. We also won’t attempt a full rewrite of consolidation or forgetting; we focus on retrieval correctness and key flows.

**Architecture Map (textual)**: 
- **Ingestion**: `FastAPI → MemoryOrchestrator.write` → `ShortTermMemory.ingest_turn` → tokens → `HippocampalStore.encode_batch` → [Postgres(pgvector) + Neo4j + FactStore]【18†L274-L283】.  
- **Storage**: Hippocampus (Postgres) stores `MemoryRecord`; Neocortex (Neo4j graph for entities/relations, Postgres for facts).  
- **Retrieval**: `MemoryOrchestrator.read → MemoryRetriever.retrieve`: *query* → `QueryClassifier` → `RetrievalPlanner.plan` → `HybridRetriever` executes steps (vector search【34†L259-L268】, graph PPR【34†L277-L286】, fact queries【34†L203-L213】, constraint search【34†L309-L318】【34†L337-L346】) → results → `MemoryReranker`【51†L42-L51】 → `MemoryPacketBuilder` to format context【55†L98-L106】.  
- **Consolidation**: `ConsolidationWorker` (async) clusters hippocampal data into semantic facts.  
- **Forgetting**: `ForgettingWorker` cleans old records.  

This document has pinpointed specific code lines to ground each issue and tied them to behavioral failure modes. The proposed fixes are modular and testable, aiming to measurably improve retrieval precision for latent constraints and ensure consistency with intended cognitive principles【104†L113-L121】【104†L96-L104】.

