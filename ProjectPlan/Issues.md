# Implementation Plan Issues, Risks, and Mitigations

> This document catalogs potential issues, failure risks, and problems identified during validation of the intrinsic memory implementation plan (Phases 1-10), along with proposed solutions.

---

## Cross-Cutting Concerns

### 0. Request Concurrency & Hook State Isolation (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Affected Phases** | 1, 2, 3, 4, 5 |
| **Description** | Hooks and injection state (e.g. `_pending_logit_bias`, steering vectors in hooks) are **process-wide**. With concurrent requests sharing the same model backend, Request A's steering vectors can be overwritten by Request B's before generation completes, or logit bias from one user can affect another. |
| **Failure Mode** | Cross-request contamination; wrong memories injected for a user; data leakage between sessions. |
| **Proposed Solution** | (1) Make injection state **request-scoped**: pass a request/session context through the bus and into hooks; store per-request injection data in context. (2) Or serialize access: ensure only one generation uses the backend at a time (queue per backend). (3) Document that multi-tenant deployments must use process-per-tenant or request-scoped state. (4) Phase 1: Define `InjectionContext(request_id=..., session_id=...)` and thread it through Controller → Bus → handlers → Backend. |

---

### 1. Single Point of Failure in Model Backend

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Affected Phases** | 1, 2, 3, 4, 5, 8 |
| **Description** | The `ModelBackend` is the foundation for all intrinsic interfaces. If it fails to initialize or becomes unavailable, the entire intrinsic memory system fails. |
| **Failure Mode** | Backend initialization failure → no hooks, no injection, system unusable. |
| **Proposed Solution** | Implement graceful degradation to RAG-only mode. Add health checks and automatic backend restart logic. Consider hot-standby backend for critical deployments. |

---

### 2. GPU Memory Exhaustion (OOM)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Affected Phases** | 3, 4, 6, 7, 8 |
| **Description** | Multiple components compete for GPU memory: steering vectors, KV pairs, LoRA adapters, L1 cache, and the model itself. Misconfigured budgets can cause OOM. |
| **Failure Mode** | CUDA OOM → process crash, service unavailable. |
| **Proposed Solution** | (1) Implement strict budget enforcement in Phase 7's `TieredMemoryStore`. (2) Add proactive memory monitoring with early warning at 80% utilization. (3) Dynamic budget reduction when memory pressure detected. (4) Document recommended budgets per GPU tier (8GB/24GB/80GB). |

---

### 3. Latency Accumulation

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Affected Phases** | 2, 3, 4, 5, 6 |
| **Description** | Each injection interface adds latency. When multiple interfaces are active simultaneously, latency compounds: gate evaluation + routing + encoding + injection + fallback chain. |
| **Failure Mode** | Response time exceeds acceptable thresholds (>1s for chat). |
| **Proposed Solution** | (1) Aggressive caching in Phase 6 (`EncodedMemoryStore`). (2) Parallel interface dispatch where possible. (3) Pre-fetch predicted memories during idle time. (4) Set strict latency budgets per interface (<10ms each). (5) Skip lower-priority interfaces if budget exhausted. |

---

## Phase-Specific Issues

### Phase 1: Foundation & Model Access Layer

#### 1.1 API-Only Backend Capability Gaps

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | `APIOnlyBackend` supports only `LOGIT_BIAS`. This means users with API-only access (e.g., OpenAI, Anthropic) cannot use activation steering, KV-cache injection, or weight adaptation. |
| **Impact** | Significantly reduced functionality for cloud-only deployments. |
| **Proposed Solution** | (1) Clearly document capability matrix by backend type. (2) Explore unofficial APIs that may expose logit distributions. (3) Consider hybrid architectures where a local model handles injection while API handles reasoning. |

#### 1.2 Model Architecture Detection Failures

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | `ModelSpec` derivation relies on heuristics to detect model architecture. New or unusual model architectures may not be correctly detected, leading to incorrect layer targeting. |
| **Failure Mode** | Steering vectors injected at wrong layers → ineffective or harmful. |
| **Proposed Solution** | (1) Maintain a manually curated registry of known architectures. (2) Add validation: run test injection and verify output changes. (3) Allow manual override of layer targeting via configuration. |

---

### Phase 2: Logit Interface

#### 2.1 kNN-LM Datastore Scaling

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The `MemoryDatastore` for kNN-LM grows with the number of memories and tokens per memory. At scale (100k+ memories), the FAISS index becomes large and slow. |
| **Impact** | Retrieval latency increases; memory usage grows unboundedly. |
| **Proposed Solution** | (1) Use FAISS IVF-PQ for approximate search. (2) Implement sharding by tenant/time. (3) Add datastore size limits with LRU eviction. (4) Periodically compact/rebuild the index. |

#### 2.2 Logit Bias Numerical Stability

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | Aggressive logit biases (e.g., +10 or -100) can dominate the softmax distribution, causing repetitive or nonsensical output. |
| **Failure Mode** | Model outputs degenerate to single repeated token. |
| **Proposed Solution** | (1) Clamp bias values within safe range ([-5, +5] as default). (2) Monitor for repetition ratio in Phase 10's `OutputQualityMonitor`. (3) Auto-reduce bias strength if repetition detected. |

---

### Phase 3: Activation Interface

#### 3.1 Steering Vector Derivation Quality

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The three derivation methods (CDD, Identity V, PCA) have different quality/cost tradeoffs. CDD requires contrastive pairs which may not be available. Identity V is fast but crude. PCA needs many samples. |
| **Impact** | Poor steering vectors → ineffective memory injection → wasted compute. |
| **Proposed Solution** | (1) Default to CDD with synthetic contrastive pairs (memory vs. negation). (2) Fall back to Identity V if CDD fails. (3) Train Phase 6's `ProjectionHead` to generate high-quality vectors from embeddings. (4) Add metrics to compare effectiveness of each method. |

#### 3.2 Layer Selection Sensitivity

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Steering effectiveness is highly sensitive to which layers are targeted. The plan uses heuristic (middle third of layers), but optimal layers vary by model and concept type. |
| **Failure Mode** | Injecting at wrong layers → negligible effect or harmful interference. |
| **Proposed Solution** | (1) Run layer sweep during initialization to find optimal injection points. (2) Store optimal layer configuration per model in registry. (3) Allow per-memory-type layer targeting (e.g., factual→deeper, style→earlier). |

---

### Phase 4: Synaptic Interface

#### 4.1 KV-Cache Budget Exhaustion

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | Each memory injects virtual tokens into the KV-cache. A 32-layer model with 64 virtual tokens per memory consumes ~2MB per memory. With 100 active memories, this is 200MB just for KV pairs. |
| **Impact** | Cache overflow → loss of older real context → degraded output quality. |
| **Proposed Solution** | (1) Strict per-turn KV budget (e.g., 32 virtual tokens total). (2) Compress KV pairs using quantization or low-rank approximation. (3) Prioritize high-relevance memories for KV injection. (4) Fall back to activation steering for lower-priority memories. |

#### 4.2 Temporal Decay Miscalibration

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The `TemporalDecayManager` fades memories over time, but decay rate is heuristic. Too fast → important memories lost. Too slow → stale memories pollute context. |
| **Failure Mode** | User's important long-term preference decays after a few turns. |
| **Proposed Solution** | (1) Use memory-type-specific decay rates (constraints decay slowly, episodic events decay fast). (2) Allow explicit "pin" flag to prevent decay. (3) Track memory access patterns to adjust decay dynamically. |

#### 4.3 RoPE Position Remapping Complexity

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | Rotary Position Embeddings (RoPE) rely on absolute position indices. Injecting virtual tokens *before* the prompt shifts all subsequent real token indices. If not handled perfectly, potential attention pattern disruption occurs. |
| **Failure Mode** | Subtly broken attention; model inability to attend to prompt correctly. |
| **Proposed Solution** | (1) Implement rigorous unit tests for `PositionRemapper`. (2) Use "position-free" virtual tokens where possible (attend via content, not position). (3) detailed integration tests comparing perplexity with/without remapping. |

---

### Phase 5: Controller & Gating Unit

#### 5.1 Relevance Gate False Negatives

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | If the `RelevanceGate` threshold is too high, relevant memories may be incorrectly skipped, resulting in the model lacking important context. |
| **Failure Mode** | User asks about a stored fact, but the memory is gated out → model hallucinates. |
| **Proposed Solution** | (1) Default to permissive threshold (0.3 instead of 0.5). (2) Monitor gate decisions in Phase 10 diagnostics. (3) Add "must inject" flag for high-value memory types (constraints). (4) User-adjustable sensitivity. |

#### 5.2 Fallback Chain Latency

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | When a primary interface fails, the fallback chain (KV→Activation→Logit→Context) adds latency by attempting each in sequence. |
| **Impact** | Worst case: 4× latency when all interfaces fail except context. |
| **Proposed Solution** | (1) Parallel probe: test interface availability before attempting. (2) Cache failure state to skip known-failed interfaces. (3) Set per-interface timeout (50ms) after which fallback triggers immediately. |

#### 5.3 Controller Latency Bottleneck

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The `MemoryController` performs sequential operations (Gate → Route → Calibrate) on the hot path. If these steps entail any standard CPU-bound heavy lifting or I/O, they add blocking latency to every generation request. |
| **Failure Mode** | TTFT (Time To First Token) increases noticeably. |
| **Proposed Solution** | (1) Ensure all Controller logic is async and non-blocking. (2) Profile `RelevanceGate` scoring; optimize vector operations. (3) Run controller in parallel with prompt pre-processing if possible. |

#### 5.4 Router: Only Two Channels Per Memory (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The Phase 5 `ROUTING_TABLE` assigns some memory types to **three** preferred channels (e.g. `semantic_fact` → KV, ACTIVATION, LOGIT). The implementation only creates a **primary** and optional **secondary** channel per memory (`available[0]`, `available[1]`). The third preferred channel is never used. |
| **Impact** | High-value memories do not get the full multi-interface reinforcement specified by the routing table; e.g. semantic facts may miss logit reinforcement when KV + Activation are chosen. |
| **Proposed Solution** | (1) Extend the router loop to assign up to N channels per memory (e.g. N=3) when budget allows, not just primary + secondary. (2) Or document that "at most two interfaces per memory" is intentional for latency/budget reasons and simplify the routing table to max two entries per type. |

---

### Phase 6: Memory Encoding Pipeline

#### 6.1 Encoding Pipeline Single Point of Failure

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | The `HippocampalEncoder` produces all representations (logit, activation, KV) in one pipeline. If the encoder fails, no representations are produced for that memory. |
| **Failure Mode** | Exception in encoder → memory cannot be used for injection. |
| **Proposed Solution** | (1) Isolate each representation stage with independent try/except. (2) Produce partial results (e.g., logit succeeds, KV fails → use logit). (3) Retry failed encodings with exponential backoff. (4) Log encoding failures prominently for debugging. |

#### 6.2 Projection Head Cold Start

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The `SteeringProjectionHead` requires training data (embedding, CDD_vector pairs). Initially, there's no data, so CDD must be used for everything (slow). |
| **Impact** | First N memories encoded slowly; projection head unusable until threshold met. |
| **Proposed Solution** | (1) Pre-train projection head on synthetic data (generated contrastive pairs). (2) Use Identity V as fast fallback until projection head is trained. (3) Lower `min_training_pairs` threshold to 50 for faster bootstrap. (4) Persist trained projection head to disk for restart. |

#### 6.3 EncodedMemory.to_memory_vector() KV Vectors Omitted (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Phase 6 `EncodedMemory.to_memory_vector()` sets `key_vectors=None` and `value_vectors=None` with a comment "Set separately for KV injection." The KV injection path (Phase 4) needs key/value tensors. No single place is specified that copies `EncodedMemory.kv_pairs` into the `MemoryVector` for the KV channel. |
| **Failure Mode** | KV injection receives MemoryVectors with null key_vectors/value_vectors; injection fails or is skipped. |
| **Proposed Solution** | (1) In `to_memory_vector()`, when `kv_pairs` is present, populate `key_vectors`/`value_vectors` from the first target layer (or a designated layer) so the returned MemoryVector is ready for KV injection. (2) Or document that the Synaptic Interface handler must resolve EncodedMemory from cache and use its kv_pairs when the bus delivers MemoryVectors produced by the encoder; ensure Controller/bus pass-through of EncodedMemory or memory_id for KV lookup. |

#### 6.4 Adaptive Encoding Mode Underspecified (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | The Hippocampal Encoder supports mode `"adaptive"`: "Choose mode based on memory type and backend capabilities." The decision matrix (memory_type × backend capabilities → mode) is not specified in the plan. |
| **Impact** | Implementers may guess; inconsistent behavior across deployments. |
| **Proposed Solution** | Add an explicit table or function spec: e.g. constraint/semantic_fact + full backend → "full"; preference + activation only → "fast"; API-only → "logit_only"; etc. |

---

### Phase 7: Memory Hierarchy & Cache Management

#### 7.1 Pre-Fetcher Accuracy

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | The `PredictivePreFetcher` uses heuristics (co-access patterns, session history) to predict needed memories. Inaccurate predictions waste I/O and cache space. |
| **Impact** | Low hit rate → pre-fetch overhead without benefit. |
| **Proposed Solution** | (1) Track pre-fetch hit rate as a metric. (2) Disable pre-fetch if hit rate falls below 30%. (3) Use topic embedding similarity as additional signal. (4) Limit pre-fetch to top-K candidates (K=10). |

#### 7.2 L3 Disk I/O Latency

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | L3 disk cache retrieval takes ~10ms. If many memories are cold, retrieval latency spikes. |
| **Impact** | First query after cold start is slow. |
| **Proposed Solution** | (1) Warm-up routine: pre-load recent session memories at startup. (2) Use async I/O to parallelize L3 reads. (3) Consider memory-mapped files for faster access. (4) SSD min requirement in documentation. |

#### 7.3 L1 GPU Cache: Recursive _to_device for EncodedMemory (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Phase 7 `L1GPUCache.put()` calls `_to_device(data, device)`. When `data` is an `EncodedMemory`, it contains nested structures: `steering_vectors: Dict[int, Tensor]`, `kv_pairs: Dict[int, Tuple[K, V]]`, and optional `embedding` tensor. A naive `_to_device` that only handles top-level `torch.Tensor` will leave nested tensors on CPU, causing device mismatch during injection. |
| **Failure Mode** | RuntimeError (tensors on different devices) when injection reads from L1 cache. |
| **Proposed Solution** | (1) Implement recursive `_to_device` that walks EncodedMemory (and CacheEntry payloads), moving every tensor to the target device. (2) Or require EncodedMemory to implement `.to(device)` and call it in L1GPUCache.put. (3) Document that all tensor-containing structures stored in L1 must be device-movable. |

---

### Phase 8: Weight Adaptation Interface

#### 8.1 LoRA Adapter Training Requirement

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | Phase 8 assumes pre-trained domain-specific LoRA adapters exist. Training these requires significant data, compute, and ML expertise. Most users won't have this capability. |
| **Impact** | Weight adaptation interface is inaccessible to most users. |
| **Proposed Solution** | (1) Ship pre-trained adapters for common domains (coding, medical, legal). (2) Provide training scripts and documentation. (3) Consider adapter marketplace/sharing. (4) Make weight adaptation optional; system works without it. |

#### 8.2 Hypernetwork Experimental Status

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | The `HyperNetwork` that generates LoRA weights on-the-fly is labeled "experimental" and "highly experimental" in the plan. It requires significant training and may produce low-quality weights. |
| **Impact** | Feature may not work reliably; could harm output quality. |
| **Proposed Solution** | (1) Disable hypernetwork by default. (2) Gate behind explicit feature flag with "experimental" warning. (3) Focus Phase 8 on pre-trained adapters; defer hypernetwork to later phase. (4) Add quality monitoring specific to hypernetwork-generated weights. |

---

### Phase 9: Integration & Migration

#### 9.1 Backward Compatibility Validation

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | The integration must maintain 100% backward compatibility with existing API clients. Any breaking change will disrupt production users. |
| **Failure Mode** | Existing `/memory/turn` calls fail or return different structure. |
| **Proposed Solution** | (1) Run full existing test suite in CI for every commit. (2) Add explicit regression tests for API contract. (3) Make `intrinsic` field in response optional (null when disabled). (4) Version the API if breaking changes are unavoidable. |

#### 9.2 Migration Mode Complexity

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Five migration modes (RAG_ONLY → HYBRID_SHADOW → HYBRID_ADDITIVE → INTRINSIC_PRIMARY → INTRINSIC_ONLY) add operational complexity. Operators may misconfigure or get stuck in intermediate states. |
| **Impact** | Confusion about system behavior; inconsistent results. |
| **Proposed Solution** | (1) Clear documentation with decision flowchart. (2) Dashboard showing current mode prominently. (3) Validation checks before mode progression (e.g., must show improvement in shadow before activating). (4) Single-button "advance migration" with safety checks. |

#### 9.3 Seamless Provider Blocking

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | The `SeamlessMemoryProvider` calls `controller.process_retrieval` on the critical request path. If the intrinsic pipeline has high latency (e.g. from cold-start encoding or complex routing), it blocks the standard RAG response. |
| **Impact** | User perception of system slowness; potential timeouts. |
| **Proposed Solution** | (1) Implement strict timeout for intrinsic processing; if timeout, proceed with RAG only. (2) Run intrinsic encoding in background if not required immediately (though pre-fill needs it). (3) Ensure `process_retrieval` is fully async. |

#### 9.4 Startup Dependencies & Initialization Order (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | Phase 9 startup code references `app.state.tokenizer`, `app.state.llm_client`, `app.state.adapter_registry`, `app.state.embedding_client`, and `app.state.seamless_provider`. The **existing** app (e.g. `src/api/app.py`) does not create these. If intrinsic init runs first, AttributeError or missing components will occur. |
| **Failure Mode** | Application fails to start; or intrinsic init runs but cannot wire into SeamlessMemoryProvider because it does not exist yet. |
| **Proposed Solution** | (1) Extend Phase 9 Task 1 to explicitly create and attach: tokenizer (from config or from backend when local), adapter_registry (Phase 8), embedding_client (use existing `get_embedding_client()`), and SeamlessMemoryProvider (create before or during intrinsic init). (2) Define strict startup order in a single lifespan: DB → Orchestrator → (optional) SeamlessProvider → Intrinsic (which then sets `seamless_provider._controller`). (3) Document that when intrinsic is disabled, tokenizer/adapter_registry need not be created. |

#### 9.5 Tokenizer Availability for API-Only Backend (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | For API-only backends, Phase 1's `APIOnlyBackend` accepts `tokenizer=None`. The Logit Interface and Phase 6's `TokenMemoryMapper` require a tokenizer to map entities to token IDs for logit bias. Without a tokenizer, logit bias cannot be produced for API-only deployments. |
| **Failure Mode** | API-only users get no logit bias; TokenMemoryMapper fails or is skipped. |
| **Proposed Solution** | (1) Maintain a mapping from API model name to HuggingFace/tiktoken tokenizer (e.g. `gpt-4o` → `cl100k_base`). (2) When backend is API-only, load a compatible tokenizer for entity→token_id mapping only. (3) Document that logit bias for API models uses best-effort tokenizer alignment. (4) If no tokenizer can be resolved, disable logit interface for that backend and document capability matrix. |

---

### Phase 10: Observability & Benchmarking

#### 10.1 Counterfactual Analysis Cost

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | The `InjectionAttributor` runs N+1 forward passes to attribute influence (one per memory). This is prohibitively expensive for real-time use. |
| **Impact** | Attribution is too slow for production; only usable for debugging. |
| **Proposed Solution** | (1) Make attribution an explicit on-demand operation (not automatic). (2) Add API endpoint for debug attribution. (3) Implement approximate attribution using gradient-based methods (faster). (4) Cache counterfactual results by query pattern. |

#### 10.2 Kill Switch Over-Triggering

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | The `EmergencyKillSwitch` may trigger too easily if perplexity thresholds are misconfigured, disabling intrinsic memory unnecessarily. |
| **Impact** | System oscillates between intrinsic and RAG-only modes. |
| **Proposed Solution** | (1) Calibrate thresholds based on baseline perplexity (adaptive). (2) Require N consecutive violations before triggering (hysteresis). (3) Auto-recover: try re-enabling after 5 minutes. (4) Manual confirmation required to trigger in production. |

#### 10.3 API Logprob Unavailability

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | `OutputQualityMonitor` relies on perplexity calculations. Many commercial APIs (e.g., Anthropic, some OpenAI models) do not return full logprobs, making perplexity calculation impossible. |
| **Failure Mode** | Safety guardrails blindly pass; quality degradation goes undetected. |
| **Proposed Solution** | (1) Implement "proxy" metric for API backends (e.g., self-consistency check). (2) Use a small local model to evaluate output coherence (model-based eval). (3) Disable perplexity-based guards when using API backends. |

---

## Dependency & Ordering Risks

### Timeline Risk

| Attribute | Details |
|-----------|---------|
| **Risk Level** | HIGH |
| **Description** | Total estimated duration: 28-42 weeks (6.5-10 months). This is a long timeline with many dependencies. Phases 1-5 must complete before 6-8 can start effectively. |
| **Impact** | Delays compound; late phases may start with incomplete dependencies. |
| **Proposed Solution** | (1) Parallelize where possible (Phase 7 can proceed alongside 6). (2) Define clear "minimally viable" milestones. (3) Add buffer time (20%) to estimates. (4) Consider phased releases (MVP with Phase 1-5 only). |

### Testing Debt

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Each phase has its own testing strategy, but integration testing across phases is not explicitly planned. Inter-phase bugs may surface late. |
| **Impact** | Integration bugs found during Phase 9 require rework in earlier phases. |
| **Proposed Solution** | (1) Start integration testing environment from Phase 2 onwards. (2) Add end-to-end smoke tests that run after each phase. (3) Dedicate Phase 9's first week to integration testing before new development. |

---

## Documentation Gaps

### 1. Error Handling Not Fully Specified

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Phase plans focus on happy path. Error scenarios (encoding failure, hook crash, cache corruption) lack detailed handling specifications. |
| **Proposed Solution** | Add "Error Handling" section to each phase document specifying: (1) Expected failure modes. (2) Recovery mechanisms. (3) Logging requirements. (4) Alerting thresholds. |

### 2. Configuration Reference Missing

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | Configuration parameters are scattered throughout the plans (thresholds, budgets, timeouts) but there's no unified configuration reference. |
| **Proposed Solution** | Create `ConfigurationReference.md` listing all parameters with: name, type, default, valid range, description, and which phase introduces it. |

### 3. Security Model Underspecified

| Attribute | Details |
|-----------|---------|
| **Risk Level** | MEDIUM |
| **Description** | Phase 10 mentions security (KV-cache doesn't leak, steering vectors can't be reversed) but lacks threat modeling and explicit security requirements. |
| **Proposed Solution** | Add security section covering: (1) Multi-tenant isolation. (2) Memory access control. (3) Injection audit logging. (4) Rate limiting. (5) Input validation for memory content. |

### 4. Two Phase Numbering Systems (NEW)

| Attribute | Details |
|-----------|---------|
| **Risk Level** | LOW |
| **Description** | `ProjectStatus.md` describes an **original** project plan with phases 1–10 (Foundation & Core Data Models, Sensory Buffer, Hippocampal Store, etc.). The **intrinsic memory** plan uses the same phase numbers 1–10 in separate documents (Phase1_Foundation_ModelAccessLayer, Phase2_LogitInterface, …) for a different scope. Readers can confuse "Phase 3" (Hippocampal Store vs Activation Interface). |
| **Impact** | Miscommunication; incorrect dependency assumptions. |
| **Proposed Solution** | (1) Rename intrinsic phase documents or add a prefix (e.g. "Intrinsic Phase 1", or keep file names but add a prominent note in ProjectStatus.md: "Intrinsic memory plan (Phases 1–10) is separate from the core CML phase list below."). (2) In ProjectStatus.md, add a subsection "Intrinsic Memory Implementation (Phases I-1 to I-10)" with a clear mapping to Phase*_*.md files. |

---

## Summary

| Category | HIGH Risk | MEDIUM Risk | LOW Risk |
|----------|-----------|-------------|----------|
| Cross-Cutting | 3 | 1 | 0 |
| Phase 1-5 | 2 | 8 | 2 |
| Phase 6-10 | 5 | 10 | 2 |
| Dependency/Timeline | 1 | 1 | 0 |
| Documentation | 0 | 2 | 2 |
| **Total** | **11** | **22** | **6** |

### Recommended Prioritization

1. **Address before starting implementation:**
   - **Request concurrency & hook state isolation** (define InjectionContext / request-scoped state)
   - GPU Memory Exhaustion (OOM) mitigation
   - Backend single point of failure handling
   - **Phase 9 startup dependencies** (tokenizer, adapter_registry, embedding_client, seamless_provider, init order)
   - LoRA adapter training requirement (or defer Phase 8)
   - Backward compatibility validation framework

2. **Address during respective phase:**
   - All phase-specific MEDIUM risks
   - Phase 5: Router multi-channel limit (5.4)
   - Phase 6: EncodedMemory KV in to_memory_vector (6.3); adaptive mode matrix (6.4)
   - Phase 7: L1 _to_device for EncodedMemory (7.3)
   - Phase 9: Tokenizer for API-only backend (9.5)

3. **Monitor during operation:**
   - Kill switch calibration
   - Pre-fetcher accuracy
   - Counterfactual analysis performance
