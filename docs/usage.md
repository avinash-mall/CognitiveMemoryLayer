# CML Usage Documentation

Last verified: 2026-03-05

This document is the canonical server-side usage guide for Cognitive Memory Layer (CML). It is aligned to the current implementation in `src/`.

## Overview

CML exposes a tenant-scoped FastAPI API and coordinates:

- PostgreSQL + pgvector for episodic memory vectors (`memory_records`)
- PostgreSQL semantic facts for neocortical facts (`semantic_facts`)
- Neo4j for entity and relation graph storage
- Redis for rate limiting, embedding cache, labile state, and dashboard counters
- Optional Celery workers for scheduled forgetting fan-out

Primary coordinator: `src/memory/orchestrator.py`.

## Runtime Modes

`FEATURES__USE_LLM_ENABLED` picks between one intelligence path and one deterministic
fallback. There is no third model tier — the custom-model ("modelpack") runtime was
removed in commit 51afd15.

### LLM-enabled mode (recommended)

`FEATURES__USE_LLM_ENABLED=true`, provider settings from `LLM_INTERNAL__*`.

`MemoryOrchestrator.create()` wires `UnifiedWritePathExtractor`, which makes one LLM call
per chunk and returns entities, relations, constraints, write-time facts, PII spans,
memory type, importance, salience, confidence, context tags and decay rate. The read path
uses the same client for query classification, conflict detection, constraint
supersession, consolidation gists and forgetting compression.

#### Write path field sources (LLM on)

```mermaid
flowchart LR
    Chunk["SemanticChunk"] --> UWE["UnifiedWritePathExtractor<br/>(one LLM call)"]
    UWE --> Fields["entities, relations, constraints, facts,<br/>pii_spans, memory_type, importance,<br/>salience, confidence, context_tags, decay_rate"]
    Fields --> Gate["WriteGate<br/>(uses unified importance + pii_spans)"]
    Gate --> Record["MemoryRecordCreate"]
```

### Heuristic mode (LLM off)

`FEATURES__USE_LLM_ENABLED=false` (the default in `src/core/config.py`). No internal LLM
client is constructed. This is the mode the hermetic unit suite runs in.

| Field | Source | Notes |
|-------|--------|-------|
| entities / relations | — | Not extracted; the graph stays empty |
| constraints | `ConstraintExtractor` regex patterns + chunk-type map | Single constraint per chunk; scope from chunk entities |
| facts | `WriteTimeFactExtractor` regex families | preference / identity / location / occupation |
| importance | Upstream chunk salience | Clamped to [0,1] |
| memory_type | Write-gate chunk_type map | Full coverage |
| context_tags | Caller-provided only | No inference |
| confidence | Chunk confidence | No inference |
| decay_rate | Type default | No inference |
| PII detection | Regex pattern table (`src/utils/ner.py`) | Typed spans, no model |
| novelty | Jaccard word overlap | See `WriteGate._compute_novelty` |
| contains_secrets | Regex patterns in write gate | Deterministic |

Consolidation produces no gists in this mode (`GistExtractor.extract_gist` returns `[]`
without an LLM); forgetting compression falls back to deterministic truncation.

## Neuro-Cognitive, Laya System 1 & ActiveCML Feature Flags

All neuro-inspired and intrinsic-memory subsystems are independently toggleable via `FEATURES__*` environment variables (see `src/core/config.py`):

| Env Var | Default | Subsystem |
| :--- | :---: | :--- |
| `FEATURES__SOMATIC_MARKERS_ENABLED` | `true` | Amygdala valence/arousal tagging, flashbulb decay modulation, state-congruent resonance (`src/memory/somatic.py`) |
| `FEATURES__WORKING_MEMORY_CONTEXT_BINDING_ENABLED` | `true` | Baddeley Episodic Buffer cross-turn coreference binding & zero-latency read overlay |
| `FEATURES__PROSPECTIVE_INDEXING_ENABLED` | `true` | BA10 causal intention triggers + read-time parent-episode ecphory (`src/memory/prospective.py`) |
| `FEATURES__ECPHORY_REACTIVATION_ENABLED` | `true` | Bjork dual-strength reactivation of dormant (`SILENT`) episodes on high-specificity cues |
| `FEATURES__REALITY_MONITORING_ENABLED` | `true` | CA1 comparator & OFC presupposition / speaker-attribution verification (`src/retrieval/reality_monitor.py`) |
| `FEATURES__DMN_THEORY_OF_MIND_ENABLED` | `true` | Default Mode Network (mPFC/TPJ) mentalizing synthesis (`src/memory/dmn.py`) |
| `FEATURES__MULTI_HOP_RECOLLECTION_ENABLED` | `true` | Feeling-of-Knowing gated 2nd-hop bridge entity recollection |
| `FEATURES__INTRINSIC_MEMORY_ENABLED` | `true` | ActiveCML tiered intrinsic memory (`src/intrinsic/`: logit bias, steering hooks, KV-prefix, LoRA routing) |
| `FEATURES__ADAPTIVE_EPISODE_BUDGET_ENABLED` | `true` | Adapts episode cap by query intent (`12` for temporal/multi-hop/common-sense, `6` for focused/cognitive) + protects verbatim somatic episodes on Cognitive cues |
| `FEATURES__LAYA_SYSTEM1_ENABLED` | `false` | Enables Convai **Laya** (`convaiinnovations/laya`, ~20ms GPU) for calibrated System 1 query routing and CA1 grounding verification (`src/core/laya_system1.py`). Requires `pip install laya` with `transformers==5.12.1` when paired with `nomic-embed-text-v2-moe`. |
| `FEATURES__THALAMIC_TRIAGE_ENABLED` | `false` | Surprise-gated System 1/System 2 write triage (`FEATURES__THALAMIC_SURPRISE_THRESHOLD=0.45`) |

## Evaluation Runners (`LoCoMo-Plus` & `LongMemEval-V2`)

Runtime evaluation outputs and committed summary JSONs are organized by benchmark under `evaluation/outputs/{locomo_plus,longmemeval_v2}` and `evaluation/results/{locomo_plus,longmemeval_v2}`.

### LoCoMo-Plus (OpenRouter Sync/Batch or Local)

Build the full 2,387-sample dataset (`cd evaluation/locomo_plus/data && python unified_input.py`) and run against either standard OpenRouter models (e.g. `openai/gpt-4o`) or `:batch` models (e.g. `z-ai/glm-5.3-flash:batch`):

```bash
OPENROUTER_API_KEY=... python evaluation/scripts/run_locomo_openrouter_batch.py \
  --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json \
  --out-dir evaluation/outputs/locomo_plus/gpt4o_gemini25flash_full_2387 \
  --cml-url http://localhost:8000 \
  --model openai/gpt-4o \
  --judge-model google/gemini-2.5-flash
```

### LongMemEval-V2 (100% Local Multimodal Agent Benchmark)

Prepare the `xiaowu0162/longmemeval-v2` dataset under `evaluation/longmemeval_v2/data/longmemeval-v2`, pre-ingest trajectories on GPU, and run all `451` (`web` + `enterprise`) questions against a local OpenAI-compatible LLM/VLM server:

```bash
python evaluation/scripts/run_longmemeval_v2_local.py --tier small --preingest-only
python evaluation/scripts/run_longmemeval_v2_local.py \
  --tier small \
  --methods cml \
  --local-base-url http://127.0.0.1:8023/v1 \
  --local-model qwen3.8-27b \
  --cml-url http://127.0.0.1:8000 \
  --memory-context-max-tokens 16500 \
  --skip-existing \
  --resume
```

## API Base And Auth

Base path: `/api/v1`

Required header:

- `X-API-Key: <AUTH__API_KEY>`

Optional headers:

- `X-Tenant-ID: <tenant>` (applies only when using admin key)
- `X-User-ID: <user>`
- `X-Eval-Mode: true` (write and session-write only)

Important tenancy rule:

- `X-Tenant-ID` override is accepted only for admin-authenticated requests.

## Endpoint Reference

### Memory endpoints

- `POST /memory/write`
- `POST /memory/write_batch`
- `POST /memory/turn`
- `POST /memory/read`
- `POST /memory/read/stream` (SSE)
- `POST /memory/update`
- `POST /memory/forget`
- `POST /memory/consolidate_windowed`
- `POST /memory/feedback`
- `POST /memory/simulate`
- `GET /memory/stats`
- `DELETE /memory/all` (admin only)

### Session convenience endpoints

- `POST /session/create`
- `POST /session/{session_id}/write`
- `POST /session/{session_id}/read`
- `GET /session/{session_id}/context`

### Admin endpoints

- `POST /admin/consolidate/{user_id}`
- `GET /admin/consolidation/status`
- `POST /admin/forget/{user_id}`

### Service and observability endpoints

- `GET /health`
- `GET /metrics` (Prometheus)

## Dashboard Surface

Dashboard API is mounted under `/api/v1/dashboard/*` and static SPA under `/dashboard/*`.

Dashboard API modules include:

- overview, timeline, components, tenants, sessions, ratelimits, request-stats
- memories list/detail/bulk-action/export
- graph stats/overview/explore/search/neo4j-config (with relationship edge labels and improved physics)
- semantic facts list/invalidate (Facts Explorer page with category/tenant filters, current-only toggle, pagination)
- config get/update
- jobs/labile/retrieval test/consolidate/forget/reconsolidate/database reset

Security behavior:

- Dashboard API endpoints require admin auth.
- State-changing dashboard requests (`POST`, `PUT`, `DELETE`, `PATCH`) require `X-Requested-With: XMLHttpRequest` (CSRF middleware).

## Request/Response Notes

- `WriteMemoryRequest.content` max length is `100_000`.
- `WriteMemoryRequest.timestamp` and `ProcessTurnRequest.timestamp` are optional and propagate through the write path.
- `MemoryRecordCreate` accepts optional `valid_from` / `valid_to`: when the described
  state of affairs *held*, as opposed to when the turn was written. Both default to
  `None`, meaning "no stated bound". `vector_search`'s `exclude_expired` filter reads
  `valid_to`, so a record whose interval has closed stops being retrieved by that path.
- `POST /admin/consolidate/{user_id}` returns `episodes_sampled`, `clusters_formed`,
  `clusters_skipped_no_recurrence`, `gists_extracted` and `details_recovered`.
- `GET /admin/consolidation/status` reports whether the background sweep is running plus
  `sweeps_run`, `tasks_enqueued`, `consolidations_run` and `pending_tasks`. Counts rather
  than config, because the scheduler previously had no caller and reading the
  configuration would have said it was enabled.
- `ReadMemoryRequest.max_results` max is `50`.
- `ReadMemoryRequest.format` supports `packet`, `list`, and `llm_context`.
- `ReadMemoryResponse` includes optional `retrieval_meta`.
- Streaming read (`/memory/read/stream`) emits item events and a final `done` event with summary metadata.
- `POST /session/create` stores session metadata in Redis when available; if Redis is unavailable it still returns a session id, but no Redis-backed session record is created.

## Write Pipeline (Current Execution Order)

For `POST /memory/write` and write steps in `/memory/turn`:

1. STM ingest and chunking (`short_term.ingest_turn`)
2. Optional unified extraction (LLM path)
3. Constraint supersession/deactivation by key
4. Batched encode/store in hippocampal store
   - write gate decision (`store`, `skip`, `redact_and_store`)
   - redaction
   - batch embed
   - batch entity/relation extraction
   - bounded upsert
5. Best-effort graph sync to Neo4j
6. Write-time facts to semantic store
7. Write-time constraints to semantic store

Key behavior:

- Memory type may come from API override, LLM extraction, or gate mapping.
- Constraints can supersede previous constraints and invalidate older semantic facts.
- Request metadata is merged with system metadata (request metadata wins on key conflicts).

## Read Pipeline (Current Execution Order)

1. Query classification (`QueryClassifier`) + System 1 routing (`LayaSystem1Engine`)
   - LLM classification when `FEATURES__USE_LLM_ENABLED` (default `true`) and an `LLM_INTERNAL__*` OpenAI-compatible client is configured
   - regex intent/constraint heuristics otherwise (and always applied on top)
2. Retrieval plan generation (`RetrievalPlanner`) with guaranteed dense vector coverage
3. Stage-1 Multi-Angle Expansion & 4-Pathway Ecphory (`MemoryRetriever` + `RetrievalAugmentersMixin`)
   - concurrent Dense `pgvector` + conjunction-boosted IDF `tsvector` BM25 + `<5ms` Cortical Hypergraph PPR + Grounded Proustian Cue Pointers
   - multi-angle query expansion (`FEATURES__LLM_MULTI_ANGLE_EXPANSION_ENABLED`, default `true`), non-hub entity-bridge boosting, and `±2` post-merge conversational contiguity
4. Wide Reranking (`MemoryReranker`) + Stage-2 Judged Delivery (`HyDEGenerator.evaluate_candidates_and_grounding`)
   - CUDA true-logit Cross-Encoder + relevance/recency/confidence/diversity over wide candidate pool
   - batched `LLM_INTERNAL` candidate selection (`FEATURES__LLM_JUDGED_DELIVERY_ENABLED`, default `true`), optional 2nd-hop bridge fetch, and CA1 evidence-grounding verdict (`supported` | `inferable` | `speaker_swap` | `unsupported_premise`)
5. Packet build (`MemoryPacketBuilder`)
   - CLS stratified slot allocation (prioritizing verbatim episodes and atomic facts while capping long chapter gists)
   - CA1 speaker-attribution & presupposition warnings, deterministic Hippocampal Time Cell deltas, DMN self-profiles, and conflict/supersession suppression

Temporal behavior:

- `today` and `yesterday` planning uses `ReadMemoryRequest.user_timezone` when provided.

## Configuration Reference

Source of truth: `src/core/config.py`.

### Config groups

- `DATABASE__*`
- `AUTH__*`
- `EMBEDDING_INTERNAL__*`
- `LLM_INTERNAL__*`
- `LLM_EVAL__*`
- `CHUNKER__*`
- `FEATURES__*`
- `RETRIEVAL__*`

### Important feature flags

- `FEATURES__USE_LLM_ENABLED` (default `true` — master switch for `LLM_INTERNAL__*` across write, consolidation, and read; accepts any OpenAI-compatible external or local endpoint)
- `FEATURES__BATCH_WRITE_SYSTEM1_FASTPATH` (default `true` — Hybrid CLS: fast-paths `/memory/write_batch` through System-1 GPU encoding at `>300 turns/s` so `WindowedMicroSleepConsolidator` handles multi-turn LLM extraction)
- `FEATURES__HYDE_RETRIEVAL_ENABLED` (default `true`)
- `FEATURES__LLM_MULTI_ANGLE_EXPANSION_ENABLED` (default `true` — Stage-1 multi-angle query expansion)
- `FEATURES__LLM_JUDGED_DELIVERY_ENABLED` (default `true` — Stage-2 batched candidate evaluation, 2nd-hop bridge fetch, and CA1 grounding verdict)
- `FEATURES__CONSTRAINT_EXTRACTION_ENABLED` (default `true`)
- `FEATURES__WRITE_TIME_FACTS_ENABLED` (default `true`)
- `FEATURES__CACHED_EMBEDDINGS_ENABLED` (default `true`)
- `FEATURES__RETRIEVAL_TIMEOUTS_ENABLED` (default `true`)
- `FEATURES__SKIP_IF_FOUND_CROSS_GROUP` (default `true`)
- `FEATURES__HNSW_EF_SEARCH_TUNING` (default `true`)
- All fine-grained `FEATURES__USE_LLM_*` flags

Consolidation (all three default to a no-op; consolidation is otherwise manual-only via
the admin route):

- `FEATURES__CONSOLIDATION_SCHEDULER_ENABLED` (default `false`) — run the background
  sweep that enqueues any tenant holding at least the episode quota in un-consolidated
  records. Assumes a single API worker: `lifespan` runs per uvicorn process and the queue
  is in-memory, so N workers means N sweeps over the same tenants.
- `FEATURES__CONSOLIDATION_RECURRENCE_MIN` (default `1`) — minimum episodes in a cluster
  before it is worth an LLM gist call. `1` gates nothing; at `2` a tenant whose episodes
  never cluster produces no gists at all.
- `FEATURES__CONSOLIDATION_DETAIL_RECOVERY_ENABLED` (default `false`) — second
  consolidation pass that uses the extracted gist as a reference to find facts it
  omitted. Costs roughly one extra LLM call per cluster.

### Dashboard config editability

Dashboard config writes through `src/core/env_file.py` and persists to `.env`, but:

- DB URLs and key/password fields are write-protected.

## Semantic Lineage

When facts are superseded during reconsolidation or consolidation, CML tracks the lineage chain via `supersedes_id`. The following APIs expose this lineage:

- `SemanticFactStore.get_fact_lineage(tenant_id, key=..., fact_id=...)` — returns the full supersession chain (oldest to newest) for a given fact key or starting fact ID.
- `SemanticFactStore.get_superseded_chain(tenant_id, fact_id)` — returns all facts that were superseded by a given fact (forward chain).

Lineage metadata is captured during consolidation and reconsolidation operations. The `supersedes_id` field is surfaced in:

- `ReadMemoryResponse` memory items (when a memory supersedes a previous version)
- Dashboard retrieval test results (`RetrievalResultItem.supersedes_id`)
- Dashboard memory detail page (visual lineage chain in the Provenance section)
- py-cml SDK `RetrievalResultItem.supersedes_id`

## Shadow Mode Logging

`src/utils/shadow_logger.py` provides `ShadowModeLogger` for running heuristic and model paths in parallel:

- `compare()` runs both paths concurrently and returns the heuristic result
- Logs latency deltas and decision disagreements
- Configurable `sample_rate` for production use (e.g. 0.1 = 10% of requests)

Used by the conflict evaluation framework and during model rollout validation.

## Operations

### Start local dependencies

```bash
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
```

### Apply migrations

```bash
alembic upgrade head
```

### Run API locally

```bash
uvicorn src.api.app:app --host 0.0.0.0 --port 8000
```

### Run API in Docker

```bash
docker compose -f docker/docker-compose.yml up -d api
```

The Docker API startup command runs:

1. `python scripts/validate_llm_endpoints.py`
2. `alembic upgrade head`
3. `uvicorn src.api.app:app ...`

Startup LLM validation controls:

- `LLM_STARTUP_VALIDATION_ENABLED` (default true)
- `LLM_STARTUP_VALIDATE_IN_CI` (default false)
- `LLM_STARTUP_VALIDATION_TIMEOUT_SEC` (default 3)

### Run tests

```bash
pytest tests/unit -v --tb=short
pytest tests/integration -v --tb=short
pytest tests/e2e -v
```

Operational note from current test setup:

- `tests/unit/test_api_ingestion.py` and `tests/unit/test_workflow.py` require a running API despite location under `unit/`.

## Examples

### Write

```bash
curl -X POST "http://localhost:8000/api/v1/memory/write" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${AUTH__API_KEY}" \
  -d '{
    "content": "I am allergic to shellfish and prefer vegetarian meals.",
    "context_tags": ["profile", "diet"],
    "session_id": "s-001"
  }'
```

### Read

```bash
curl -X POST "http://localhost:8000/api/v1/memory/read" \
  -H "Content-Type: application/json" \
  -H "X-API-Key: ${AUTH__API_KEY}" \
  -d '{
    "query": "suggest dinner options",
    "max_results": 10,
    "format": "packet",
    "user_timezone": "America/New_York"
  }'
```

## Current Caveats

- `POST /memory/forget` accepts `delete|archive|silence`, but non-`delete` actions currently execute the same soft-delete path in storage.

## Related Docs

- Root overview: [README.md](../README.md)
- SDK docs: [packages/py-cml/README.md](../packages/py-cml/README.md)
- SDK eval module: [packages/py-cml/docs/evaluation.md](../packages/py-cml/docs/evaluation.md) (`cml-eval` CLI and Python API)
- Changelog: [CHANGELOG.md](../CHANGELOG.md)
