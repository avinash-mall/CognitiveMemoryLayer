# Base CML Status

Last verified: 2026-03-05

This status file reflects the current implementation in `src/`, `migrations/`, and `docker/`.

## Executive Summary

Cognitive Memory Layer is running with a complete FastAPI memory API, retrieval stack, dashboard/admin surface, and production storage backends (PostgreSQL + pgvector, Neo4j, Redis). The major architecture pieces are implemented and integrated, with a few known gaps where code exists but is not fully wired into runtime paths.

## Current Delivery Status

| Area | Status | Notes |
|---|---|---|
| API surface (`/api/v1`) | Complete | Memory write/read/turn/update/forget/stats, session endpoints, health, streaming read, admin endpoints. |
| Auth and tenancy | Complete | API key auth, admin key permissions, admin-only tenant override via `X-Tenant-ID`. |
| Write pipeline | Complete | STM ingest, write gate, redaction, batch embeddings, extraction, fact/constraint write-time storage, graph sync. |
| Retrieval pipeline | Complete | Classifier -> planner -> hybrid retriever -> reranker -> packet builder with retrieval diagnostics. |
| Constraint layer | Complete | Constraint extraction, supersession/deactivation, constraint-first retrieval and formatting. |
| Consolidation | Complete | Sampling, clustering, gist extraction, migration, consolidation-specific supersession handling. |
| Forgetting | Complete | Scoring, policy planning, execution, duplicate resolution, optional Celery scheduling. |
| Reconsolidation | Complete | Labile tracking, conflict detection, belief revision operation planning and application. |
| Dashboard API + SPA mount | Complete | Operational, memory, events, graph, config, jobs, fact management endpoints. |
| Storage schema/migrations | Complete | Unified initial migration includes `event_log`, `memory_records`, `semantic_facts`, `dashboard_jobs`. |
| Observability | Complete | `/metrics`, request logging, retrieval metrics, DB pool metrics, optional OTel tracing. |
| Docker runtime | Complete | API container runs LLM endpoint validation + migration + uvicorn. |

## Implemented API Inventory

### Core memory and session routes

- `POST /api/v1/memory/write`
- `POST /api/v1/memory/turn`
- `POST /api/v1/memory/read`
- `POST /api/v1/memory/read/stream`
- `POST /api/v1/memory/update`
- `POST /api/v1/memory/forget`
- `GET /api/v1/memory/stats`
- `DELETE /api/v1/memory/all` (admin)
- `POST /api/v1/session/create`
- `POST /api/v1/session/{session_id}/write`
- `POST /api/v1/session/{session_id}/read` (deprecated)
- `GET /api/v1/session/{session_id}/context`
- `GET /api/v1/health`
- `GET /metrics`

### Admin routes

- `POST /api/v1/admin/consolidate/{user_id}`
- `POST /api/v1/admin/forget/{user_id}`

### Dashboard routes (`/api/v1/dashboard/*`)

- Overview/timeline/components/tenants/sessions/ratelimits/request-stats
- Memory list/detail/bulk-action/export
- Events list
- Graph stats/overview/explore/search/neo4j-config
- Config get/update
- Jobs, labile state, retrieval testing, consolidate/forget/reconsolidate, database reset
- Semantic facts list and invalidate

## Core Runtime Behavior Snapshot

### LLM controls

- Master switch: `FEATURES__USE_LLM_ENABLED` (default `false`).
- When disabled, orchestrator does not create internal LLM clients.
- Fine-grained `FEATURES__USE_LLM_*` flags gate specific subpaths when master switch is enabled.

### Write path (high-level)

1. Ingest and chunk through short-term memory.
2. Optional unified extraction for LLM path.
3. Deactivate superseded constraints/facts.
4. Encode and store via batched hippocampal pipeline.
5. Sync entities/relations to Neo4j (best effort).
6. Persist write-time semantic facts.
7. Persist write-time constraints.

### Read path (high-level)

1. Query classification (modelpack-first, optional LLM fallback).
2. Retrieval planning with parallel groups and timeout budgets.
3. Hybrid retrieval across vector/facts/graph/constraints/cache.
4. Rerank by relevance/recency/confidence/diversity plus constraint boosts.
5. Build packet and formatted context with constraints-first budget.

## Storage And Schema Status

Implemented tables:

- `event_log`
- `memory_records`
- `semantic_facts`
- `dashboard_jobs`

Storage behaviors currently active:

- content-hash dedupe and key-based upsert updates
- query-time HNSW `ef_search` tuning (feature-flagged)
- vector filters for type/session/time/confidence/expiration
- bulk dependency counting for forgetting paths
- constraint deactivation by key with supersession lineage metadata

## Dashboard Config Status

Dashboard config updates persist to `.env` through `src/core/env_file.py`.

Current protection rules:

- DB URLs and key/password fields are non-editable.
- Secret fields are masked.
- `features.use_llm_enabled` (master LLM switch) is not currently in the editable dashboard feature set.

## Background Processing Status

- Celery app and beat schedule implemented for forgetting fan-out.
- Consolidation worker implemented and callable from admin/dashboard flows.
- Forgetting worker implemented with dry-run support.
- Reconsolidation service implemented with labile tracker and revision operations.

## Known Gaps And Caveats

1. `FEATURES__STORE_ASYNC` is implemented in config and `AsyncStoragePipeline`, but `/api/v1/memory/turn` currently does not inject async pipeline; writes remain synchronous.
2. `src/core/tenant_flags.py` (per-tenant Redis overrides) exists but is not wired into active runtime decision points.
3. Event log schema and repository exist, but write/read/update/forget runtime paths do not currently auto-append event-log records.
4. `POST /memory/forget` accepts actions `delete|archive|silence`, but non-`delete` actions currently use the same storage soft-delete path.
5. `POST /session/{session_id}/read` is still present but marked deprecated.

## Recommended Next Upgrades

1. Wire `AsyncStoragePipeline` into `/memory/turn` when `FEATURES__STORE_ASYNC=true`.
2. Integrate `tenant_flags` override resolution into orchestrator/retrieval/write-gate paths.
3. Add event-log append hooks for write/read/update/forget and background jobs.
4. Align forget action semantics with API contract (`archive` and `silence` should map to distinct statuses).
5. Expose `features.use_llm_enabled` in dashboard config (with explicit warning and restart requirement).
