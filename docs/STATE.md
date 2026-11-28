# Working state

Current snapshot of active work and load-bearing architecture invariants. Keep this file current as work progresses; delete notes when they stop being true.

## Where things stand

- **Release & Version**: Synchronized at **`3.0.1`** across `VERSION`, `.env.minimal`, and `packages/py-cml/src/cml/_version.py` (`hatch_build.get_version()` resolves `VERSION` env -> `.env` -> `VERSION` file).
- **CI**: Both `CI/CD Pipeline` (`lint`, `test` with `>=75%` coverage gate across `src/` + `py-cml`, and GHCR `build`) and `py-cml: CI and release` (`lint`, Python 3.11/3.12/3.13 matrix with `>=55%` branch coverage gate, wheel/sdist build) are green on `main`.
- **Docker vs. Source Testing**: The Docker `api` container bakes `src/` in without a volume mount (published on `:6000`). To test working-tree changes against local Postgres/Neo4j/Redis, run `uvicorn src.api.app:app --port 8000` from source with explicit env overrides for injected secrets (see `AGENTS.md`).

## Shipped Architecture (CML v3.0 Whole-Brain + Refactoring Pass)

- **Whole-Brain v3.0 Subsystems**:
  - `src/consolidation/windowed_consolidator.py`: CRC32 content-defined `WindowedMicroSleepConsolidator` + SQLite content-addressable extraction cache (`POST /api/v1/memory/consolidate_windowed`, gated by `FEATURES__WINDOWED_CONSOLIDATION_ENABLED`).
  - `src/memory/neocortical/hypergraph.py` & `packages/py-cml/src/cml/storage/sqlite_neocortical.py`: N-ary `HyperedgeEngram` + `<5ms` Bipartite Cortical Hypergraph Personalized PageRank + full embedded SQLite parity (`SQLiteGraphStore`, `SQLiteFactStore`).
  - `src/storage/postgres.py` (`tsvector` GIN `ix_memory_records_fts`, migration `003_whole_brain_v3.py`) & `packages/py-cml/src/cml/storage/sqlite_store.py` (`FTS5` `memories_fts` + `bm25()`): Dentate Gyrus sparse lexical pattern separation.
  - `src/extraction/temporal_resolver.py` & `src/retrieval/temporal_engine.py`: Granularity-aware Hippocampal Time Cells (`day`, `week`, `month`, `season`, `year`; skipping `vague`) + deterministic calendar deltas.
  - `src/retrieval/cross_encoder.py`, `src/retrieval/memory_retriever.py`, `src/retrieval/augmenters.py`, `src/retrieval/graph_expansion.py`, `src/retrieval/reranker.py`: True-logit CUDA Cross-Encoder (`activation_fn=Identity()`), 4-Pathway Ecphory + Synaptic Co-Activation Resonance, CLS stratified slot allocation, and ACC up-to-2-hop prefrontal loop.
  - `src/memory/dmn.py`, `src/memory/neuromodulation.py`, `src/memory/procedural.py`, `src/reconsolidation/dopamine_rpe.py` (`POST /memory/feedback`), `src/retrieval/preplay.py` (`POST /memory/simulate`), `src/memory/sensory/perirhinal.py` (`media=` on `write`), `src/intrinsic/` (ActiveCML 4-depth intrinsic controller), and `src/core/laya_system1.py` (optional Convai Laya System 1 routing).
- **v3.0 Codebase Refactoring & Cleanup (2026-11-25)**:
  - Removed 8 unwired/dead v1 modules (`scratch_pad.py`, `conversation.py`, `tool_memory.py`, `knowledge_base.py`, `entity_extractor.py`, `relation_extractor.py`, `storage/redis.py`, `api/dependencies.py`), `cml/utils/deprecation.py`, 6 legacy evaluation wrapper scripts (canonical CLI is `cml-eval`), obsolete `ProjectPlan/ActiveCML/`, `requirements-docker.txt`, `evaluation/requirements.txt`, and 20+ dead classes/methods.
  - Centralized shared SDK request/query-param builders in `packages/py-cml/src/cml/_endpoints.py` (~1,670 lines of duplicate sync/async client code eliminated) and extracted cohesive mixins (`src/memory/_write_pipeline.py`, `src/retrieval/augmenters.py`, `src/retrieval/explain.py`, `src/retrieval/graph_expansion.py`).
- **Benchmark Baselines (`evaluation/` organized into `locomo_plus/` & `longmemeval_v2/`)**:
  - **LoCoMo-Plus Official Table 1 Protocol — Hybrid CLS + 2-Stage `LLM_INTERNAL` (`2026-11-27`, `openai/gpt-4o` QA + `google/gemini-2.5-flash` Judge via OpenRouter, `2,387` samples, `0` errors)**:
    - **LoCoMo Factual Avg (`1,986` samples): `69.99%`** (`64.48%` macro; `single-hop 71.70%`, `multi-hop 47.87%`, `temporal 60.59%`, `common-sense 50.52%` [`58.85%` open], `adversarial 91.70%`) — **#1 among all `gpt-4o` memory systems** (`A-Mem 59.64%`, `SeCom 57.53%`, `Mem0 57.24%`) and ahead of full-context `GPT-4o` (`62.99%`), `GPT-4.1` (`62.21%`), and 1M-context `Gemini-2.5-Flash` (`69.25%`).
    - **`Cognitive` (`401` samples): `50.12%`** (**#1 on Table 1**, ahead of `SeCom 42.63%`, `A-Mem 42.44%`, `Mem0 41.44%`, `GPT-4o 41.94%`, and 1M-context `Gemini-2.5-Pro 45.72%`) | **Gap: `19.87pp` (`14.35pp` macro)** | **Overall (`2,387` samples): `66.65%` (`1,591.0 / 2,387`)**.
    - Committed summary: `evaluation/results/locomo_plus/locomo_plus_2026-11-27_cml_v3_1_gpt4o_gemini25flash_full_2387_summary.json`.
  - **LoCoMo-Plus 100%-Local Zero-API-Cost Run (`qwen3.8-27b` QA + Judge, `\$0.00` API cost)**:
    - **LoCoMo Factual Avg (`1,986` samples): `68.40%`** (`single-hop 70.21%`, `multi-hop 47.52%`, `temporal 70.25%`, `common-sense 44.27%`, `adversarial 82.06%`)
    - **`Cognitive` (`401` samples): `39.40%`** | **Overall (`2,387` samples): `63.53%`**
    - Committed summary: `evaluation/results/locomo_plus/locomo_plus_2026-11-25_cml_v3_qwen38_full_2387_summary.json`.
  - **LongMemEval-V2 100%-Local Zero-API-Cost Run (`2026-11-27`, `small` tier, `451` questions: `240` `web` + `211` `enterprise`, `qwen3.8-27b` Multimodal Reader + Judge, `0` errors)**:
    - **Non-Abstention Overall (`323` questions): `56.97%`** (`web 59.52%`, `enterprise 54.19%`; `procedure 77.03%` [`web 90.48%`], `static 52.99%` [`enterprise 56.76%`], `gotchas 51.72%` [`web 66.67%`], `dynamic 47.67%` [`enterprise 51.43%`]).
    - **Abstention (`128` questions): `24.22%`** | **Full-Set (`451` questions): `47.67%` (`215.0 / 451`)** | **Avg Query Latency: `19.65s` (`web 13.17s`)**.
    - Committed summary: `evaluation/results/longmemeval_v2/longmemeval_v2_2026-11-27_cml_qwen3827b_small_full_451_summary.json`.

## Architecture facts (load-bearing)

- **Auth is the `X-API-Key` header** (`src/api/auth.py`), not `Authorization: Bearer`. Admin keys additionally accept `X-Tenant-Id`.
- **Single LLM switch**: `FEATURES__USE_LLM_ENABLED` controls whether `UnifiedWritePathExtractor`, `QueryClassifier`, HyDE, and LLM consolidation run via `LLM_INTERNAL__*` vs. deterministic heuristic mode (used by the hermetic unit suite).
- **Standalone `[embedded]` import isolation**: `src/storage/__init__.py` uses PEP 562 lazy `__getattr__` and `src/memory/orchestrator.py` wraps server-only stores (`PostgresMemoryStore`, `Neo4jGraphStore`, `DatabaseManager`, `SemanticFactStore`) in `try ... except ImportError` so `pip install "cognitive-memory-layer[embedded]"` runs without `redis`, `neo4j`, or `pgvector`.
- **Similarity primitives**: Live in `src/utils/similarity.py` (`word_set`, `jaccard`, `cosine_similarity`); `packages/py-cml/src/cml/storage/sqlite_store.py` keeps its own local cosine helper so `py-cml` never imports `src.*` at module scope.

## Invariants — do not "simplify" these

- **`vector_search(min_similarity=-1.0)`** must stay at `-1.0`. Cosine similarity is valid on `[-1, 1]`; `0.0` silently discards negatively-correlated hashed mock embeddings in CI (`tests/unit/test_vector_search_similarity_floor.py`).
- **Graph relevance normalization**: Neo4j traversal scores are rank-normalized into `[GRAPH_RELEVANCE_FLOOR, GRAPH_RELEVANCE_CEILING]` (`[0.55, 0.85]` in `src/retrieval/graph_expansion.py`) and scored by query cosine (`_graph_relevance`), while `MemoryReranker._score_components` clamps relevance to `[0, 1]`.
- **Reranker breakdown keys** must match `RetrievalExplainRerankItem` (`id`, `source_type`, etc.) for the dashboard explain endpoint.
- **`encode_batch` returns a 4-tuple on every path**, including early returns when the write gate skips all chunks.
- **`get_internal_llm_client()` never returns `None`** — it raises when credentials/config are invalid.
- **`hippocampal/store.py` reads `get_settings` via function-local imports** so `monkeypatch.setattr("src.core.config.get_settings", ...)` reaches per-call reads in tests.
- **Cross-Encoder raw logits**: `CrossEncoderScorer` (`src/retrieval/cross_encoder.py`) passes `activation_fn=torch.nn.Identity()` in `sentence-transformers 5.x` to avoid double-sigmoid compression before manual calibration.

## Dashboard, Offline & Docker notes

- **Dashboard**: Third-party assets are vendored in `src/dashboard/static/vendor/` with zero CDN calls (`grep -rn "https://" src/dashboard/static` excluding `vendor/` must stay empty). Dashboard `POST` routes require `X-Requested-With: XMLHttpRequest` (CSRF middleware).
- **Offline posture**: Embedding weights, `semchunk` tokenizer, and `tiktoken` `cl100k_base` ranks persist on `/cache/hf` and `/cache/tiktoken` (chowned to `appuser` before `USER` switch in `docker/Dockerfile`).
- **Docker PyTorch wheel**: `requirements-runtime.txt` pins `--extra-index-url https://download.pytorch.org/whl/cu129` and `torch==2.13.0+cu129` so pip resolves the explicit CUDA 12.9 build cleanly across platforms.
- **Orphan DB table**: `event_log` in `migrations/versions/001_initial_schema.py` has no active writer or reader; left intact to avoid a destructive migration.

## Notes index

- [usage.md](usage.md) — durable reference: server API, endpoints, and configuration (linked by anchor from README, CONTRIBUTING, and `py-cml` docs).
