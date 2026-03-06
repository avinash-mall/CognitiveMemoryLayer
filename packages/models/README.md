# Model Pipeline

This folder contains the consolidated custom-model pipeline used by CML runtime.

- Prepare script: `packages/models/scripts/prepare.py`
- Train script: `packages/models/scripts/train.py`
- Config: `packages/models/model_pipeline.toml`
- Output models: `packages/models/trained_models`

## Model Families (Baseline)

The pipeline trains three model families that serve as the backbone for all classification tasks:

1. `router_model.joblib`
- Tasks: `memory_type`, `query_intent`, `query_domain`, `constraint_dimension`, `context_tag`, `salience_bin`, `importance_bin`, `confidence_bin`, `decay_profile`

2. `extractor_model.joblib`
- Tasks: `constraint_type`, `constraint_scope`, `constraint_stability`, `fact_type`, `pii_presence`

3. `pair_model.joblib`
- Tasks: `conflict_detection`, `constraint_rerank`, `scope_match`, `supersession`

All families use TF-IDF + `SGDClassifier` for composite labels (`task::label`).

## Task-Level Models

Beyond the three family-level classifiers, the pipeline supports 10 task-specific models that replace heuristic and rule-based logic throughout the runtime. Each model targets a specific subsystem and can be trained independently.

### Model Inventory

| Model | Replacement Target | Objective | Public Datasets | Synthetic? |
|---|---|---|---|---|
| `retrieval_constraint_relevance_pair` | Domain keyword bonus in retriever | `pair_ranking` | MS MARCO, BEIR, Quora duplicates | Yes |
| `memory_rerank_pair` | Weighted reranker core | `pair_ranking` | MS MARCO, BEIR, SNLI, MultiNLI, ANLI | Yes |
| `novelty_pair` | Jaccard novelty in write gate and interference detector | `pair_ranking` | Quora duplicates, PAWS, GLUE (QQP/MRPC/STS-B) | Yes |
| `fact_extraction_structured` *(planned)* | Regex/spaCy fallback extraction and dependency-based relation extraction | `token_classification` | DocRED, Re-TACRED | Yes |
| `schema_match_pair` | Jaccard schema similarity in consolidation | `pair_ranking` | STS-B, FEVER, SNLI, MultiNLI | Yes |
| `reconsolidation_candidate_pair` | Word-overlap top-k in reconsolidation | `pair_ranking` | MS MARCO, FEVER, SNLI, ANLI | Yes |
| `write_importance_regression` *(deferred)* | Fixed importance bins in write gate | `single_regression` | None (internal labels only) | Yes (after score supervision) |
| `pii_span_detection` *(planned)* | Regex+NER span redaction | `token_classification` | PII-Masking-200k | Yes |
| `consolidation_gist_quality` | String-match gist blacklist and mixed-topic detection | `classification` | SummEval, FRANK, TRUE | Yes |
| `forgetting_action_policy` | Heuristic action choice in forgetting scorer | `classification` | None (internal labels only) | Yes (mandatory) |

### Dataset Strategy by Model

**`retrieval_constraint_relevance_pair`** — MS MARCO positives/negatives for general relevance. BEIR for zero-shot domain-shift robustness. Synthesize constraint-shaped examples: positives where query aligns with policy/goal/value/state/causal constraint text; hard negatives that are lexically similar but conflicting in scope or intent.

**`memory_rerank_pair`** — Same IR backbone datasets plus NLI/FEVER contradiction signals so the reranker avoids semantically incompatible memories. Synthesize "same topic, wrong memory type/timeframe" hard negatives. Model must implicitly learn stability-aware recency behavior currently encoded in `_get_recency_weight` (value/policy constraints have zero recency decay, preferences age slowly).

**`novelty_pair`** — Quora/PAWS/GLUE give paraphrase-vs-non-paraphrase signal. Synthetic memory-style perturbations: same fact rephrased, updated fact (temporal change), contradiction fact. Outputs both class and calibrated novelty score. Also serves `InterferenceDetector`: replaces word-level Jaccard overlap and cosine-threshold duplicate detection with learned similarity.

**`fact_extraction_structured`** — DocRED/Re-TACRED give relation extraction supervision. Mandatory synthesis maps extracted relations into CML fact schema (`key = user:{category}:{predicate}`, `category in FactCategory`, stable predicate normalization). Replaces `_PREDICATE_KEYWORDS` mapping and dependency-based relation extraction with fixed `confidence=0.65`.

**`schema_match_pair`** — STS-B for semantic similarity baseline. FEVER/NLI for support-vs-refute behavior. Synthesize gist-to-existing-fact pairs from real migration outputs to capture project-specific key structure.

**`reconsolidation_candidate_pair`** — MS MARCO covers retrieval candidate ranking. FEVER/NLI/ANLI improve conflict candidate prioritization. Synthesize from reconsolidation logs: positives are memory/fact pairs that actually resulted in revision operations.

**`write_importance_regression`** — Fully internal labels built from replay logs and annotation: downstream retrieval frequency, survival/forgetting outcomes, human criticality labels for safety-relevant memories. This task is currently disabled by default until score-supervision parquet columns are available.

**`pii_span_detection`** — PII-Masking-200k as primary span supervision. Synthesis for secrets patterns not fully covered by generic PII datasets (`api_key=`, tokens, credentials).

**`consolidation_gist_quality`** — SummEval/FRANK/TRUE provide consistency/factuality quality supervision. Synthesize CML-specific labels (accept/reject, fallback-needed) using consolidation replay + reviewer labels.

**`forgetting_action_policy`** — Fully internal replay and simulation. Labels: `keep`, `decay`, `silence`, `compress`, `delete`. Inputs include text + metadata features (`importance`, `access_count`, `age`, `type`, `dependency_count`).

## Runtime Wiring

Modelpack inference is consumed from `src/utils/modelpack.py`.

### Existing integrations (family-level)

- `query_domain`: classifier/planner/retriever
- `scope_match`: retriever/reranker
- `constraint_stability`: reranker
- `supersession`: constraint extraction/supersession checks
- `pii_presence` + `importance_bin`: write gate non-LLM path
- `context_tag`: write path content-derived categorization (via `LocalUnifiedWriteExtractor`)
- `confidence_bin`: write path content-derived confidence (via `LocalUnifiedWriteExtractor`)
- `decay_profile`: write path per-memory decay rate (via `LocalUnifiedWriteExtractor`)

### Task-level integrations

| Task | Primary Files/Functions | Integration Behavior |
|---|---|---|
| `retrieval_constraint_relevance_pair` | `retriever.py::_rescore_constraints` | Replace domain keyword bonus with learned relevance score; keep deterministic tie-break floor. |
| `memory_rerank_pair` | `reranker.py::_calculate_score` | Model score as primary relevance component; subsumes stability-based recency weights. MMR diversity post-pass retained. |
| `novelty_pair` | `write_gate.py::_compute_novelty`, `interference.py::detect_duplicates`, `detect_overlapping` | Replace Jaccard heuristic with pair scoring. Replace interference detector's word-overlap and cosine-threshold detection. |
| `fact_extraction_structured` | `write_time_facts.py::extract`, `ner.py::extract_relations`, `service.py::_extract_new_facts` | Replace regex/spaCy fallback, `_PREDICATE_KEYWORDS` mapping, dependency-based relation extraction, and reconsolidation sentence-split fallback. |
| `schema_match_pair` | `schema_aligner.py::_calculate_similarity` | Replace Jaccard with semantic pair match score. |
| `reconsolidation_candidate_pair` | `service.py::_top_k_similar_memories` | Replace word-overlap top-k with pair candidate relevance. |
| `write_importance_regression` | `write_gate.py::_predict_importance` | Calibrated regression output; retain clipping/floor. |
| `pii_span_detection` | `redactor.py::redact`, `write_gate.py::_predict_pii` | Merge model spans with regex/NER; regex remains safety baseline. |
| `consolidation_gist_quality` | `worker.py::_is_valid_gist` | Model score gates gist acceptance; replaces string-match blacklist and rule-based mixed-topic detection. Deterministic overlap ratio check remains as guardrail. |
| `forgetting_action_policy` | `scorer.py::_suggest_action` | Model proposes action; hard protected-type rules remain deterministic. |

If model artifacts are not present, runtime falls back to safe heuristic defaults for all paths. All model calls are wrapped in `try/except` with graceful fallthrough.

### Dashboard integration

The dashboard Components page displays live modelpack status via `GET /api/v1/dashboard/models/status` (admin-only). This endpoint returns:

- `available: bool` — whether the modelpack runtime loaded successfully
- `families: list[str]` — loaded family model names (e.g. `router`, `extractor`, `pair`)
- `task_models: list[str]` — loaded task-specific model names
- `load_errors: dict[str, str]` — any models that failed to load with error messages
- `models_dir: str` — path to the models directory on disk

Backend: `src/api/dashboard/models_routes.py`.

### Modelpack APIs

Runtime capability helpers:

- `supports_task(task)` checks whether a task can be served by either a loaded family model or a dedicated task model.
- `capability_report()` returns load diagnostics (`available_families`, `available_tasks`, `pending_families`, `load_errors`, `manifest_schema_version`).

- `predict_single(task, text)` — single-text classification (existing)
- `predict_pair(task, text_a, text_b)` — text-pair classification (existing)
- `predict_score_pair(task, text_a, text_b)` — text-pair numeric score output
- `predict_score_single(task, text)` — single-text numeric score output
- `predict_spans(task, text)` — token/span extraction output
- `has_task_model(task)` — check if a task-specific model is loaded

### Local Unified Write Extractor

`src/extraction/local_unified_extractor.py` provides `LocalUnifiedWriteExtractor`, a model-based alternative to the LLM-centric `UnifiedWritePathExtractor`. It composes:

- `fact_extraction_structured` for structured fact extraction
- `write_importance_regression` for importance scoring
- `pii_span_detection` for PII detection
- `memory_type` router task for memory type classification
- `context_tag` router task for content-derived categorization
- `confidence_bin` router task for content-derived confidence (maps low/medium/high → 0.35/0.65/0.9)
- `decay_profile` router task for per-memory decay rate (maps very_fast/fast/medium/slow/very_slow → 0.35/0.2/0.1/0.05/0.02)

Deferred tasks (`write_importance_regression`, token objectives) remain optional: local extraction uses them only when corresponding task artifacts are present.

Wired into `HippocampalStore` via `MemoryOrchestrator.create()` and `create_lite()` when LLM is disabled. Both `encode_chunk` and `encode_batch` consume context_tags, confidence, and decay_rate from the local extractor when the LLM unified result is unavailable.

```mermaid
flowchart LR
    subgraph localExtract ["LocalUnifiedWriteExtractor"]
        R[Router Model] --> CT["context_tag → context_tags"]
        R --> CB["confidence_bin → confidence"]
        R --> DP["decay_profile → decay_rate"]
        R --> MT["memory_type"]
        TM["Task Models"] --> FE["fact_extraction_structured"]
        TM --> WI["write_importance_regression"]
        TM --> PII["pii_span_detection"]
    end
    localExtract --> Store["HippocampalStore.encode_chunk / encode_batch"]
```

## Data Preparation

`prepare.py` performs:

1. Dataset loading (auto-download via Hugging Face IDs in config)
2. Merge with existing local prepared data when available
3. Missing-only balancing to target counts (default `10000` per task-label)
4. LLM-only synthetic backfill for deficits
5. Stratified split output (`train`, `test`, `eval`)

Prepared outputs are written to `packages/models/prepared_data/modelpack`.

### Task registries

The script maintains separate task registries:

- **Single-text tasks** (`ROUTER_TASK_LABELS` + `NEW_SINGLE_TASK_LABELS`): classification and regression on individual text inputs.
- **Pair tasks** (`PAIR_TASK_LABELS` + `NEW_PAIR_TASK_LABELS`): classification, ranking, and scoring on text pairs.
- **Token-level tasks**: span/entity tagging via `_TokenTaskStore`.

### Dataset mappers

Public dataset mappers transform standard NLP datasets into CML task formats:

- `PAWS/GLUE (QQP, MRPC, STS-B)` → `novelty_pair`
- `DocRED/Re-TACRED` → `fact_extraction_structured`
- `FEVER/NLI` → `schema_match_pair`, `reconsolidation_candidate_pair`

### Output split formats

- **Classification/ranking**: parquet schema (`text` or `text_a`/`text_b`, `task`, `label`)
- **Regression**: adds numeric target column (`score`) via `_RegressionTaskStore`
- **Token**: adds token-level fields (`tokens`, `tags`) and span metadata via `_TokenTaskStore`

### Manifest fields

The preparation manifest includes:

- Per-task row counts and task-label distributions
- Source mix per task (public/synthetic/internal)
- Synthetic ratio per label
- Top source breakdown

### Existing dataset folder

Raw dataset cache/downloads use existing flat folder: `packages/models/datasets`

## Synthetic Generation (LLM-only)

Synthetic generation does not use rule-based labeling.

Flow:

1. Randomly select seed samples from related datasets.
2. Prompt LLM for target `(task, label)` generation.
3. Parse/validate and keep accepted rows until target count is reached.

Current behavior:

1. Requests are sent concurrently (`synthetic_llm.concurrency`).
2. Per-label generation uses adaptive batch sizing when outputs are repeatedly unparseable.
3. Best-effort recovery extracts `text` / `text_a` / `text_b` from partial JSON responses.
4. Periodic LLM telemetry is printed with request, parse, and acceptance stats.

Expected env settings (OpenAI-compatible endpoint example, including vLLM):

```bash
LLM_EVAL__PROVIDER=openai_compatible
LLM_EVAL__MODEL=meta-llama/Llama-3.2-3B-Instruct
LLM_EVAL__BASE_URL=http://localhost:8001/v1
```

Example telemetry line:

```text
[prepare] LLM stats: req=... fail=... retries=... parse_fail=... req/s=... gen=... acc=... parse_recovered=... finish_reason=stop:...,length:...
```

Quick interpretation:

1. `parse_fail` high: output format is breaking badly; lower `temperature` and `batch_size`.
2. `finish_reason=length` high: responses are getting truncated; lower `batch_size` or `max_tokens`.
3. `acc/gen` low: many candidates are duplicates/invalid for the label; reduce randomness and inspect seed diversity.

### Multilingual data generation

When `[multilingual]` is enabled in config (default), synthetic LLM generation is distributed across **15 languages** (top by global internet usage): English, Chinese (Simplified), Spanish, Arabic, Hindi, Portuguese, French, Japanese, Russian, German, Korean, Turkish, Indonesian, Vietnamese, Italian. Each concurrent job picks a language via weighted random selection (English ~30%, others ~5% each) so models stay strong in English while covering multiple languages.

- **Prompt definitions**: `packages/models/scripts/multilingual_prompts.py` defines `SUPPORTED_LANGUAGES`, `pick_language()`, and language-specific system/user prompt builders (`system_prompt_single`, `user_prompt_single`, etc.). Label guidance stays in English (semantic intent); the LLM generates text in the chosen language.
- **Config**: `model_pipeline.toml` section `[multilingual]` with `enabled = true` and `english_weight = 0.30`. Disable with `--no-multilingual` for English-only synthetic data (e.g. faster iteration).
- **Output**: Prepared parquet includes a `language` column (ISO 639-1 code, e.g. `en`, `zh`, `es`). Existing prepared data without `language` is treated as `en` when loaded.

## Training

`train.py` supports two modes:

By default, training runs in strict mode (`--strict`) with preflight validation and post-train artifact validation. Use `--allow-skips` only for exploratory runs.

### Family-level training (baseline)

Trains all selected families (`router`, `extractor`, `pair`) using TF-IDF + `SGDClassifier` with composite labels. This is the default when no task specs are present.

### Task-level training

When `[[tasks]]` blocks are defined in `model_pipeline.toml`, the trainer dispatches to objective-specific trainers:

| Objective | Trainer | Model |
|---|---|---|
| `classification` | `_train_classification_task` | TF-IDF + SGDClassifier |
| `pair_ranking` | `_train_pair_ranking` | TF-IDF + SGDClassifier (logistic score) |
| `single_regression` | `_train_single_regression` | TF-IDF + SGDRegressor |
| `token_classification` | `_train_token_classification` | Placeholder (transformer path) |

Task specs are defined via `TaskSpec` dataclass: `task_name`, `family`, `input_type`, `objective`, `labels`, `artifact_name`, `metrics`.

### Evaluation metrics

Metrics are computed per objective type:

- **Classification**: accuracy, macro/weighted F1, Expected Calibration Error (ECE), per-class precision/recall.
- **Ranking**: proxy accuracy and F1 from pair classification (true MRR@k/NDCG@k require list-wise evaluation data).
- **Regression**: MAE, RMSE, calibration buckets.
- **Token**: entity/span F1 and strict span exact-match (when transformer path is active).

### Artifacts

Per-family artifacts:

- `*_model.joblib`
- `*_label_map.json`
- `*_metrics_test.json`
- `*_metrics_eval.json`
- `*_report_test.json`
- `*_report_eval.json`
- `*_epoch_stats.json`
- `*_training_metadata.json`

Per-task artifacts:

- `<task>_model.joblib`
- `<task>_epoch_stats.json`
- `<task>_thresholds.json` (when `--export-thresholds` is set)

Manifest (`manifest.json`):

- `manifest_schema_version` (v2)
- `configured_tasks` from `model_pipeline.toml`
- `families` map with artifact paths, metrics summary, labels.
- `task_models` map with objective type, artifact path, train rows, and test metrics.
- `task_training_status` with per-task status (`trained`, `disabled`, `filtered_out`, `failed`, `skipped`) and reason.
- `preflight_validation` with objective/data/coverage checks run before task training.
- `build_metadata` with Python/dependency versions and git state.
- `runtime_thresholds` section with per-task confidence thresholds and calibration metadata.

Per-epoch training logs are printed to console and persisted in epoch stats files.

## Usage

From repository root:

```bash
python -m packages.models.scripts.prepare
python -m packages.models.scripts.train
```

Common overrides:

```bash
python -m packages.models.scripts.prepare --target-per-task-label 10000 --llm-temperature 1.35
python -m packages.models.scripts.prepare --target-per-task-label 5002 --llm-temperature 0.3 --llm-concurrency 32
python -m packages.models.scripts.prepare --force-full
python -m packages.models.scripts.prepare --no-multilingual   # English-only synthetic data
python -m packages.models.scripts.train --max-iter 25 --max-features 250000 --strict
python -m packages.models.scripts.train --max-iter 25 --max-features 250000 --allow-skips
```

Task-level training overrides:

```bash
python -m packages.models.scripts.train --tasks retrieval_constraint_relevance_pair,novelty_pair
python -m packages.models.scripts.train --objective-types pair_ranking,single_regression
python -m packages.models.scripts.train --max-seq-length 512 --learning-rate 5e-5
python -m packages.models.scripts.train --calibration-split eval --export-thresholds
python -m packages.models.scripts.train --tasks novelty_pair --strict
```

Contract and probe checks:

```bash
python scripts/models_artifact_probe.py --fail-on-mismatch
python scripts/package_surface_probe.py embedded --write "User prefers tea" --query tea --expect-min-memories 1
python scripts/constraint_retrieval_probe.py --scenario budget_decision --mode compare
```

## Rollout Plan

### Phase 0: Pipeline foundations

1. Add task schemas and dataset configs to `model_pipeline.toml`.
2. Refactor `prepare.py` and `train.py` to support task objectives.
3. Ship new manifest format with backward-compatible loading.

### Phase 1: Retrieval + reconsolidation (highest impact)

1. `retrieval_constraint_relevance_pair`
2. `memory_rerank_pair`
3. `reconsolidation_candidate_pair`

Exit criteria: improved NDCG/MRR/candidate recall; no regression in p95 latency with fallbacks enabled.

### Phase 2: Write/extraction quality

1. `novelty_pair`
2. `fact_extraction_structured`
3. `write_importance_regression`
4. `pii_span_detection`

Exit criteria: write decision quality improves; no regression in PII/secret safety checks.

### Phase 3: Consolidation + forgetting

1. `schema_match_pair`
2. `consolidation_gist_quality`
3. `forgetting_action_policy`

Exit criteria: consolidation acceptance precision improves; forgetting policy does not regress protected-memory handling.

### Phase 4: Remaining items and hardening

1. Local unified write extractor replacement complete.
2. Semantic lineage API complete.
3. Conflict detector offline benchmark integrated in CI.

## Validation and Safety Gates

For every model-assisted path:

1. **Offline**: objective-specific metrics + calibration.
2. **Runtime**: latency, fallback rate, error rate.
3. **Safety**: secret/PII hard regex checks intact; bounded output/truncation guaranteed; deterministic keys and control-plane behavior unchanged.
4. **Shadow mode**: run heuristic + model in parallel via `ShadowModeLogger`; compare decisions and log deltas before switching defaults.

## Evaluation Infrastructure

### Conflict detector offline evaluation

`src/evaluation/conflict_eval.py` provides `ConflictDetectorEvaluator` for comparing heuristic vs. model-based conflict detection:

- Loads evaluation corpus from JSONL files (`old_memory`, `new_statement`, `expected_label`)
- Computes precision/recall/F1 for both heuristic and model paths
- Generates shadow comparison reports

### Shadow mode logging

`src/utils/shadow_logger.py` provides `ShadowModeLogger` for parallel heuristic/model execution:

- Runs both paths concurrently via `compare()`
- Logs latency deltas and decision disagreements
- Configurable `sample_rate` for production use

### Semantic lineage

`src/memory/neocortical/fact_store.py` provides lineage query APIs:

- `get_fact_lineage(tenant_id, key)` — full supersession chain (oldest to newest)
- `get_superseded_chain(tenant_id, fact_id)` — forward chain of all facts superseded by a given fact

Lineage metadata is captured during reconsolidation and consolidation operations and surfaced in retrieval API responses via the `supersedes_id` field.

## Deterministic Paths (Preserved)

The following paths are intentionally kept deterministic and are not candidates for model replacement:

| Area | Rationale |
|---|---|
| Query intent fallback (`classifier.py`) | Deterministic fallback; expand training coverage for rare intents. |
| Retrieval planner (`planner.py`) | Hard source constraints; optional learned re-ordering under SLA guardrails. |
| Packet budgets (`packet_builder.py`) | Control-plane guardrail; deterministic token budgets. |
| Secret detection (`write_gate.py`) | Regex hard block; never delegated to model. |
| Stable key generation (`store.py`) | Deterministic hash key path. |
| Belief revision strategy (`belief_revision.py`) | Deterministic strategy selection for auditability. |
| Gist fallback (`worker.py`, `summarizer.py`) | Deterministic fallback gist construction. |
| Scheduler triggers (`triggers.py`) | Deterministic schedule/quota/event triggers. |
| Compression fallback (`compression.py`, `actions.py`) | Deterministic truncation as final fallback. |
| Neo4j fallback (`storage/neo4j.py`) | Non-GDS fallback path. |
| Write gate type mapping (`write_gate.py`) | `_determine_memory_types` is deterministic schema logic. |
| Reranker stability rules (`reranker.py`) | `_get_recency_weight` stability sets kept as guardrails during rollout. |
| Interference resolution (`interference.py`) | `_recommend_resolution` strategy is deterministic; uses `novelty_pair` for similarity scoring input only. |
| Consolidation clustering (`clusterer.py`) | Deterministic greedy cosine clustering; tune thresholds via offline calibration. |
| Consolidation sampling (`sampler.py`) | Deterministic weighted sampling; tune weights via offline replay. |
| Seamless relevance threshold (`seamless_provider.py`) | Configurable control-plane parameter (`relevance_threshold=0.3`). |
| NER entity aliases (`ner.py`) | `_ENTITY_ALIAS_MAP` deterministic; expand via data-driven alias mining. |
| Consolidation fallback type mapping (`worker.py`) | Dominant-type-to-gist-type mapping is deterministic. |

## Config Reference

All settings are in `packages/models/model_pipeline.toml`:

- Paths
- Preparation targets/splits
- Training hyperparameters
- Per-task `[[tasks]]` blocks (family, input type, objective, labels, artifact name, metrics)
- `[[datasets]]` entries with links, HF IDs, required/optional flags
- Synthetic LLM parameters
- `[multilingual]`: `enabled`, `english_weight` for multilingual synthetic generation

Key synthetic knobs:

- `batch_size`
- `concurrency`
- `max_tokens`
- `temperature`
- `max_attempts_per_label`
- `log_stats_every_seconds`
- `log_zero_progress_every`
- `parse_failure_log_every`
- `log_request_failures`

## Dataset References

Primary public dataset sources used for model training:

- [MS MARCO](https://huggingface.co/datasets/microsoft/ms_marco)
- [BEIR benchmark](https://arxiv.org/abs/2104.08663)
- [Quora duplicates](https://huggingface.co/datasets/sentence-transformers/quora-duplicates)
- [PAWS](https://aclanthology.org/N19-1131/)
- [GLUE (QQP/MRPC/STS-B)](https://huggingface.co/datasets/nyu-mll/glue)
- [SNLI](https://nlp.stanford.edu/projects/snli/)
- [MultiNLI](https://cims.nyu.edu/~sbowman/multinli/)
- [ANLI](https://aclanthology.org/2020.acl-main.441/)
- [FEVER](https://aclanthology.org/N18-1074/)
- [DocRED](https://aclanthology.org/P19-1074/)
- [Re-TACRED](https://arxiv.org/abs/2104.08398)
- [PII-Masking-200k](https://huggingface.co/datasets/ai4privacy/pii-masking-200k)
- [SummEval](https://arxiv.org/abs/2007.12626)
- [FRANK](https://aclanthology.org/2021.naacl-main.383/)
- [TRUE benchmark](https://arxiv.org/abs/2204.04991)
