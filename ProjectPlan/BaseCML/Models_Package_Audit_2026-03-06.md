# packages/models Audit - 2026-03-06

Scope: `packages/models` and the modeling pipeline code that produces and consumes its artifacts:
- `packages/models/*`
- `packages/py-cml/src/cml/modeling/*`
- runtime consumers in `src/utils/modelpack.py`, `src/extraction/local_unified_extractor.py`, `src/retrieval/*`, and `src/memory/hippocampal/write_gate.py`

Method:
- Static inspection of config, README, prepare/train code, prepared manifests, trained manifests, and runtime consumers.
- Runtime inspection of committed parquet/artifact contents.
- Targeted test execution for modeling package wrappers and CLI routing.

Verification run:
- `pytest packages/py-cml/tests/unit/test_modeling_cli.py packages/py-cml/tests/unit/test_modeling_pipeline.py packages/py-cml/tests/unit/test_modeling_prepare_api.py packages/py-cml/tests/unit/test_modeling_train_api.py -q --tb=short` -> `10 passed`
- Runtime data checks:
  - `router_train.parquet`, `extractor_train.parquet`, and `pair_train.parquet` contain only baseline family tasks
  - none of those parquet files contain a `score` column
  - `packages/models/trained_models` contains only `pair_model.joblib`
  - `packages/models/trained_models/manifest.json` lists `router`, `extractor`, and `pair`, but `task_models` is empty
- Runtime probe:
  - `src.utils.modelpack.ModelPackRuntime` loaded only `pair`
  - router/extractor single-task predictions returned `None`

## Architecture Map

### Modules and responsibilities
- `packages/models/model_pipeline.toml`
  - Central spec for:
    - output paths
    - prep targets
    - train hyperparameters
    - task-level model definitions
    - dataset sources
    - synthetic LLM generation settings
- `packages/py-cml/src/cml/modeling/prepare.py`
  - Builds family-level training rows.
  - Mixes existing prepared data, public datasets, and LLM-synthetic backfill.
  - Writes `router_*`, `extractor_*`, `pair_*` parquet splits plus a prep manifest.
- `packages/py-cml/src/cml/modeling/train.py`
  - Trains:
    - family-level TF-IDF + `SGDClassifier` models
    - optional task-level models by objective type
  - Writes `manifest.json`, family artifacts, and optional task-model artifacts.
- `packages/models/scripts/prepare.py`
  - Legacy wrapper to `cml.modeling.prepare`.
- `packages/models/scripts/train.py`
  - Legacy wrapper to `cml.modeling.train`.
- `packages/models/trained_models/*`
  - Committed training outputs used by runtime modelpack loading.
- `src/utils/modelpack.py`
  - Runtime loader for family and task artifacts.
- Runtime consumers
  - `src/extraction/local_unified_extractor.py`
  - `src/retrieval/retriever.py`
  - `src/retrieval/reranker.py`
  - `src/memory/hippocampal/write_gate.py`

### Data flow
1. `model_pipeline.toml`
2. `cml.modeling.prepare`
3. `packages/models/prepared_data/modelpack/{router,extractor,pair}_{train,test,eval}.parquet`
4. `cml.modeling.train`
5. `packages/models/trained_models/*`
6. `src.utils.modelpack.ModelPackRuntime`
7. runtime retrieval/write paths use loaded models or silently fall back to heuristics

### Runtime touchpoints that matter for constraint consistency
- `src/retrieval/retriever.py:562-610`
  - constraint facts are rescored through modelpack-backed relevance logic when available
- `src/retrieval/reranker.py:123-129`
  - `memory_rerank_pair` can override raw retrieval score
- `src/memory/hippocampal/write_gate.py:238-297`
  - `write_importance_regression` and `novelty_pair` influence storage and duplicate handling
- `src/extraction/local_unified_extractor.py:82-162`
  - `fact_extraction_structured`, `write_importance_regression`, `pii_span_detection`, plus router tasks affect the non-LLM write path

## Executive Summary

`packages/models` is currently a planning surface plus partial artifacts, not a coherent deployed modelpack. The strongest failures are contract mismatches:

1. The config and README define ten task-level models, but the committed prepared data contains none of those tasks and the trained manifest contains zero `task_models`.
2. Several advertised tasks are structurally unsupported by the current pipeline:
   - `token_classification` is still a placeholder
   - `write_importance_regression` has no numeric target pipeline
   - `forgetting_action_policy` is documented as using metadata features, but training only uses text
3. The committed artifact set is incomplete: only `pair_model.joblib` exists, while the manifest advertises `router` and `extractor` model files too.
4. The baseline family datasets are heavily synthetic, especially router (`98.21%` LLM-generated rows), and exactly balanced per label. That makes the training distribution easy to optimize but poorly matched to real constraint retrieval workloads.
5. Ranking objectives are still evaluated with classification proxies, so the current package cannot verify the retrieval improvements its README promises.

The practical result is that the runtime keeps falling back to heuristics on many of the paths this package claims to replace.

## Architecture Findings

### Family-only prepared data
- Evidence:
  - `packages/py-cml/src/cml/modeling/prepare.py:75-93` defines baseline family tasks.
  - `packages/py-cml/src/cml/modeling/prepare.py:157-171` defines new pair and single tasks that should extend the baseline families.
  - Runtime inspection of committed parquet files showed:
    - router tasks: `memory_type`, `query_intent`, `query_domain`, `constraint_dimension`, `context_tag`, `salience_bin`, `importance_bin`, `confidence_bin`, `decay_profile`
    - extractor tasks: `constraint_type`, `constraint_scope`, `constraint_stability`, `fact_type`, `pii_presence`
    - pair tasks: `conflict_detection`, `constraint_rerank`, `scope_match`, `supersession`
  - No committed parquet split contains:
    - `retrieval_constraint_relevance_pair`
    - `memory_rerank_pair`
    - `novelty_pair`
    - `schema_match_pair`
    - `reconsolidation_candidate_pair`
    - `consolidation_gist_quality`
    - `forgetting_action_policy`
    - `write_importance_regression`
    - `fact_extraction_structured`
    - `pii_span_detection`
- Explanation:
  - The checked-in prepared data is still a family-only dataset, even though the config and README describe an expanded task-level pipeline.

### Partial trained modelpack
- Evidence:
  - `packages/models/trained_models/manifest.json` lists `router`, `extractor`, and `pair` families.
  - Actual directory contents include only `pair_model.joblib`; `router_model.joblib` and `extractor_model.joblib` are absent.
  - All ten configured task model files are absent.
- Explanation:
  - The artifact directory is internally inconsistent. It contains metrics and metadata for router/extractor, but not the actual model binaries, and no task-model binaries at all.

## Issue Register

### MOD-01
- Severity: Critical
- Category: Logical / artifact integrity
- Evidence:
  - `packages/models/model_pipeline.toml:95-183` defines ten task-level models.
  - `packages/models/README.md:27-42` states the pipeline supports ten task-specific models.
  - `packages/models/trained_models/manifest.json` has `task_models = {}` and no runtime thresholds.
  - All configured `<task>_model.joblib` files are absent.
- Explanation:
  - The package claims task-level model support, but the committed trained artifact set contains zero task models. Runtime paths that depend on them therefore never activate.
- Repro:
  - Check `packages/models/trained_models/manifest.json`.
  - Check file existence for each configured `<artifact_name>_model.joblib`.
- Fix plan:
  - Add an artifact completeness check that validates:
    - every configured task with prepared rows either has an artifact or is explicitly marked unsupported
    - manifest entries match the filesystem
  - Fail training if the config asks for task models but zero task models were actually produced.
- Test plan:
  - Add a manifest/artifact consistency unit test.
  - Add a CI test that trains a tiny config and asserts task-model outputs exist when configured.

### MOD-02
- Severity: Critical
- Category: Conceptual / logical
- Evidence:
  - `packages/py-cml/src/cml/modeling/prepare.py:1973-2004` includes `NEW_SINGLE_TASK_LABELS` in router-row generation.
  - `packages/py-cml/src/cml/modeling/prepare.py:2071-2095` includes `NEW_PAIR_TASK_LABELS` in pair-row generation.
  - Runtime inspection of committed parquet files found only the baseline family tasks and none of the new tasks.
  - `packages/models/prepared_data/modelpack/manifest.json` lists only baseline task names in `router.tasks`, `extractor.tasks`, and `pair.tasks`.
- Explanation:
  - The committed prepared dataset does not reflect the current config or generation logic. That leaves the modeling package in a broken intermediate state where the public config suggests task-level training is ready, but the data contract feeding that training is not.
- Repro:
  - Inspect the unique `task` values in:
    - `packages/models/prepared_data/modelpack/router_train.parquet`
    - `packages/models/prepared_data/modelpack/extractor_train.parquet`
    - `packages/models/prepared_data/modelpack/pair_train.parquet`
- Fix plan:
  - Regenerate the prepared dataset from the current config.
  - Add a validation pass after prepare that compares configured tasks with tasks present in parquet output.
  - Persist a machine-readable `configured_tasks` section in the prep manifest and compare it with `observed_tasks`.
- Test plan:
  - Add a prep integration test that runs on a small synthetic config and asserts every configured task appears in the expected family split.

### MOD-03
- Severity: Critical
- Category: Logical / pipeline feasibility
- Evidence:
  - `packages/models/model_pipeline.toml:150-156` defines `write_importance_regression` as `single_regression`.
  - `packages/py-cml/src/cml/modeling/prepare.py:165-168` does not include `write_importance_regression` in `NEW_SINGLE_TASK_LABELS`.
  - The committed parquet splits have no `score` column.
  - `packages/py-cml/src/cml/modeling/train.py:647-649` expects `score` or numeric `label` for regression and drops non-numeric rows.
- Explanation:
  - `write_importance_regression` is not just missing from the current artifacts. With the current prep schema, it cannot be trained: there are no rows for the task and no numeric target column.
- Repro:
  - Inspect current parquet columns.
  - Inspect `prepare.py` task-label definitions and `train.py` regression loader.
- Fix plan:
  - Add a dedicated regression data path in `prepare.py` that writes numeric `score`.
  - Add `write_importance_regression` to the generated dataset plan only when numeric supervision is available.
  - Fail fast if regression tasks are configured but no numeric rows exist.
- Test plan:
  - Add a tiny regression fixture dataset and verify:
    - prepare writes `score`
    - train produces `write_importance_regression_model.joblib`

### MOD-04
- Severity: High
- Category: Logical / pipeline feasibility
- Evidence:
  - `packages/models/model_pipeline.toml:123-128` defines `fact_extraction_structured` as `token_classification`.
  - `packages/models/model_pipeline.toml:159-164` defines `pii_span_detection` as `token_classification`.
  - `packages/py-cml/src/cml/modeling/train.py:741-749` raises `NotImplementedError` for `token_classification`.
  - `packages/py-cml/src/cml/modeling/train.py:767-771` catches the exception and silently skips the task.
- Explanation:
  - Two of the most important write-time tasks are configured as default task models, but the trainer does not implement their objective type. The package currently advertises capabilities it cannot train.
- Repro:
  - Run training with a config that includes the token tasks.
  - Observe skip messages and absent artifacts.
- Fix plan:
  - Short term:
    - mark token-classification tasks as unsupported in the default config
    - fail loudly instead of silently skipping
  - Long term:
    - implement a real span/token training path
- Test plan:
  - Add a unit test that default training fails if unsupported configured objectives are present.

### MOD-05
- Severity: High
- Category: Conceptual / feature mismatch
- Evidence:
  - `packages/models/README.md:58-64` says:
    - `write_importance_regression` uses internal replay/annotation outcomes
    - `forgetting_action_policy` uses text plus metadata features such as `importance`, `access_count`, `age`, `type`, `dependency_count`
  - `packages/py-cml/src/cml/modeling/train.py:186-192` encodes only text or text-pair features.
  - `packages/py-cml/src/cml/modeling/train.py:655-663` and `713-717` build regression input from text only.
  - Current parquet schemas are:
    - router/extractor: `text`, `task`, `label`, `source`
    - pair: `text_a`, `text_b`, `task`, `label`, `source`
- Explanation:
  - The training pipeline does not support the structured features the README says some tasks need. Even if those tasks were present, the current feature encoder would discard the metadata that defines the decision.
- Repro:
  - Inspect parquet columns and feature encoding in `train.py`.
- Fix plan:
  - Introduce typed structured features for policy/regression tasks.
  - Use a combined featurization path:
    - text encoder
    - numeric/categorical metadata encoder
  - Update task schemas to declare required feature fields.
- Test plan:
  - Add a task fixture with metadata columns and assert training consumes them.

### MOD-06
- Severity: High
- Category: Conceptual / data quality
- Evidence:
  - `packages/models/prepared_data/modelpack/manifest.json` records exact label balancing at `10000` rows for nearly every router/extractor label.
  - Runtime inspection of training rows showed:
    - router: `448000` train rows, `440001` from `llm:*` sources (`98.21%`)
    - extractor: `288000` train rows, `208000` from `llm:*` sources (`72.22%`)
    - pair: `320000` train rows, `0` from `llm:*`, all public datasets
  - `packages/py-cml/src/cml/modeling/prepare.py:2355-2358` defaults `target_per_task_label = 10000` and `max_per_task_label = 50000`.
  - `packages/py-cml/src/cml/modeling/prepare.py:1993-2003` and `2084-2094` backfill missing rows with LLM generation.
- Explanation:
  - The baseline datasets are almost perfectly balanced and, for router/extractor, overwhelmingly synthetic. That optimizes for training symmetry, not for real runtime distributions. For constraint retrieval, this is especially risky:
    - rare classes are overrepresented
    - surface-form shortcuts from synthetic prompts dominate
    - the training distribution differs from the retrieval/query distribution that matters in production
- Repro:
  - Inspect `packages/models/prepared_data/modelpack/manifest.json`.
  - Compute source mix from the parquet files.
- Fix plan:
  - Add source-aware reporting and hard caps on synthetic fraction per task.
  - Preserve natural frequency splits alongside balanced experimental splits.
  - Hold out a real-only validation set for each task.
- Test plan:
  - Add source-mix assertions in the prep manifest validation.
  - Benchmark real-only validation versus synthetic-heavy validation.

### MOD-07
- Severity: Medium
- Category: Conceptual / evaluation mismatch
- Evidence:
  - `packages/models/README.md:46-56` promises IR-style gains for retrieval and reranking tasks.
  - `packages/models/README.md:332-338` uses NDCG/MRR/candidate recall as rollout exit criteria.
  - `packages/py-cml/src/cml/modeling/train.py:568-626` trains `pair_ranking` tasks with TF-IDF + classifier and then reports classification metrics plus a `ranking_proxy`.
  - `packages/models/README.md:264-267` explicitly notes ranking metrics are only proxy accuracy/F1 without list-wise data.
- Explanation:
  - The current training/evaluation loop cannot validate the retrieval improvements its rollout plan claims to target. The proxy metrics may correlate weakly with ranking quality, but they are not a substitute for list-wise evaluation on retrieval tasks.
- Repro:
  - Inspect `pair_ranking` trainer and README metrics section.
- Fix plan:
  - Add list-wise eval sets for:
    - `retrieval_constraint_relevance_pair`
    - `memory_rerank_pair`
    - `reconsolidation_candidate_pair`
  - Export actual MRR/NDCG/Recall@k from evaluation, not only classification proxy metrics.
- Test plan:
  - Add a benchmark harness that reads grouped candidate lists and computes ranking metrics.

### MOD-08
- Severity: High
- Category: Logical / artifact integrity
- Evidence:
  - `packages/models/README.md:271-286` says family training writes `*_model.joblib` for each family and per-task models for task entries.
  - `packages/py-cml/src/cml/modeling/train.py:479-501` writes family artifacts including `*_model.joblib`.
  - Committed `packages/models/trained_models` directory contains:
    - `pair_model.joblib`
    - router/extractor metrics and metadata files
    - no `router_model.joblib`
    - no `extractor_model.joblib`
- Explanation:
  - The committed artifact directory violates the trainer’s own output contract. That in turn breaks runtime capability in a silent way because metadata files make the directory look complete.
- Repro:
  - Inspect `packages/models/trained_models`.
- Fix plan:
  - Add post-train validation that ensures every manifest model path exists before writing the final manifest.
  - Refuse to commit or publish a partial artifact directory.
- Test plan:
  - Add a training integration test that asserts every summary entry points to an existing file.

### MOD-09
- Severity: Medium
- Category: Reliability / reproducibility
- Evidence:
  - Embedded/runtime loading emitted `InconsistentVersionWarning` while unpickling the committed `pair_model.joblib` under `scikit-learn 1.7.2`.
  - `packages/models/trained_models/*_training_metadata.json` contains no library-version metadata.
  - `packages/py-cml/src/cml/modeling/train.py:454-477` records training config and epoch stats, but not package versions.
- Explanation:
  - The model package does not record enough provenance to make binary compatibility diagnosable. That is already visible in the current environment via `scikit-learn` unpickle warnings.
- Repro:
  - Load the committed pair model with `src.utils.modelpack.ModelPackRuntime`.
- Fix plan:
  - Record:
    - Python version
    - scikit-learn version
    - joblib version
    - pandas version
    - commit SHA / dirty flag if available
  - Validate version compatibility on load and warn clearly.
- Test plan:
  - Add metadata completeness tests and loader behavior tests.

### MOD-10
- Severity: Medium
- Category: Testing gap
- Evidence:
  - `packages/py-cml/tests/unit/test_modeling_prepare_api.py`
  - `packages/py-cml/tests/unit/test_modeling_train_api.py`
  - `packages/py-cml/tests/unit/test_modeling_cli.py`
  - `packages/py-cml/tests/unit/test_modeling_pipeline.py`
  - These tests only validate argument forwarding and call routing.
- Explanation:
  - The modeling package has almost no tests that validate actual prepared outputs, artifact completeness, objective support, or runtime compatibility. That is why the current drift between config, data, and artifacts was able to persist.
- Repro:
  - Inspect the modeling unit tests.
- Fix plan:
  - Add end-to-end miniature pipeline tests with tiny local fixtures.
  - Validate:
    - configured tasks appear in prepared data
    - trained artifacts match manifest
    - runtime loader can score each shipped task/family
- Test plan:
  - Add small, deterministic fixture datasets under `packages/models/testdata`.

## Prioritized Roadmap

### Quick wins
- Add config/prepared/trained consistency validation.
- Make unsupported objectives fail loudly.
- Add artifact existence checks before writing manifests.
- Add provenance metadata to trained artifacts.

### Medium
- Regenerate prepared data so configured tasks actually exist.
- Introduce real regression and structured-feature schemas.
- Add list-wise ranking evaluation harness.

### Larger refactors
- Replace family-level composite-label training with per-task heads for the retrieval-critical tasks.
- Implement true token/span training for extraction and PII tasks.
- Separate natural-distribution validation from balanced synthetic training data.

## Patch Plan

### PR-01: Integrity guardrails
- Add a `packages/models` validator that checks:
  - config task list
  - prepared parquet task coverage
  - artifact completeness
  - manifest/file consistency
- Fail CI on mismatches.

### PR-02: Data-contract repair
- Extend `prepare.py` to generate:
  - task rows for every configured task
  - numeric `score` columns for regression
  - structured metadata columns for policy tasks
- Add manifest sections for configured-vs-observed tasks and source mix.

### PR-03: Training-contract repair
- Make unsupported objectives fail by default.
- Split text-only versus structured-feature trainers.
- Record dependency versions and dataset lineage in metadata.

### PR-04: Evaluation realism
- Add grouped ranking evaluation data and metrics.
- Report true MRR/NDCG/Recall@k for retrieval tasks.
- Add real-only validation subsets to measure synthetic drift.

### PR-05: Runtime activation benchmark
- Add a benchmark harness that verifies each shipped model changes runtime behavior on:
  - constraint retrieval
  - memory reranking
  - novelty/write gating
  - local unified extraction

## Risks and Non-goals

- Non-goal:
  - I am not recommending immediate removal of heuristic fallbacks in runtime. The package is not ready for that.
- Tradeoff:
  - Better task realism and structured features will increase prep/training complexity and artifact size.
- Compatibility:
  - Tightening validation will likely break the current checked-in artifact flow until the dataset and modelpack are regenerated.
