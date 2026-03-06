# Packages Audit - 2026-03-06

Scope: `packages/py-cml` and `packages/models`, plus the package-facing runtime seams they depend on in `src/`.

Method:
- Static inspection of package code, build config, READMEs, tests, and committed model artifacts.
- Runtime spot checks of package tests and modelpack loading.
- No external services or web research used.

Verification run:
- `pytest packages/py-cml/tests/unit -q --tb=short` -> `192 passed`
- `pytest packages/py-cml/tests/embedded -q --tb=short` -> `6 passed`
- Runtime probe:
  - `src.utils.modelpack.ModelPackRuntime` loaded only `pair`
  - `predict_single("memory_type", ...)`, `predict_single("query_intent", ...)`, `predict_single("constraint_type", ...)` all returned `None`
- Wheel build verification was attempted with `python -m build --wheel --no-isolation -o .tmp_pkg_audit_dist` but could not be completed because `hatchling.build` was not installed in this environment.

## Architecture Map

### Module map
- `packages/py-cml/src/cml/client.py`
  - Sync HTTP client for `/memory/*`, `/session/*`, admin, dashboard, graph, and batch routes.
- `packages/py-cml/src/cml/async_client.py`
  - Async mirror of the sync client.
- `packages/py-cml/src/cml/transport/http.py`
  - Low-level HTTP transport and exception mapping.
- `packages/py-cml/src/cml/transport/retry.py`
  - Retry/backoff policy for sync and async transport calls.
- `packages/py-cml/src/cml/embedded.py`
  - In-process package API for embedded mode.
  - Delegates real work to repo-internal engine code under `src.memory.*`.
- `packages/py-cml/src/cml/embedded_utils.py`
  - Export/import helpers for embedded migration.
- `packages/py-cml/src/cml/integrations/openai_helper.py`
  - Memory-augmented OpenAI chat helper.
- `packages/py-cml/src/cml/eval/*`
  - Package wrappers around repo evaluation flows.
- `packages/py-cml/src/cml/modeling/*`
  - Package wrappers around repo model prepare/train flows.
- `packages/models/model_pipeline.toml`
  - Single config for dataset prep and model training.
- `packages/models/scripts/*.py`
  - Legacy wrappers that forward into `cml.modeling.*`.
- `packages/models/trained_models/*`
  - Committed model artifacts and metrics used by runtime modelpack loading.
- `src/utils/modelpack.py`
  - Runtime loader for package-produced model artifacts. This is the bridge from `packages/models` back into live runtime behavior.

### Data flow
1. Application -> `cml.CognitiveMemoryLayer` or `cml.AsyncCognitiveMemoryLayer`
2. Client -> `cml.transport.http` -> FastAPI server routes
3. Server runtime -> retrieval / write / turn logic under `src/`
4. Optional package-side embedded path:
   - Application -> `cml.EmbeddedCognitiveMemoryLayer`
   - `cml.embedded` -> `src.memory.orchestrator.MemoryOrchestrator.create_lite(...)`
   - Retrieval path for `turn()` -> `src.memory.seamless_provider.SeamlessMemoryProvider`
5. Optional model path:
   - `cml.modeling.prepare/train` -> `packages/models/model_pipeline.toml`
   - outputs -> `packages/models/trained_models/*`
   - runtime consumers -> `src.utils.modelpack.ModelPackRuntime`

### Test coverage map
- Covered well:
  - client request/response mapping
  - CLI argument routing for `cml.eval` and `cml.modeling`
  - basic embedded lifecycle and embedded roundtrip
- Missing or weak:
  - installed-wheel smoke tests outside the repo
  - artifact completeness tests for committed modelpack files
  - migration fidelity tests for `embedded_utils`
  - tests that assert real router/extractor modelpack capability rather than mocked behavior

## Executive Summary

The `packages/` subtree is useful, but it is not actually modular in the way the package docs imply. The public package is still tightly coupled to repo-internal engine code, repo-relative assets, and an incomplete artifact set. The biggest problems are:

1. The published package boundary is blurred. `pyproject.toml` builds both `src` and `cml`, and embedded mode imports `src.memory.*` directly.
2. The committed model artifacts are incomplete. `packages/models/trained_models/manifest.json` advertises `router`, `extractor`, and `pair`, but only `pair_model.joblib` exists on disk.
3. The runtime overstates model availability. `src/utils/modelpack.py` reports the modelpack as available if any family loads, which masks missing router/extractor capability.
4. The modeling package overclaims support for token-classification tasks that the trainer explicitly skips.
5. The migration utilities are lossy and discard fields that matter for retrieval quality and temporal fidelity.

For package consumers, the result is a package that works best inside the monorepo and silently degrades outside it.

## Issue Register

### PKG-01
- Severity: High
- Category: Conceptual / packaging
- Evidence:
  - `packages/py-cml/README.md:3-6` markets `py-cml` as a Python SDK with optional embedded/eval/modeling features.
  - `packages/py-cml/NewModules.md:4-10` states the goal is to keep the base SDK install small.
  - `pyproject.toml:28-52` puts server and ML stack dependencies in the base package: `fastapi`, `uvicorn`, `sqlalchemy`, `asyncpg`, `pgvector`, `neo4j`, `redis`, `celery`, `sentence-transformers`, `spacy`, `semchunk`, `transformers`.
  - `pyproject.toml:121-122` builds both `src` and `packages/py-cml/src/cml` into the wheel.
- Explanation:
  - This is not an SDK-light package boundary. The base install drags server/runtime concerns and the wheel ships repo-internal engine code. That makes dependency resolution heavier, weakens semver boundaries, and makes it difficult to reason about which APIs are stable package APIs versus internal engine internals.
- Repro:
  - Static inspection of `pyproject.toml`.
- Fix plan:
  - Split package responsibilities:
    - Base `cognitive-memory-layer`: HTTP clients, models, transport, integrations.
    - `embedded` extra or separate package: engine/runtime code and heavy ML dependencies.
    - `eval` and `modeling`: keep as optional extras, but make them truly path-independent.
  - Stop shipping `src` in the base wheel.
- Test plan:
  - Add isolated CI jobs for:
    - `pip install cognitive-memory-layer`
    - `pip install cognitive-memory-layer[embedded]`
  - Assert which modules import successfully in each environment.

### PKG-02
- Severity: High
- Category: Conceptual / packaging
- Evidence:
  - `packages/py-cml/src/cml/embedded.py:56-71` checks for `src.memory.orchestrator`.
  - `packages/py-cml/src/cml/embedded.py:209-223` imports `src.utils.llm.OpenAICompatibleClient` and `src.memory.orchestrator.MemoryOrchestrator`.
  - `packages/py-cml/src/cml/embedded.py:348-356` imports `src.memory.seamless_provider.SeamlessMemoryProvider`.
  - `src/utils/modelpack.py:105-120` assumes model artifacts live at `repo_root / "packages" / "models" / "trained_models"`.
- Explanation:
  - Embedded mode is presented as a package feature, but it is an API wrapper over repo-internal engine modules plus repo-relative model artifacts. This is not a clean distributable boundary. Even if the wheel ships `src`, quality-sensitive behavior still depends on repo-local model assets.
- Repro:
  - Static inspection of `cml.embedded` and `src/utils/modelpack.py`.
  - Inference: since no package-data directives for `packages/models/trained_models` were found in `pyproject.toml` or `hatch_build.py`, an installed wheel is unlikely to contain those artifacts.
  - The wheel-build confirmation step could not be completed here because `hatchling.build` was unavailable.
- Fix plan:
  - Move embedded engine code behind an explicit package boundary with its own packaged assets.
  - Add explicit `models_dir` configuration for embedded mode and fail clearly when assets are missing.
  - Package required artifacts intentionally or remove hidden dependence on them.
- Test plan:
  - Build a wheel in CI and inspect contents.
  - Install into a fresh venv outside the repo and run an embedded write/read smoke test.

### PKG-03
- Severity: Critical
- Category: Logical / reliability
- Evidence:
  - `packages/models/trained_models/manifest.json:26` references `router_model.joblib`.
  - `packages/models/trained_models/manifest.json:6665` references `extractor_model.joblib`.
  - `packages/models/trained_models/manifest.json:8432` references `pair_model.joblib`.
  - Directory listing of `packages/models/trained_models` showed only `pair_model.joblib`.
  - `Test-Path packages/models/trained_models/router_model.joblib` -> `False`
  - `Test-Path packages/models/trained_models/extractor_model.joblib` -> `False`
  - `Test-Path packages/models/trained_models/pair_model.joblib` -> `True`
  - `src/utils/modelpack.py:125-129` reports `available` as `bool(self._models)`.
  - `src/utils/modelpack.py:244-257` silently loads whichever family files exist.
  - Runtime probe output: `modelpack_loaded ... families: ['pair']` and `predict_single(...) -> None` for router/extractor tasks.
- Explanation:
  - The committed artifact set is internally inconsistent. The manifest describes a full family set; the directory only contains the pair family model. Because `ModelPackRuntime.available` only checks whether any family loaded, the process looks healthy while router/extractor-backed paths silently degrade.
- Repro:
  - Inspect `packages/models/trained_models`.
  - Run:
    - `from src.utils.modelpack import get_modelpack_runtime`
    - `mp = get_modelpack_runtime()`
    - `mp.available`
    - `mp.predict_single("memory_type", "user prefers vegetarian food")`
- Fix plan:
  - Add artifact completeness validation:
    - Fail load when manifest-advertised files are missing.
    - Expose `available_families` and `available_tasks` explicitly.
    - Make dashboard and logs report partial capability, not boolean-only health.
  - Either ship the missing router/extractor models or remove stale manifest entries.
- Test plan:
  - Add unit tests for partial artifact sets.
  - Add CI test asserting manifest and filesystem agree.
  - Add runtime smoke test that router/extractor predictions succeed when artifacts are present.

### PKG-04
- Severity: Medium
- Category: Logical
- Evidence:
  - `packages/py-cml/src/cml/config.py:48-51` defines `max_retry_delay`.
  - `packages/py-cml/src/cml/transport/retry.py:106-119` hardcodes `MAX_RETRY_DELAY = 60.0` and ignores config.
  - `packages/py-cml/tests/unit/test_retry.py:102-104` asserts the constant instead of config behavior.
- Explanation:
  - The public config exposes a tunable retry cap, but transport code never uses it. The existing test suite has normalized the bug by asserting the constant.
- Repro:
  - Static inspection is sufficient.
- Fix plan:
  - Pass `config.max_retry_delay` into `_sleep_with_backoff` and `_async_sleep_with_backoff`.
  - Keep a fallback default only when config omits it.
- Test plan:
  - Replace the constant test with one that monkeypatches sleep and verifies the configured cap is honored.

### PKG-05
- Severity: High
- Category: Logical / data fidelity
- Evidence:
  - `packages/py-cml/src/cml/embedded_utils.py:73-82` imports each record using only `text` and `metadata`.
  - `packages/py-cml/src/cml/embedded_utils.py:21-39` exports full scanned records.
  - `packages/py-cml/tests/e2e/test_migration.py:25-55` validates only count-level import success, not field fidelity.
- Explanation:
  - Export captures richer records than import restores. Timestamps, memory type, namespace, confidence, context tags, session/provenance, and any overwrite-sensitive fields are discarded on import. For a memory system, this materially harms retrieval quality and temporal consistency after migration.
- Repro:
  - Static inspection of `embedded_utils.py`.
- Fix plan:
  - Add a typed export schema and a typed import path.
  - Preserve at least: `text`, `type`, `timestamp`, `metadata`, `namespace`, `context_tags`, `confidence`, `session_id`, and tenant/source identifiers where supported.
  - If the live API cannot accept those fields today, add a dedicated admin import endpoint rather than silently discarding them.
- Test plan:
  - Add unit tests for roundtrip fidelity.
  - Extend migration E2E to assert restored timestamp/type/metadata and retrieval behavior.

### PKG-06
- Severity: High
- Category: Conceptual / logical
- Evidence:
  - `packages/models/README.md:27-40` states the pipeline supports 10 task-specific models, including `fact_extraction_structured` and `pii_span_detection`.
  - `packages/models/README.md:121-131` presents those tasks as part of `LocalUnifiedWriteExtractor`.
  - `packages/models/model_pipeline.toml:123-164` configures `fact_extraction_structured` and `pii_span_detection` as `token_classification`.
  - `packages/py-cml/src/cml/modeling/train.py:741-749` raises `NotImplementedError` for token classification.
  - `packages/py-cml/src/cml/modeling/train.py:767-771` catches the error and silently returns `{}` for the skipped task.
- Explanation:
  - The default package docs and config describe capabilities that the trainer does not implement. This is more than documentation drift because the default config actively invites unsupported task training and the trainer degrades into silent skipping.
- Repro:
  - Run `cml-models train --config packages/models/model_pipeline.toml` and inspect stderr / output artifacts.
  - Static inspection already shows the mismatch.
- Fix plan:
  - Short term:
    - Remove unsupported token-classification tasks from the default config and README, or mark them as planned but not trainable.
    - Make skipped default tasks a hard failure, not a silent no-op.
  - Longer term:
    - Implement a transformer/span-based token classification trainer.
- Test plan:
  - Add a test that default config training fails if any configured objective type is unsupported.
  - Add artifact expectation tests for each default task entry.

### PKG-07
- Severity: Medium
- Category: Conceptual / model architecture
- Evidence:
  - `packages/models/README.md:12-23` states each family is a single TF-IDF + `SGDClassifier` over composite `task::label` outputs.
  - `packages/py-cml/src/cml/modeling/train.py:137-198` encodes all family tasks into composite labels.
  - `packages/py-cml/src/cml/modeling/train.py:201-220` builds one TF-IDF + SGD pipeline per family.
  - Committed metrics are only moderate:
    - router test accuracy `0.6295`, macro-F1 `0.628700918893886` at `packages/models/trained_models/manifest.json:34-35`
    - extractor eval accuracy `0.5809722222222222`, macro-F1 `0.4617083871408387` at `packages/models/trained_models/manifest.json:7552-7553`
    - pair test accuracy `0.66135`, macro-F1 `0.6601257733156894` at `packages/models/trained_models/manifest.json:8440-8441`
- Explanation:
  - This architecture forces unrelated tasks to share a single classifier head and calibration space. For constraint-heavy behavior, that is a weak fit: `query_intent`, `memory_type`, `constraint_dimension`, `constraint_type`, and `fact_type` are semantically different tasks with different error costs. The committed metrics support the concern that these heads are not especially strong.
- Repro:
  - Inspect training code and committed manifest metrics.
- Fix plan:
  - Keep the shared vectorizer if desired, but split to per-task heads with per-task calibration.
  - Prioritize per-task models for:
    - `query_intent`
    - `constraint_dimension`
    - `memory_type`
    - `constraint_type`
    - `constraint_scope`
    - `constraint_rerank`
  - Evaluate those heads on semantic-disconnect and constraint-consistency benchmarks, not only label accuracy.
- Test plan:
  - Add offline benchmark harness comparing current family model versus per-task heads on held-out task metrics and downstream retrieval outcomes.

### PKG-08
- Severity: Medium
- Category: Packaging / developer experience
- Evidence:
  - `packages/py-cml/src/cml/eval/config.py:8-19` and `packages/py-cml/src/cml/modeling/config.py:8-19` detect repo root via repo markers such as `docker/docker-compose.yml` and `evaluation/locomo_plus`.
  - `packages/py-cml/src/cml/eval/cli.py:46-56` defaults to repo-relative evaluation paths.
  - `packages/py-cml/src/cml/eval/pipeline.py:36-50` hardcodes repo-relative docker/eval paths.
  - `packages/models/scripts/prepare.py:11-19` mutates `sys.path` to import `cml.modeling.prepare`.
- Explanation:
  - Eval/modeling are advertised as package modules, but their defaults are still monorepo-first. Outside the repo they require manual path repair, and the legacy wrapper strategy depends on path mutation.
- Repro:
  - Static inspection.
- Fix plan:
  - Make non-repo usage explicit:
    - Require explicit `--config`, `--repo-root`, or `--out-dir` outside repo detection.
    - Remove `sys.path` mutation from legacy wrappers.
    - Keep wrappers as tiny process-level shims that invoke installed console scripts.
- Test plan:
  - Add temp-directory CLI tests that run from outside the repo tree.

### PKG-09
- Severity: Medium
- Category: Efficiency / maintainability
- Evidence:
  - `packages/py-cml/src/cml/client.py` and `packages/py-cml/src/cml/async_client.py` expose near-mirror method surfaces:
    - `health`, `write`, `read`, `turn`, `get_context`, `batch_write`, `batch_read`
    - mirrored namespaced/session helpers and dashboard/admin helpers
  - Example mirrored read methods:
    - `packages/py-cml/src/cml/client.py:199-264`
    - `packages/py-cml/src/cml/async_client.py:216-283`
- Explanation:
  - This is not a correctness bug today, but it increases parity-drift risk and patch cost. The package surface is already large, so duplicated sync/async implementations are expensive to keep behaviorally aligned.
- Repro:
  - Static inspection.
- Fix plan:
  - Extract shared request-building and response-parsing helpers.
  - Generate thin sync/async wrappers over a single endpoint schema where practical.
- Test plan:
  - Add a parity test that checks sync and async clients forward the same payload fields for matching methods.

## Prioritized Roadmap

### Quick wins
- Fix `max_retry_delay` wiring.
- Add model artifact completeness validation.
- Make `ModelPackRuntime` report partial capability instead of a single `available` boolean.
- Make default-task skips in `cml.modeling.train` fail loudly.
- Add migration fidelity tests for `embedded_utils`.

### Medium
- Preserve timestamp/type/session/context fields in import/export.
- Make `cml.eval` and `cml.modeling` explicitly support non-repo execution with cleaner path handling.
- Add package install smoke tests and artifact-capability tests to CI.

### Larger refactors
- Separate HTTP SDK from embedded/server engine packaging.
- Replace family-level composite-label training with per-task heads plus task-specific calibration.
- Package model assets intentionally, or redesign runtime so package behavior does not depend on repo-relative artifacts.

## Patch Plan

### PR-01: Package boundary and capability truthfulness
- Scope:
  - Add explicit package capability reporting.
  - Fail or warn clearly on partial model artifact sets.
  - Update docs to describe actual install/runtime boundaries.
- Code:
  - `src/utils/modelpack.py`
  - `packages/py-cml/README.md`
  - `packages/models/README.md`
  - `pyproject.toml`
- Tests:
  - artifact completeness unit test
  - install smoke tests in isolated environments

### PR-02: Migration fidelity and retry correctness
- Scope:
  - Preserve memory fields in `embedded_utils`.
  - Honor `config.max_retry_delay`.
- Code:
  - `packages/py-cml/src/cml/embedded_utils.py`
  - `packages/py-cml/src/cml/config.py`
  - `packages/py-cml/src/cml/transport/retry.py`
- Tests:
  - migration roundtrip fidelity unit test
  - retry-cap unit test
  - extended E2E migration assertions

### PR-03: Modeling package correctness
- Scope:
  - Remove unsupported default tasks or implement proper token classification.
  - Fail hard when default config requests unsupported objectives.
- Code:
  - `packages/models/model_pipeline.toml`
  - `packages/models/README.md`
  - `packages/py-cml/src/cml/modeling/train.py`
- Tests:
  - default-config training coverage
  - expected artifact set assertions

### PR-04: Standalone package usability
- Scope:
  - Remove repo-only assumptions from eval/modeling defaults.
  - Replace `sys.path` mutation wrappers with explicit installed-entrypoint behavior.
- Code:
  - `packages/py-cml/src/cml/eval/*`
  - `packages/py-cml/src/cml/modeling/*`
  - `packages/models/scripts/*.py`
- Tests:
  - temp-dir standalone CLI tests

### PR-05: Model architecture upgrade
- Scope:
  - Introduce per-task heads for constraint-relevant tasks and benchmark them.
- Code:
  - `packages/py-cml/src/cml/modeling/train.py`
  - `packages/py-cml/src/cml/modeling/prepare.py`
  - `packages/models/model_pipeline.toml`
  - benchmark harness under `scripts/` or `packages/models/benchmarks/`
- Tests and benchmarks:
  - held-out task metrics
  - semantic-disconnect retrieval benchmark
  - constraint-consistency evaluation against package-generated model outputs

## Risks and Non-goals

- Latency vs accuracy:
  - Packaging and preserving more structured fields will improve fidelity but may increase payload size and import cost.
- Backward compatibility:
  - Making unsupported modeling tasks fail loudly may break existing scripts that currently rely on silent skipping.
- Non-goals for the first pass:
  - I am not recommending immediate replacement of every heuristic path in `src/`.
  - I am not recommending shipping large transformer/span models in the base package.
  - I am not claiming current embedded mode is unusable; the finding is that it is more monorepo-dependent and more silently degraded than the package docs suggest.
