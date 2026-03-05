# New Modules for py-cml

## Objective
Add two optional, separately installable modules under `py-cml` so users can install only what they need:

1. `cml.eval`: run evaluation with core features.
2. `cml.modeling`: prepare data and train custom models.

This keeps base SDK install small while exposing eval/training workflows as first-class package modules.

## Proposed Install Experience
Use optional dependency groups from the root `pyproject.toml`.

```bash
# Evaluation only
pip install "cognitive-memory-layer[eval]"

# Custom model prep + training only
pip install "cognitive-memory-layer[modeling]"

# Both
pip install "cognitive-memory-layer[eval,modeling]"
```

## Module 1: `cml.eval` (Basic Evaluation Features)

### Scope
Wrap current evaluation workflows from `evaluation/scripts/*` into importable APIs + CLI.

### Features to include
- Full evaluation orchestration (docker optional): equivalent to `run_full_eval.py`.
- Dataset evaluation flow: equivalent to `eval_locomo_plus.py`.
- Output validation: equivalent to `validate_outputs.py`.
- Report generation: equivalent to `generate_locomo_report.py`.
- Score comparison helper: equivalent to `compare_locomo_scores.py`.
- Resume support using run state file (`run_full_eval_state.json`).
- Parallel ingestion workers (`--ingestion-workers`).
- Optional modes: `--skip-docker`, `--skip-ingestion`, `--skip-consolidation`, `--score-only`, `--limit-samples`.

### Suggested package layout
```text
packages/py-cml/src/cml/eval/
  __init__.py
  config.py
  runner.py
  validate.py
  report.py
  compare.py
  cli.py
```

### Public Python API (suggested)
- `run_full_eval(config: EvalRunConfig) -> EvalRunResult`
- `run_locomo_plus(config: LocomoEvalConfig) -> EvalArtifacts`
- `validate_outputs(outputs_dir: Path) -> ValidationResult`
- `generate_report(summary_file: Path, method: str) -> ReportTable`
- `compare_scores(summary_file: Path) -> ComparisonTable`

### CLI (suggested)
- `cml-eval run-full ...`
- `cml-eval run-locomo ...`
- `cml-eval validate ...`
- `cml-eval report ...`
- `cml-eval compare ...`

## Module 2: `cml.modeling` (Custom Model Prepare + Train)

### Scope
Wrap existing pipeline logic from `packages/models/scripts/prepare.py` and `packages/models/scripts/train.py` into importable APIs + CLI.

### Features to include
- Data preparation pipeline with config-driven datasets.
- Missing-only balancing + synthetic backfill.
- Optional `--force-full` rebuild.
- Multilingual generation toggle (`--no-multilingual`).
- Family-level training (`router`, `extractor`, `pair`).
- Task-level training filters (`--tasks`, `--objective-types`).
- Artifact + manifest generation.
- Optional threshold export (`--export-thresholds`).

### Suggested package layout
```text
packages/py-cml/src/cml/modeling/
  __init__.py
  config.py
  prepare.py
  train.py
  pipeline.py
  cli.py
```

### Public Python API (suggested)
- `prepare_data(config: ModelPipelineConfig, overrides: PrepareOverrides) -> PrepareResult`
- `train_models(config: ModelPipelineConfig, overrides: TrainOverrides) -> TrainResult`
- `run_pipeline(config: ModelPipelineConfig, run_prepare: bool, run_train: bool) -> PipelineResult`

### CLI (suggested)
- `cml-models prepare ...`
- `cml-models train ...`
- `cml-models pipeline ...`

## Packaging Changes Required

### 1) Optional dependencies
Add extras in root `pyproject.toml`:

```toml
[project.optional-dependencies]
eval = [
  "requests>=2.32",
  "tqdm>=4.66",
  "python-dotenv>=1.0",
  "openai>=1.0",
]
modeling = [
  "pandas>=2.2",
  "scikit-learn>=1.5",
  "joblib>=1.4",
  "pyarrow>=17.0",
  "datasets>=2.20",
]
```

### 2) Console entry points
```toml
[project.scripts]
cml-eval = "cml.eval.cli:main"
cml-models = "cml.modeling.cli:main"
```

### 3) Backward compatibility
Keep existing scripts in:
- `evaluation/scripts/*`
- `packages/models/scripts/*`

Convert them into thin wrappers that call the new package APIs. This avoids breaking current docs and CI commands.

## Implementation Steps

1. Extract reusable logic from current scripts into `cml.eval` and `cml.modeling` modules.
2. Keep file-path defaults aligned with current repo layout (same defaults as existing scripts).
3. Add typed config/result dataclasses for stable API contracts.
4. Add CLI entrypoints and argument parity with current scripts.
5. Add tests in `packages/py-cml/tests/unit` for APIs and CLI argument parsing.
6. Update `packages/py-cml/README.md` and docs pages with new install options and examples.

## Acceptance Criteria
- `pip install "cognitive-memory-layer[eval]"` enables `cml-eval` commands without modeling dependencies.
- `pip install "cognitive-memory-layer[modeling]"` enables `cml-models` commands without eval dependencies.
- Existing direct script commands still work.
- New module APIs are importable and covered by unit tests.
- Eval artifacts and model artifacts remain compatible with current output formats.

---

## Implementation Status

**Status: Implemented and hardened.**

All features described above have been implemented. The following post-implementation improvements were applied:

### Robustness Fixes
- **Lazy imports throughout** — `cml.eval.__init__`, `cml.modeling.__init__`, `cml.eval.cli`, and `cml.modeling.cli` all use lazy imports for heavy dependencies. Running a CLI without the corresponding extras produces a clear error message with install instructions.
- **`locomo.py` standalone fallback** — `run_locomo_plus` works outside the CML repository via a built-in sample loader fallback. `phase_c_judge` raises a descriptive `ImportError` when the repo-local `task_eval` module is unavailable.
- **`locomo.py` retry/error handling** — `_dashboard_post` and `_cml_read` handle exhausted retry loops correctly. `main()` catches `FileNotFoundError`, `ImportError`, and general exceptions with non-zero exit codes.
- **`train.py` token_classification** — Stub now raises `NotImplementedError` (caught gracefully by the dispatcher) instead of silently returning.
- **`modeling/cli.py` pipeline** — The `pipeline` subcommand correctly uses `run_pipeline` with typed configs instead of raw argv delegation.

### Documentation
- `docs/evaluation.md` — Full CLI flag tables, Python API reference, standalone usage notes.
- `docs/modeling.md` — Full CLI flag tables, Python API reference, dataclass field tables, `token_classification` note.
- `README.md` — Updated Optional Modules section with expanded examples.
- `docs/getting-started.md` — New Optional Modules section with cross-references.
- Root `README.md` — Updated Evaluation, Training, and Documentation sections to reference `cml-eval`/`cml-models`.
- `CHANGELOG.md` — Added entries under `[Unreleased]` for both new modules and all fixes.

### Tests
- Unit tests cover CLI argument parsing and routing for both `cml.eval.cli` and `cml.modeling.cli`.
- Unit tests cover `cml.modeling.pipeline.run_pipeline`.
- All monkeypatching targets updated to match the lazy-import structure.
