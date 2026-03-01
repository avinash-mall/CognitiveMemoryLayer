# Model Pipeline

This folder contains the consolidated custom-model pipeline used by CML runtime.

- Prepare script: `packages/models/scripts/prepare.py`
- Train script: `packages/models/scripts/train.py`
- Config: `packages/models/model_pipeline.toml`
- Output models: `packages/models/trained_models`

## Model families

1. `router_model.joblib`
- Tasks: `memory_type`, `query_intent`, `query_domain`, `constraint_dimension`, `context_tag`, `salience_bin`, `importance_bin`, `confidence_bin`, `decay_profile`

2. `extractor_model.joblib`
- Tasks: `constraint_type`, `constraint_scope`, `constraint_stability`, `fact_type`, `pii_presence`

3. `pair_model.joblib`
- Tasks: `conflict_detection`, `constraint_rerank`, `scope_match`, `supersession`

## Runtime wiring

Modelpack inference is consumed from `src/utils/modelpack.py`.

Current runtime integrations include:

- `query_domain`: classifier/planner/retriever
- `scope_match`: retriever/reranker
- `constraint_stability`: reranker
- `supersession`: constraint extraction/supersession checks
- `pii_presence` + `importance_bin`: write gate non-LLM path

If model artifacts are not present, runtime uses safe non-LLM defaults for those paths.

## Data preparation

`prepare.py` performs:

1. dataset loading (auto-download via Hugging Face IDs in config)
2. merge with existing local prepared data when available
3. missing-only balancing to target counts (default `10000` per task-label)
4. LLM-only synthetic backfill for deficits
5. stratified split output (`train`, `test`, `eval`)

Prepared outputs are written to `packages/models/prepared_data/modelpack`.

### Existing dataset folder

Raw dataset cache/downloads use existing flat folder:

- `packages/models/datasets`

## Synthetic generation (LLM-only)

Synthetic generation does not use rule-based labeling.

Flow:

1. Randomly select seed samples from related datasets.
2. Prompt LLM for target `(task, label)` generation.
3. Parse/validate and keep accepted rows until target count is reached.

Expected env settings (example):

```bash
LLM_EVAL__PROVIDER=ollama
LLM_EVAL__MODEL=gemma3:12b
LLM_EVAL__BASE_URL=http://localhost:11434/v1
```

## Training

`train.py` trains all selected families and writes:

- `*_model.joblib`
- `*_label_map.json`
- `*_metrics_test.json`
- `*_metrics_eval.json`
- `*_report_test.json`
- `*_report_eval.json`
- `*_epoch_stats.json`
- `*_training_metadata.json`
- `manifest.json`

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
python -m packages.models.scripts.prepare --force-full
python -m packages.models.scripts.train --max-iter 25 --max-features 250000
```

## Config reference

All settings are in `packages/models/model_pipeline.toml`:

- paths
- preparation targets/splits
- training hyperparameters
- synthetic LLM parameters
- dataset source definitions (links + IDs + required/optional)
