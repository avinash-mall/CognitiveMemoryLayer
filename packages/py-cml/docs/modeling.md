# Modeling Module (`cml.modeling`)

`cml.modeling` provides typed APIs and CLI commands for model data preparation and training in `packages/models`.

## Install

```bash
pip install "cognitive-memory-layer[modeling]"
```

The modeling extra installs `pandas`, `scikit-learn`, `joblib`, `pyarrow`, and `datasets`. Running `cml-models` without the extra prints an actionable install error.

## CLI

All subcommands are available via `cml-models`.

### `prepare` - Prepare modeling datasets

Downloads public NLP datasets, generates synthetic data, applies balancing, and writes train/test/eval splits.

```bash
cml-models prepare --config packages/models/model_pipeline.toml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | auto-detected `model_pipeline.toml` | Path to model pipeline TOML config |
| `--seed` | from config | Random seed for reproducibility |
| `--max-rows-per-source` | from config | Cap rows per source dataset |
| `--max-per-task-label` | from config | Cap rows per task label |
| `--target-per-task-label` | from config | Target count per label for balancing |
| `--llm-temperature` | from config | Temperature for synthetic generation |
| `--llm-concurrency` | from config | Concurrent LLM requests |
| `--disable-download` | off | Skip dataset download (use existing files) |
| `--allow-missing-datasets-package` | off | Proceed even if `datasets` package is unavailable |
| `--force-full` | off | Rebuild all datasets from scratch |
| `--no-multilingual` | off | Skip multilingual synthetic generation |

### `train` - Train custom models

Trains family models (`router`, `extractor`, `pair`) and task models defined in `[[tasks]]`.

```bash
cml-models train --config packages/models/model_pipeline.toml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | auto-detected `model_pipeline.toml` | Path to model pipeline TOML config |
| `--families` | all | Comma-separated families (`router`, `extractor`, `pair`) |
| `--seed` | from config | Random seed |
| `--max-iter` | from config | Max SGD iterations |
| `--max-features` | from config | Max TF-IDF features |
| `--predict-batch-size` | from config | Prediction batch size |
| `--prepared-dir` | from config | Directory with prepared parquet splits |
| `--output-dir` | from config | Output directory for trained artifacts |
| `--tasks` | all | Comma-separated task names to train |
| `--objective-types` | all | Comma-separated objective types (`classification`, `pair_ranking`, `single_regression`, `token_classification`) |
| `--max-seq-length` | from config | Max input sequence length (reserved for token trainers) |
| `--learning-rate` | from config | Learning rate override (reserved for token trainers) |
| `--calibration-split` | none | Split to use for threshold calibration |
| `--export-thresholds` | off | Export decision thresholds alongside models |
| `--strict` | on | Hard-fail on preflight, unsupported objectives, disabled selected tasks, empty/missing task rows, or missing artifacts |
| `--allow-skips` | off | Legacy behavior; continue despite preflight/task/artifact errors |

Strict mode is the default (`TrainConfig.strict=True`). Preflight validates objective support, required columns, configured-vs-observed task coverage, and regression score requirements before training task models. The train manifest is schema v2 and includes `configured_tasks`, `preflight_validation`, `task_training_status`, and `build_metadata`.

Tasks with objective `token_classification` are currently deferred. In default config they are disabled and strict mode fails if they are explicitly selected.

### `pipeline` - Run prepare + train in one command

```bash
# Run both prepare and train
cml-models pipeline --config packages/models/model_pipeline.toml

# Skip prepare (train only)
cml-models pipeline --config packages/models/model_pipeline.toml --skip-prepare

# Skip train (prepare only)
cml-models pipeline --config packages/models/model_pipeline.toml --skip-train

# Pass args through to prepare/train
cml-models pipeline --config packages/models/model_pipeline.toml -- --seed 42 --strict
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | auto-detected `model_pipeline.toml` | Path to model pipeline TOML config |
| `--skip-prepare` | off | Skip the preparation step |
| `--skip-train` | off | Skip the training step |
| `-- ARGS...` | none | Extra arguments forwarded to both prepare and train parsers |

## Python API

```python
from pathlib import Path
from cml.modeling import PrepareConfig, TrainConfig, prepare_data, train_models, run_pipeline
```

### `prepare_data(config: PrepareConfig) -> int`

```python
rc = prepare_data(
    PrepareConfig(
        config_path=Path("packages/models/model_pipeline.toml"),
        seed=42,
        force_full=True,
    )
)
```

### `train_models(config: TrainConfig) -> int`

```python
rc = train_models(
    TrainConfig(
        config_path=Path("packages/models/model_pipeline.toml"),
        families="router,pair",
        strict=True,
        export_thresholds=True,
    )
)
```

### `run_pipeline(prepare_cfg: PrepareConfig | None, train_cfg: TrainConfig | None) -> int`

```python
rc = run_pipeline(
    PrepareConfig(config_path=Path("packages/models/model_pipeline.toml")),
    TrainConfig(config_path=Path("packages/models/model_pipeline.toml"), strict=True),
)
```

## Dataclass Fields

### `PrepareConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config_path` | `Path` | required | Path to `model_pipeline.toml` |
| `seed` | `int \| None` | `None` | Random seed |
| `max_rows_per_source` | `int \| None` | `None` | Cap rows per source |
| `max_per_task_label` | `int \| None` | `None` | Cap rows per task label |
| `target_per_task_label` | `int \| None` | `None` | Target count for balancing |
| `llm_temperature` | `float \| None` | `None` | Synthetic generation temperature |
| `llm_concurrency` | `int \| None` | `None` | LLM concurrency |
| `disable_download` | `bool` | `False` | Skip downloads |
| `allow_missing_datasets_package` | `bool` | `False` | Tolerate missing `datasets` |
| `force_full` | `bool` | `False` | Full rebuild |
| `no_multilingual` | `bool` | `False` | Skip multilingual generation |

### `TrainConfig`

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config_path` | `Path` | required | Path to `model_pipeline.toml` |
| `families` | `str` | `""` (all) | Comma-separated families |
| `seed` | `int \| None` | `None` | Random seed |
| `max_iter` | `int \| None` | `None` | Max SGD iterations |
| `max_features` | `int \| None` | `None` | Max TF-IDF features |
| `predict_batch_size` | `int \| None` | `None` | Prediction batch size |
| `prepared_dir` | `Path \| None` | `None` | Input data directory |
| `output_dir` | `Path \| None` | `None` | Output model directory |
| `tasks` | `str` | `""` (all) | Comma-separated task names |
| `objective_types` | `str` | `""` (all) | Comma-separated objective types |
| `max_seq_length` | `int \| None` | `None` | Reserved for token trainers |
| `learning_rate` | `float \| None` | `None` | Reserved for token trainers |
| `calibration_split` | `str \| None` | `None` | Calibration split name |
| `export_thresholds` | `bool` | `False` | Export thresholds |
| `strict` | `bool` | `True` | Hard-fail modeling contract checks |

## Legacy script compatibility

Legacy wrapper entry points remain available:

- `python -m packages.models.scripts.prepare ...`
- `python -m packages.models.scripts.train ...`

Internal helpers used by tests (for example `_missing_task_labels`) are re-exported from wrapper modules.
