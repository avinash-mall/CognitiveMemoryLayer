# Modeling Module (`cml.modeling`)

`cml.modeling` exposes custom model data preparation and training from the consolidated model pipeline.

## Install

```bash
pip install "cognitive-memory-layer[modeling]"
```

The modeling extras install `pandas`, `scikit-learn`, `joblib`, `pyarrow`, and `datasets`. If you run `cml-models` without the extras, you will see a clear message telling you which dependency is missing.

## CLI

All subcommands are available via the `cml-models` entry point.

### `prepare` — Prepare modeling datasets

Downloads public NLP datasets, generates LLM-synthetic data, applies missing-only balancing, and writes stratified splits.

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
| `--llm-temperature` | from config | Temperature for LLM-based synthetic data |
| `--llm-concurrency` | from config | Concurrent LLM requests |
| `--disable-download` | off | Skip dataset download (use existing files) |
| `--allow-missing-datasets-package` | off | Proceed even if `datasets` package is unavailable |
| `--force-full` | off | Rebuild all datasets from scratch (ignore existing) |
| `--no-multilingual` | off | Skip multilingual data generation |

### `train` — Train custom models

Trains TF-IDF + SGDClassifier/SGDRegressor models for the 10 task-specific model objectives.

```bash
cml-models train --config packages/models/model_pipeline.toml
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | auto-detected `model_pipeline.toml` | Path to model pipeline TOML config |
| `--families` | all | Comma-separated model families to train (`router`, `extractor`, `pair`) |
| `--seed` | from config | Random seed |
| `--max-iter` | from config | Max SGD iterations |
| `--max-features` | from config | Max TF-IDF features |
| `--predict-batch-size` | from config | Prediction batch size |
| `--prepared-dir` | from config | Directory with prepared data |
| `--output-dir` | from config | Output directory for trained models |
| `--tasks` | all | Comma-separated task names to train |
| `--objective-types` | all | Comma-separated objective types (`classification`, `regression`, `pair`) |
| `--max-seq-length` | from config | Max input sequence length |
| `--learning-rate` | from config | SGD learning rate |
| `--calibration-split` | none | Split to use for calibration |
| `--export-thresholds` | off | Export decision thresholds alongside models |

**Note:** The `token_classification` objective is currently a stub and will raise `NotImplementedError` if selected. The dispatcher logs a skip message and continues with other tasks.

### `pipeline` — Run prepare + train as a single step

```bash
# Run both prepare and train
cml-models pipeline --config packages/models/model_pipeline.toml

# Skip prepare (train only)
cml-models pipeline --config packages/models/model_pipeline.toml --skip-prepare

# Skip train (prepare only)
cml-models pipeline --config packages/models/model_pipeline.toml --skip-train

# Pass extra args to both sub-steps
cml-models pipeline --config packages/models/model_pipeline.toml -- --seed 42
```

| Flag | Default | Description |
|------|---------|-------------|
| `--config` | auto-detected `model_pipeline.toml` | Path to model pipeline TOML config |
| `--skip-prepare` | off | Skip the preparation step |
| `--skip-train` | off | Skip the training step |
| `-- ARGS...` | none | Extra arguments forwarded to both prepare and train |

## Python API

All public functions and types are importable from `cml.modeling`:

```python
from pathlib import Path
from cml.modeling import PrepareConfig, TrainConfig, prepare_data, train_models, run_pipeline
```

### `prepare_data(config: PrepareConfig) -> int`

Run the data preparation pipeline. Returns 0 on success, non-zero on failure.

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

Train models according to the pipeline config. Returns 0 on success, non-zero on failure.

```python
rc = train_models(
    TrainConfig(
        config_path=Path("packages/models/model_pipeline.toml"),
        families="router,pair",
        export_thresholds=True,
    )
)
```

### `run_pipeline(prepare_cfg: PrepareConfig | None, train_cfg: TrainConfig | None) -> int`

Run prepare and/or train in sequence. Pass `None` to skip a step. Returns the first non-zero exit code, or 0 on success.

```python
rc = run_pipeline(
    PrepareConfig(config_path=Path("packages/models/model_pipeline.toml")),
    TrainConfig(config_path=Path("packages/models/model_pipeline.toml")),
)
```

### `PrepareConfig` dataclass fields

| Field | Type | Default | Description |
|-------|------|---------|-------------|
| `config_path` | `Path` | required | Path to `model_pipeline.toml` |
| `seed` | `int \| None` | `None` | Random seed |
| `max_rows_per_source` | `int \| None` | `None` | Cap rows per source |
| `max_per_task_label` | `int \| None` | `None` | Cap rows per task label |
| `target_per_task_label` | `int \| None` | `None` | Target count for balancing |
| `llm_temperature` | `float \| None` | `None` | LLM temperature |
| `llm_concurrency` | `int \| None` | `None` | LLM concurrency |
| `disable_download` | `bool` | `False` | Skip downloads |
| `allow_missing_datasets_package` | `bool` | `False` | Tolerate missing `datasets` |
| `force_full` | `bool` | `False` | Full rebuild |
| `no_multilingual` | `bool` | `False` | Skip multilingual |

### `TrainConfig` dataclass fields

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
| `max_seq_length` | `int \| None` | `None` | Max input length |
| `learning_rate` | `float \| None` | `None` | SGD learning rate |
| `calibration_split` | `str \| None` | `None` | Calibration split name |
| `export_thresholds` | `bool` | `False` | Export thresholds |

## Legacy script compatibility

Legacy script entry points remain available as wrappers:

- `python -m packages.models.scripts.prepare ...`
- `python -m packages.models.scripts.train ...`

Internal helpers used by tests (for example `_missing_task_labels`) are re-exported from wrapper modules.
