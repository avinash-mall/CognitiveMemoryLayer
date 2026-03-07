"""Typed configs for custom modeling workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class PrepareConfig:
    config_path: Path
    seed: int | None = None
    max_rows_per_source: int | None = None
    max_per_task_label: int | None = None
    target_per_task_label: int | None = None
    llm_temperature: float | None = None
    llm_concurrency: int | None = None
    disable_download: bool = False
    allow_missing_datasets_package: bool = False
    force_full: bool = False
    no_multilingual: bool = False


@dataclass(slots=True)
class TrainConfig:
    config_path: Path
    families: str = ""
    seed: int | None = None
    max_iter: int | None = None
    max_features: int | None = None
    predict_batch_size: int | None = None
    prepared_dir: Path | None = None
    output_dir: Path | None = None
    tasks: str = ""
    objective_types: str = ""
    max_seq_length: int | None = None
    learning_rate: float | None = None
    token_model_name_or_path: str | None = None
    token_num_train_epochs: int | None = None
    token_per_device_train_batch_size: int | None = None
    token_per_device_eval_batch_size: int | None = None
    token_stride: int | None = None
    token_warmup_ratio: float | None = None
    token_weight_decay: float | None = None
    token_gradient_accumulation_steps: int | None = None
    calibration_split: str | None = None
    export_thresholds: bool = False
    strict: bool = True
