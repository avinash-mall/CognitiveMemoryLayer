"""Custom model prep/training helpers for CML."""

from cml.modeling.pipeline import run_pipeline
from cml.modeling.types import PrepareConfig, TrainConfig


def prepare_data(config: PrepareConfig) -> int:
    from cml.modeling.prepare import prepare_data as _prepare_data

    return _prepare_data(config)


def train_models(config: TrainConfig) -> int:
    from cml.modeling.train import train_models as _train_models

    return _train_models(config)


__all__ = [
    "PrepareConfig",
    "TrainConfig",
    "prepare_data",
    "run_pipeline",
    "train_models",
]
