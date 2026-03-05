"""High-level orchestration helpers for modeling workflows."""

from __future__ import annotations

from cml.modeling.types import PrepareConfig, TrainConfig


def run_pipeline(prepare_cfg: PrepareConfig | None, train_cfg: TrainConfig | None) -> int:
    """Run prepare and/or train in order; return non-zero on first failure."""
    if prepare_cfg is not None:
        from cml.modeling.prepare import prepare_data

        rc = prepare_data(prepare_cfg)
        if rc != 0:
            return rc
    if train_cfg is not None:
        from cml.modeling.train import train_models

        rc = train_models(train_cfg)
        if rc != 0:
            return rc
    return 0
