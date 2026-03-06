"""Unit tests for modeling train API wrappers."""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import cml.modeling.train as train_module
from cml.modeling.types import TrainConfig


def test_train_models_builds_expected_argv(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(train_module, "main", _fake_main)
    cfg = TrainConfig(
        config_path=tmp_path / "model_pipeline.toml",
        families="router,pair",
        max_iter=20,
        tasks="novelty_pair",
        objective_types="classification",
        export_thresholds=True,
    )
    rc = train_module.train_models(cfg)
    assert rc == 0
    assert "--families" in captured["argv"]
    assert "router,pair" in captured["argv"]
    assert "--export-thresholds" in captured["argv"]
