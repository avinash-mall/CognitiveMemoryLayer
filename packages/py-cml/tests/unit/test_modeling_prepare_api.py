"""Unit tests for modeling prepare API wrappers."""

from __future__ import annotations

from pathlib import Path

import cml.modeling.prepare as prepare_module
from cml.modeling.types import PrepareConfig


def test_prepare_data_builds_expected_argv(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(prepare_module, "main", _fake_main)
    cfg = PrepareConfig(
        config_path=tmp_path / "model_pipeline.toml",
        seed=42,
        target_per_task_label=100,
        force_full=True,
        no_multilingual=True,
    )
    rc = prepare_module.prepare_data(cfg)
    assert rc == 0
    assert "--config" in captured["argv"]
    assert "--seed" in captured["argv"]
    assert "--force-full" in captured["argv"]
    assert "--no-multilingual" in captured["argv"]
