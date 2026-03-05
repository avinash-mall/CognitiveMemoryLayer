"""Unit tests for cml.modeling.pipeline.run_pipeline."""

from __future__ import annotations

from pathlib import Path

from cml.modeling.types import PrepareConfig, TrainConfig


def test_run_pipeline_prepare_only(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_prepare(config):
        calls.append("prepare")
        return 0

    monkeypatch.setattr("cml.modeling.prepare.prepare_data", _fake_prepare)
    from cml.modeling.pipeline import run_pipeline

    cfg = PrepareConfig(config_path=tmp_path / "cfg.toml")
    rc = run_pipeline(cfg, None)
    assert rc == 0
    assert calls == ["prepare"]


def test_run_pipeline_train_only(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_train(config):
        calls.append("train")
        return 0

    monkeypatch.setattr("cml.modeling.train.train_models", _fake_train)
    from cml.modeling.pipeline import run_pipeline

    cfg = TrainConfig(config_path=tmp_path / "cfg.toml")
    rc = run_pipeline(None, cfg)
    assert rc == 0
    assert calls == ["train"]


def test_run_pipeline_both(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_prepare(config):
        calls.append("prepare")
        return 0

    def _fake_train(config):
        calls.append("train")
        return 0

    monkeypatch.setattr("cml.modeling.prepare.prepare_data", _fake_prepare)
    monkeypatch.setattr("cml.modeling.train.train_models", _fake_train)
    from cml.modeling.pipeline import run_pipeline

    prep_cfg = PrepareConfig(config_path=tmp_path / "cfg.toml")
    train_cfg = TrainConfig(config_path=tmp_path / "cfg.toml")
    rc = run_pipeline(prep_cfg, train_cfg)
    assert rc == 0
    assert calls == ["prepare", "train"]


def test_run_pipeline_prepare_failure_skips_train(monkeypatch, tmp_path: Path) -> None:
    calls: list[str] = []

    def _fake_prepare(config):
        calls.append("prepare")
        return 1

    def _fake_train(config):
        calls.append("train")
        return 0

    monkeypatch.setattr("cml.modeling.prepare.prepare_data", _fake_prepare)
    monkeypatch.setattr("cml.modeling.train.train_models", _fake_train)
    from cml.modeling.pipeline import run_pipeline

    prep_cfg = PrepareConfig(config_path=tmp_path / "cfg.toml")
    train_cfg = TrainConfig(config_path=tmp_path / "cfg.toml")
    rc = run_pipeline(prep_cfg, train_cfg)
    assert rc == 1
    assert calls == ["prepare"]


def test_run_pipeline_none_none() -> None:
    from cml.modeling.pipeline import run_pipeline

    rc = run_pipeline(None, None)
    assert rc == 0
