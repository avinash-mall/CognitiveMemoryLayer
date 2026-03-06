"""Unit tests for cml.modeling CLI routing."""

from __future__ import annotations

from pathlib import Path

import cml.modeling.cli as cli


def test_modeling_prepare_cli_routes(monkeypatch, tmp_path: Path) -> None:
    captured = {"config": None}

    def _fake(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("cml.modeling.prepare.prepare_data", _fake)
    rc = cli.main(["prepare", "--config", str(tmp_path / "model_pipeline.toml"), "--seed", "7"])
    assert rc == 0
    assert captured["config"].seed == 7


def test_modeling_train_cli_routes(monkeypatch, tmp_path: Path) -> None:
    captured = {"config": None}

    def _fake(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("cml.modeling.train.train_models", _fake)
    rc = cli.main(
        [
            "train",
            "--config",
            str(tmp_path / "model_pipeline.toml"),
            "--families",
            "router,pair",
            "--max-iter",
            "10",
        ]
    )
    assert rc == 0
    assert captured["config"].families == "router,pair"
    assert captured["config"].max_iter == 10
    assert captured["config"].strict is True


def test_modeling_train_cli_allow_skips(monkeypatch, tmp_path: Path) -> None:
    captured = {"config": None}

    def _fake(config):
        captured["config"] = config
        return 0

    monkeypatch.setattr("cml.modeling.train.train_models", _fake)
    rc = cli.main(
        [
            "train",
            "--config",
            str(tmp_path / "model_pipeline.toml"),
            "--allow-skips",
        ]
    )
    assert rc == 0
    assert captured["config"].strict is False


def test_modeling_pipeline_cli_passthrough(monkeypatch, tmp_path: Path) -> None:
    from cml.modeling.types import PrepareConfig, TrainConfig

    captured: dict = {}

    def _fake_run_pipeline(prep_cfg, train_cfg):
        captured["prep"] = prep_cfg
        captured["train"] = train_cfg
        return 0

    monkeypatch.setattr("cml.modeling.pipeline.run_pipeline", _fake_run_pipeline)

    rc = cli.main(
        [
            "pipeline",
            "--config",
            str(tmp_path / "model_pipeline.toml"),
            "--",
            "--seed",
            "11",
        ]
    )
    assert rc == 0
    assert isinstance(captured["prep"], PrepareConfig)
    assert isinstance(captured["train"], TrainConfig)
    assert captured["prep"].seed == 11
    assert captured["train"].seed == 11
    assert captured["train"].strict is True
    assert captured["prep"].config_path == tmp_path / "model_pipeline.toml"


def test_modeling_pipeline_cli_passthrough_allow_skips(monkeypatch, tmp_path: Path) -> None:
    captured: dict = {}

    def _fake_run_pipeline(prep_cfg, train_cfg):
        captured["prep"] = prep_cfg
        captured["train"] = train_cfg
        return 0

    monkeypatch.setattr("cml.modeling.pipeline.run_pipeline", _fake_run_pipeline)

    rc = cli.main(
        [
            "pipeline",
            "--config",
            str(tmp_path / "model_pipeline.toml"),
            "--",
            "--allow-skips",
        ]
    )
    assert rc == 0
    assert captured["train"].strict is False
