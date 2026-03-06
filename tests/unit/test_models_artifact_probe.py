"""Tests for scripts/models_artifact_probe.py mismatch gating."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _write_min_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[[tasks]]",
                'task_name = "novelty_pair"',
                'family = "pair"',
                'input_type = "pair"',
                'objective = "classification"',
                "enabled = true",
                'artifact_name = "novelty_pair"',
                'metrics = ["accuracy"]',
            ]
        ),
        encoding="utf-8",
    )


def test_models_artifact_probe_fail_on_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "model_pipeline.toml"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"
    prepared.mkdir(parents=True, exist_ok=True)
    trained.mkdir(parents=True, exist_ok=True)
    _write_min_config(config)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/models_artifact_probe.py",
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--trained-dir",
            str(trained),
            "--fail-on-mismatch",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Mismatch:" in result.stderr


def test_models_artifact_probe_no_fail_flag_returns_zero(tmp_path: Path) -> None:
    config = tmp_path / "model_pipeline.toml"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"
    prepared.mkdir(parents=True, exist_ok=True)
    trained.mkdir(parents=True, exist_ok=True)
    _write_min_config(config)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/models_artifact_probe.py",
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--trained-dir",
            str(trained),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert '"mismatches"' in result.stdout
