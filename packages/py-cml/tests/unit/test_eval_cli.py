"""Unit tests for cml.eval CLI routing."""

from __future__ import annotations

from pathlib import Path

import cml.eval.cli as cli


def test_run_full_cli_routes_to_pipeline(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr("cml.eval.pipeline.main", _fake)
    rc = cli.main(
        [
            "run-full",
            "--repo-root",
            str(tmp_path),
            "--skip-docker",
            "--ingestion-workers",
            "7",
            "--score-only",
        ]
    )
    assert rc == 0
    assert "--repo-root" in captured["argv"]
    assert str(tmp_path) in captured["argv"]
    assert "--skip-docker" in captured["argv"]
    assert "--score-only" in captured["argv"]


def test_run_locomo_cli_routes_to_locomo(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr("cml.eval.locomo.main", _fake)
    rc = cli.main(
        [
            "run-locomo",
            "--unified-file",
            str(tmp_path / "u.json"),
            "--out-dir",
            str(tmp_path),
            "--cml-url",
            "http://localhost:8000",
            "--cml-api-key",
            "test-key",
            "--max-results",
            "10",
            "--ingestion-workers",
            "3",
        ]
    )
    assert rc == 0
    assert "--unified-file" in captured["argv"]
    assert str(tmp_path / "u.json") in captured["argv"]
    assert "--out-dir" in captured["argv"]


def test_validate_cli_routes_to_validate(monkeypatch, tmp_path: Path) -> None:
    called = {"count": 0}

    def _fake(argv=None):
        called["count"] += 1
        assert argv == ["--outputs-dir", str(tmp_path)]
        return 0

    monkeypatch.setattr("cml.eval.validate.main", _fake)
    rc = cli.main(["validate", "--outputs-dir", str(tmp_path)])
    assert rc == 0
    assert called["count"] == 1
