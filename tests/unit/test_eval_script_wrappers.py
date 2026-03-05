"""Compatibility tests for evaluation legacy script wrappers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _run(script_path: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script_path, "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )


def test_run_full_eval_wrapper_help() -> None:
    result = _run("evaluation/scripts/run_full_eval.py")
    assert result.returncode == 0


def test_eval_locomo_wrapper_help() -> None:
    result = _run("evaluation/scripts/eval_locomo_plus.py")
    assert result.returncode == 0


def test_validate_outputs_wrapper_help() -> None:
    result = _run("evaluation/scripts/validate_outputs.py")
    assert result.returncode == 0
