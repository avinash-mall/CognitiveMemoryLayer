"""Compatibility tests for modeling legacy script wrappers."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("pandas", reason="modeling scripts require [modeling] deps")


def test_prepare_wrapper_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "packages.models.scripts.prepare", "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_train_wrapper_module_help() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "packages.models.scripts.train", "--help"],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0


def test_prepare_wrapper_reexports_internal_functions() -> None:
    from packages.models.scripts import prepare as p

    assert hasattr(p, "_missing_task_labels")
    assert callable(p._missing_task_labels)
