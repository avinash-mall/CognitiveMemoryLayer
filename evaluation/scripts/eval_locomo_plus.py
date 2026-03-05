#!/usr/bin/env python3
"""Legacy wrapper for cml.eval run-locomo command."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any


def _load_module():
    try:
        return import_module("cml.eval.locomo")
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sdk_src = repo_root / "packages" / "py-cml" / "src"
        if sdk_src.exists():
            sys.path.insert(0, str(sdk_src))
        return import_module("cml.eval.locomo")


def _bootstrap_import():
    try:
        from cml.eval.cli import main_legacy_eval_locomo

        return main_legacy_eval_locomo
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sdk_src = repo_root / "packages" / "py-cml" / "src"
        if sdk_src.exists():
            sys.path.insert(0, str(sdk_src))
        from cml.eval.cli import main_legacy_eval_locomo

        return main_legacy_eval_locomo


_module = _load_module()

# Explicit legacy symbols used by tests/tools.
_cml_write = _module._cml_write
_cml_read = _module._cml_read
phase_a_ingestion = _module.phase_a_ingestion
phase_b_qa = _module.phase_b_qa
phase_c_judge = _module.phase_c_judge


def __getattr__(name: str) -> Any:
    """Fallback for attributes moved into cml.eval.locomo."""
    return getattr(_module, name)


if __name__ == "__main__":
    runner = _bootstrap_import()
    raise SystemExit(runner(sys.argv[1:]))
