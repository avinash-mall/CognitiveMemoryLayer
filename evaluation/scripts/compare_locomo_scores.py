#!/usr/bin/env python3
"""Legacy wrapper for cml.eval compare command."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import():
    try:
        from cml.eval.cli import main_legacy_compare

        return main_legacy_compare
    except ImportError:
        repo_root = Path(__file__).resolve().parents[2]
        sdk_src = repo_root / "packages" / "py-cml" / "src"
        if sdk_src.exists():
            sys.path.insert(0, str(sdk_src))
        from cml.eval.cli import main_legacy_compare

        return main_legacy_compare


if __name__ == "__main__":
    runner = _bootstrap_import()
    raise SystemExit(runner(sys.argv[1:]))
