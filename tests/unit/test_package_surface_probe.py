"""Unit tests for scripts/package_surface_probe.py CLI flags."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "package_surface_probe.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("package_surface_probe", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib edge case
        raise RuntimeError("Failed to load package_surface_probe module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_parser_accepts_expect_min_memories() -> None:
    mod = _load_module()
    parser = mod.build_parser()
    ns = parser.parse_args(
        [
            "embedded",
            "--write",
            "hello",
            "--query",
            "hello",
            "--expect-min-memories",
            "1",
        ]
    )
    assert ns.expect_min_memories == 1
