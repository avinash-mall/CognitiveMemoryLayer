"""Tests for scripts/update_readme_badges.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
SCRIPT_PATH = REPO_ROOT / "scripts" / "update_readme_badges.py"


def _load_module():
    spec = importlib.util.spec_from_file_location("update_readme_badges", SCRIPT_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib edge case
        raise RuntimeError("Failed to load update_readme_badges module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_update_badge_text_rewrites_current_badges() -> None:
    mod = _load_module()
    original = "\n".join(
        [
            "[![Tests](https://img.shields.io/badge/Tests-761-brightgreen?style=for-the-badge&logo=pytest)](./tests/README.md)",
            "[![Version](https://img.shields.io/badge/version-1.3.6-blue?style=for-the-badge)](#)",
        ]
    )
    updated, changed = mod.update_badge_text(original, version="2.0.1", tests=999)
    assert changed is True
    assert "Tests-999-brightgreen" in updated
    assert "version-2.0.1-blue" in updated


def test_update_badge_text_is_noop_when_values_match() -> None:
    mod = _load_module()
    original = "\n".join(
        [
            "[![Tests](https://img.shields.io/badge/Tests-12-brightgreen?logo=pytest)](./tests/README.md)",
            "[![Version](https://img.shields.io/badge/version-1.0.0-blue)](#)",
        ]
    )
    updated, changed = mod.update_badge_text(original, version="1.0.0", tests=12)
    assert changed is False
    assert updated == original
