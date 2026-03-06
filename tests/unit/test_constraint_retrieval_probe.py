"""Deterministic checks for scripts/constraint_retrieval_probe.py helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
PROBE_PATH = REPO_ROOT / "scripts" / "constraint_retrieval_probe.py"


def _load_probe_module():
    spec = importlib.util.spec_from_file_location("constraint_retrieval_probe", PROBE_PATH)
    if spec is None or spec.loader is None:  # pragma: no cover - importlib edge case
        raise RuntimeError("Failed to load constraint_retrieval_probe module")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_evaluate_constraint_probe_hit_metrics() -> None:
    probe = _load_probe_module()
    memories = [
        {"text": "User has severe shellfish allergy and carries epipen."},
        {"text": "User likes Thai food."},
    ]
    context = "Active Constraints: severe shellfish allergy."
    metrics = probe.evaluate_constraint_probe(
        memories,
        context,
        expected_terms=["shellfish", "allergy", "epipen"],
        top_k=2,
    )
    assert metrics["memory_hit_rate"] >= 0.66
    assert metrics["context_hit_rate"] >= 0.66
    assert metrics["all_terms_hit"] is True


def test_probe_scenarios_include_shellfish_fixture() -> None:
    probe = _load_probe_module()
    scenario = probe.SCENARIOS["shellfish_restaurant"]
    assert scenario.query
    assert len(scenario.writes) >= 2
