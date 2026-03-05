"""Unit tests for eval report/compare helpers."""

from __future__ import annotations

import json
from pathlib import Path

from cml.eval.compare import compare_locomo_scores
from cml.eval.report import generate_locomo_report


def _summary() -> dict:
    return {
        "total_score": 3.0,
        "total_samples": 6,
        "max_possible": 6,
        "overall_avg": 0.5,
        "by_category": {
            "single-hop": {"score": 1.0, "count": 1, "avg": 1.0},
            "multi-hop": {"score": 0.5, "count": 1, "avg": 0.5},
            "temporal": {"score": 0.0, "count": 1, "avg": 0.0},
            "common-sense": {"score": 0.5, "count": 1, "avg": 0.5},
            "adversarial": {"score": 0.0, "count": 1, "avg": 0.0},
            "Cognitive": {"score": 1.0, "count": 1, "avg": 1.0},
        },
    }


def test_generate_locomo_report_contains_method(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    table = generate_locomo_report(summary_path, method="CML+test", no_title=False)
    assert "CML+test" in table
    assert "LoCoMo-Plus" in table


def test_compare_locomo_scores_contains_baselines(tmp_path: Path) -> None:
    summary_path = tmp_path / "summary.json"
    summary_path.write_text(json.dumps(_summary()), encoding="utf-8")

    out = compare_locomo_scores(summary_path, method="CML+test")
    assert "CML+test" in out
    assert "Qwen2.5-3B-Instruct" in out
