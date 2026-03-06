"""Unit tests for eval validation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from cml.eval.validate import validate_outputs


def test_validate_outputs_happy_path(tmp_path: Path) -> None:
    preds = [
        {
            "question_input": "q",
            "evidence": "e",
            "category": "single-hop",
            "ground_truth": "a",
            "prediction": "a",
            "model": "m",
        }
    ]
    judged = [
        {
            **preds[0],
            "judge_label": "correct",
            "judge_reason": "ok",
            "judge_score": 1.0,
        }
    ]
    summary = {
        "total_score": 1.0,
        "total_samples": 1,
        "max_possible": 1,
        "overall_avg": 1.0,
        "by_category": {"single-hop": {"score": 1.0, "count": 1, "avg": 1.0}},
    }

    (tmp_path / "locomo_plus_qa_cml_predictions.json").write_text(
        json.dumps(preds), encoding="utf-8"
    )
    (tmp_path / "locomo_plus_qa_cml_judged.json").write_text(json.dumps(judged), encoding="utf-8")
    (tmp_path / "locomo_plus_qa_cml_judge_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    assert validate_outputs(tmp_path) == []


def test_validate_outputs_detects_mismatch(tmp_path: Path) -> None:
    preds = [
        {
            "question_input": "q",
            "evidence": "e",
            "category": "single-hop",
            "ground_truth": "a",
            "prediction": "a",
            "model": "m",
        }
    ]
    judged = [
        {
            **preds[0],
            "judge_label": "wrong",
            "judge_reason": "no",
            "judge_score": 0.0,
        }
    ]
    summary = {
        "total_score": 1.0,
        "total_samples": 1,
        "max_possible": 1,
        "overall_avg": 1.0,
        "by_category": {"single-hop": {"score": 1.0, "count": 1, "avg": 1.0}},
    }

    (tmp_path / "locomo_plus_qa_cml_predictions.json").write_text(
        json.dumps(preds), encoding="utf-8"
    )
    (tmp_path / "locomo_plus_qa_cml_judged.json").write_text(json.dumps(judged), encoding="utf-8")
    (tmp_path / "locomo_plus_qa_cml_judge_summary.json").write_text(
        json.dumps(summary), encoding="utf-8"
    )

    errors = validate_outputs(tmp_path)
    assert any("total_score" in e for e in errors)
