#!/usr/bin/env python3
"""
Validate evaluation outputs in evaluation/outputs.

Checks:
- predictions.json: valid JSON array, required fields per record.
- judged.json: same length and keys as predictions + judge_label, judge_reason, judge_score.
- judge_summary.json: total_samples/counts/scores consistent with judged.json.
- Cross-file: predictions and judged align by index; summary aggregates match judged.

Usage (from project root):
  python evaluation/scripts/validate_outputs.py [--outputs-dir evaluation/outputs]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REQUIRED_PREDICTION_KEYS = {
    "question_input",
    "evidence",
    "category",
    "ground_truth",
    "prediction",
    "model",
}
REQUIRED_JUDGED_KEYS = REQUIRED_PREDICTION_KEYS | {"judge_label", "judge_reason", "judge_score"}
VALID_JUDGE_LABELS = {"correct", "partial", "wrong", ""}
VALID_JUDGE_SCORES = {0.0, 0.5, 1.0}


def load_json(path: Path) -> dict | list:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def validate_predictions(path: Path) -> list[str]:
    errors = []
    if not path.exists():
        errors.append(f"Missing: {path}")
        return errors
    try:
        data = load_json(path)
    except json.JSONDecodeError as e:
        errors.append(f"{path}: Invalid JSON: {e}")
        return errors
    if not isinstance(data, list):
        errors.append(f"{path}: Root must be a JSON array, got {type(data).__name__}")
        return errors
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            errors.append(f"{path}: Record[{i}] is not an object")
            continue
        missing = REQUIRED_PREDICTION_KEYS - set(rec.keys())
        if missing:
            errors.append(f"{path}: Record[{i}] missing keys: {sorted(missing)}")
    return errors


def validate_judged(path: Path, predictions_path: Path) -> list[str]:
    errors = []
    if not path.exists():
        errors.append(f"Missing: {path}")
        return errors
    try:
        data = load_json(path)
    except json.JSONDecodeError as e:
        errors.append(f"{path}: Invalid JSON: {e}")
        return errors
    if not isinstance(data, list):
        errors.append(f"{path}: Root must be a JSON array, got {type(data).__name__}")
        return errors

    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            errors.append(f"{path}: Record[{i}] is not an object")
            continue
        missing = REQUIRED_JUDGED_KEYS - set(rec.keys())
        if missing:
            errors.append(f"{path}: Record[{i}] missing keys: {sorted(missing)}")
            continue
        label = (rec.get("judge_label") or "").strip().lower()
        if label and label not in VALID_JUDGE_LABELS:
            errors.append(f"{path}: Record[{i}] invalid judge_label={rec.get('judge_label')!r}")
        try:
            score = float(rec.get("judge_score", 0))
        except (TypeError, ValueError):
            errors.append(
                f"{path}: Record[{i}] judge_score not numeric: {rec.get('judge_score')!r}"
            )
        else:
            if not any(abs(score - v) < 0.001 for v in VALID_JUDGE_SCORES):
                errors.append(f"{path}: Record[{i}] judge_score={score} not in {{0, 0.5, 1.0}}")

    if predictions_path.exists():
        try:
            preds = load_json(predictions_path)
        except Exception:
            pass
        else:
            if isinstance(preds, list) and len(preds) != len(data):
                errors.append(f"{path}: length {len(data)} != predictions length {len(preds)}")
            elif isinstance(preds, list) and len(preds) == len(data):
                for i in range(len(data)):
                    p, j = preds[i], data[i]
                    if p.get("question_input") != j.get("question_input"):
                        errors.append(
                            f"{path}: Record[{i}] question_input mismatch with predictions"
                        )
                        break
                    if p.get("prediction") != j.get("prediction"):
                        errors.append(f"{path}: Record[{i}] prediction mismatch with predictions")
                        break
    return errors


def validate_state_file(path: Path) -> list[str]:
    """Optional: full_eval_state.json if present (pipeline resume state)."""
    errors: list[str] = []
    if not path.exists():
        return errors
    try:
        data = load_json(path)
    except json.JSONDecodeError as e:
        errors.append(f"{path}: Invalid JSON: {e}")
        return errors
    if not isinstance(data, dict):
        errors.append(f"{path}: Root must be a JSON object, got {type(data).__name__}")
        return errors
    if "pipeline_step" in data and not isinstance(data["pipeline_step"], (int, float)):
        errors.append(f"{path}: pipeline_step must be numeric")
    return errors


def validate_summary(path: Path, judged_path: Path) -> list[str]:
    errors = []
    if not path.exists():
        errors.append(f"Missing: {path}")
        return errors
    try:
        summary = load_json(path)
    except json.JSONDecodeError as e:
        errors.append(f"{path}: Invalid JSON: {e}")
        return errors
    if not isinstance(summary, dict):
        errors.append(f"{path}: Root must be a JSON object, got {type(summary).__name__}")
        return errors

    for key in ("total_score", "total_samples", "max_possible", "overall_avg", "by_category"):
        if key not in summary:
            errors.append(f"{path}: Missing key {key!r}")

    by_cat = summary.get("by_category") or {}
    if not isinstance(by_cat, dict):
        errors.append(f"{path}: by_category must be an object")
    else:
        total_count = 0
        total_score_from_cat = 0.0
        for cat, v in by_cat.items():
            if not isinstance(v, dict):
                errors.append(f"{path}: by_category.{cat} must be an object")
                continue
            c = v.get("count", 0)
            s = v.get("score", 0)
            total_count += c
            total_score_from_cat += float(s)
        if summary.get("total_samples") is not None and total_count != summary["total_samples"]:
            errors.append(
                f"{path}: sum(by_category.count)={total_count} != total_samples={summary['total_samples']}"
            )
        if summary.get("total_score") is not None:
            if abs(total_score_from_cat - float(summary["total_score"])) > 0.01:
                errors.append(
                    f"{path}: sum(by_category.score)={total_score_from_cat} != total_score={summary['total_score']}"
                )

    if judged_path.exists():
        try:
            judged = load_json(judged_path)
        except Exception:
            pass
        else:
            if isinstance(judged, list):
                if summary.get("total_samples") is not None and summary["total_samples"] != len(
                    judged
                ):
                    errors.append(
                        f"{path}: total_samples={summary['total_samples']} != judged length {len(judged)}"
                    )
                else:
                    computed_total = sum(float(r.get("judge_score", 0)) for r in judged)
                    if summary.get("total_score") is not None:
                        if abs(computed_total - float(summary["total_score"])) > 0.01:
                            errors.append(
                                f"{path}: total_score={summary['total_score']} != sum(judge_score)={computed_total}"
                            )
    return errors


def main() -> None:
    p = argparse.ArgumentParser(description="Validate evaluation outputs")
    p.add_argument(
        "--outputs-dir",
        type=str,
        default=None,
        help="Directory containing predictions, judged, and judge_summary JSON (default: evaluation/outputs)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    out_dir = Path(args.outputs_dir) if args.outputs_dir else root / "outputs"

    pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
    judged_file = out_dir / "locomo_plus_qa_cml_judged.json"
    summary_file = out_dir / "locomo_plus_qa_cml_judge_summary.json"
    state_file = out_dir / "full_eval_state.json"

    all_errors: list[str] = []
    all_errors.extend(validate_predictions(pred_file))
    all_errors.extend(validate_judged(judged_file, pred_file))
    all_errors.extend(validate_summary(summary_file, judged_file))
    all_errors.extend(validate_state_file(state_file))

    # Optional: report stats
    stats: list[str] = []
    if pred_file.exists():
        try:
            preds = load_json(pred_file)
            if isinstance(preds, list):
                empty = sum(1 for r in preds if not (r.get("prediction") or "").strip())
                stats.append(f"predictions: {len(preds)} records, {empty} empty predictions")
        except Exception:
            pass
    if judged_file.exists():
        try:
            judged = load_json(judged_file)
            if isinstance(judged, list):
                labels: dict[str, int] = {}
                for r in judged:
                    label = (r.get("judge_label") or "").strip().lower() or "(empty)"
                    labels[label] = labels.get(label, 0) + 1
                stats.append(f"judged: {len(judged)} records, labels: {labels}")
        except Exception:
            pass
    if summary_file.exists():
        try:
            s = load_json(summary_file)
            if isinstance(s, dict):
                stats.append(
                    f"summary: total_score={s.get('total_score')}, total_samples={s.get('total_samples')}, overall_avg={s.get('overall_avg')}"
                )
        except Exception:
            pass
    if state_file.exists():
        try:
            st = load_json(state_file)
            if isinstance(st, dict):
                stats.append(
                    f"full_eval_state: pipeline_step={st.get('pipeline_step')}, eval_phase={st.get('eval_phase')}"
                )
        except Exception:
            pass

    if all_errors:
        print("Validation FAILED:", file=sys.stderr, flush=True)
        for e in all_errors:
            print(f"  - {e}", file=sys.stderr, flush=True)
        if stats:
            print("\nStats:", file=sys.stderr, flush=True)
            for stat_line in stats:
                print(f"  {stat_line}", file=sys.stderr, flush=True)
        sys.exit(1)
    print("Validation passed.", flush=True)
    if stats:
        for stat_line in stats:
            print(f"  {stat_line}", flush=True)
    sys.exit(0)


if __name__ == "__main__":
    main()
