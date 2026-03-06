"""Validate evaluation artifacts."""

from __future__ import annotations

import argparse
import json
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
    errors: list[str] = []
    if not path.exists():
        return [f"Missing: {path}"]
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path}: Invalid JSON: {exc}"]
    if not isinstance(data, list):
        return [f"{path}: Root must be a JSON array, got {type(data).__name__}"]
    for i, rec in enumerate(data):
        if not isinstance(rec, dict):
            errors.append(f"{path}: Record[{i}] is not an object")
            continue
        missing = REQUIRED_PREDICTION_KEYS - set(rec.keys())
        if missing:
            errors.append(f"{path}: Record[{i}] missing keys: {sorted(missing)}")
    return errors


def validate_judged(path: Path, predictions_path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing: {path}"]
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path}: Invalid JSON: {exc}"]
    if not isinstance(data, list):
        return [f"{path}: Root must be a JSON array, got {type(data).__name__}"]

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
            preds = None
        if isinstance(preds, list) and len(preds) != len(data):
            errors.append(f"{path}: length {len(data)} != predictions length {len(preds)}")
        elif isinstance(preds, list) and len(preds) == len(data):
            for i, (pred, judged) in enumerate(zip(preds, data, strict=False)):
                if pred.get("question_input") != judged.get("question_input"):
                    errors.append(f"{path}: Record[{i}] question_input mismatch with predictions")
                    break
                if pred.get("prediction") != judged.get("prediction"):
                    errors.append(f"{path}: Record[{i}] prediction mismatch with predictions")
                    break

    return errors


def validate_state_file(path: Path) -> list[str]:
    if not path.exists():
        return []
    try:
        data = load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path}: Invalid JSON: {exc}"]
    if not isinstance(data, dict):
        return [f"{path}: Root must be a JSON object, got {type(data).__name__}"]
    return []


def validate_summary(path: Path, judged_path: Path) -> list[str]:
    errors: list[str] = []
    if not path.exists():
        return [f"Missing: {path}"]
    try:
        summary = load_json(path)
    except json.JSONDecodeError as exc:
        return [f"{path}: Invalid JSON: {exc}"]
    if not isinstance(summary, dict):
        return [f"{path}: Root must be a JSON object, got {type(summary).__name__}"]

    for key in ("total_score", "total_samples", "max_possible", "overall_avg", "by_category"):
        if key not in summary:
            errors.append(f"{path}: Missing key {key!r}")

    by_cat = summary.get("by_category") or {}
    if isinstance(by_cat, dict):
        total_count = 0
        total_score = 0.0
        for cat, value in by_cat.items():
            if not isinstance(value, dict):
                errors.append(f"{path}: by_category.{cat} must be an object")
                continue
            total_count += int(value.get("count", 0))
            total_score += float(value.get("score", 0))
        if summary.get("total_samples") is not None and total_count != summary["total_samples"]:
            errors.append(
                f"{path}: sum(by_category.count)={total_count} != total_samples={summary['total_samples']}"
            )
        if (
            summary.get("total_score") is not None
            and abs(total_score - float(summary["total_score"])) > 0.01
        ):
            errors.append(
                f"{path}: sum(by_category.score)={total_score} != total_score={summary['total_score']}"
            )
    else:
        errors.append(f"{path}: by_category must be an object")

    if judged_path.exists():
        try:
            judged = load_json(judged_path)
        except Exception:
            judged = None
        if isinstance(judged, list):
            if summary.get("total_samples") is not None and summary["total_samples"] != len(judged):
                errors.append(
                    f"{path}: total_samples={summary['total_samples']} != judged length {len(judged)}"
                )
            computed_total = sum(float(item.get("judge_score", 0)) for item in judged)
            if (
                summary.get("total_score") is not None
                and abs(computed_total - float(summary["total_score"])) > 0.01
            ):
                errors.append(
                    f"{path}: total_score={summary['total_score']} != sum(judge_score)={computed_total}"
                )
    return errors


def validate_outputs(outputs_dir: Path) -> list[str]:
    pred_file = outputs_dir / "locomo_plus_qa_cml_predictions.json"
    judged_file = outputs_dir / "locomo_plus_qa_cml_judged.json"
    summary_file = outputs_dir / "locomo_plus_qa_cml_judge_summary.json"
    state_legacy = outputs_dir / "full_eval_state.json"
    state_current = outputs_dir / "run_full_eval_state.json"

    errors: list[str] = []
    errors.extend(validate_predictions(pred_file))
    errors.extend(validate_judged(judged_file, pred_file))
    errors.extend(validate_summary(summary_file, judged_file))
    errors.extend(validate_state_file(state_legacy))
    errors.extend(validate_state_file(state_current))
    return errors


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate evaluation outputs")
    parser.add_argument(
        "--outputs-dir",
        type=Path,
        default=Path("evaluation") / "outputs",
        help="Directory containing predictions, judged, and judge_summary JSON",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    errors = validate_outputs(Path(args.outputs_dir))
    if errors:
        print("Validation FAILED:", flush=True)
        for err in errors:
            print(f"  - {err}", flush=True)
        return 1
    print("Validation passed.", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
