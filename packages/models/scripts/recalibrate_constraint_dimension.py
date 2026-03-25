"""Recalibrate constraint_dimension model temperature using a wider grid.

The original temperature grid [0.7..2.0] found T=2.0 as optimal, giving ECE=0.083
which fails the ≤0.06 gate. A wider grid reveals T=2.8 gives ECE≈0.044.

This script:
  1. Loads the existing HF checkpoint and finds the best temperature on eval data.
  2. Updates the model.joblib with the new temperature.
  3. Re-evaluates on test and eval splits.
  4. Writes updated metrics files.

Run from repo root:
    python packages/models/scripts/recalibrate_constraint_dimension.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import joblib  # noqa: E402
import numpy as np  # noqa: E402

from cml.modeling.runtime_models import TransformerTextClassifier  # noqa: E402
from cml.modeling.train import (  # noqa: E402
    _encode_features,
    _encode_targets,
    _evaluate,
    _load_split,
    _write_task_metrics,
)

MODELS_DIR = REPO_ROOT / "packages" / "models" / "trained_models"
PREPARED_DIR = REPO_ROOT / "packages" / "models" / "prepared_data" / "modelpack"
TASK = "constraint_dimension"
ARTIFACT = "constraint_dimension"
MAX_SEQ_LENGTH = 128
BATCH_SIZE = 16


def _ece(y_true: list[str], probs: np.ndarray, classes: list[str], n_bins: int = 10) -> float:
    confidences = probs.max(axis=1)
    preds = np.array([classes[int(i)] for i in probs.argmax(axis=1)])
    accuracies = (preds == np.array(y_true)).astype(float)
    edges = np.linspace(0, 1, n_bins + 1)
    result = 0.0
    for i in range(n_bins):
        mask = (confidences >= edges[i]) & (confidences < edges[i + 1])
        if mask.sum() > 0:
            result += (mask.sum() / len(confidences)) * abs(
                accuracies[mask].mean() - confidences[mask].mean()
            )
    return float(result)


def main() -> None:
    hf_dir = MODELS_DIR / f"{ARTIFACT}_hf"
    cfg = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    id2label: dict[str, str] = cfg["id2label"]
    classes = [id2label[str(i)] for i in range(len(id2label))]
    print(f"[{TASK}] classes: {classes}")

    # ── Load eval split for calibration ──────────────────────────────────────
    eval_df = _load_split(PREPARED_DIR, "router", "eval")
    eval_df = eval_df[eval_df["task"] == TASK].reset_index(drop=True)
    test_df = _load_split(PREPARED_DIR, "router", "test")
    test_df = test_df[test_df["task"] == TASK].reset_index(drop=True)
    print(f"[{TASK}] eval rows: {len(eval_df)}, test rows: {len(test_df)}")

    eval_features = _encode_features(eval_df, "router")
    eval_y = _encode_targets(eval_df)

    # ── Find best temperature on eval data ────────────────────────────────────
    model = TransformerTextClassifier(
        task_name=TASK,
        model_dir=str(hf_dir),
        classes_=classes,
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        temperature=1.0,
    )
    model._ensure_loaded()  # load once; then swap temperature cheaply

    # Finer grid around the previous best (T=2.0 was the old grid's maximum)
    candidate_temps = [float(x) / 10.0 for x in range(20, 51)]  # 2.0 to 5.0 step 0.1
    best_t = 2.0
    best_ece = float("inf")
    print(f"[{TASK}] Scanning {len(candidate_temps)} temperature candidates …")
    for t in candidate_temps:
        model.temperature = t
        probs = model.predict_proba(eval_features)
        e = _ece(eval_y, probs, classes)
        if e < best_ece:
            best_ece = e
            best_t = t

    print(f"[{TASK}] Best temperature: {best_t:.1f} → eval ECE = {best_ece:.5f}")

    # ── Load existing joblib and update temperature ───────────────────────────
    model_path = MODELS_DIR / f"{ARTIFACT}_model.joblib"
    payload = joblib.load(str(model_path))
    runtime_model: TransformerTextClassifier = payload["model"]
    old_temp = runtime_model.temperature
    runtime_model.temperature = best_t
    print(f"[{TASK}] Updated temperature {old_temp} → {best_t}")

    # ── Re-evaluate ───────────────────────────────────────────────────────────
    print(f"[{TASK}] Evaluating on eval split …")
    metrics_eval, _ = _evaluate(
        runtime_model, eval_df, family="router", split_name="eval", predict_batch_size=BATCH_SIZE
    )
    print(f"  eval accuracy: {metrics_eval['overall']['accuracy']:.4f}")
    print(f"  eval macro_f1: {metrics_eval['overall']['macro_f1']:.4f}")
    print(f"  eval calibration_error: {metrics_eval['overall']['calibration_error']:.5f}")

    print(f"[{TASK}] Evaluating on test split …")
    metrics_test, _ = _evaluate(
        runtime_model, test_df, family="router", split_name="test", predict_batch_size=BATCH_SIZE
    )
    print(f"  test accuracy: {metrics_test['overall']['accuracy']:.4f}")
    print(f"  test macro_f1: {metrics_test['overall']['macro_f1']:.4f}")
    print(f"  test calibration_error: {metrics_test['overall']['calibration_error']:.5f}")

    gate_pass = metrics_test["overall"]["calibration_error"] <= 0.06
    print(f"[{TASK}] Gate (calibration_error ≤ 0.06): {'PASS' if gate_pass else 'FAIL'}")

    # ── Save updated joblib ───────────────────────────────────────────────────
    payload["model"] = runtime_model
    joblib.dump(payload, str(model_path))
    print(f"[{TASK}] Saved updated model to {model_path}")

    # Update calibration field in metrics
    calibration = {
        "method": "temperature_grid_search",
        "rows": len(eval_df),
        "temperature": best_t,
        "loss": None,  # NLL not recomputed
    }
    _write_task_metrics(
        MODELS_DIR,
        ARTIFACT,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
        calibration=calibration,
    )
    print(f"[{TASK}] Done. Metrics written.")


if __name__ == "__main__":
    main()
