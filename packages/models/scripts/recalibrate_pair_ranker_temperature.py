"""Post-hoc temperature recalibration for pair_ranker models.

The NLL-optimal temperature (T=2.0) does not minimize ECE for these models.
This script sweeps temperatures on the eval set to find ECE-optimal T,
updates the joblib model, and regenerates metrics JSON files.

Usage:
    python packages/models/scripts/recalibrate_pair_ranker_temperature.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_CML_SRC = REPO_ROOT / "packages" / "py-cml" / "src"
MODELS_DIR = REPO_ROOT / "packages" / "models"
PREPARED_DIR = MODELS_DIR / "prepared_data" / "modelpack"
TRAINED_DIR = MODELS_DIR / "trained_models"

TASKS = [
    "retrieval_constraint_relevance_pair",
    "reconsolidation_candidate_pair",
]

TEMPERATURE_GRID = [
    0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0
]


def _load_sequence_inputs_from_features():
    if str(PY_CML_SRC) not in sys.path:
        sys.path.insert(0, str(PY_CML_SRC))
    from cml.modeling.train import _sequence_inputs_from_features

    return _sequence_inputs_from_features


def compute_ece(p_pos: np.ndarray, binary_labels: np.ndarray, n_bins: int = 10) -> float:
    confidences = np.maximum(p_pos, 1 - p_pos)
    preds = (p_pos > 0.5).astype(float)
    accuracies = (preds == binary_labels).astype(float)
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if not mask.any():
            continue
        ece += mask.sum() * abs(accuracies[mask].mean() - confidences[mask].mean())
    return float(ece / max(1, len(binary_labels)))


def get_raw_logits(
    model_dir: str,
    task_df: pd.DataFrame,
    task_name: str,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_dir, use_fast=True)
    model = AutoModelForSequenceClassification.from_pretrained(model_dir, num_labels=1)
    model.eval()
    device = torch.device("cpu")
    model.to(device)

    features = [
        f"task={row.task} [a] {row.text_a} [b] {row.text_b}"
        for row in task_df.itertuples()
    ]
    sequence_inputs_from_features = _load_sequence_inputs_from_features()
    left, right = sequence_inputs_from_features(features, input_type="pair")
    assert right is not None

    outputs = []
    batch_size = 32
    for start in range(0, len(left), batch_size):
        bl = left[start : start + batch_size]
        br = right[start : start + batch_size]
        enc = tokenizer(
            bl, br, truncation=True, max_length=256, padding=True, return_tensors="pt"
        )
        enc = {k: v.to(device) for k, v in enc.items()}
        with torch.no_grad():
            logits = model(**enc).logits
        outputs.append(logits.detach().cpu().numpy())
    return np.vstack(outputs).reshape(-1)


def find_optimal_temperature(
    raw_logits: np.ndarray,
    binary_labels: np.ndarray,
    grid: list[float],
) -> tuple[float, float]:
    best_ece = float("inf")
    best_t = 1.0
    for temperature in grid:
        p_pos = 1.0 / (1.0 + np.exp(-raw_logits / temperature))
        ece = compute_ece(p_pos, binary_labels)
        if ece < best_ece:
            best_ece = ece
            best_t = temperature
    return best_t, best_ece


def update_metrics_file(path: Path, new_ece: float, new_temp: float) -> None:
    with open(path) as f:
        data = json.load(f)
    data["overall"]["calibration_error"] = new_ece
    data["calibration"]["temperature"] = new_temp
    data["calibration"]["method"] = "ece_grid_search"
    with open(path, "w") as f:
        json.dump(data, f, indent=2)
    print(f"  Updated {path.name}: ECE={new_ece:.4f} T={new_temp}")


def main() -> None:
    eval_df_all = pd.read_parquet(PREPARED_DIR / "pair_eval.parquet")
    test_df_all = pd.read_parquet(PREPARED_DIR / "pair_test.parquet")

    for task in TASKS:
        print(f"\n=== {task} ===")
        model_dir = str(TRAINED_DIR / f"{task}_hf")
        joblib_path = TRAINED_DIR / f"{task}_model.joblib"

        eval_df = eval_df_all[eval_df_all["task"] == task].copy()
        test_df = test_df_all[test_df_all["task"] == task].copy()

        eval_binary = np.array(
            [1.0 if lbl == "relevant" else 0.0 for lbl in eval_df["label"]]
        )
        test_binary = np.array(
            [1.0 if lbl == "relevant" else 0.0 for lbl in test_df["label"]]
        )

        print("  Computing eval logits...")
        eval_logits = get_raw_logits(model_dir, eval_df, task)
        print("  Computing test logits...")
        test_logits = get_raw_logits(model_dir, test_df, task)

        # Find ECE-optimal temperature on eval set
        best_t, best_eval_ece = find_optimal_temperature(
            eval_logits, eval_binary, TEMPERATURE_GRID
        )
        print(f"  ECE-optimal T={best_t:.2f}, eval ECE={best_eval_ece:.4f}")

        # Compute test ECE at optimal temperature
        p_pos_test = 1.0 / (1.0 + np.exp(-test_logits / best_t))
        test_ece = compute_ece(p_pos_test, test_binary)
        print(f"  Test ECE at T={best_t:.2f}: {test_ece:.4f} (gate: <=0.08)")

        if test_ece > 0.08:
            print(f"  WARNING: test ECE {test_ece:.4f} still above gate. Keeping original.")
            continue

        # Update joblib model temperature
        payload = joblib.load(joblib_path)
        runtime_model = payload["model"] if isinstance(payload, dict) else payload
        old_temp = runtime_model.temperature
        runtime_model.temperature = float(best_t)
        joblib.dump(payload, joblib_path)
        print(f"  Updated joblib: temperature {old_temp} -> {best_t}")

        # Update metrics JSON files
        update_metrics_file(
            TRAINED_DIR / f"{task}_metrics_eval.json", best_eval_ece, best_t
        )
        update_metrics_file(
            TRAINED_DIR / f"{task}_metrics_test.json", test_ece, best_t
        )


if __name__ == "__main__":
    main()
