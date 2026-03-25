"""Reconstruct memory_type model artifacts from existing HF checkpoint directories.

This script loads the pre-trained HF model directories for memory_type and generates:
  - memory_type_model.joblib
  - memory_type_metrics_test.json
  - memory_type_metrics_eval.json
  - memory_type_epoch_stats.json (placeholder with no epoch data)

Run from the repository root:
    python packages/models/scripts/reconstruct_memory_type.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Ensure src is on sys.path
REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import joblib  # noqa: E402
import pandas as pd  # noqa: E402

from cml.modeling.runtime_models import (  # noqa: E402
    HierarchicalTextClassifier,
    TransformerTextClassifier,
)
from cml.modeling.train import (  # noqa: E402
    _MEMORY_TYPE_MACRO_GROUPS,
    _evaluate,
    _load_split,
    _write_task_metrics,
    _write_json,
)

PREPARED_DIR = REPO_ROOT / "packages" / "models" / "prepared_data" / "modelpack"
OUTPUT_DIR = REPO_ROOT / "packages" / "models" / "trained_models"
FAMILY = "router"
TASK = "memory_type"
ARTIFACT = "memory_type"

# Max sequence length and batch size used during training (transformer defaults)
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 16


def _load_hf_classes(hf_dir: Path) -> list[str]:
    """Read ordered class list from a saved HF config.json."""
    config = json.loads((hf_dir / "config.json").read_text(encoding="utf-8"))
    id2label: dict[str, str] = config["id2label"]
    return [id2label[str(i)] for i in range(len(id2label))]


def main() -> None:
    # ── Load prepared splits ──────────────────────────────────────────────────
    print("[memory_type] Loading prepared router splits …")
    test_df = _load_split(PREPARED_DIR, FAMILY, "test")
    eval_df = _load_split(PREPARED_DIR, FAMILY, "eval")
    test_df = test_df[test_df["task"] == TASK].reset_index(drop=True)
    eval_df = eval_df[eval_df["task"] == TASK].reset_index(drop=True)
    print(f"  test rows: {len(test_df)}, eval rows: {len(eval_df)}")

    # ── Build stage1 (macro) model ────────────────────────────────────────────
    macro_dir = OUTPUT_DIR / "memory_type_macro_hf"
    macro_classes = _load_hf_classes(macro_dir)
    print(f"[memory_type] Macro classes: {macro_classes}")
    stage1_model = TransformerTextClassifier(
        task_name=f"{TASK}:macro",
        model_dir=str(macro_dir),
        classes_=macro_classes,
        max_seq_length=MAX_SEQ_LENGTH,
        batch_size=BATCH_SIZE,
        temperature=1.0,
    )

    # ── Build stage2 models ───────────────────────────────────────────────────
    stage2_models: dict[str, TransformerTextClassifier] = {}
    stage2_dirs: dict[str, str] = {}
    for macro_name in _MEMORY_TYPE_MACRO_GROUPS:
        hf_dir = OUTPUT_DIR / f"memory_type_{macro_name}_hf"
        if not hf_dir.exists():
            print(f"[memory_type] No HF dir for macro '{macro_name}', skipping.")
            continue
        fine_classes = _load_hf_classes(hf_dir)
        print(f"[memory_type] Stage2 '{macro_name}' classes: {fine_classes}")
        stage2_models[macro_name] = TransformerTextClassifier(
            task_name=f"{TASK}:{macro_name}",
            model_dir=str(hf_dir),
            classes_=fine_classes,
            max_seq_length=MAX_SEQ_LENGTH,
            batch_size=BATCH_SIZE,
            temperature=1.0,
        )
        stage2_dirs[macro_name] = str(hf_dir)

    # ── Task spec labels (composite) ──────────────────────────────────────────
    spec_labels = [
        "episodic_event", "semantic_fact", "preference", "constraint", "procedure",
        "hypothesis", "task_state", "conversation", "message", "tool_result",
        "reasoning_step", "scratch", "knowledge", "observation", "plan",
    ]
    composite_classes = [f"{TASK}::{label}" for label in spec_labels]

    # ── Assemble HierarchicalTextClassifier ───────────────────────────────────
    runtime_model = HierarchicalTextClassifier(
        task_name=TASK,
        stage1_model=stage1_model,
        stage2_models=stage2_models,
        macro_to_labels=_MEMORY_TYPE_MACRO_GROUPS,
        classes_=composite_classes,
    )

    # ── Evaluate ──────────────────────────────────────────────────────────────
    print("[memory_type] Evaluating on test split …")
    metrics_test, _ = _evaluate(
        runtime_model, test_df, family=FAMILY, split_name="test", predict_batch_size=BATCH_SIZE
    )
    print(f"  test accuracy: {metrics_test['overall']['accuracy']:.4f}")
    print(f"  test macro_f1: {metrics_test['overall']['macro_f1']:.4f}")

    print("[memory_type] Evaluating on eval split …")
    metrics_eval, _ = _evaluate(
        runtime_model, eval_df, family=FAMILY, split_name="eval", predict_batch_size=BATCH_SIZE
    )
    print(f"  eval accuracy: {metrics_eval['overall']['accuracy']:.4f}")
    print(f"  eval macro_f1: {metrics_eval['overall']['macro_f1']:.4f}")

    # ── Save joblib ───────────────────────────────────────────────────────────
    model_path = OUTPUT_DIR / f"{ARTIFACT}_model.joblib"
    joblib.dump(
        {
            "model": runtime_model,
            "task_spec": {
                "task_name": TASK,
                "family": FAMILY,
                "input_type": "single",
                "objective": "classification",
                "labels": spec_labels,
                "artifact_name": ARTIFACT,
                "metrics": ["accuracy", "macro_f1", "weighted_f1"],
                "trainer": "hierarchical_transformer",
                "backbone_model_name": "microsoft/deberta-v3-base",
            },
            "hf_model_dir": str(macro_dir),
            "model_kind": "hierarchical_transformer",
            "stage_model_dirs": stage2_dirs,
        },
        model_path,
    )
    print(f"[memory_type] Saved model to {model_path}")

    # ── Write epoch_stats placeholder ─────────────────────────────────────────
    _write_json(
        OUTPUT_DIR / f"{ARTIFACT}_epoch_stats.json",
        {
            "task": TASK,
            "note": "Reconstructed from existing HF checkpoints; no epoch-level stats available.",
            "stage_model_dirs": stage2_dirs,
        },
    )

    # ── Write metrics files ───────────────────────────────────────────────────
    _write_task_metrics(
        OUTPUT_DIR,
        ARTIFACT,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
        calibration=None,
    )
    print("[memory_type] Done. Metrics written to trained_models/.")


if __name__ == "__main__":
    main()
