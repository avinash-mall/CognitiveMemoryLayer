"""Train only the personal stage2 model for memory_type, then patch the existing model.joblib.

This script:
1. Loads existing memory_type_model.joblib (has macro + 4 stage2 models, missing personal)
2. Trains a new DeBERTa binary classifier for preference/episodic_event
3. Adds the personal stage2 model to the HierarchicalTextClassifier
4. Re-evaluates on test + eval splits
5. Saves updated model.joblib + metrics

Run from repo root:
    python packages/models/scripts/train_memory_type_personal_stage2.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import joblib
import pandas as pd

import cml.modeling.train as train_module
from cml.modeling.train import (
    _encode_memory_type_features,
    _load_task_splits,
    _evaluate,
    _train_transformer_sequence_classifier,
    _write_task_metrics,
    _transformer_cfg,
    _MEMORY_TYPE_MACRO_GROUPS,
)
from cml.modeling.runtime_models import HierarchicalTextClassifier

MODELS_DIR = REPO_ROOT / "packages" / "models"
PREPARED_DIR = MODELS_DIR / "prepared_data" / "modelpack"
OUTPUT_DIR = MODELS_DIR / "trained_models"


def main() -> None:
    # Load pipeline config
    import tomllib

    with open(MODELS_DIR / "model_pipeline.toml", "rb") as f:
        pipeline_cfg = tomllib.load(f)

    train_cfg = {k: v for k, v in pipeline_cfg.items() if not isinstance(v, (dict, list))}

    # ── Load existing model ──────────────────────────────────────────────────
    model_path = OUTPUT_DIR / "memory_type_model.joblib"
    print(f"[personal-stage2] Loading existing model from {model_path}")
    artifact = joblib.load(model_path)
    existing_model: HierarchicalTextClassifier = artifact["model"]
    print(f"[personal-stage2] Existing stage2 keys: {list(existing_model.stage2_models.keys())}")

    if "personal" in existing_model.stage2_models:
        print("[personal-stage2] Personal stage2 model already exists! Exiting.")
        return

    # ── Load training data ───────────────────────────────────────────────────
    print("[personal-stage2] Loading training splits...")
    train_df, test_df, eval_df = _load_task_splits(PREPARED_DIR, "router", "memory_type")

    personal_labels = _MEMORY_TYPE_MACRO_GROUPS["personal"]  # ['preference', 'episodic_event']
    sub_train = train_df[train_df["label"].astype(str).isin(personal_labels)].reset_index(drop=True)
    sub_eval = eval_df[eval_df["label"].astype(str).isin(personal_labels)].reset_index(drop=True)
    print(f"[personal-stage2] personal train={len(sub_train)}, eval={len(sub_eval)}")
    print(f"[personal-stage2] train labels: {sub_train['label'].value_counts().to_dict()}")

    # ── Encode features ──────────────────────────────────────────────────────
    sub_train_x = _encode_memory_type_features(sub_train, "router")
    sub_eval_x = _encode_memory_type_features(sub_eval, "router")

    backbone = "microsoft/deberta-v3-base"
    tokenizer_name = ""

    # ── Train personal stage2 ────────────────────────────────────────────────
    print("[personal-stage2] Training personal stage2 (preference/episodic_event)...")
    personal_model, personal_dir, epoch_stats, summary, calibration = (
        _train_transformer_sequence_classifier(
            features=sub_train_x,
            targets=sub_train["label"].astype(str).tolist(),
            eval_features=sub_eval_x,
            eval_targets=sub_eval["label"].astype(str).tolist(),
            input_type="single",
            classes=list(personal_labels),
            task_name="memory_type:personal",
            output_dir=OUTPUT_DIR,
            artifact_stem="memory_type_personal",
            backbone_model_name=backbone,
            tokenizer_name=tokenizer_name,
            train_cfg=train_cfg,
            runtime_kind="text",
        )
    )
    print(f"[personal-stage2] Personal stage2 trained: {personal_dir}")

    # ── Reconstruct HierarchicalTextClassifier with personal model ──────────
    new_stage2_models = dict(existing_model.stage2_models)
    new_stage2_models["personal"] = personal_model

    new_model = HierarchicalTextClassifier(
        task_name=existing_model.task_name,
        stage1_model=existing_model.stage1_model,
        stage2_models=new_stage2_models,
        macro_to_labels=existing_model.macro_to_labels,
        classes_=existing_model.classes_,
    )

    # ── Evaluate ─────────────────────────────────────────────────────────────
    batch_size = max(1, int(_transformer_cfg(train_cfg)["per_device_eval_batch_size"]))
    print("[personal-stage2] Evaluating on test set...")
    metrics_test, _ = _evaluate(
        new_model, test_df, family="router", split_name="test", predict_batch_size=batch_size
    )
    print(f"[personal-stage2] test macro_f1: {metrics_test['overall']['macro_f1']:.4f}")
    print("[personal-stage2] Evaluating on eval set...")
    metrics_eval, _ = _evaluate(
        new_model, eval_df, family="router", split_name="eval", predict_batch_size=batch_size
    )

    # Print per-class results
    cr = metrics_test["overall"].get("classification_report", {})
    for label in ["preference", "episodic_event", "plan", "hypothesis"]:
        full_label = f"memory_type::{label}"
        if full_label in cr and isinstance(cr[full_label], dict):
            v = cr[full_label]
            print(f"  {label}: f1={v.get('f1-score', 0):.3f}, recall={v.get('recall', 0):.3f}")

    # ── Save updated model ───────────────────────────────────────────────────
    print(f"[personal-stage2] Saving updated model to {model_path}")
    joblib.dump(
        {
            "model": new_model,
            "task_spec": artifact.get("task_spec", {}),
            "hf_model_dir": artifact.get("hf_model_dir", ""),
            "model_kind": "hierarchical_transformer",
            "stage_model_dirs": {
                **artifact.get("stage_model_dirs", {}),
                "personal": personal_dir,
            },
        },
        model_path,
    )
    print("[personal-stage2] Saved model.joblib.")

    _write_task_metrics(
        OUTPUT_DIR,
        "memory_type",
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
    )
    print("[personal-stage2] Done.")


if __name__ == "__main__":
    main()
