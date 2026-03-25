"""Regenerate manifest.json release_gates from current trained model metrics.

Run this after any targeted retrains or post-hoc recalibrations to update
the manifest without re-running full training.

Usage:
    python packages/models/scripts/regenerate_manifest.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
PY_CML_SRC = REPO_ROOT / "packages" / "py-cml" / "src"

TRAINED_DIR = REPO_ROOT / "packages" / "models" / "trained_models"
MANIFEST_PATH = TRAINED_DIR / "manifest.json"

# All tasks with release gates (from _RELEASE_GATES in train.py)
GATED_TASKS = [
    "schema_match_pair",
    "memory_type",
    "novelty_pair",
    "confidence_bin",
    "decay_profile",
    "pii_span_detection",
    "forgetting_action_policy",
    "constraint_dimension",
    "context_tag",
    "retrieval_constraint_relevance_pair",
    "memory_rerank_pair",
    "reconsolidation_candidate_pair",
    "write_importance_regression",
]


def _load_release_gate_results():
    if str(PY_CML_SRC) not in sys.path:
        sys.path.insert(0, str(PY_CML_SRC))
    from cml.modeling.train import _release_gate_results

    return _release_gate_results


def load_task_metrics(task: str) -> dict | None:
    """Load combined test metrics for a task, merged with special fields."""
    metrics_path = TRAINED_DIR / f"{task}_metrics_test.json"
    if not metrics_path.exists():
        return None
    with open(metrics_path) as f:
        data = json.load(f)
    overall = data.get("overall", {})
    # Flatten to summary dict matching _release_gate_results expected format
    summary: dict = dict(overall)
    # For write_importance_regression, metrics are nested under 'metrics'
    if "metrics" in data:
        summary.update(data["metrics"])
    # Include classification_report for per-label recall gates
    if "classification_report" in overall:
        summary["classification_report"] = overall["classification_report"]
    # For memory_type: plan_f1 from classification_report
    report = overall.get("classification_report", {})
    for label, label_metrics in report.items():
        if "::" in label:
            short = label.split("::", 1)[1]
            summary[f"{short}_recall"] = label_metrics.get("recall")
            summary[f"{short}_f1"] = label_metrics.get("f1-score")
    return summary


def main() -> None:
    if not MANIFEST_PATH.exists():
        print(f"ERROR: manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    release_gate_results = _load_release_gate_results()

    with open(MANIFEST_PATH) as f:
        manifest = json.load(f)

    new_gates: dict[str, dict] = {}
    all_passed = True

    print("=== Release gate check ===")
    for task in GATED_TASKS:
        summary = load_task_metrics(task)
        if summary is None:
            print(f"  MISSING  {task} (no metrics file)")
            all_passed = False
            continue

        gate_result = release_gate_results(task, {"test": summary})
        new_gates[task] = gate_result

        def fmt_val(v):
            return f"{v:.4f}" if isinstance(v, float) else str(v)
        checks_str = ", ".join(
            f'{c["metric"]}={fmt_val(c["actual"])}({"✓" if c["passed"] else "✗"})'
            for c in gate_result["checks"]
        )
        status = "PASS" if gate_result["passed"] else "FAIL"
        print(f"  {status:<6} {task}: {checks_str}")
        if not gate_result["passed"]:
            all_passed = False

    manifest["release_gates"] = new_gates
    with open(MANIFEST_PATH, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\nManifest updated: {MANIFEST_PATH}")
    print(f"Overall: {'ALL GATES PASS ✓' if all_passed else 'SOME GATES FAIL ✗'}")


if __name__ == "__main__":
    main()
