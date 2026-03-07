"""Validate config/prepared/trained/runtime alignment for `packages/models`.

Examples:

    python scripts/models_artifact_probe.py
    python scripts/models_artifact_probe.py --runtime-probe
"""

from __future__ import annotations

import argparse
import json
import sys
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from cml.modeling.token_training import load_token_task_split


def _token_task_probe(prepared_dir: Path, task_name: str) -> dict[str, Any]:
    path = prepared_dir / f"{task_name}_train.parquet"
    if not path.exists():
        return {"exists": False}
    df = load_token_task_split(prepared_dir, task_name, "train")
    source_counts = Counter(df["source"].astype(str)) if "source" in df.columns else Counter()
    label_counts: Counter[str] = Counter()
    for spans in df["spans"].tolist():
        for span in spans or []:
            label_counts[str(span.get("label", ""))] += 1
    return {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
        "tasks": sorted(df["task"].astype(str).unique().tolist()) if "task" in df.columns else [],
        "span_label_counts": dict(label_counts),
        "top_sources": dict(source_counts.most_common(10)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_jsonable(v) for v in value]
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    return value


def _load_toml(path: Path) -> dict[str, Any]:
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _read_manifest(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _family_probe(prepared_dir: Path, family: str) -> dict[str, Any]:
    train_path = prepared_dir / f"{family}_train.parquet"
    if not train_path.exists():
        return {"exists": False}
    df = pd.read_parquet(train_path)
    source_counts = Counter(df["source"].astype(str)) if "source" in df.columns else Counter()
    llm_rows = sum(count for src, count in source_counts.items() if src.startswith("llm:"))
    hf_rows = sum(count for src, count in source_counts.items() if src.startswith("hf:"))
    return {
        "exists": True,
        "rows": len(df),
        "columns": list(df.columns),
        "tasks": sorted(df["task"].astype(str).unique().tolist()) if "task" in df.columns else [],
        "llm_rows": int(llm_rows),
        "hf_rows": int(hf_rows),
        "llm_pct": round(llm_rows / max(1, len(df)), 4),
        "hf_pct": round(hf_rows / max(1, len(df)), 4),
        "top_sources": dict(source_counts.most_common(10)),
    }


def _trained_probe(config_tasks: list[dict[str, Any]], trained_dir: Path) -> dict[str, Any]:
    manifest = _read_manifest(trained_dir / "manifest.json") or {}
    family_models = {
        family: (trained_dir / f"{family}_model.joblib").exists()
        for family in ("router", "extractor", "pair")
    }
    task_models = {
        task["task_name"]: (trained_dir / f"{task['artifact_name']}_model.joblib").exists()
        for task in config_tasks
    }
    return {
        "manifest_exists": (trained_dir / "manifest.json").exists(),
        "manifest_family_keys": sorted((manifest.get("families") or {}).keys()),
        "manifest_task_model_keys": sorted((manifest.get("task_models") or {}).keys()),
        "family_models": family_models,
        "task_models": task_models,
        "runtime_thresholds": manifest.get("runtime_thresholds"),
    }


def _collect_mismatches(
    *,
    config_tasks: list[dict[str, Any]],
    prepared: dict[str, dict[str, Any]],
    token_prepared: dict[str, dict[str, Any]],
    trained: dict[str, Any],
) -> list[str]:
    mismatches: list[str] = []

    observed_tasks: set[str] = set()
    for family, payload in prepared.items():
        if not payload.get("exists", False):
            mismatches.append(f"[prepared:{family}] missing <family>_train.parquet")
            continue
        observed_tasks.update(str(x) for x in payload.get("tasks", []))

    enabled_tasks = [t for t in config_tasks if bool(t.get("enabled", True))]
    enabled_task_names = [str(t.get("task_name", "")) for t in enabled_tasks]
    for raw in enabled_tasks:
        task_name = str(raw.get("task_name", ""))
        objective = str(raw.get("objective", ""))
        if objective == "token_classification":
            payload = token_prepared.get(task_name, {})
            if not payload.get("exists", False):
                mismatches.append(
                    f"[prepared] configured enabled token task missing split: {task_name}"
                )
                continue
            if task_name not in payload.get("tasks", []):
                mismatches.append(
                    f"[prepared] configured enabled token task missing from token splits: {task_name}"
                )
            continue
        if task_name and task_name not in observed_tasks:
            mismatches.append(
                f"[prepared] configured enabled task missing from train splits: {task_name}"
            )

    family_models = trained.get("family_models", {})
    for family in trained.get("manifest_family_keys", []):
        if not bool(family_models.get(family)):
            mismatches.append(
                f"[trained] manifest declares family '{family}' but model file is missing"
            )

    task_models = trained.get("task_models", {})
    for task_name in enabled_task_names:
        if task_name and not bool(task_models.get(task_name)):
            mismatches.append(f"[trained] enabled task model missing: {task_name}")

    return mismatches


def _runtime_probe() -> dict[str, Any]:
    from src.utils.modelpack import get_modelpack_runtime

    mp = get_modelpack_runtime()
    result: dict[str, Any] = {
        "available": bool(mp.available),
        "load_errors": list(getattr(mp, "_load_errors", [])),
    }
    checks: dict[str, Any] = {}
    single_tasks = [
        "memory_type",
        "query_intent",
        "constraint_dimension",
        "constraint_type",
        "constraint_scope",
        "importance_bin",
    ]
    for task in single_tasks:
        pred = mp.predict_single(task, "User prefers vegetarian food and avoids pork.")
        checks[task] = (
            None if pred is None else {"label": pred.label, "confidence": pred.confidence}
        )
    pair_tasks = [
        "scope_match",
        "constraint_rerank",
        "memory_rerank_pair",
        "retrieval_constraint_relevance_pair",
        "novelty_pair",
    ]
    for task in pair_tasks:
        score_pred = mp.predict_score_pair(
            task,
            "What should I cook tonight?",
            "User prefers vegetarian food and avoids pork.",
        )
        checks[task] = (
            None
            if score_pred is None
            else {"score": score_pred.score, "confidence": score_pred.confidence}
        )
    token_checks = {
        "pii_span_detection": "Contact me at alice@example.com and do not share the api_key sk-live-secret0001.",
        "fact_extraction_structured": "I live in Paris and prefer vegetarian food.",
    }
    for task, text in token_checks.items():
        span_pred = mp.predict_spans(task, text)
        checks[task] = (
            None
            if span_pred is None
            else {
                "spans": [list(span) for span in span_pred.spans],
                "confidence": span_pred.confidence,
            }
        )
    result["checks"] = checks
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Probe packages/models artifact alignment.")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("packages/models/model_pipeline.toml"),
    )
    parser.add_argument(
        "--prepared-dir",
        type=Path,
        default=Path("packages/models/prepared_data/modelpack"),
    )
    parser.add_argument(
        "--trained-dir",
        type=Path,
        default=Path("packages/models/trained_models"),
    )
    parser.add_argument("--runtime-probe", action="store_true")
    parser.add_argument(
        "--fail-on-mismatch",
        action="store_true",
        help="Exit with code 1 when config/prepared/trained mismatches are detected.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = _load_toml(args.config)
    config_tasks = cfg.get("tasks", [])

    prepared = {
        family: _family_probe(args.prepared_dir, family)
        for family in ("router", "extractor", "pair")
    }
    token_prepared = {
        task_name: _token_task_probe(args.prepared_dir, task_name)
        for task_name in (
            str(task.get("task_name", ""))
            for task in config_tasks
            if str(task.get("objective", "")) == "token_classification"
        )
    }
    trained = _trained_probe(config_tasks, args.trained_dir)
    mismatches = _collect_mismatches(
        config_tasks=config_tasks,
        prepared=prepared,
        token_prepared=token_prepared,
        trained=trained,
    )

    result = {
        "config_path": args.config,
        "configured_task_names": [task["task_name"] for task in config_tasks],
        "prepared": prepared,
        "token_prepared": token_prepared,
        "trained": trained,
        "mismatches": mismatches,
    }

    if args.runtime_probe:
        runtime = _runtime_probe()
        result["runtime"] = runtime
        token_checks = runtime.get("checks", {})
        for task_name in ("pii_span_detection", "fact_extraction_structured"):
            check = token_checks.get(task_name)
            spans = [] if not isinstance(check, dict) else list(check.get("spans") or [])
            if not spans:
                mismatches.append(f"[runtime] token task produced no spans: {task_name}")

    print(json.dumps(_jsonable(result), indent=2))
    if args.fail_on_mismatch and mismatches:
        for mismatch in mismatches:
            print(f"Mismatch: {mismatch}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
