"""Preflight and artifact validation for modeling train workflows."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import pandas as pd

_KNOWN_OBJECTIVES = {
    "classification",
    "pair_ranking",
    "single_regression",
    "token_classification",
}
_DEFERRED_OBJECTIVES = {"token_classification"}
_FAMILIES = {"router", "extractor", "pair"}


@dataclass(slots=True)
class TaskPreflightStatus:
    task_name: str
    family: str
    input_type: str
    objective: str
    enabled: bool
    status: str
    reason: str | None = None
    rows_found: int = 0
    valid_score_rows: int = 0


@dataclass(slots=True)
class PreflightValidationResult:
    ok: bool
    strict: bool
    errors: list[str]
    warnings: list[str]
    task_checks: list[TaskPreflightStatus]
    observed_tasks_by_family: dict[str, list[str]]
    coverage_vs_config: dict[str, dict[str, list[str]]]

    def as_dict(self) -> dict[str, Any]:
        return {
            "ok": self.ok,
            "strict": self.strict,
            "errors": self.errors,
            "warnings": self.warnings,
            "task_checks": [asdict(t) for t in self.task_checks],
            "observed_tasks_by_family": self.observed_tasks_by_family,
            "coverage_vs_config": self.coverage_vs_config,
        }


def _required_columns(*, input_type: str, objective: str) -> list[str]:
    cols = ["task", "text_a", "text_b"] if input_type == "pair" else ["task", "text"]
    if objective == "single_regression":
        cols.append("score")
    else:
        cols.append("label")
    return cols


def _load_family_train_df(
    prepared_dir: Path, family: str
) -> tuple[pd.DataFrame | None, str | None]:
    path = prepared_dir / f"{family}_train.parquet"
    if not path.exists():
        return None, f"Missing prepared split: {path}"
    try:
        return pd.read_parquet(path), None
    except Exception as exc:  # pragma: no cover - depends on parquet engine/data files
        return None, f"Failed to read {path}: {exc}"


def run_preflight_validation(
    *,
    task_specs_raw: list[dict[str, Any]],
    prepared_dir: Path,
    strict: bool,
) -> PreflightValidationResult:
    errors: list[str] = []
    warnings: list[str] = []
    task_checks: list[TaskPreflightStatus] = []

    family_frames: dict[str, pd.DataFrame | None] = {}
    for family in _FAMILIES:
        df, err = _load_family_train_df(prepared_dir, family)
        family_frames[family] = df
        if err:
            errors.append(err)

    observed_by_family: dict[str, list[str]] = {}
    for family, df in family_frames.items():
        if df is None or "task" not in df.columns:
            observed_by_family[family] = []
            continue
        observed_by_family[family] = sorted(df["task"].dropna().astype(str).unique().tolist())

    enabled_tasks_by_family: dict[str, set[str]] = {f: set() for f in _FAMILIES}

    for raw in task_specs_raw:
        task_name = str(raw.get("task_name", "")).strip()
        family = str(raw.get("family", "")).strip()
        input_type = str(raw.get("input_type", "")).strip()
        objective = str(raw.get("objective", "")).strip()
        enabled = bool(raw.get("enabled", True))

        if family in enabled_tasks_by_family and enabled and task_name:
            enabled_tasks_by_family[family].add(task_name)

        if not task_name:
            errors.append(f"Configured task is missing task_name: {raw}")
            continue

        status = TaskPreflightStatus(
            task_name=task_name,
            family=family,
            input_type=input_type,
            objective=objective,
            enabled=enabled,
            status="ok",
        )

        if not enabled:
            status.status = "disabled"
            status.reason = "Task disabled in config"
            task_checks.append(status)
            continue

        if family not in _FAMILIES:
            status.status = "error"
            status.reason = f"Unknown family '{family}'"
            errors.append(f"[task:{task_name}] unknown family '{family}'")
            task_checks.append(status)
            continue

        if objective not in _KNOWN_OBJECTIVES:
            status.status = "error"
            status.reason = f"Unknown objective '{objective}'"
            errors.append(f"[task:{task_name}] unknown objective '{objective}'")
            task_checks.append(status)
            continue

        if objective in _DEFERRED_OBJECTIVES:
            status.status = "error"
            status.reason = f"Objective '{objective}' is deferred; keep task disabled until trainer support lands"
            errors.append(f"[task:{task_name}] objective '{objective}' is not supported yet")
            task_checks.append(status)
            continue

        if input_type not in {"single", "pair"}:
            status.status = "error"
            status.reason = f"Invalid input_type '{input_type}'"
            errors.append(f"[task:{task_name}] invalid input_type '{input_type}'")
            task_checks.append(status)
            continue

        df = family_frames.get(family)
        if df is None:
            status.status = "error"
            status.reason = f"Missing training split for family '{family}'"
            errors.append(f"[task:{task_name}] missing family split for '{family}'")
            task_checks.append(status)
            continue

        required = _required_columns(input_type=input_type, objective=objective)
        missing_cols = [col for col in required if col not in df.columns]
        if missing_cols:
            status.status = "error"
            status.reason = f"Missing required columns in {family}_train.parquet: {missing_cols}"
            errors.append(
                f"[task:{task_name}] missing columns in {family}_train.parquet: {missing_cols}"
            )
            task_checks.append(status)
            continue

        subset = df[df["task"].astype(str) == task_name]
        status.rows_found = len(subset)
        if subset.empty:
            status.status = "error"
            status.reason = "No rows found for configured task in prepared train split"
            errors.append(f"[task:{task_name}] no prepared rows found in {family}_train.parquet")
            task_checks.append(status)
            continue

        if objective == "single_regression":
            numeric = pd.to_numeric(subset["score"], errors="coerce")
            valid_rows = int(numeric.notna().sum())
            status.valid_score_rows = valid_rows
            if valid_rows == 0:
                status.status = "error"
                status.reason = "single_regression requires numeric score rows"
                errors.append(
                    f"[task:{task_name}] single_regression requires numeric score values in score column"
                )
                task_checks.append(status)
                continue

        task_checks.append(status)

    coverage: dict[str, dict[str, list[str]]] = {}
    for family in sorted(_FAMILIES):
        configured = sorted(enabled_tasks_by_family[family])
        observed = observed_by_family.get(family, [])
        missing = sorted(set(configured) - set(observed))
        coverage[family] = {
            "configured_enabled_tasks": configured,
            "observed_tasks": observed,
            "missing_configured_tasks": missing,
        }
        if missing:
            errors.append(
                f"[family:{family}] configured tasks missing from prepared data: {missing}"
            )

    if strict and errors:
        warnings.append("Strict mode enabled: preflight errors will fail training.")

    return PreflightValidationResult(
        ok=(len(errors) == 0),
        strict=strict,
        errors=errors,
        warnings=warnings,
        task_checks=task_checks,
        observed_tasks_by_family=observed_by_family,
        coverage_vs_config=coverage,
    )


def validate_manifest_artifacts(
    *,
    families_summary: dict[str, dict[str, Any]],
    task_summaries: dict[str, dict[str, Any]],
) -> list[str]:
    errors: list[str] = []

    for family, summary in families_summary.items():
        model_path = Path(str(summary.get("model_path", ""))).expanduser()
        if not model_path.exists():
            errors.append(f"[family:{family}] missing model artifact: {model_path}")

    for task_name, summary in task_summaries.items():
        model_path = Path(str(summary.get("model_path", ""))).expanduser()
        if not model_path.exists():
            errors.append(f"[task:{task_name}] missing model artifact: {model_path}")

    return errors
