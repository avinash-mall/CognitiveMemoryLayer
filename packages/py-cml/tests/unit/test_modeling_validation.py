"""Unit tests for modeling preflight validation and artifact checks."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from cml.modeling.validation import (  # noqa: E402
    run_preflight_validation,
    validate_manifest_artifacts,
)


def _write_minimal_family_splits(prepared_dir: Path) -> None:
    prepared_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{"text": "query text", "task": "query_intent", "label": "factual"}]).to_parquet(
        prepared_dir / "router_train.parquet", index=False
    )
    pd.DataFrame(
        [{"text": "constraint text", "task": "constraint_type", "label": "policy"}]
    ).to_parquet(prepared_dir / "extractor_train.parquet", index=False)
    pd.DataFrame(
        [
            {
                "text_a": "a",
                "text_b": "b",
                "task": "constraint_rerank",
                "label": "relevant",
            }
        ]
    ).to_parquet(prepared_dir / "pair_train.parquet", index=False)


def test_preflight_rejects_deferred_token_objective(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "pii_span_detection",
                "family": "extractor",
                "input_type": "single",
                "objective": "token_classification",
                "enabled": True,
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is False
    assert any("not supported yet" in err for err in result.errors)


def test_preflight_requires_score_for_single_regression(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "write_importance_regression",
                "family": "router",
                "input_type": "single",
                "objective": "single_regression",
                "enabled": True,
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is False
    assert any("missing columns" in err for err in result.errors)


def test_preflight_allows_disabled_tasks(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "pii_span_detection",
                "family": "extractor",
                "input_type": "single",
                "objective": "token_classification",
                "enabled": False,
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is True
    assert result.errors == []
    assert result.task_checks[0].status == "disabled"


def test_validate_manifest_artifacts_reports_missing(tmp_path: Path) -> None:
    model_path = tmp_path / "router_model.joblib"
    model_path.write_bytes(b"ok")

    errors = validate_manifest_artifacts(
        families_summary={
            "router": {"model_path": str(model_path)},
            "pair": {"model_path": str(tmp_path / "pair_model.joblib")},
        },
        task_summaries={
            "novelty_pair": {"model_path": str(tmp_path / "novelty_pair_model.joblib")}
        },
    )

    assert any("family:pair" in err for err in errors)
    assert any("task:novelty_pair" in err for err in errors)
