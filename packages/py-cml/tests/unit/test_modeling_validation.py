"""Unit tests for modeling preflight validation and artifact checks."""

from __future__ import annotations

from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from cml.modeling.token_training import load_token_task_split  # noqa: E402
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


def _write_token_split(prepared_dir: Path, task_name: str) -> None:
    pd.DataFrame(
        [
            {
                "text": "Contact me at alice@example.com",
                "task": task_name,
                "spans": [{"start": 14, "end": 31, "label": "EMAIL"}],
                "source": "unit",
                "language": "en",
            }
        ]
    ).to_parquet(prepared_dir / f"{task_name}_train.parquet", index=False)


def test_preflight_accepts_enabled_token_objective_with_dedicated_split(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)
    _write_token_split(prepared_dir, "pii_span_detection")

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

    assert result.ok is True
    assert result.errors == []
    assert result.task_checks[0].rows_found == 1


def test_load_token_task_split_preserves_spans_after_parquet_roundtrip(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    _write_token_split(prepared_dir, "pii_span_detection")

    df = load_token_task_split(prepared_dir, "pii_span_detection", "train")

    assert len(df) == 1
    assert df.iloc[0]["spans"] == [{"start": 14, "end": 31, "label": "EMAIL"}]


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


def test_preflight_rejects_missing_token_split(tmp_path: Path) -> None:
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
    assert any("Missing prepared token split" in err for err in result.errors)


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


def test_preflight_rejects_embedding_pair_without_cache(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "memory_rerank_pair",
                "family": "pair",
                "input_type": "pair",
                "objective": "pair_ranking",
                "enabled": True,
                "trainer": "embedding_pair",
                "feature_backend": "embedding_pair",
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is False
    assert any("missing embedding cache" in err.lower() for err in result.errors)


def test_preflight_accepts_embedding_pair_with_cache(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)
    pd.DataFrame(
        [
            {
                "text_a": "query",
                "text_b": "memory",
                "task": "memory_rerank_pair",
                "label": "relevant",
            }
        ]
    ).to_parquet(prepared_dir / "pair_train.parquet", index=False)
    pd.DataFrame(
        [
            {
                "text_hash": "abc123",
                "text": "a",
                "embedding": [0.1, 0.2, 0.3],
                "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            }
        ]
    ).to_parquet(prepared_dir / "pair_text_embeddings.parquet", index=False)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "memory_rerank_pair",
                "family": "pair",
                "input_type": "pair",
                "objective": "pair_ranking",
                "enabled": True,
                "trainer": "embedding_pair",
                "feature_backend": "embedding_pair",
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is True
    assert result.errors == []


def test_preflight_accepts_embedding_pair_classification_with_cache(tmp_path: Path) -> None:
    """embedding_pair trainer with objective=classification (e.g. schema_match_pair) is accepted when cache exists."""
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)
    pd.DataFrame(
        [
            {"text_a": "gist A", "text_b": "fact key A", "task": "schema_match_pair", "label": "match"},
            {"text_a": "gist B", "text_b": "fact key B", "task": "schema_match_pair", "label": "no_match"},
        ]
    ).to_parquet(prepared_dir / "pair_train.parquet", index=False)
    pd.DataFrame(
        [
            {"text_hash": "h1", "text": "gist A", "embedding": [0.1] * 64, "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"},
            {"text_hash": "h2", "text": "fact key A", "embedding": [0.2] * 64, "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2"},
        ]
    ).to_parquet(prepared_dir / "pair_text_embeddings.parquet", index=False)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "schema_match_pair",
                "family": "pair",
                "input_type": "pair",
                "objective": "classification",
                "labels": ["match", "no_match"],
                "enabled": True,
                "trainer": "embedding_pair",
                "feature_backend": "embedding_pair",
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is True
    assert result.errors == []


def test_preflight_rejects_ordinal_threshold_with_mismatched_label_order(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    _write_minimal_family_splits(prepared_dir)

    result = run_preflight_validation(
        task_specs_raw=[
            {
                "task_name": "importance_bin",
                "family": "router",
                "input_type": "single",
                "objective": "classification",
                "enabled": True,
                "trainer": "ordinal_threshold",
                "labels": ["low", "medium", "high"],
                "label_order": ["low", "high"],
            }
        ],
        prepared_dir=prepared_dir,
        strict=True,
    )

    assert result.ok is False
    assert any("label_order" in err for err in result.errors)
