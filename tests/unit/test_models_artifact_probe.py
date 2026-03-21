"""Tests for scripts/models_artifact_probe.py mismatch gating."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

from scripts.models_artifact_probe import _collect_runtime_mismatches

REPO_ROOT = Path(__file__).resolve().parents[2]

pytest.importorskip("pandas", reason="models_artifact_probe requires pandas")
pytest.importorskip("pyarrow", reason="models_artifact_probe parquet probe requires pyarrow")


def _write_min_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[[tasks]]",
                'task_name = "novelty_pair"',
                'family = "pair"',
                'input_type = "pair"',
                'objective = "classification"',
                "enabled = true",
                'artifact_name = "novelty_pair"',
                'metrics = ["accuracy"]',
            ]
        ),
        encoding="utf-8",
    )


def _write_fact_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[[tasks]]",
                'task_name = "fact_extraction_structured"',
                'family = "extractor"',
                'input_type = "single"',
                'objective = "token_classification"',
                "enabled = true",
                'artifact_name = "fact_extraction_structured"',
                'metrics = ["span_f1"]',
            ]
        ),
        encoding="utf-8",
    )


def _write_embedding_pair_config(path: Path) -> None:
    path.write_text(
        "\n".join(
            [
                "[[tasks]]",
                'task_name = "memory_rerank_pair"',
                'family = "pair"',
                'input_type = "pair"',
                'objective = "pair_ranking"',
                "enabled = true",
                'artifact_name = "memory_rerank_pair"',
                'metrics = ["mrr@10"]',
                'trainer = "embedding_pair"',
                'feature_backend = "embedding_pair"',
            ]
        ),
        encoding="utf-8",
    )


def test_models_artifact_probe_fail_on_mismatch(tmp_path: Path) -> None:
    config = tmp_path / "model_pipeline.toml"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"
    prepared.mkdir(parents=True, exist_ok=True)
    trained.mkdir(parents=True, exist_ok=True)
    _write_min_config(config)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/models_artifact_probe.py",
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--trained-dir",
            str(trained),
            "--fail-on-mismatch",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "Mismatch:" in result.stderr


def test_models_artifact_probe_no_fail_flag_returns_zero(tmp_path: Path) -> None:
    config = tmp_path / "model_pipeline.toml"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"
    prepared.mkdir(parents=True, exist_ok=True)
    trained.mkdir(parents=True, exist_ok=True)
    _write_min_config(config)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/models_artifact_probe.py",
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--trained-dir",
            str(trained),
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 0
    assert '"mismatches"' in result.stdout


def test_models_artifact_probe_flags_english_only_fact_token_rows(tmp_path: Path) -> None:
    config = tmp_path / "model_pipeline.toml"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"
    prepared.mkdir(parents=True, exist_ok=True)
    trained.mkdir(parents=True, exist_ok=True)
    _write_fact_config(config)

    token_df = pd.DataFrame(
        [
            {
                "text": "I live in Paris.",
                "task": "fact_extraction_structured",
                "spans": [{"start": 10, "end": 15, "label": "location"}],
                "source": "template:fact_extraction_structured:location:en",
                "language": "en",
            }
        ]
    )
    token_df.to_parquet(prepared / "fact_extraction_structured_train.parquet", index=False)
    token_df.to_parquet(prepared / "fact_extraction_structured_test.parquet", index=False)
    token_df.to_parquet(prepared / "fact_extraction_structured_eval.parquet", index=False)

    result = subprocess.run(
        [
            sys.executable,
            "scripts/models_artifact_probe.py",
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--trained-dir",
            str(trained),
            "--fail-on-mismatch",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "fact_extraction_structured token split has no non-English rows" in result.stderr


def test_models_artifact_probe_flags_missing_pair_embedding_cache(tmp_path: Path) -> None:
    config = tmp_path / "model_pipeline.toml"
    prepared = tmp_path / "prepared"
    trained = tmp_path / "trained"
    prepared.mkdir(parents=True, exist_ok=True)
    trained.mkdir(parents=True, exist_ok=True)
    _write_embedding_pair_config(config)

    pd.DataFrame(
        [
            {
                "text_a": "query",
                "text_b": "memory",
                "task": "memory_rerank_pair",
                "label": "relevant",
                "source": "unit",
            }
        ]
    ).to_parquet(prepared / "pair_train.parquet", index=False)
    pd.DataFrame([{"text": "x", "task": "query_intent", "label": "factual"}]).to_parquet(
        prepared / "router_train.parquet", index=False
    )
    pd.DataFrame([{"text": "x", "task": "constraint_type", "label": "policy"}]).to_parquet(
        prepared / "extractor_train.parquet", index=False
    )

    result = subprocess.run(
        [
            sys.executable,
            "scripts/models_artifact_probe.py",
            "--config",
            str(config),
            "--prepared-dir",
            str(prepared),
            "--trained-dir",
            str(trained),
            "--fail-on-mismatch",
        ],
        cwd=str(REPO_ROOT),
        text=True,
        capture_output=True,
    )

    assert result.returncode == 1
    assert "pair embedding cache missing" in result.stderr


def test_collect_runtime_mismatches_flags_dependency_drift_and_bad_sanity_checks() -> None:
    runtime = {
        "load_errors": [
            "dependency version mismatch: scikit-learn trained_with=trained-version installed=installed-version"
        ],
        "checks": {
            "pii_span_detection": {"spans": [[0, 5, "email"]], "confidence": 0.9},
            "fact_extraction_structured": {"spans": [[0, 5, "location"]], "confidence": 0.9},
            "fact_extraction_structured_multilingual": {
                "spans": [[0, 5, "location"]],
                "confidence": 0.9,
            },
            "pii_presence_positive": {"label": "no_pii", "confidence": 0.9},
            "pii_presence_negative": {"label": "pii", "confidence": 0.99},
            "constraint_type_positive": {"label": "value", "confidence": 0.8},
            "constraint_type_negative": {"label": "goal", "confidence": 0.8},
        },
    }

    mismatches = _collect_runtime_mismatches(runtime)

    assert any("dependency version mismatch" in mismatch for mismatch in mismatches)
    assert any("positive sample was not classified as PII" in mismatch for mismatch in mismatches)
    assert any("negative sample was classified as PII" in mismatch for mismatch in mismatches)
    assert any("expected goal label" in mismatch for mismatch in mismatches)
    assert any("negative sample produced a non-none label" in mismatch for mismatch in mismatches)


def test_collect_runtime_mismatches_accepts_guardrailed_negative_samples() -> None:
    runtime = {
        "load_errors": [],
        "checks": {
            "pii_span_detection": {"spans": [[0, 5, "email"]], "confidence": 0.9},
            "fact_extraction_structured": {"spans": [[0, 5, "location"]], "confidence": 0.9},
            "fact_extraction_structured_multilingual": {
                "spans": [[0, 5, "location"]],
                "confidence": 0.9,
            },
            "pii_presence_positive": {"label": "pii", "confidence": 0.99},
            "pii_presence_negative": {"label": "pii", "confidence": 0.2},
            "constraint_type_positive": {"label": "goal", "confidence": 0.2},
            "constraint_type_negative": {"label": "goal", "confidence": 0.2},
        },
    }

    mismatches = _collect_runtime_mismatches(runtime)

    assert mismatches == []
