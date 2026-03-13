"""Tests for scripts/models_artifact_probe.py mismatch gating."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest

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
