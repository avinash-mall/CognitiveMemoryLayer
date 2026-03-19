"""Unit tests for eval dataset preparation helpers."""

from __future__ import annotations

import json
from pathlib import Path

from cml.eval.config import ensure_unified_eval_data


def _make_repo(tmp_path: Path) -> tuple[Path, Path]:
    repo_root = tmp_path / "repo"
    data_dir = repo_root / "evaluation" / "locomo_plus" / "data"
    (repo_root / "docker").mkdir(parents=True)
    (repo_root / "packages" / "models").mkdir(parents=True)
    data_dir.mkdir(parents=True)
    (repo_root / "docker" / "docker-compose.yml").write_text("services: {}", encoding="utf-8")
    (repo_root / "packages" / "models" / "model_pipeline.toml").write_text(
        "name = 'test'", encoding="utf-8"
    )
    (data_dir / "locomo_plus.json").write_text("[]", encoding="utf-8")
    (data_dir / "unified_input.py").write_text(
        """
from __future__ import annotations

import json


def build_unified_samples(data_dir):
    with open(data_dir / "locomo10.json", encoding="utf-8") as f:
        rows = json.load(f)
    qa = rows[0]["qa"][0]
    return [
        {
            "input_prompt": "stub",
            "trigger": qa["question"],
            "evidence": "Alice: I live in Paris.",
            "category": "single-hop",
            "answer": qa["answer"],
        }
    ]
""".strip(),
        encoding="utf-8",
    )
    return repo_root, data_dir


def test_ensure_unified_eval_data_downloads_and_builds(monkeypatch, tmp_path: Path) -> None:
    repo_root, data_dir = _make_repo(tmp_path)
    unified_path = data_dir / "unified_input_samples_v2.json"

    downloaded_locomo = [
        {
            "conversation": {
                "session_1_date_time": "January 15, 2024",
                "session_1": [{"speaker": "Alice", "text": "I live in Paris."}],
            },
            "qa": [
                {
                    "question": "Where does Alice live?",
                    "category": 4,
                    "answer": "Paris",
                    "evidence": ["D1:1"],
                }
            ],
        }
    ]

    def _fake_download(_url: str, destination: Path) -> None:
        destination.write_text(json.dumps(downloaded_locomo), encoding="utf-8")

    monkeypatch.setattr("cml.eval.config._download_file", _fake_download)

    resolved = ensure_unified_eval_data(unified_path, repo_root=repo_root)

    assert resolved == unified_path
    assert (data_dir / "locomo10.json").exists()
    payload = json.loads(unified_path.read_text(encoding="utf-8"))
    assert len(payload) == 1
    assert payload[0]["trigger"] == "Where does Alice live?"
    assert payload[0]["category"] == "single-hop"


def test_ensure_unified_eval_data_keeps_existing_file(tmp_path: Path) -> None:
    repo_root, data_dir = _make_repo(tmp_path)
    unified_path = data_dir / "unified_input_samples_v2.json"
    unified_path.write_text("[]", encoding="utf-8")

    resolved = ensure_unified_eval_data(unified_path, repo_root=repo_root)

    assert resolved == unified_path
    assert unified_path.read_text(encoding="utf-8") == "[]"
