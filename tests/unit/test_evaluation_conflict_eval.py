from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from src.evaluation.conflict_eval import ConflictDetectorEvaluator, ConflictEvalCase


class _FakeModelPack:
    def __init__(self, *, heuristic_label: str = "conflict", score: float | None = None) -> None:
        self.heuristic_label = heuristic_label
        self.score = score

    def predict_pair(self, task: str, old_memory: str, new_statement: str):
        assert task == "conflict_detection"
        assert old_memory
        assert new_statement
        return SimpleNamespace(label=self.heuristic_label)

    def has_task_model(self, task: str) -> bool:
        return task == "reconsolidation_candidate_pair" and self.score is not None

    def predict_score_pair(self, task: str, new_statement: str, old_memory: str):
        assert task == "reconsolidation_candidate_pair"
        assert new_statement
        assert old_memory
        return SimpleNamespace(score=self.score)


def test_load_corpus_skips_invalid_lines(tmp_path: Path) -> None:
    path = tmp_path / "corpus.jsonl"
    path.write_text(
        json.dumps(
            {
                "old_memory": "User likes tea",
                "new_statement": "User hates tea",
                "expected_conflict": True,
                "expected_operation": "replace",
            }
        )
        + "\n"
        + "{not json}\n"
        + json.dumps({"old_memory": "missing keys"})
        + "\n",
        encoding="utf-8",
    )

    evaluator = ConflictDetectorEvaluator(modelpack=_FakeModelPack())
    cases = evaluator.load_corpus(path)

    assert len(cases) == 1
    assert cases[0].expected_operation == "replace"


def test_load_corpus_returns_empty_for_missing_file(tmp_path: Path) -> None:
    evaluator = ConflictDetectorEvaluator(modelpack=_FakeModelPack())
    assert evaluator.load_corpus(tmp_path / "missing.jsonl") == []


def test_evaluate_case_with_model_agreement() -> None:
    evaluator = ConflictDetectorEvaluator(modelpack=_FakeModelPack(score=0.9))
    case = ConflictEvalCase(
        old_memory="User lives in Paris",
        new_statement="User lives in Berlin",
        expected_conflict=True,
    )

    result = evaluator.evaluate_case(case)

    assert result.heuristic_detected is True
    assert result.model_detected is True
    assert result.heuristic_label == "conflict"
    assert result.model_label == "conflict"
    assert result.agreed is True


def test_evaluate_case_without_model_returns_none_model_detection() -> None:
    evaluator = ConflictDetectorEvaluator(modelpack=_FakeModelPack(heuristic_label="no_conflict"))
    case = ConflictEvalCase(
        old_memory="User likes tea",
        new_statement="User still likes tea",
        expected_conflict=False,
    )

    result = evaluator.evaluate_case(case)

    assert result.heuristic_detected is False
    assert result.model_detected is None
    assert result.agreed is False


def test_evaluate_corpus_computes_metrics_and_disagreements() -> None:
    class _AlternatingModelPack(_FakeModelPack):
        def __init__(self) -> None:
            super().__init__(score=0.8)
            self._scores = iter([0.8, 0.2])

        def predict_score_pair(self, task: str, new_statement: str, old_memory: str):
            _ = task
            _ = new_statement
            _ = old_memory
            return SimpleNamespace(score=next(self._scores))

    evaluator = ConflictDetectorEvaluator(modelpack=_AlternatingModelPack())
    cases = [
        ConflictEvalCase("A", "B", expected_conflict=True),
        ConflictEvalCase("C", "D", expected_conflict=False),
    ]

    report = evaluator.evaluate_corpus(cases)

    assert report["total_cases"] == 2
    assert report["heuristic"]["tp"] == 1
    assert report["heuristic"]["fp"] == 1
    assert report["model"]["evaluated"] == 2
    assert report["shadow_comparison"]["agreement_count"] == 1
    assert len(report["shadow_comparison"]["disagreements"]) == 1


def test_save_report_writes_json(tmp_path: Path) -> None:
    evaluator = ConflictDetectorEvaluator(modelpack=_FakeModelPack())
    report_path = tmp_path / "reports" / "conflict.json"

    evaluator.save_report({"status": "ok", "count": 2}, report_path)

    assert json.loads(report_path.read_text(encoding="utf-8")) == {"status": "ok", "count": 2}
