"""Offline evaluation framework for conflict detection accuracy.

Compares heuristic-based conflict detection with model-based detection,
producing precision/recall/F1 metrics and a shadow comparison report.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from ..utils.logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ConflictEvalCase:
    """Single evaluation case for conflict detection."""

    old_memory: str
    new_statement: str
    expected_conflict: bool
    expected_operation: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)


@dataclass
class EvalResult:
    """Result of evaluating a single case."""

    case: ConflictEvalCase
    heuristic_detected: bool
    model_detected: bool | None
    heuristic_label: str = ""
    model_label: str = ""
    agreed: bool = False


class ConflictDetectorEvaluator:
    """Runs offline evaluation of conflict detection."""

    def __init__(self, modelpack=None):
        from ..utils.modelpack import get_modelpack_runtime

        self.modelpack = modelpack or get_modelpack_runtime()

    def load_corpus(self, path: Path) -> list[ConflictEvalCase]:
        """Load evaluation corpus from JSONL file.

        Each line: {"old_memory": "...", "new_statement": "...",
                     "expected_conflict": true/false, "expected_operation": "..."}
        """
        cases: list[ConflictEvalCase] = []
        if not path.exists():
            logger.warning("conflict_eval_corpus_missing", extra={"path": str(path)})
            return cases
        with open(path, encoding="utf-8") as f:
            for line_num, raw_line in enumerate(f, 1):
                raw_line = raw_line.strip()
                if not raw_line:
                    continue
                try:
                    data = json.loads(raw_line)
                    cases.append(
                        ConflictEvalCase(
                            old_memory=data["old_memory"],
                            new_statement=data["new_statement"],
                            expected_conflict=bool(data.get("expected_conflict", False)),
                            expected_operation=str(data.get("expected_operation", "")),
                            metadata=data.get("metadata", {}),
                        )
                    )
                except (json.JSONDecodeError, KeyError) as exc:
                    logger.debug(
                        "conflict_eval_parse_error",
                        extra={"line": line_num, "error": str(exc)},
                    )
        return cases

    def evaluate_case(self, case: ConflictEvalCase) -> EvalResult:
        """Evaluate a single case using both heuristic and model approaches."""
        heuristic_detected = False
        heuristic_label = "no_conflict"
        try:
            pred = self.modelpack.predict_pair(
                "conflict_detection", case.old_memory, case.new_statement
            )
            if pred is not None:
                heuristic_detected = pred.label == "conflict"
                heuristic_label = pred.label
        except Exception:
            pass

        model_detected: bool | None = None
        model_label = ""
        try:
            has_model = getattr(self.modelpack, "has_task_model", lambda _: False)
            if has_model("reconsolidation_candidate_pair"):
                score_pred = self.modelpack.predict_score_pair(
                    "reconsolidation_candidate_pair",
                    case.new_statement,
                    case.old_memory,
                )
                if score_pred is not None:
                    model_detected = score_pred.score >= 0.5
                    model_label = "conflict" if model_detected else "no_conflict"
        except Exception:
            pass

        agreed = model_detected is not None and heuristic_detected == model_detected

        return EvalResult(
            case=case,
            heuristic_detected=heuristic_detected,
            model_detected=model_detected,
            heuristic_label=heuristic_label,
            model_label=model_label,
            agreed=agreed,
        )

    def evaluate_corpus(self, cases: list[ConflictEvalCase]) -> dict[str, Any]:
        """Evaluate an entire corpus and produce metrics report."""
        results = [self.evaluate_case(c) for c in cases]

        h_tp = sum(1 for r in results if r.heuristic_detected and r.case.expected_conflict)
        h_fp = sum(1 for r in results if r.heuristic_detected and not r.case.expected_conflict)
        h_fn = sum(1 for r in results if not r.heuristic_detected and r.case.expected_conflict)
        h_tn = sum(1 for r in results if not r.heuristic_detected and not r.case.expected_conflict)

        h_precision = h_tp / max(1, h_tp + h_fp)
        h_recall = h_tp / max(1, h_tp + h_fn)
        h_f1 = 2 * h_precision * h_recall / max(1e-9, h_precision + h_recall)

        report: dict[str, Any] = {
            "total_cases": len(results),
            "heuristic": {
                "tp": h_tp,
                "fp": h_fp,
                "fn": h_fn,
                "tn": h_tn,
                "precision": round(h_precision, 4),
                "recall": round(h_recall, 4),
                "f1": round(h_f1, 4),
            },
        }

        model_results = [r for r in results if r.model_detected is not None]
        if model_results:
            m_tp = sum(1 for r in model_results if r.model_detected and r.case.expected_conflict)
            m_fp = sum(
                1 for r in model_results if r.model_detected and not r.case.expected_conflict
            )
            m_fn = sum(
                1 for r in model_results if not r.model_detected and r.case.expected_conflict
            )
            m_tn = sum(
                1 for r in model_results if not r.model_detected and not r.case.expected_conflict
            )

            m_precision = m_tp / max(1, m_tp + m_fp)
            m_recall = m_tp / max(1, m_tp + m_fn)
            m_f1 = 2 * m_precision * m_recall / max(1e-9, m_precision + m_recall)

            report["model"] = {
                "evaluated": len(model_results),
                "tp": m_tp,
                "fp": m_fp,
                "fn": m_fn,
                "tn": m_tn,
                "precision": round(m_precision, 4),
                "recall": round(m_recall, 4),
                "f1": round(m_f1, 4),
            }

        agreed = sum(1 for r in results if r.agreed)
        report["shadow_comparison"] = {
            "model_available_cases": len(model_results),
            "agreement_count": agreed,
            "agreement_rate": round(agreed / max(1, len(model_results)), 4),
            "disagreements": [
                {
                    "old_memory": r.case.old_memory[:100],
                    "new_statement": r.case.new_statement[:100],
                    "expected": r.case.expected_conflict,
                    "heuristic": r.heuristic_detected,
                    "model": r.model_detected,
                }
                for r in results
                if r.model_detected is not None and not r.agreed
            ][:50],
        }

        return report

    def save_report(self, report: dict[str, Any], path: Path) -> None:
        """Write evaluation report to JSON file."""
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)
        logger.info("conflict_eval_report_saved", extra={"path": str(path)})
