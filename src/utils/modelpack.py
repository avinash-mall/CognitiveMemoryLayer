"""Runtime adapter for task-specific inference from trained modelpack artifacts."""

from __future__ import annotations

import math as _math_module
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .logging_config import get_logger

_math_exp = _math_module.exp

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelPrediction:
    task: str
    label: str
    confidence: float


@dataclass(frozen=True)
class ScorePrediction:
    """Score output for ranking / regression tasks."""

    task: str
    score: float
    confidence: float = 1.0


@dataclass(frozen=True)
class SpanPrediction:
    """Span output for token-classification tasks (PII, fact extraction)."""

    task: str
    spans: tuple[tuple[int, int, str], ...] = ()  # (start_char, end_char, label)
    confidence: float = 1.0


_TASK_FAMILY: dict[str, str] = {
    # Router tasks
    "memory_type": "router",
    "query_intent": "router",
    "query_domain": "router",
    "constraint_dimension": "router",
    "context_tag": "router",
    "salience_bin": "router",
    "importance_bin": "router",
    "confidence_bin": "router",
    "decay_profile": "router",
    # Extractor tasks
    "constraint_type": "extractor",
    "constraint_scope": "extractor",
    "constraint_stability": "extractor",
    "fact_type": "extractor",
    "pii_presence": "extractor",
    # Pair tasks
    "conflict_detection": "pair",
    "constraint_rerank": "pair",
    "scope_match": "pair",
    "supersession": "pair",
    # New Phase 1-3 tasks (task-level models loaded from per-task artifacts)
    "retrieval_constraint_relevance_pair": "_task",
    "memory_rerank_pair": "_task",
    "novelty_pair": "_task",
    "fact_extraction_structured": "_task",
    "schema_match_pair": "_task",
    "reconsolidation_candidate_pair": "_task",
    "write_importance_regression": "_task",
    "pii_span_detection": "_task",
    "consolidation_gist_quality": "_task",
    "forgetting_action_policy": "_task",
}

_MODEL_FILE = {
    "router": "router_model.joblib",
    "extractor": "extractor_model.joblib",
    "pair": "pair_model.joblib",
}

_TASK_MODEL_FILE: dict[str, str] = {
    "retrieval_constraint_relevance_pair": "retrieval_constraint_relevance_pair_model.joblib",
    "memory_rerank_pair": "memory_rerank_pair_model.joblib",
    "novelty_pair": "novelty_pair_model.joblib",
    "fact_extraction_structured": "fact_extraction_structured_model.joblib",
    "schema_match_pair": "schema_match_pair_model.joblib",
    "reconsolidation_candidate_pair": "reconsolidation_candidate_pair_model.joblib",
    "write_importance_regression": "write_importance_regression_model.joblib",
    "pii_span_detection": "pii_span_detection_model.joblib",
    "consolidation_gist_quality": "consolidation_gist_quality_model.joblib",
    "forgetting_action_policy": "forgetting_action_policy_model.joblib",
}


def _split_composite_label(raw: str) -> tuple[str, str]:
    if "::" not in raw:
        return "", raw
    task, label = raw.split("::", 1)
    return task.strip(), label.strip()


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


class ModelPackRuntime:
    """Loads trained models and serves task-level predictions.

    The training pipeline writes composite labels in format `task::label`.
    Runtime inference encodes input with the same template used by training:
    - single-text families: `task=<task> [text] <text>`
    - pair-text family: `task=<task> [a] <text_a> [b] <text_b>`
    """

    def __init__(self, models_dir: Path | None = None):
        self.models_dir = models_dir or (_repo_root() / "packages" / "models" / "trained_models")
        self._models: dict[str, Any] = {}
        self._task_models: dict[str, Any] = {}
        self._loaded = False
        self._load_errors: list[str] = []

    @property
    def available(self) -> bool:
        if not self._loaded:
            self._load_all()
        return bool(self._models)

    def predict_single(self, task: str, text: str) -> ModelPrediction | None:
        if not text.strip():
            return None
        family = _TASK_FAMILY.get(task)
        if family not in {"router", "extractor"}:
            return None
        model = self._get_family_model(family)
        if model is None:
            return None
        feature = f"task={task} [text] {text.strip()}"
        return self._predict_from_model(model, task=task, feature=feature)

    def predict_pair(self, task: str, text_a: str, text_b: str) -> ModelPrediction | None:
        if not text_a.strip() or not text_b.strip():
            return None
        family = _TASK_FAMILY.get(task)
        if family != "pair":
            return None
        model = self._get_family_model(family)
        if model is None:
            return None
        feature = f"task={task} [a] {text_a.strip()} [b] {text_b.strip()}"
        return self._predict_from_model(model, task=task, feature=feature)

    def predict_score_pair(self, task: str, text_a: str, text_b: str) -> ScorePrediction | None:
        """Predict a relevance/ranking score for a text pair (ranking/regression tasks)."""
        if not text_a.strip() or not text_b.strip():
            return None
        model = self._get_task_model(task)
        if model is None:
            pred = self.predict_pair(task, text_a, text_b)
            if pred is None:
                return None
            return ScorePrediction(task=task, score=pred.confidence, confidence=pred.confidence)

        feature = f"task={task} [a] {text_a.strip()} [b] {text_b.strip()}"
        try:
            if hasattr(model, "predict_proba"):
                prob = model.predict_proba([feature])[0]
                score = float(max(prob)) if len(prob) > 0 else 0.5
            elif hasattr(model, "decision_function"):
                raw = model.decision_function([feature])[0]
                score = 1.0 / (1.0 + _math_exp(-float(raw)))
            else:
                raw = model.predict([feature])[0]
                score = float(raw)
            return ScorePrediction(task=task, score=max(0.0, min(1.0, score)))
        except Exception:
            return None

    def predict_score_single(self, task: str, text: str) -> ScorePrediction | None:
        """Predict a regression score for single text (e.g. importance regression)."""
        if not text.strip():
            return None
        model = self._get_task_model(task)
        if model is None:
            pred = self.predict_single(task, text)
            if pred is None:
                return None
            return ScorePrediction(task=task, score=pred.confidence, confidence=pred.confidence)

        feature = f"task={task} [text] {text.strip()}"
        try:
            raw = model.predict([feature])[0]
            return ScorePrediction(task=task, score=max(0.0, min(1.0, float(raw))))
        except Exception:
            return None

    def predict_spans(self, task: str, text: str) -> SpanPrediction | None:
        """Predict labelled spans for token-classification tasks (PII, fact extraction)."""
        if not text.strip():
            return None
        model = self._get_task_model(task)
        if model is None:
            return None
        try:
            result = model.predict([text])[0]
            if isinstance(result, (list, tuple)):
                spans = tuple(
                    (int(s[0]), int(s[1]), str(s[2]))
                    for s in result
                    if isinstance(s, (list, tuple)) and len(s) >= 3
                )
                return SpanPrediction(task=task, spans=spans)
            return SpanPrediction(task=task, spans=())
        except Exception:
            return None

    def has_task_model(self, task: str) -> bool:
        """Check whether a dedicated per-task model is available."""
        return self._get_task_model(task) is not None

    def _get_task_model(self, task: str) -> Any | None:
        if not self._loaded:
            self._load_all()
        return self._task_models.get(task)

    def _get_family_model(self, family: str) -> Any | None:
        if not self._loaded:
            self._load_all()
        return self._models.get(family)

    def _load_all(self) -> None:
        if self._loaded:
            return
        self._loaded = True

        try:
            import joblib
        except Exception as exc:  # pragma: no cover - optional runtime dependency
            logger.warning("modelpack_joblib_unavailable", extra={"error": str(exc)})
            return

        for family, filename in _MODEL_FILE.items():
            path = self.models_dir / filename
            if not path.exists():
                continue
            try:
                payload = joblib.load(path)
                model = payload.get("model") if isinstance(payload, dict) else payload
                if model is None:
                    self._load_errors.append(f"{family}: missing model key")
                    continue
                self._models[family] = model
            except Exception as exc:
                self._load_errors.append(f"{family}: {exc}")

        for task_name, filename in _TASK_MODEL_FILE.items():
            path = self.models_dir / filename
            if not path.exists():
                continue
            try:
                payload = joblib.load(path)
                model = payload.get("model") if isinstance(payload, dict) else payload
                if model is None:
                    self._load_errors.append(f"task:{task_name}: missing model key")
                    continue
                self._task_models[task_name] = model
            except Exception as exc:
                self._load_errors.append(f"task:{task_name}: {exc}")

        loaded_families = sorted(self._models.keys())
        loaded_tasks = sorted(self._task_models.keys())
        if self._models or self._task_models:
            logger.info(
                "modelpack_loaded",
                extra={
                    "models_dir": str(self.models_dir),
                    "families": loaded_families,
                    "task_models": loaded_tasks,
                },
            )
        elif self._load_errors:
            logger.warning(
                "modelpack_load_failed",
                extra={"models_dir": str(self.models_dir), "errors": self._load_errors[:5]},
            )

    def _predict_from_model(self, model: Any, *, task: str, feature: str) -> ModelPrediction | None:
        try:
            raw_pred = model.predict([feature])[0]
        except Exception:
            return None

        pred_task, pred_label = _split_composite_label(str(raw_pred))
        confidence = 0.5

        # If the top class belongs to another task, choose the best class among task-specific labels.
        if pred_task != task or not pred_label:
            best = self._best_task_class(model, task=task, feature=feature)
            if best is None:
                return None
            chosen_label, confidence = best
            return ModelPrediction(task=task, label=chosen_label, confidence=confidence)

        conf_from_proba = self._class_probability(
            model, feature=feature, target=f"{task}::{pred_label}"
        )
        if conf_from_proba is not None:
            confidence = conf_from_proba

        return ModelPrediction(
            task=task, label=pred_label, confidence=max(0.0, min(1.0, confidence))
        )

    def _best_task_class(self, model: Any, *, task: str, feature: str) -> tuple[str, float] | None:
        classes = self._classes(model)
        if not classes:
            return None
        task_classes = [c for c in classes if c.startswith(f"{task}::")]
        if not task_classes:
            return None

        probs = self._predict_proba(model, feature=feature)
        if probs is None:
            first = task_classes[0]
            _, label = _split_composite_label(first)
            return label, 0.5

        best_class = max(task_classes, key=lambda c: probs.get(c, 0.0))
        _, label = _split_composite_label(best_class)
        return label, float(probs.get(best_class, 0.5))

    def _class_probability(self, model: Any, *, feature: str, target: str) -> float | None:
        probs = self._predict_proba(model, feature=feature)
        if probs is None:
            return None
        return probs.get(target)

    def _predict_proba(self, model: Any, *, feature: str) -> dict[str, float] | None:
        classifier = getattr(model, "named_steps", {}).get("classifier")
        classes = self._classes(model)
        if classifier is None or not classes:
            return None
        if not hasattr(model, "predict_proba"):
            return None

        try:
            row = model.predict_proba([feature])[0]
        except Exception:
            return None

        out: dict[str, float] = {}
        for idx, class_name in enumerate(classes):
            try:
                out[str(class_name)] = float(row[idx])
            except Exception:
                continue
        return out

    @staticmethod
    def _classes(model: Any) -> list[str]:
        classifier = getattr(model, "named_steps", {}).get("classifier")
        classes = getattr(classifier, "classes_", None)
        if classes is None:
            return []
        return [str(x) for x in classes]


@lru_cache(maxsize=1)
def get_modelpack_runtime() -> ModelPackRuntime:
    """Return process-cached runtime modelpack loader."""
    return ModelPackRuntime()
