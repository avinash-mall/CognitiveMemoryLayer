"""Runtime adapter for task-specific inference from trained modelpack artifacts."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class ModelPrediction:
    task: str
    label: str
    confidence: float


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
}

_MODEL_FILE = {
    "router": "router_model.joblib",
    "extractor": "extractor_model.joblib",
    "pair": "pair_model.joblib",
}

try:  # pragma: no cover - optional dependency surface
    from sklearn.exceptions import InconsistentVersionWarning
except Exception:  # pragma: no cover - sklearn may be unavailable

    class InconsistentVersionWarning(Warning):  # type: ignore[no-redef]
        """Fallback warning type when sklearn is unavailable."""

        pass


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
                with warnings.catch_warnings(record=True) as caught:
                    warnings.simplefilter("always")
                    payload = joblib.load(path)

                version_mismatch = any(
                    isinstance(w.message, InconsistentVersionWarning) for w in caught
                )
                if version_mismatch:
                    self._load_errors.append(
                        f"{family}: skipped due sklearn version mismatch for {path.name}"
                    )
                    continue

                model = payload.get("model") if isinstance(payload, dict) else payload
                if model is None:
                    self._load_errors.append(f"{family}: missing model key")
                    continue
                self._models[family] = model
            except Exception as exc:
                self._load_errors.append(f"{family}: {exc}")

        if self._models:
            logger.info(
                "modelpack_loaded",
                extra={
                    "models_dir": str(self.models_dir),
                    "families": sorted(self._models.keys()),
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
