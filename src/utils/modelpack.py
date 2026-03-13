"""Runtime adapter for task-specific inference from trained modelpack artifacts."""

from __future__ import annotations

import importlib.metadata
import json
import math as _math_module
import os
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
    "memory_type": "memory_type_model.joblib",
    "salience_bin": "salience_bin_model.joblib",
    "importance_bin": "importance_bin_model.joblib",
    "confidence_bin": "confidence_bin_model.joblib",
    "decay_profile": "decay_profile_model.joblib",
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


def _safety_override_forgetting_policy(
    pred: ModelPrediction,
    metadata: dict[str, Any] | None,
) -> ModelPrediction | None:
    """Deterministic guardrails: avoid compress/delete on high-value, recently-accessed memories."""
    if pred.label not in ("compress", "delete"):
        return None
    meta = metadata or {}
    importance = meta.get("importance")
    access_count = meta.get("access_count")
    age_days = meta.get("age_days")
    dependency_count = meta.get("dependency_count")
    try:
        imp = float(importance) if importance is not None else 0.0
        acc = int(access_count) if access_count is not None else 0
        age = int(age_days) if age_days is not None else 999
        dep = int(dependency_count) if dependency_count is not None else 0
    except (TypeError, ValueError):
        return None
    if pred.label == "delete" and dep >= 2:
        logger.info(
            "safety_override_forgetting_policy",
            extra={"original": pred.label, "override": "decay", "reason": "dependency_count>=2"},
        )
        return ModelPrediction(task=pred.task, label="decay", confidence=pred.confidence)
    if imp >= 0.7 and acc >= 5 and age <= 30:
        logger.info(
            "safety_override_forgetting_policy",
            extra={"original": pred.label, "override": "keep", "reason": "high_value_recent"},
        )
        return ModelPrediction(task=pred.task, label="keep", confidence=pred.confidence)
    return None


def _safety_override_gist_quality(pred: ModelPrediction) -> ModelPrediction | None:
    """Downgrade low-confidence accept to reject."""
    if pred.label != "accept" or pred.confidence >= 0.6:
        return None
    logger.info(
        "safety_override_gist_quality",
        extra={"original": "accept", "override": "reject", "confidence": pred.confidence},
    )
    return ModelPrediction(task=pred.task, label="reject", confidence=pred.confidence)


_SINGLE_META_FIELDS = (
    "memory_type",
    "namespace",
    "importance",
    "confidence",
    "access_count",
    "age_days",
    "dependency_count",
    "support_count",
    "mixed_topic",
    "context_tags",
)


def _ratio_bucket(value: float) -> str:
    if value >= 0.9:
        return "very_high"
    if value >= 0.7:
        return "high"
    if value >= 0.4:
        return "medium"
    if value >= 0.15:
        return "low"
    return "very_low"


def _count_bucket(value: int, *, high: int, medium: int, zero_label: str = "none") -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    if value > 0:
        return "low"
    return zero_label


def _safe_float(value: object) -> float | None:
    try:
        if value is None:
            return None
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None:
            return None
        return int(value)  # type: ignore[call-overload]
    except Exception:
        return None


def _normalize_tags(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                parsed = json.loads(text)
            except Exception:
                parsed = None
            if isinstance(parsed, list):
                return [str(x).strip() for x in parsed if str(x).strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


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
        env_dir = os.environ.get("CML_MODELS_DIR")
        if models_dir is not None:
            self.models_dir = models_dir
        elif env_dir:
            self.models_dir = Path(env_dir)
        else:
            self.models_dir = _repo_root() / "packages" / "models" / "trained_models"
        self._models: dict[str, Any] = {}
        self._task_models: dict[str, Any] = {}
        self._loaded = False
        self._load_errors: list[str] = []
        self._manifest: dict[str, Any] | None = None

    @property
    def available(self) -> bool:
        """True when at least one family or per-task model loaded successfully."""
        if not self._loaded:
            self._load_all()
        return bool(self._models) or bool(self._task_models)

    @property
    def available_families(self) -> list[str]:
        """Family names that loaded successfully (e.g. ['pair'])."""
        if not self._loaded:
            self._load_all()
        return sorted(self._models.keys())

    @property
    def available_tasks(self) -> list[str]:
        """Task names that have a backing model (per-task or via family)."""
        if not self._loaded:
            self._load_all()
        tasks: set[str] = set(self._task_models.keys())
        for task_name, family in _TASK_FAMILY.items():
            if family in self._models:
                tasks.add(task_name)
        return sorted(tasks)

    def supports_task(self, task: str) -> bool:
        """Check whether this runtime can score a task via task-model or loaded family."""
        if not self._loaded:
            self._load_all()
        family = _TASK_FAMILY.get(task)
        if family is None:
            return False
        if task in self._task_models:
            return True
        return family in self._models

    def capability_report(self) -> dict[str, Any]:
        """Return load/capability diagnostics for probes and health checks."""
        if not self._loaded:
            self._load_all()
        pending_families = sorted(set(_MODEL_FILE.keys()) - set(self._models.keys()))
        return {
            "models_dir": str(self.models_dir),
            "available": bool(self._models) or bool(self._task_models),
            "available_families": sorted(self._models.keys()),
            "available_tasks": self.available_tasks,
            "pending_families": pending_families,
            "load_errors": list(self._load_errors),
            "manifest_schema_version": (
                self._manifest.get("manifest_schema_version")
                if isinstance(self._manifest, dict)
                else None
            ),
        }

    def predict_single(
        self, task: str, text: str, metadata: dict[str, Any] | None = None
    ) -> ModelPrediction | None:
        if not text.strip():
            return None
        pred: ModelPrediction | None = None
        task_model = self._get_task_model(task)
        if task_model is not None:
            feature = self._serialize_single_feature(task, text, metadata=metadata)
            pred = self._predict_from_model(task_model, task=task, feature=feature)
        else:
            family = _TASK_FAMILY.get(task)
            if family in {"router", "extractor"}:
                model = self._get_family_model(family)
                if model is not None:
                    feature = self._serialize_single_feature(task, text, metadata=metadata)
                    pred = self._predict_from_model(model, task=task, feature=feature)
        if pred is None:
            return None
        if task == "forgetting_action_policy":
            override = _safety_override_forgetting_policy(pred, metadata)
            if override is not None:
                pred = override
        if task == "consolidation_gist_quality":
            override = _safety_override_gist_quality(pred)
            if override is not None:
                pred = override
        return pred

    def predict_pair(self, task: str, text_a: str, text_b: str) -> ModelPrediction | None:
        if not text_a.strip() or not text_b.strip():
            return None
        task_model = self._get_task_model(task)
        if task_model is not None:
            feature = f"task={task} [a] {text_a.strip()} [b] {text_b.strip()}"
            return self._predict_from_model(task_model, task=task, feature=feature)
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
            score = self._score_from_task_model(model, task=task, feature=feature)
            return ScorePrediction(task=task, score=max(0.0, min(1.0, score)))
        except Exception:
            return None

    def predict_score_single(
        self, task: str, text: str, metadata: dict[str, Any] | None = None
    ) -> ScorePrediction | None:
        """Predict a regression score for single text (e.g. importance regression)."""
        if not text.strip():
            return None
        model = self._get_task_model(task)
        if model is None:
            pred = self.predict_single(task, text, metadata=metadata)
            if pred is None:
                return None
            return ScorePrediction(task=task, score=pred.confidence, confidence=pred.confidence)

        feature = self._serialize_single_feature(task, text, metadata=metadata)
        try:
            score = self._score_from_task_model(model, task=task, feature=feature)
            return ScorePrediction(task=task, score=max(0.0, min(1.0, float(score))))
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
        except ImportError as exc:
            msg = f"task:{task}: {exc}"
            if msg not in self._load_errors:
                self._load_errors.append(msg)
            return None
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

        manifest_path = self.models_dir / "manifest.json"
        if manifest_path.exists():
            try:
                self._manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except Exception as exc:
                self._load_errors.append(f"manifest: {exc}")
            else:
                self._validate_manifest_versions()

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
        pending_families = sorted(set(_MODEL_FILE.keys()) - set(self._models.keys()))

        if self._models or self._task_models:
            log_extra: dict[str, Any] = {
                "models_dir": str(self.models_dir),
                "families": loaded_families,
                "task_models": loaded_tasks,
            }
            if pending_families:
                log_extra["pending_families"] = pending_families
            logger.info("modelpack_loaded", extra=log_extra)
        elif self._load_errors:
            logger.warning(
                "modelpack_load_failed",
                extra={"models_dir": str(self.models_dir), "errors": self._load_errors[:5]},
            )

        if not self.models_dir.exists():
            logger.warning(
                "modelpack_dir_missing",
                extra={"models_dir": str(self.models_dir)},
            )

        self._validate_manifest_artifacts()

    def _validate_manifest_artifacts(self) -> None:
        """Warn when the manifest references model files that don't exist on disk."""
        if not isinstance(self._manifest, dict):
            return

        missing: list[str] = []
        families = self._manifest.get("families", {})
        if isinstance(families, dict):
            for family_name, fmeta in families.items():
                if not isinstance(fmeta, dict):
                    continue
                model_path = fmeta.get("model_path")
                if model_path and not Path(model_path).exists():
                    expected_local = self.models_dir / f"{family_name}_model.joblib"
                    if not expected_local.exists():
                        missing.append(f"family:{family_name}")

        task_models_cfg = self._manifest.get("task_models", {})
        if isinstance(task_models_cfg, dict):
            for task_name, tmeta in task_models_cfg.items():
                if not isinstance(tmeta, dict):
                    continue
                model_path = tmeta.get("model_path")
                if model_path and not Path(model_path).exists():
                    expected_local = self.models_dir / f"{task_name}_model.joblib"
                    if not expected_local.exists():
                        missing.append(f"task:{task_name}")

        if missing:
            self._load_errors.extend(
                [f"manifest references missing artifact: {m}" for m in missing]
            )
            logger.warning(
                "modelpack_manifest_artifact_mismatch",
                extra={
                    "models_dir": str(self.models_dir),
                    "missing": missing,
                },
            )

    def _validate_manifest_versions(self) -> None:
        if not isinstance(self._manifest, dict):
            return
        build_meta = self._manifest.get("build_metadata")
        if not isinstance(build_meta, dict):
            return
        deps = build_meta.get("dependencies")
        if not isinstance(deps, dict):
            return

        expected = {
            "scikit-learn": deps.get("scikit_learn"),
            "joblib": deps.get("joblib"),
            "pandas": deps.get("pandas"),
        }
        mismatches: dict[str, dict[str, str]] = {}
        for pkg_name, expected_version in expected.items():
            if not expected_version:
                continue
            try:
                installed_version = importlib.metadata.version(pkg_name)
            except importlib.metadata.PackageNotFoundError:
                continue
            if installed_version != expected_version:
                mismatches[pkg_name] = {
                    "trained_with": str(expected_version),
                    "installed": str(installed_version),
                }
        if mismatches:
            logger.warning(
                "modelpack_dependency_version_mismatch", extra={"mismatches": mismatches}
            )

    def _serialize_single_feature(
        self, task: str, text: str, *, metadata: dict[str, Any] | None = None
    ) -> str:
        feature = f"task={task} [text] {text.strip()}"
        if not metadata:
            return feature
        tokens: list[str] = []
        for key in _SINGLE_META_FIELDS:
            value = metadata.get(key)
            if value is None:
                continue
            if key == "memory_type":
                tokens.append(f"memory_type={str(value).strip()}")
            elif key == "namespace":
                tokens.append(f"namespace={str(value).strip()}")
            elif key == "context_tags":
                tokens.extend(f"context_tag={tag}" for tag in _normalize_tags(value)[:3])
            elif key in {"importance", "confidence"}:
                ratio = _safe_float(value)
                if ratio is not None:
                    tokens.append(f"{key}_bin={_ratio_bucket(ratio)}")
            elif key in {"access_count", "dependency_count", "support_count"}:
                count = _safe_int(value)
                if count is not None:
                    tokens.append(
                        f"{key}={_count_bucket(count, high=6 if key == 'access_count' else 4, medium=2 if key != 'dependency_count' else 1)}"
                    )
            elif key == "age_days":
                age_days = _safe_int(value)
                if age_days is not None:
                    tokens.append(
                        f"age_days={_count_bucket(age_days, high=90, medium=21, zero_label='fresh')}"
                    )
            elif key == "mixed_topic":
                tokens.append(f"mixed_topic={bool(value)}")
        if not tokens:
            return feature
        return feature + " [meta] " + " ".join(token for token in tokens if token)

    def _score_from_task_model(self, model: Any, *, task: str, feature: str) -> float:
        probs = self._predict_proba(model, feature=feature)
        if probs:
            preferred_positive = {
                "retrieval_constraint_relevance_pair": "relevant",
                "memory_rerank_pair": "relevant",
                "reconsolidation_candidate_pair": "relevant",
                "schema_match_pair": "match",
                "consolidation_gist_quality": "accept",
            }
            if task == "novelty_pair":
                weighted = 0.0
                for class_name, prob in probs.items():
                    pred_task, pred_label = _split_composite_label(class_name)
                    if pred_task != task:
                        continue
                    weighted += {
                        "duplicate": 1.0,
                        "changed": 0.85,
                        "novel": 0.05,
                    }.get(pred_label, 0.5) * float(prob)
                return weighted
            positive_label = preferred_positive.get(task)
            if positive_label:
                positive_key = f"{task}::{positive_label}"
                if positive_key in probs:
                    return float(probs[positive_key])
            return float(max(probs.values()))

        if hasattr(model, "decision_function"):
            raw = model.decision_function([feature])[0]
            return 1.0 / (1.0 + _math_exp(-float(raw)))

        raw = model.predict([feature])[0]
        try:
            return float(raw)
        except Exception:
            pred = self._predict_from_model(model, task=task, feature=feature)
            return 0.5 if pred is None else float(pred.confidence)

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
        classes = self._classes(model)
        if not classes or not hasattr(model, "predict_proba"):
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
        classes = getattr(model, "classes_", None)
        if classes is None:
            classifier = getattr(model, "named_steps", {}).get("classifier")
            classes = getattr(classifier, "classes_", None)
        if classes is None:
            return []
        return [str(x) for x in classes]


@lru_cache(maxsize=1)
def get_modelpack_runtime() -> ModelPackRuntime:
    """Return process-cached runtime modelpack loader."""
    return ModelPackRuntime()
