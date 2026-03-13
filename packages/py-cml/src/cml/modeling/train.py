"""
Unified train/test/eval script for all custom model families.

Families trained:
1) router
2) extractor
3) pair

Configuration:
  packages/models/model_pipeline.toml
"""

from __future__ import annotations

import argparse
import copy
import importlib.metadata
import json
import logging
import math
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

try:
    import joblib
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: joblib. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: numpy/pandas. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

try:
    from sklearn.calibration import CalibratedClassifierCV
    from sklearn.ensemble import HistGradientBoostingClassifier
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression, SGDClassifier, SGDRegressor
    from sklearn.metrics import (
        accuracy_score,
        classification_report,
        confusion_matrix,
        f1_score,
        log_loss,
        mean_absolute_error,
        mean_squared_error,
    )
    from sklearn.pipeline import Pipeline
    from sklearn.utils.class_weight import compute_class_weight, compute_sample_weight
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: scikit-learn. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

from cml.modeling.config import find_repo_root
from cml.modeling.memory_type_features import derive_memory_type_feature_tokens_from_row
from cml.modeling.pair_features import (
    build_pair_dense_features,
    hash_text,
    pair_embedding_cache_path,
)
from cml.modeling.runtime_models import (
    CumulativeOrdinalClassifier,
    EmbeddingPairClassifier,
    HierarchicalTextClassifier,
    TaskConditionalCalibratedClassifier,
    build_task_conditional_calibration_features,
)
from cml.modeling.token_training import train_token_task
from cml.modeling.types import TrainConfig
from cml.modeling.validation import run_preflight_validation, validate_manifest_artifacts

REPO_ROOT = find_repo_root(Path(__file__).resolve()) or Path.cwd()
MODELS_ROOT = REPO_ROOT / "packages" / "models"
DEFAULT_CONFIG_PATH = MODELS_ROOT / "model_pipeline.toml"

FAMILY_SINGLE_TEXT = {"router", "extractor"}
FAMILY_PAIR_TEXT = {"pair"}
ALL_FAMILIES = ("router", "extractor", "pair")

_logger = logging.getLogger("cml.modeling.train")
_TRAINER_DEFAULTS = {
    "classification": "classification",
    "pair_ranking": "pair_ranking",
    "single_regression": "single_regression",
    "token_classification": "token_classification",
}
_ADVERSARIAL_FIXTURES = {
    "consolidation_gist_quality": MODELS_ROOT / "adversarial" / "adversarial_gist_quality.jsonl",
    "forgetting_action_policy": MODELS_ROOT / "adversarial" / "adversarial_forgetting_policy.jsonl",
    "schema_match_pair": MODELS_ROOT / "adversarial" / "adversarial_schema_match_pair.jsonl",
}
_MEMORY_TYPE_MACRO_GROUPS = {
    "factual": ["semantic_fact", "knowledge", "observation"],
    "procedural": ["procedure", "plan", "task_state", "constraint"],
    "conversational": ["conversation", "message", "tool_result", "scratch"],
    "analytical": ["hypothesis", "reasoning_step"],
    "personal": ["preference", "episodic_event"],
}
_PAIR_GROUP_TOP1_TASKS = {
    "retrieval_constraint_relevance_pair",
    "memory_rerank_pair",
    "reconsolidation_candidate_pair",
}
_ALL_METADATA_COLUMNS = {
    "memory_type",
    "namespace",
    "context_tags",
    "importance",
    "confidence",
    "access_count",
    "age_days",
    "dependency_count",
    "support_count",
    "mixed_topic",
}
_TASK_METADATA_KEEP_COLUMNS: dict[str, set[str]] = {
    "salience_bin": {"age_days", "access_count", "support_count", "mixed_topic", "context_tags"},
    "importance_bin": {
        "age_days",
        "access_count",
        "dependency_count",
        "support_count",
        "mixed_topic",
        "context_tags",
    },
    "confidence_bin": {
        "age_days",
        "access_count",
        "dependency_count",
        "support_count",
        "mixed_topic",
        "context_tags",
    },
    "decay_profile": {
        "age_days",
        "access_count",
        "dependency_count",
        "support_count",
        "mixed_topic",
        "context_tags",
    },
}
_TASK_METADATA_EXCLUDE_COLUMNS: dict[str, set[str]] = {
    "consolidation_gist_quality": {"memory_type", "importance", "confidence"},
    "forgetting_action_policy": {"memory_type", "importance", "confidence"},
}


@dataclass
class TaskSpec:
    task_name: str
    family: str
    input_type: str  # "single" or "pair"
    objective: str  # "classification", "pair_ranking", "single_regression", "token_classification"
    labels: list[str]
    artifact_name: str
    metrics: list[str]
    enabled: bool = field(default=True)
    trainer: str = field(default="")
    feature_backend: str = field(default="")
    label_order: list[str] = field(default_factory=list)
    embedding_model_name: str = field(default="")


def _resolved_trainer(spec: TaskSpec) -> str:
    return spec.trainer.strip() or _TRAINER_DEFAULTS.get(spec.objective, spec.objective)


def _resolved_feature_backend(spec: TaskSpec) -> str:
    if spec.feature_backend.strip():
        return spec.feature_backend.strip()
    if _resolved_trainer(spec) == "embedding_pair":
        return "embedding_pair"
    return "tfidf"


class _NoopProgress:
    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")

    def update(self, n: int = 1) -> None:
        return

    def set_description(self, desc: str) -> None:
        return

    def close(self) -> None:
        return


def _progress(*, total: int, desc: str, unit: str):
    if _tqdm is None:
        return _NoopProgress(total=total, desc=desc, unit=unit)
    return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def _resolve_path(path_like: str, *, base: Path) -> Path:
    value = Path(path_like)
    if value.is_absolute():
        return value
    return (base / value).resolve()


def _load_config(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Config not found: {path}")
    with open(path, "rb") as f:
        cfg = tomllib.load(f)
    if "paths" not in cfg or "train" not in cfg:
        raise ValueError("Config must contain [paths] and [train] sections.")
    return cfg


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def _safe_pkg_version(dist_name: str) -> str | None:
    try:
        return importlib.metadata.version(dist_name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _git_build_metadata() -> dict[str, str | bool | None]:
    meta: dict[str, str | bool | None] = {"commit_sha": None, "dirty": None}
    try:
        proc_sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if proc_sha.returncode == 0:
            meta["commit_sha"] = proc_sha.stdout.strip()
    except Exception:
        return meta

    try:
        proc_dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            check=False,
            capture_output=True,
            text=True,
            cwd=str(REPO_ROOT),
        )
        if proc_dirty.returncode == 0:
            meta["dirty"] = bool(proc_dirty.stdout.strip())
    except Exception:
        return meta

    return meta


def _build_metadata() -> dict:
    return {
        "python_version": sys.version,
        "dependencies": {
            "scikit_learn": _safe_pkg_version("scikit-learn"),
            "joblib": _safe_pkg_version("joblib"),
            "pandas": _safe_pkg_version("pandas"),
        },
        **_git_build_metadata(),
    }


def _task_label_counts(df: pd.DataFrame) -> dict[str, dict[str, int]]:
    out: dict[str, dict[str, int]] = {}
    grouped = df.groupby(["task", "label"]).size()
    for (task, label), count in grouped.items():
        out.setdefault(str(task), {})[str(label)] = int(count)
    return out


def _composite_label(task: str, label: str) -> str:
    return f"{task}::{label}"


def _split_composite(value: str) -> tuple[str, str]:
    if "::" not in value:
        return "unknown", value
    a, b = value.split("::", 1)
    return a, b


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


def _safe_float(value: object) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
        return float(value)  # type: ignore[arg-type]
    except Exception:
        return None


def _safe_int(value: object) -> int | None:
    try:
        if value is None or pd.isna(value):
            return None
        return int(value)  # type: ignore[call-overload]
    except Exception:
        return None


def _normalize_tag_list(value: object) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        if text.startswith("[") and text.endswith("]"):
            try:
                loaded = json.loads(text)
            except Exception:
                loaded = None
            if isinstance(loaded, list):
                return [str(x).strip() for x in loaded if str(x).strip()]
        return [text]
    if isinstance(value, (list, tuple, set)):
        return [str(x).strip() for x in value if str(x).strip()]
    return []


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


def _count_bucket(value: int, *, high: int, medium: int) -> str:
    if value >= high:
        return "high"
    if value >= medium:
        return "medium"
    if value > 0:
        return "low"
    return "none"


def _metadata_excluded_columns(task_name: str) -> set[str]:
    keep = _TASK_METADATA_KEEP_COLUMNS.get(str(task_name).strip())
    if keep is not None:
        return set(_ALL_METADATA_COLUMNS - keep)
    return _TASK_METADATA_EXCLUDE_COLUMNS.get(str(task_name).strip(), set())


def _single_metadata_tokens(
    record: object,
    available_cols: set[str],
    *,
    task_name: str,
) -> list[str]:
    tokens: list[str] = []
    excluded = _metadata_excluded_columns(task_name)

    if "memory_type" in available_cols and "memory_type" not in excluded:
        memory_type = str(getattr(record, "memory_type", "") or "").strip()
        if memory_type:
            tokens.append(f"memory_type={memory_type}")
    if "namespace" in available_cols and "namespace" not in excluded:
        namespace = str(getattr(record, "namespace", "") or "").strip()
        if namespace:
            tokens.append(f"namespace={namespace}")
    if "context_tags" in available_cols and "context_tags" not in excluded:
        for tag in _normalize_tag_list(getattr(record, "context_tags", None))[:3]:
            tokens.append(f"context_tag={tag}")

    importance = (
        _safe_float(getattr(record, "importance", None)) if "importance" in available_cols else None
    )
    if importance is not None and "importance" not in excluded:
        tokens.append(f"importance_bin={_ratio_bucket(importance)}")
    confidence = (
        _safe_float(getattr(record, "confidence", None)) if "confidence" in available_cols else None
    )
    if confidence is not None and "confidence" not in excluded:
        tokens.append(f"confidence_bin={_ratio_bucket(confidence)}")
    access_count = (
        _safe_int(getattr(record, "access_count", None))
        if "access_count" in available_cols
        else None
    )
    if access_count is not None and "access_count" not in excluded:
        tokens.append(f"access_count={_count_bucket(access_count, high=6, medium=2)}")
    age_days = (
        _safe_int(getattr(record, "age_days", None)) if "age_days" in available_cols else None
    )
    if age_days is not None and "age_days" not in excluded:
        tokens.append(
            f"age_days={_count_bucket(age_days, high=90, medium=21).replace('none', 'fresh')}"
        )
    dependency_count = (
        _safe_int(getattr(record, "dependency_count", None))
        if "dependency_count" in available_cols
        else None
    )
    if dependency_count is not None and "dependency_count" not in excluded:
        tokens.append(f"dependency_count={_count_bucket(dependency_count, high=4, medium=1)}")
    support_count = (
        _safe_int(getattr(record, "support_count", None))
        if "support_count" in available_cols
        else None
    )
    if support_count is not None and "support_count" not in excluded:
        tokens.append(f"support_count={_count_bucket(support_count, high=4, medium=2)}")
    if "mixed_topic" in available_cols and "mixed_topic" not in excluded:
        mixed = getattr(record, "mixed_topic", None)
        if mixed is not None and not pd.isna(mixed):
            tokens.append(f"mixed_topic={bool(mixed)}")
    return tokens


def _load_split(prepared_dir: Path, family: str, split_name: str) -> pd.DataFrame:
    path = prepared_dir / f"{family}_{split_name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"Missing {family}_{split_name}: {path}")
    df = pd.read_parquet(path)

    if family in FAMILY_SINGLE_TEXT:
        required = {"text", "task", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{family}_{split_name} missing columns: {sorted(missing)}")
        df = df.dropna(subset=["text", "task", "label"]).copy()
        df["text"] = df["text"].astype(str).str.strip()
        df["task"] = df["task"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        df = df[(df["text"] != "") & (df["task"] != "") & (df["label"] != "")]

    elif family in FAMILY_PAIR_TEXT:
        required = {"text_a", "text_b", "task", "label"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{family}_{split_name} missing columns: {sorted(missing)}")
        df = df.dropna(subset=["text_a", "text_b", "task", "label"]).copy()
        df["text_a"] = df["text_a"].astype(str).str.strip()
        df["text_b"] = df["text_b"].astype(str).str.strip()
        df["task"] = df["task"].astype(str).str.strip()
        df["label"] = df["label"].astype(str).str.strip()
        df = df[
            (df["text_a"] != "") & (df["text_b"] != "") & (df["task"] != "") & (df["label"] != "")
        ]
    else:
        raise ValueError(f"Unknown family: {family}")

    if df.empty:
        raise ValueError(f"{family}_{split_name} is empty after cleanup.")
    return df.reset_index(drop=True)


def _encode_features(df: pd.DataFrame, family: str) -> list[str]:
    if family in FAMILY_SINGLE_TEXT:
        available_cols = set(df.columns)
        features: list[str] = []
        for row in df.itertuples(index=False):
            task = str(getattr(row, "task", ""))
            text = str(getattr(row, "text", ""))
            feature = f"task={task} [text] {text}"
            meta_tokens = _single_metadata_tokens(row, available_cols, task_name=task)
            if meta_tokens:
                feature += " [meta] " + " ".join(meta_tokens)
            features.append(feature)
        return features
    return [
        f"task={t} [a] {a} [b] {b}"
        for t, a, b in zip(df["task"], df["text_a"], df["text_b"], strict=False)
    ]


def _encode_targets(df: pd.DataFrame) -> list[str]:
    return [
        _composite_label(task, label) for task, label in zip(df["task"], df["label"], strict=False)
    ]


def _build_vectorizer(train_cfg: dict) -> TfidfVectorizer:
    max_features = max(1000, int(train_cfg["max_features"]))
    min_df = max(1, int(train_cfg["min_df"]))
    ngram_min = max(1, int(train_cfg.get("ngram_min", 1)))
    ngram_max = max(ngram_min, int(train_cfg.get("ngram_max", 2)))
    return TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )


def _build_sgd_classifier(
    train_cfg: dict, *, class_weight: dict[str, float] | str | None = None
) -> SGDClassifier:
    alpha = max(1e-8, float(train_cfg["alpha"]))
    seed = int(train_cfg["seed"])
    return SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        tol=None,
        shuffle=True,
        early_stopping=False,
        class_weight=class_weight,
        random_state=seed,
    )


def _build_pipeline(
    train_cfg: dict, *, class_weight: dict[str, float] | str | None = None
) -> Pipeline:
    return Pipeline(
        steps=[
            ("vectorizer", _build_vectorizer(train_cfg)),
            ("classifier", _build_sgd_classifier(train_cfg, class_weight=class_weight)),
        ]
    )


def _balanced_class_weight_map(targets: list[str]) -> dict[str, float]:
    import numpy as np

    observed_classes = sorted(set(targets))
    if not observed_classes:
        return {}
    weights = compute_class_weight(
        class_weight="balanced",
        classes=np.asarray(observed_classes, dtype=object),
        y=np.asarray(targets, dtype=object),
    )
    return {
        label: float(weight)
        for label, weight in zip(observed_classes, weights, strict=False)
    }


def _build_sgd_regressor(train_cfg: dict) -> SGDRegressor:
    alpha = max(1e-8, float(train_cfg["alpha"]))
    seed = int(train_cfg["seed"])
    return SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        tol=None,
        shuffle=True,
        random_state=seed,
    )


def _classification_metrics_for_matrix(
    classifier: SGDClassifier,
    matrix: Any,
    targets: list[str],
    *,
    classes: list[str],
) -> dict[str, float | None]:
    pred = classifier.predict(matrix).tolist()
    metrics: dict[str, float | None] = {
        "accuracy": float(accuracy_score(targets, pred)),
        "macro_f1": float(f1_score(targets, pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(targets, pred, average="weighted", zero_division=0)),
        "loss": None,
    }
    try:
        probs = classifier.predict_proba(matrix)
        metrics["loss"] = float(log_loss(targets, probs, labels=classes))
    except Exception:
        metrics["loss"] = None
    return metrics


def _improved_metric(
    value: float | None,
    best_value: float | None,
    *,
    metric_name: str,
    min_delta: float,
) -> bool:
    if value is None:
        return best_value is None
    if best_value is None:
        return True
    if metric_name in {"loss", "mae", "rmse", "ordinal_mae"}:
        return value < (best_value - min_delta)
    return value > (best_value + min_delta)


def _monitored_value(
    metrics: dict[str, float | None],
    *,
    metric_name: str,
    fallback_key: str,
) -> float | None:
    value = metrics.get(metric_name)
    if value is not None:
        return float(value)
    fallback = metrics.get(fallback_key)
    return None if fallback is None else float(fallback)


def _train_classifier_with_monitoring(
    *,
    train_x: list[str],
    train_y: list[str],
    valid_x: list[str],
    valid_y: list[str],
    train_cfg: dict,
    run_name: str,
    sample_weight: list[float] | np.ndarray | None = None,
) -> tuple[Any, list[dict[str, Any]], dict[str, Any]]:
    vectorizer = _build_vectorizer(train_cfg)
    x_train = vectorizer.fit_transform(train_x)
    x_valid = vectorizer.transform(valid_x) if valid_x else None
    classes = sorted(set(train_y) | set(valid_y))
    classifier = _build_sgd_classifier(
        train_cfg,
        class_weight=_balanced_class_weight_map(train_y) or None,
    )
    epochs = max(2, int(train_cfg["max_iter"]))
    metric_name = str(train_cfg.get("early_stopping_metric", "macro_f1") or "macro_f1").strip()
    patience = max(1, int(train_cfg.get("early_stopping_patience", 3)))
    min_delta = max(0.0, float(train_cfg.get("early_stopping_min_delta", 0.0)))
    use_early_stopping = bool(train_cfg.get("early_stopping", False)) and x_valid is not None and bool(valid_y)

    epoch_stats: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = 0
    best_classifier: SGDClassifier | None = None
    no_improve = 0

    sw = np.asarray(sample_weight, dtype=np.float64) if sample_weight is not None else None
    if sw is not None and (sw.size != len(train_y) or sw.ndim != 1):
        sw = None
    pbar = _progress(total=epochs, desc=f"Epochs {run_name}", unit="epoch")
    try:
        for epoch in range(1, epochs + 1):
            if epoch == 1:
                classifier.partial_fit(
                    x_train, train_y, classes=classes, sample_weight=sw
                )
            else:
                classifier.partial_fit(x_train, train_y, sample_weight=sw)

            train_metrics = _classification_metrics_for_matrix(
                classifier, x_train, train_y, classes=classes
            )
            valid_metrics = (
                _classification_metrics_for_matrix(classifier, x_valid, valid_y, classes=classes)
                if x_valid is not None and valid_y
                else {}
            )
            monitored = _monitored_value(
                valid_metrics if valid_metrics else train_metrics,
                metric_name=metric_name,
                fallback_key="macro_f1",
            )
            improved = _improved_metric(
                monitored, best_value, metric_name=metric_name, min_delta=min_delta
            )
            if improved:
                best_value = monitored
                best_epoch = epoch
                best_classifier = copy.deepcopy(classifier)
                no_improve = 0
            else:
                no_improve += 1

            stat = {
                "epoch": epoch,
                "train_loss": train_metrics.get("loss"),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "train_weighted_f1": train_metrics["weighted_f1"],
                "valid_loss": valid_metrics.get("loss"),
                "valid_accuracy": valid_metrics.get("accuracy"),
                "valid_macro_f1": valid_metrics.get("macro_f1"),
                "valid_weighted_f1": valid_metrics.get("weighted_f1"),
                "monitor_metric": metric_name,
                "monitor_value": monitored,
                "improved": bool(improved),
            }
            epoch_stats.append(stat)
            acc = train_metrics.get("accuracy") or 0.0
            val_acc = valid_metrics.get("accuracy", acc) or 0.0
            f1 = train_metrics.get("macro_f1") or 0.0
            val_f1 = valid_metrics.get("macro_f1", f1) or 0.0
            print(
                f"[{run_name}] epoch {epoch}/{epochs} | "
                f"train_acc={float(acc):.4f} "
                f"valid_acc={float(val_acc):.4f} "
                f"train_macro_f1={float(f1):.4f} "
                f"valid_macro_f1={float(val_f1):.4f}"
            )
            pbar.update(1)
            if use_early_stopping and no_improve >= patience:
                break
    finally:
        pbar.close()

    chosen = best_classifier if best_classifier is not None else classifier
    model = Pipeline(steps=[("vectorizer", vectorizer), ("classifier", chosen)])
    summary = {
        "actual_epochs": len(epoch_stats),
        "best_epoch": best_epoch or len(epoch_stats),
        "best_metric": best_value,
        "monitor_metric": metric_name,
        "early_stopped": bool(use_early_stopping and len(epoch_stats) < epochs),
    }
    return model, epoch_stats, summary


def _regression_metrics_for_matrix(regressor: SGDRegressor, matrix: Any, targets: Any) -> dict[str, float]:
    import numpy as np

    pred = np.asarray(regressor.predict(matrix), dtype=float)
    return {
        "mae": float(mean_absolute_error(targets, pred)),
        "rmse": float(np.sqrt(mean_squared_error(targets, pred))),
    }


def _train_regressor_with_monitoring(
    *,
    train_x: list[str],
    train_y: Any,
    valid_x: list[str],
    valid_y: Any,
    train_cfg: dict,
    run_name: str,
) -> tuple[Pipeline, list[dict[str, Any]], dict[str, Any]]:
    vectorizer = _build_vectorizer(train_cfg)
    regressor = _build_sgd_regressor(train_cfg)
    x_train = vectorizer.fit_transform(train_x)
    x_valid = vectorizer.transform(valid_x) if valid_x else None
    epochs = max(2, int(train_cfg["max_iter"]))
    metric_name = str(train_cfg.get("early_stopping_metric", "mae") or "mae").strip()
    if metric_name not in {"mae", "rmse"}:
        metric_name = "mae"
    patience = max(1, int(train_cfg.get("early_stopping_patience", 3)))
    min_delta = max(0.0, float(train_cfg.get("early_stopping_min_delta", 0.0)))
    use_early_stopping = bool(train_cfg.get("early_stopping", False)) and x_valid is not None and len(valid_y) > 0

    epoch_stats: list[dict[str, Any]] = []
    best_value: float | None = None
    best_epoch = 0
    best_regressor: SGDRegressor | None = None
    no_improve = 0

    pbar = _progress(total=epochs, desc=f"Epochs {run_name}", unit="epoch")
    try:
        for epoch in range(1, epochs + 1):
            regressor.partial_fit(x_train, train_y)
            train_metrics = _regression_metrics_for_matrix(regressor, x_train, train_y)
            valid_metrics = (
                _regression_metrics_for_matrix(regressor, x_valid, valid_y)
                if x_valid is not None and len(valid_y) > 0
                else {}
            )
            monitored = _monitored_value(
                cast("dict[str, float | None]", valid_metrics if valid_metrics else train_metrics),
                metric_name=metric_name,
                fallback_key="mae",
            )
            improved = _improved_metric(
                monitored, best_value, metric_name=metric_name, min_delta=min_delta
            )
            if improved:
                best_value = monitored
                best_epoch = epoch
                best_regressor = copy.deepcopy(regressor)
                no_improve = 0
            else:
                no_improve += 1
            epoch_stats.append(
                {
                    "epoch": epoch,
                    "train_mae": train_metrics["mae"],
                    "train_rmse": train_metrics["rmse"],
                    "valid_mae": valid_metrics.get("mae"),
                    "valid_rmse": valid_metrics.get("rmse"),
                    "monitor_metric": metric_name,
                    "monitor_value": monitored,
                    "improved": bool(improved),
                }
            )
            print(
                f"[{run_name}] epoch {epoch}/{epochs} | "
                f"train_mae={train_metrics['mae']:.4f} "
                f"valid_mae={float(valid_metrics.get('mae', train_metrics['mae'])):.4f}"
            )
            pbar.update(1)
            if use_early_stopping and no_improve >= patience:
                break
    finally:
        pbar.close()

    chosen = best_regressor if best_regressor is not None else regressor
    model = Pipeline(steps=[("vectorizer", vectorizer), ("regressor", chosen)])
    summary = {
        "actual_epochs": len(epoch_stats),
        "best_epoch": best_epoch or len(epoch_stats),
        "best_metric": best_value,
        "monitor_metric": metric_name,
        "early_stopped": bool(use_early_stopping and len(epoch_stats) < epochs),
    }
    return model, epoch_stats, summary


def _predict_batched(
    model: Any, features: list[str], *, batch_size: int, desc: str
) -> list[str]:
    pbar = _progress(total=len(features), desc=desc, unit="row")
    pred: list[str] = []
    try:
        for start in range(0, len(features), batch_size):
            batch = features[start : start + batch_size]
            pred.extend(model.predict(batch).tolist())
            pbar.update(len(batch))
    finally:
        pbar.close()
    return pred


def _predict_proba_batched(
    model: Any, features: list[str], *, batch_size: int
) -> list[list[float]] | None:
    if not hasattr(model, "predict_proba"):
        return None
    probs: list[list[float]] = []
    for start in range(0, len(features), batch_size):
        batch = features[start : start + batch_size]
        rows = model.predict_proba(batch)
        if hasattr(rows, "tolist"):
            rows = rows.tolist()
        probs.extend([[float(v) for v in row] for row in rows])
    return probs


def _model_classes(model: Any) -> list[str]:
    classes = getattr(model, "classes_", None)
    if classes is not None:
        return [str(x) for x in classes]
    classifier = getattr(model, "named_steps", {}).get("classifier")
    if classifier is not None:
        return [str(x) for x in getattr(classifier, "classes_", [])]
    return []


def _metric_block(
    y_true: list[str], y_pred: list[str], labels_for_cm: list[str] | None = None
) -> dict:
    labels = labels_for_cm if labels_for_cm is not None else sorted(set(y_true))
    return {
        "rows": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "labels": labels,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    }


def _positive_class_scores(
    model: Any,
    features: list[str],
    *,
    positive_class: str,
) -> list[float]:
    classes = _model_classes(model)
    if hasattr(model, "predict_proba") and positive_class in classes:
        probs = model.predict_proba(features)
        idx = classes.index(positive_class)
        return [float(row[idx]) for row in probs]
    if hasattr(model, "decision_function"):
        raw = model.decision_function(features)
        if hasattr(raw, "tolist"):
            raw = raw.tolist()
        if isinstance(raw, list):
            return [float(x) for x in raw]
    pred = model.predict(features).tolist()
    return [1.0 if str(x) == positive_class else 0.0 for x in pred]


def _ranking_metrics_from_groups(
    df: pd.DataFrame,
    scores: list[float],
    *,
    positive_label: str,
    k: int = 10,
) -> dict[str, float] | None:
    if "group_id" not in df.columns or not scores:
        return None

    scored = df.copy()
    scored["__score"] = scores
    valid_groups = []
    for _, group in scored.groupby("group_id", sort=False):
        labels = group["label"].astype(str).tolist()
        positives = sum(1 for label in labels if label == positive_label)
        if len(group) <= 1 or positives == 0:
            continue
        valid_groups.append(group.sort_values("__score", ascending=False))

    if not valid_groups:
        return None

    mrr_total = 0.0
    ndcg_total = 0.0
    recall_total = 0.0
    for group in valid_groups:
        labels = group["label"].astype(str).tolist()
        ranked = labels[:k]
        positive_positions = [
            idx for idx, label in enumerate(ranked, start=1) if label == positive_label
        ]
        if positive_positions:
            mrr_total += 1.0 / positive_positions[0]
        gains = [1.0 if label == positive_label else 0.0 for label in ranked]
        dcg = sum(
            gain / (1.0 if idx == 1 else math.log2(idx + 1))
            for idx, gain in enumerate(gains, start=1)
        )
        ideal_positives = min(sum(1 for label in labels if label == positive_label), k)
        idcg = sum(
            1.0 / (1.0 if idx == 1 else math.log2(idx + 1)) for idx in range(1, ideal_positives + 1)
        )
        ndcg_total += (dcg / idcg) if idcg > 0 else 0.0
        recall_total += sum(1 for label in ranked if label == positive_label) / max(
            1, sum(1 for label in labels if label == positive_label)
        )

    groups = len(valid_groups)
    return {
        "mrr@10": float(mrr_total / groups),
        "ndcg@10": float(ndcg_total / groups),
        "recall@10": float(recall_total / groups),
        "groups_evaluated": int(groups),
    }


def _calibration_error(
    y_true: list[str],
    y_pred: list[str],
    y_prob: list[list[float]] | None,
    *,
    n_bins: int = 10,
) -> float | None:
    """Compute Expected Calibration Error (ECE)."""
    if y_prob is None or not y_prob:
        return None
    import numpy as np

    confidences = np.array([max(row) for row in y_prob])
    accuracies = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred, strict=True)])
    bin_edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        mask = (confidences > bin_edges[i]) & (confidences <= bin_edges[i + 1])
        if not mask.any():
            continue
        bin_acc = accuracies[mask].mean()
        bin_conf = confidences[mask].mean()
        ece += mask.sum() * abs(bin_acc - bin_conf)
    return float(ece / max(1, len(y_true)))


def _maybe_calibrate_classifier(
    model: Any,
    *,
    calibration_x: list[str],
    calibration_y: list[str],
    train_cfg: dict,
    predict_batch_size: int,
) -> tuple[Any, dict[str, Any] | None]:
    method = str(train_cfg.get("calibration_method", "") or "").strip().lower()
    if not method or not calibration_x or not calibration_y:
        return model, None

    pre_pred = _predict_batched(
        model, calibration_x, batch_size=predict_batch_size, desc="Calibration pre-check"
    )
    pre_probs = _predict_proba_batched(model, calibration_x, batch_size=predict_batch_size)
    pre_ece = _calibration_error(calibration_y, pre_pred, pre_probs)
    pre_acc = float(accuracy_score(calibration_y, pre_pred))

    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(calibration_x, calibration_y)

    post_pred = _predict_batched(
        calibrated, calibration_x, batch_size=predict_batch_size, desc="Calibration post-check"
    )
    post_probs = _predict_proba_batched(calibrated, calibration_x, batch_size=predict_batch_size)
    post_ece = _calibration_error(calibration_y, post_pred, post_probs)
    post_acc = float(accuracy_score(calibration_y, post_pred))

    summary = {
        "method": method,
        "split": str(train_cfg.get("calibration_split", "eval") or "eval"),
        "rows": len(calibration_y),
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "pre_accuracy": pre_acc,
        "post_accuracy": post_acc,
        "accuracy_delta": float(post_acc - pre_acc),
    }
    return calibrated, summary


def _maybe_calibrate_matrix_classifier(
    model: Any,
    *,
    calibration_x: Any,
    calibration_y: list[str],
    train_cfg: dict,
) -> tuple[Any, dict[str, Any] | None]:
    method = str(train_cfg.get("calibration_method", "") or "").strip().lower()
    if not method or calibration_x is None or not calibration_y:
        return model, None

    pre_pred = model.predict(calibration_x).tolist()
    pre_probs = model.predict_proba(calibration_x).tolist()
    pre_ece = _calibration_error(calibration_y, pre_pred, pre_probs)
    pre_acc = float(accuracy_score(calibration_y, pre_pred))

    calibrated = CalibratedClassifierCV(model, method=method, cv="prefit")
    calibrated.fit(calibration_x, calibration_y)

    post_pred = calibrated.predict(calibration_x).tolist()
    post_probs = calibrated.predict_proba(calibration_x).tolist()
    post_ece = _calibration_error(calibration_y, post_pred, post_probs)
    post_acc = float(accuracy_score(calibration_y, post_pred))

    summary = {
        "method": method,
        "split": str(train_cfg.get("calibration_split", "eval") or "eval"),
        "rows": len(calibration_y),
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "pre_accuracy": pre_acc,
        "post_accuracy": post_acc,
        "accuracy_delta": float(post_acc - pre_acc),
    }
    return calibrated, summary


def _task_label_indices(classes: list[str]) -> dict[str, list[int]]:
    out: dict[str, list[int]] = {}
    for idx, class_name in enumerate(classes):
        task, _ = _split_composite(str(class_name))
        out.setdefault(task, []).append(idx)
    return out


def _maybe_calibrate_task_conditional_classifier(
    model: Any,
    *,
    calibration_x: list[str],
    calibration_y: list[str],
    train_cfg: dict,
) -> tuple[Any, dict[str, Any] | None]:
    method = str(train_cfg.get("calibration_method", "") or "").strip().lower()
    if not method or not calibration_x or not calibration_y:
        return model, None

    import numpy as np

    classes = _model_classes(model)
    if not classes:
        return model, None

    raw_probs = np.asarray(model.predict_proba(calibration_x), dtype=np.float64)
    pre_pred = model.predict(calibration_x).tolist()
    pre_ece = _calibration_error(calibration_y, pre_pred, raw_probs.tolist())
    pre_acc = float(accuracy_score(calibration_y, pre_pred))

    parsed = [_split_composite(value) for value in calibration_y]
    task_indices = _task_label_indices(classes)
    calibrators: dict[str, Any] = {}
    calibrator_classes: dict[str, list[int]] = {}
    task_summaries: dict[str, Any] = {}

    for task_name, indices in task_indices.items():
        row_ids = [idx for idx, (task, _) in enumerate(parsed) if task == task_name]
        if not row_ids or len(indices) <= 1:
            continue
        local_labels = [_split_composite(calibration_y[idx])[1] for idx in row_ids]
        label_to_local = {
            _split_composite(classes[class_idx])[1]: local_idx
            for local_idx, class_idx in enumerate(indices)
        }
        if any(label not in label_to_local for label in local_labels):
            continue
        task_probs = raw_probs[np.ix_(row_ids, indices)]
        task_probs = task_probs / np.maximum(task_probs.sum(axis=1, keepdims=True), 1e-12)
        y_local = np.asarray([label_to_local[label] for label in local_labels], dtype=int)
        pre_local_pred = task_probs.argmax(axis=1)
        pre_local_ece = _calibration_error(
            [str(value) for value in y_local],
            [str(value) for value in pre_local_pred.tolist()],
            task_probs.tolist(),
        )
        pre_local_acc = float(accuracy_score(y_local, pre_local_pred))
        if len(set(y_local.tolist())) < 2 or len(row_ids) < max(30, len(indices) * 4):
            task_summaries[task_name] = {
                "rows": len(row_ids),
                "skipped": True,
                "pre_ece": pre_local_ece,
                "post_ece": pre_local_ece,
                "pre_accuracy": pre_local_acc,
                "post_accuracy": pre_local_acc,
            }
            continue

        calibrator = LogisticRegression(
            max_iter=500,
            class_weight="balanced",
            multi_class="auto",
            random_state=int(train_cfg["seed"]),
        )
        cal_x = build_task_conditional_calibration_features(task_probs)
        calibrator.fit(cal_x, y_local)
        post_raw = np.asarray(calibrator.predict_proba(cal_x), dtype=np.float64)
        aligned = np.zeros_like(task_probs)
        local_classes = [int(value) for value in getattr(calibrator, "classes_", [])]
        for src_idx, class_idx in enumerate(local_classes):
            if 0 <= class_idx < aligned.shape[1]:
                aligned[:, class_idx] = post_raw[:, src_idx]
        aligned = aligned / np.maximum(aligned.sum(axis=1, keepdims=True), 1e-12)
        post_local_pred = aligned.argmax(axis=1)
        post_local_ece = _calibration_error(
            [str(value) for value in y_local],
            [str(value) for value in post_local_pred.tolist()],
            aligned.tolist(),
        )
        post_local_acc = float(accuracy_score(y_local, post_local_pred))
        calibrators[task_name] = calibrator
        calibrator_classes[task_name] = local_classes
        task_summaries[task_name] = {
            "rows": len(row_ids),
            "pre_ece": pre_local_ece,
            "post_ece": post_local_ece,
            "pre_accuracy": pre_local_acc,
            "post_accuracy": post_local_acc,
            "accuracy_delta": float(post_local_acc - pre_local_acc),
        }

    wrapped = TaskConditionalCalibratedClassifier(
        base_model=model,
        classes_=classes,
        task_label_indices=task_indices,
        calibrators=calibrators,
        calibrator_classes=calibrator_classes,
    )
    post_pred = wrapped.predict(calibration_x).tolist()
    post_probs = wrapped.predict_proba(calibration_x).tolist()
    post_ece = _calibration_error(calibration_y, post_pred, post_probs)
    post_acc = float(accuracy_score(calibration_y, post_pred))

    summary = {
        "method": f"task_conditional_{method}",
        "split": str(train_cfg.get("calibration_split", "eval") or "eval"),
        "rows": len(calibration_y),
        "pre_ece": pre_ece,
        "post_ece": post_ece,
        "pre_accuracy": pre_acc,
        "post_accuracy": post_acc,
        "accuracy_delta": float(post_acc - pre_acc),
        "tasks": task_summaries,
    }
    return wrapped, summary


def _with_calibration_payload(
    metrics: dict[str, Any] | None,
    calibration_summary: dict[str, Any] | None,
) -> dict[str, Any] | None:
    if metrics is None or calibration_summary is None:
        return metrics
    payload = copy.deepcopy(metrics)
    payload["calibration"] = copy.deepcopy(calibration_summary)
    return payload


def _evaluate(
    model: Any, df: pd.DataFrame, *, family: str, split_name: str, predict_batch_size: int
) -> tuple[dict, dict]:
    x = _encode_features(df, family)
    y_true = _encode_targets(df)
    y_pred = _predict_batched(
        model, x, batch_size=predict_batch_size, desc=f"Predict {family}:{split_name}"
    )

    overall_labels = sorted(set(y_true) | set(y_pred))
    overall_metrics = _metric_block(y_true, y_pred, overall_labels)

    cal_error = _calibration_error(
        y_true,
        y_pred,
        _predict_proba_batched(model, x, batch_size=predict_batch_size),
    )
    overall_metrics["calibration_error"] = cal_error

    overall_report = classification_report(
        y_true,
        y_pred,
        labels=overall_labels,
        output_dict=True,
        zero_division=0,
    )

    y_true_task, y_true_label = zip(*[_split_composite(v) for v in y_true], strict=False)
    y_pred_task, y_pred_label = zip(*[_split_composite(v) for v in y_pred], strict=False)
    tasks = sorted(set(y_true_task))

    per_task_metrics: dict[str, dict] = {}
    per_task_reports: dict[str, dict] = {}
    for task in tasks:
        idx = [i for i, t in enumerate(y_true_task) if t == task]
        task_true = [y_true_label[i] for i in idx]
        task_pred: list[str] = []
        wrong_task = 0
        for i in idx:
            if y_pred_task[i] != task:
                task_pred.append("__wrong_task__")
                wrong_task += 1
            else:
                task_pred.append(y_pred_label[i])

        labels_for_cm = sorted(set(task_true))
        if "__wrong_task__" in task_pred:
            labels_for_cm = [*labels_for_cm, "__wrong_task__"]

        task_metrics = _metric_block(task_true, task_pred, labels_for_cm)
        task_metrics["wrong_task_predictions"] = int(wrong_task)
        task_metrics["wrong_task_rate"] = float(wrong_task / max(1, len(task_true)))
        per_task_metrics[task] = task_metrics

        report_labels = sorted(set(task_true))
        per_task_reports[task] = classification_report(
            task_true,
            task_pred,
            labels=report_labels,
            output_dict=True,
            zero_division=0,
        )

    metrics = {
        "family": family,
        "split": split_name,
        "overall": overall_metrics,
        "per_task": per_task_metrics,
    }
    reports = {
        "family": family,
        "split": split_name,
        "overall": overall_report,
        "per_task": per_task_reports,
    }
    return metrics, reports


def _train_one_family(
    family: str, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    print(f"Training family: {family}")
    train_df = _load_split(prepared_dir, family, "train")
    test_df = _load_split(prepared_dir, family, "test")
    eval_df = _load_split(prepared_dir, family, "eval")

    task_counts = train_df.groupby("task").size().to_dict()
    print(f"[{family}] task rows (train): {task_counts}")

    train_x = _encode_features(train_df, family)
    train_y = _encode_targets(train_df)
    eval_x = _encode_features(eval_df, family)
    eval_y = _encode_targets(eval_df)
    epochs = max(2, int(train_cfg["max_iter"]))
    print(f"Fitting {family} on {len(train_df)} rows for {epochs} epochs...")
    model, epoch_stats, training_summary = _train_classifier_with_monitoring(
        train_x=train_x,
        train_y=train_y,
        valid_x=eval_x,
        valid_y=eval_y,
        train_cfg=train_cfg,
        run_name=family,
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    model, calibration_summary = _maybe_calibrate_task_conditional_classifier(
        model,
        calibration_x=eval_x,
        calibration_y=eval_y,
        train_cfg=train_cfg,
    )
    metrics_test, reports_test = _evaluate(
        model, test_df, family=family, split_name="test", predict_batch_size=batch_size
    )
    metrics_eval, reports_eval = _evaluate(
        model, eval_df, family=family, split_name="eval", predict_batch_size=batch_size
    )

    all_labels = sorted(
        set(train_y) | set(_encode_targets(test_df)) | set(_encode_targets(eval_df))
    )
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {str(i): label for label, i in label_to_id.items()}

    metadata = {
        "family": family,
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "rows": {"train": len(train_df), "test": len(test_df), "eval": len(eval_df)},
        "tasks": sorted(set(train_df["task"]) | set(test_df["task"]) | set(eval_df["task"])),
        "task_label_counts": {
            "train": _task_label_counts(train_df),
            "test": _task_label_counts(test_df),
            "eval": _task_label_counts(eval_df),
        },
        "labels": all_labels,
        "label_to_id": label_to_id,
        "train_config": {
            "max_features": int(train_cfg["max_features"]),
            "min_df": int(train_cfg["min_df"]),
            "ngram_min": int(train_cfg.get("ngram_min", 1)),
            "ngram_max": int(train_cfg.get("ngram_max", 2)),
            "max_iter": epochs,
            "alpha": float(train_cfg["alpha"]),
            "seed": int(train_cfg["seed"]),
            "predict_batch_size": int(train_cfg["predict_batch_size"]),
            "early_stopping": bool(train_cfg.get("early_stopping", False)),
            "early_stopping_patience": int(train_cfg.get("early_stopping_patience", 3)),
            "early_stopping_metric": str(train_cfg.get("early_stopping_metric", "macro_f1")),
            "early_stopping_min_delta": float(train_cfg.get("early_stopping_min_delta", 0.0)),
            "calibration_method": str(train_cfg.get("calibration_method", "") or ""),
        },
        "training_summary": training_summary,
        "calibration": calibration_summary,
        "epoch_stats": epoch_stats,
    }
    metrics_test = _with_calibration_payload(metrics_test, calibration_summary) or metrics_test
    metrics_eval = _with_calibration_payload(metrics_eval, calibration_summary) or metrics_eval

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{family}_model.joblib"
    joblib.dump(
        {
            "model": model,
            "metadata": metadata,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        },
        model_path,
    )
    _write_json(
        output_dir / f"{family}_label_map.json",
        {"label_to_id": label_to_id, "id_to_label": id_to_label},
    )
    _write_json(output_dir / f"{family}_metrics_test.json", metrics_test)
    _write_json(output_dir / f"{family}_metrics_eval.json", metrics_eval)
    _write_json(output_dir / f"{family}_report_test.json", reports_test)
    _write_json(output_dir / f"{family}_report_eval.json", reports_eval)
    _write_json(
        output_dir / f"{family}_epoch_stats.json", {"family": family, "epoch_stats": epoch_stats}
    )
    _write_json(output_dir / f"{family}_training_metadata.json", metadata)

    return {
        "family": family,
        "model_path": str(model_path),
        "rows": metadata["rows"],
        "test": metrics_test["overall"],
        "eval": metrics_eval["overall"],
        "actual_epochs": training_summary["actual_epochs"],
        "best_epoch": training_summary["best_epoch"],
        "early_stopped": training_summary["early_stopped"],
        "calibration": calibration_summary,
    }


def _load_task_splits(
    prepared_dir: Path, family: str, task_name: str
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_df = _load_split(prepared_dir, family, "train")
    test_df = _load_split(prepared_dir, family, "test")
    eval_df = _load_split(prepared_dir, family, "eval")
    return (
        train_df[train_df["task"] == task_name].reset_index(drop=True),
        test_df[test_df["task"] == task_name].reset_index(drop=True),
        eval_df[eval_df["task"] == task_name].reset_index(drop=True),
    )


def _load_regression_task_split(prepared_dir: Path, family: str, split_name: str, task_name: str) -> pd.DataFrame:
    path = prepared_dir / f"{family}_{split_name}.parquet"
    if not path.exists():
        return pd.DataFrame()
    df = pd.read_parquet(path)
    if "task" not in df.columns:
        return pd.DataFrame()
    df = df[df["task"].astype(str) == task_name].reset_index(drop=True)
    if df.empty:
        return df
    score_col = "score" if "score" in df.columns else "label"
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    return df.dropna(subset=[score_col]).reset_index(drop=True)


def _write_task_metrics(
    output_dir: Path,
    artifact_name: str,
    *,
    metrics_test: dict[str, Any] | None = None,
    metrics_eval: dict[str, Any] | None = None,
    adversarial_metrics: dict[str, Any] | None = None,
    calibration: dict[str, Any] | None = None,
) -> None:
    metrics_test = _with_calibration_payload(metrics_test, calibration)
    metrics_eval = _with_calibration_payload(metrics_eval, calibration)
    if metrics_test is not None:
        _write_json(output_dir / f"{artifact_name}_metrics_test.json", metrics_test)
    if metrics_eval is not None:
        _write_json(output_dir / f"{artifact_name}_metrics_eval.json", metrics_eval)
    if adversarial_metrics is not None:
        _write_json(output_dir / f"{artifact_name}_metrics_adversarial.json", adversarial_metrics)


def _load_adversarial_fixture(path: Path, *, task_name: str) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            item = json.loads(text)
            row = {"task": task_name, **item}
            rows.append(row)
    return pd.DataFrame(rows)


def _evaluate_adversarial_task(
    *,
    spec: TaskSpec,
    model: Any,
    family: str,
    predict_batch_size: int,
) -> dict[str, Any] | None:
    fixture_path = _ADVERSARIAL_FIXTURES.get(spec.task_name)
    if fixture_path is None:
        return None
    heldout_path = fixture_path.parent / f"{fixture_path.stem}_heldout{fixture_path.suffix}"
    path_to_load = heldout_path if heldout_path.exists() else fixture_path
    df = _load_adversarial_fixture(path_to_load, task_name=spec.task_name)
    if df.empty:
        return None
    metrics, _ = _evaluate(
        model,
        df,
        family=family,
        split_name="adversarial",
        predict_batch_size=predict_batch_size,
    )
    return metrics


def _ordinal_metric_block(
    *,
    true_labels: list[str],
    pred_labels: list[str],
    label_order: list[str],
) -> dict[str, Any]:
    index = {label: idx for idx, label in enumerate(label_order)}
    base = _metric_block(true_labels, pred_labels, label_order)
    true_idx = [index[label] for label in true_labels]
    pred_idx = [index[label] for label in pred_labels]
    diffs = [abs(a - b) for a, b in zip(true_idx, pred_idx, strict=False)]
    base["ordinal_mae"] = float(sum(diffs) / max(1, len(diffs)))
    base["off_by_two_rate"] = float(sum(1 for d in diffs if d >= 2) / max(1, len(diffs)))
    return base


def _load_pair_embedding_cache(prepared_dir: Path) -> tuple[dict[str, Any], str]:
    path = pair_embedding_cache_path(prepared_dir)
    if not path.exists():
        raise FileNotFoundError(f"Missing pair embedding cache: {path}")
    df = pd.read_parquet(path)
    if df.empty or "text_hash" not in df.columns or "embedding" not in df.columns:
        raise ValueError(f"Invalid pair embedding cache schema: {path}")
    cache = {
        str(row["text_hash"]): row["embedding"]
        for row in df.to_dict(orient="records")
        if row.get("text_hash") is not None and row.get("embedding") is not None
    }
    model_names = [
        str(value)
        for value in df.get("embedding_model_name", pd.Series(dtype=str)).astype(str).tolist()
        if str(value).strip()
    ]
    model_name = model_names[0] if model_names else "sentence-transformers/all-MiniLM-L6-v2"
    return cache, model_name


def _dense_pair_matrix(df: pd.DataFrame, cache: dict[str, Any]) -> Any:
    rows = []
    for item in df.itertuples(index=False):
        text_a = str(getattr(item, "text_a", ""))
        text_b = str(getattr(item, "text_b", ""))
        hash_a = hash_text(text_a)
        hash_b = hash_text(text_b)
        emb_a = cache.get(hash_a)
        emb_b = cache.get(hash_b)
        if emb_a is None or emb_b is None:
            raise KeyError("Missing embedding cache entry for pair row.")
        rows.append(
            build_pair_dense_features(
                emb_a,
                emb_b,
                text_a=text_a,
                text_b=text_b,
            )
        )
    if not rows:
        raise ValueError("No dense pair rows available.")
    import numpy as np

    return np.vstack(rows).astype("float32", copy=False)


def _pair_group_top1_predictions(
    df: pd.DataFrame,
    probs: Any,
    *,
    task_name: str,
    classes: list[str],
) -> list[str]:
    relevant_label = _composite_label(task_name, "relevant")
    not_relevant_label = _composite_label(task_name, "not_relevant")
    if (
        task_name not in _PAIR_GROUP_TOP1_TASKS
        or relevant_label not in classes
        or not_relevant_label not in classes
        or "text_a" not in df.columns
    ):
        return [classes[int(idx)] for idx in probs.argmax(axis=1)]

    pos_idx = classes.index(relevant_label)
    pred = [not_relevant_label for _ in range(len(df))]
    grouped = df.reset_index(drop=True).groupby("text_a", sort=False).indices
    for row_indices in grouped.values():
        rows = [int(idx) for idx in row_indices]
        if len(rows) <= 1:
            row_idx = rows[0]
            pred[row_idx] = classes[int(probs[row_idx].argmax())]
            continue
        best_idx = max(rows, key=lambda row_idx: float(probs[row_idx, pos_idx]))
        pred[best_idx] = relevant_label
    return pred


def _memory_type_macro(label: str) -> str:
    cleaned = str(label).strip()
    for macro, labels in _MEMORY_TYPE_MACRO_GROUPS.items():
        if cleaned in labels:
            return macro
    return "personal"


def _encode_memory_type_features(df: pd.DataFrame, family: str) -> list[str]:
    base = _encode_features(df, family)
    enriched: list[str] = []
    for feature, row in zip(base, df.itertuples(index=False), strict=False):
        hints = derive_memory_type_feature_tokens_from_row(row)
        enriched.append(feature if not hints else feature + " [hint] " + " ".join(hints))
    return enriched


def _train_classification_task(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train a dedicated TF-IDF+SGD classifier for a single task subset."""
    print(f"[task:{spec.task_name}] classification training")
    family = spec.family
    train_df, test_df, eval_df = _load_task_splits(prepared_dir, family, spec.task_name)

    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    train_x = _encode_features(train_df, family)
    train_y = _encode_targets(train_df)
    eval_x = _encode_features(eval_df, family)
    eval_y = _encode_targets(eval_df)

    model, epoch_stats, training_summary = _train_classifier_with_monitoring(
        train_x=train_x,
        train_y=train_y,
        valid_x=eval_x,
        valid_y=eval_y,
        train_cfg=train_cfg,
        run_name=spec.task_name,
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    model, calibration_summary = _maybe_calibrate_classifier(
        model,
        calibration_x=eval_x,
        calibration_y=eval_y,
        train_cfg=train_cfg,
        predict_batch_size=batch_size,
    )
    metrics_test, _ = _evaluate(
        model, test_df, family=family, split_name="test", predict_batch_size=batch_size
    )
    metrics_eval, _ = _evaluate(
        model, eval_df, family=family, split_name="eval", predict_batch_size=batch_size
    )
    adversarial_metrics = _evaluate_adversarial_task(
        spec=spec,
        model=model,
        family=family,
        predict_batch_size=batch_size,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    all_labels = sorted(set(train_y))
    joblib.dump(
        {"model": model, "labels": all_labels, "task_spec": spec.__dict__},
        model_path,
    )
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {
            "task": spec.task_name,
            "epoch_stats": epoch_stats,
            "training_summary": training_summary,
            "calibration": calibration_summary,
        },
    )
    _write_task_metrics(
        output_dir,
        spec.artifact_name,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
        adversarial_metrics=adversarial_metrics,
        calibration=calibration_summary,
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "trainer": _resolved_trainer(spec),
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": metrics_test["overall"],
        "eval": metrics_eval["overall"],
        "actual_epochs": training_summary["actual_epochs"],
        "best_epoch": training_summary["best_epoch"],
        "early_stopped": training_summary["early_stopped"],
        "calibration": calibration_summary,
        "adversarial_metrics": adversarial_metrics.get("overall") if adversarial_metrics else None,
    }


def _train_pair_ranking(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train a TF-IDF+SGD model for pair ranking using predict_proba as ranking score."""
    print(f"[task:{spec.task_name}] pair_ranking training")
    family = spec.family
    train_df, test_df, eval_df = _load_task_splits(prepared_dir, family, spec.task_name)

    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    train_x = _encode_features(train_df, family)
    train_y = _encode_targets(train_df)
    eval_x = _encode_features(eval_df, family)
    eval_y = _encode_targets(eval_df)

    model, epoch_stats, training_summary = _train_classifier_with_monitoring(
        train_x=train_x,
        train_y=train_y,
        valid_x=eval_x,
        valid_y=eval_y,
        train_cfg=train_cfg,
        run_name=spec.task_name,
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    model, calibration_summary = _maybe_calibrate_classifier(
        model,
        calibration_x=eval_x,
        calibration_y=eval_y,
        train_cfg=train_cfg,
        predict_batch_size=batch_size,
    )
    metrics_test, _ = _evaluate(
        model, test_df, family=family, split_name="test", predict_batch_size=batch_size
    )
    metrics_eval, _ = _evaluate(
        model, eval_df, family=family, split_name="eval", predict_batch_size=batch_size
    )

    for df, metrics in ((test_df, metrics_test), (eval_df, metrics_eval)):
        split_x = _encode_features(df, family)
        ranking = _ranking_metrics_from_groups(
            df,
            _positive_class_scores(
                model,
                split_x,
                positive_class=_composite_label(spec.task_name, "relevant"),
            ),
            positive_label="relevant",
        )
        if ranking is not None:
            metrics["overall"].update(ranking)
        else:
            metrics["overall"]["ranking_proxy"] = {
                "note": "Classification metrics used as ranking proxy; true MRR/NDCG require grouped candidate lists",
                "proxy_accuracy": metrics["overall"].get("accuracy", 0.0),
                "proxy_f1": metrics["overall"].get("macro_f1", 0.0),
            }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    all_labels = sorted(set(train_y))
    joblib.dump(
        {"model": model, "labels": all_labels, "task_spec": spec.__dict__},
        model_path,
    )
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {
            "task": spec.task_name,
            "epoch_stats": epoch_stats,
            "training_summary": training_summary,
            "calibration": calibration_summary,
        },
    )
    _write_task_metrics(
        output_dir,
        spec.artifact_name,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
        calibration=calibration_summary,
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "trainer": _resolved_trainer(spec),
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": metrics_test["overall"],
        "eval": metrics_eval["overall"],
        "actual_epochs": training_summary["actual_epochs"],
        "best_epoch": training_summary["best_epoch"],
        "early_stopped": training_summary["early_stopped"],
        "calibration": calibration_summary,
    }


def _train_single_regression(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train an SGDRegressor on single-text features with a numeric target."""
    import numpy as np

    print(f"[task:{spec.task_name}] single_regression training")
    family = spec.family
    train_df = _load_regression_task_split(prepared_dir, family, "train", spec.task_name)
    test_df = _load_regression_task_split(prepared_dir, family, "test", spec.task_name)
    eval_df = _load_regression_task_split(prepared_dir, family, "eval", spec.task_name)

    if train_df.empty:
        print(f"[task:{spec.task_name}] no valid regression rows after filtering; skipping.")
        return {}

    score_col = "score" if "score" in train_df.columns else "label"
    train_y = train_df[score_col].values.astype(float)
    train_features = _encode_features(train_df, family)
    eval_y = eval_df[score_col].values.astype(float) if not eval_df.empty else np.asarray([])
    eval_features = _encode_features(eval_df, family) if not eval_df.empty else []
    model, epoch_stats, training_summary = _train_regressor_with_monitoring(
        train_x=train_features,
        train_y=train_y,
        valid_x=eval_features,
        valid_y=eval_y,
        train_cfg=train_cfg,
        run_name=spec.task_name,
    )

    vectorizer = model.named_steps["vectorizer"]
    regressor = model.named_steps["regressor"]
    x_train = vectorizer.transform(train_features)
    train_metrics = {
        "train_mae": float(mean_absolute_error(train_y, regressor.predict(x_train))),
        "train_rmse": float(np.sqrt(mean_squared_error(train_y, regressor.predict(x_train)))),
    }

    test_metrics: dict[str, Any] = dict(train_metrics)
    eval_metrics: dict[str, Any] = dict(train_metrics)
    if not test_df.empty:
        test_y = test_df[score_col].values.astype(float)
        x_test = vectorizer.transform(_encode_features(test_df, family))
        test_pred = regressor.predict(x_test)
        test_metrics["test_mae"] = float(mean_absolute_error(test_y, test_pred))
        test_metrics["test_rmse"] = float(np.sqrt(mean_squared_error(test_y, test_pred)))
    if not eval_df.empty:
        x_eval = vectorizer.transform(eval_features)
        eval_pred = regressor.predict(x_eval)
        eval_metrics["eval_mae"] = float(mean_absolute_error(eval_y, eval_pred))
        eval_metrics["eval_rmse"] = float(np.sqrt(mean_squared_error(eval_y, eval_pred)))
        test_metrics["eval_mae"] = eval_metrics["eval_mae"]
        test_metrics["eval_rmse"] = eval_metrics["eval_rmse"]

    train_mean = float(np.mean(train_y))
    train_median = float(np.median(train_y))
    data_profile = {
        "rows": len(train_y),
        "min": float(np.min(train_y)),
        "max": float(np.max(train_y)),
        "std": float(np.std(train_y)),
        "p05": float(np.percentile(train_y, 5)),
        "p50": float(np.percentile(train_y, 50)),
        "p95": float(np.percentile(train_y, 95)),
        "mean": train_mean,
        "median": train_median,
    }
    warnings: list[str] = []
    baselines: dict[str, Any] = {
        "mean_prediction": {"value": train_mean},
        "median_prediction": {"value": train_median},
    }
    if not test_df.empty:
        test_y = test_df[score_col].values.astype(float)
        baselines["mean_prediction"]["test_mae"] = float(
            mean_absolute_error(test_y, np.full_like(test_y, train_mean))
        )
        baselines["median_prediction"]["test_mae"] = float(
            mean_absolute_error(test_y, np.full_like(test_y, train_median))
        )
        if data_profile["std"] < 0.1:
            warnings.append("low_target_variance")
        if test_metrics.get("test_mae") is not None and baselines["mean_prediction"]["test_mae"] - float(test_metrics["test_mae"]) < 0.01:
            warnings.append("weak_improvement_over_mean_baseline")

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    joblib.dump({"model": model, "task_spec": spec.__dict__}, model_path)
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {"task": spec.task_name, "epoch_stats": epoch_stats, "training_summary": training_summary},
    )
    _write_json(
        output_dir / f"{spec.artifact_name}_metrics_test.json",
        {
            "task": spec.task_name,
            "metrics": test_metrics,
            "data_profile": data_profile,
            "baselines": baselines,
            "warnings": warnings,
        },
    )
    _write_json(
        output_dir / f"{spec.artifact_name}_metrics_eval.json",
        {
            "task": spec.task_name,
            "metrics": eval_metrics,
            "data_profile": data_profile,
            "baselines": baselines,
            "warnings": warnings,
        },
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "trainer": _resolved_trainer(spec),
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": test_metrics,
        "eval": eval_metrics,
        "metrics": test_metrics,
        "actual_epochs": training_summary["actual_epochs"],
        "best_epoch": training_summary["best_epoch"],
        "early_stopped": training_summary["early_stopped"],
        "data_profile": data_profile,
        "baselines": baselines,
        "warnings": warnings,
    }


def _train_embedding_pair(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train a dense classifier on cached sentence-embedding pair features."""
    print(f"[task:{spec.task_name}] embedding_pair training")
    family = spec.family
    train_df, test_df, eval_df = _load_task_splits(prepared_dir, family, spec.task_name)
    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    cache, default_model_name = _load_pair_embedding_cache(prepared_dir)
    train_y = _encode_targets(train_df)
    classes = sorted(set(train_y))
    sample_weight = compute_sample_weight(class_weight="balanced", y=train_y)
    classifier = HistGradientBoostingClassifier(
        learning_rate=0.05,
        max_depth=6,
        max_iter=300,
        min_samples_leaf=20,
        random_state=int(train_cfg["seed"]),
    )
    dense_train = _dense_pair_matrix(train_df, cache)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")
    with joblib.parallel_backend("threading", n_jobs=1):
        classifier.fit(dense_train, train_y, sample_weight=sample_weight)

    # Score margin refinement for pair_ranking: upweight groups with small positive-negative gap
    score_margin_threshold = 0.15
    if spec.task_name in _PAIR_GROUP_TOP1_TASKS and "group_id" in train_df.columns:
        rel_label = _composite_label(spec.task_name, "relevant")
        if rel_label in classes:
            rel_idx = classes.index(rel_label)
            probs = classifier.predict_proba(dense_train)
            train_df = train_df.reset_index(drop=True)
            hard_indices: set[int] = set()
            for _gid, group in train_df.groupby("group_id", sort=False):
                if _gid is None or str(_gid).strip() == "":
                    continue
                idx = group.index.tolist()
                labels = [train_y[i] for i in group.index]
                pos_idx = [i for i, lab in zip(idx, labels, strict=False) if lab == rel_label]
                neg_idx = [i for i, lab in zip(idx, labels, strict=False) if lab != rel_label]
                if not pos_idx or not neg_idx:
                    continue
                pos_scores = [float(probs[i, rel_idx]) for i in pos_idx]
                neg_scores = [float(probs[i, rel_idx]) for i in neg_idx]
                gap = min(pos_scores) - max(neg_scores)
                if gap < score_margin_threshold:
                    hard_indices.update(pos_idx)
                    hard_indices.update(neg_idx)
            if hard_indices:
                margin_weights = np.ones(len(train_df), dtype=np.float64)
                for i in hard_indices:
                    if 0 <= i < len(margin_weights):
                        margin_weights[i] = 2.0
                with joblib.parallel_backend("threading", n_jobs=1):
                    classifier.fit(
                        dense_train,
                        train_y,
                        sample_weight=np.asarray(margin_weights) * np.asarray(sample_weight),
                    )

    test_dense = _dense_pair_matrix(test_df, cache) if not test_df.empty else None
    eval_dense = _dense_pair_matrix(eval_df, cache) if not eval_df.empty else None
    calibration_y = _encode_targets(eval_df) if not eval_df.empty else []
    with joblib.parallel_backend("threading", n_jobs=1):
        classifier, calibration_summary = _maybe_calibrate_matrix_classifier(
            classifier,
            calibration_x=eval_dense,
            calibration_y=calibration_y,
            train_cfg=train_cfg,
        )

    def _dense_metrics(df: pd.DataFrame, dense: Any | None, estimator: Any) -> dict[str, Any]:
        if dense is None or df.empty:
            return {"overall": {"rows": 0}}
        true_y = _encode_targets(df)
        probs = estimator.predict_proba(dense)
        classes_for_probs = [str(value) for value in getattr(estimator, "classes_", [])]
        pred_y = _pair_group_top1_predictions(
            df,
            probs,
            task_name=spec.task_name,
            classes=classes_for_probs,
        )
        overall = _metric_block(true_y, pred_y)
        overall["calibration_error"] = _calibration_error(true_y, pred_y, probs.tolist())
        ranking = _ranking_metrics_from_groups(
            df,
            [
                float(row[classes_for_probs.index(_composite_label(spec.task_name, "relevant"))])
                for row in probs
            ]
            if _composite_label(spec.task_name, "relevant") in classes_for_probs
            else [],
            positive_label="relevant",
        )
        if ranking is not None:
            overall.update(ranking)
        return {"task": spec.task_name, "overall": overall}

    metrics_test = _dense_metrics(test_df, test_dense, classifier)
    metrics_eval = _dense_metrics(eval_df, eval_dense, classifier)
    model_name = spec.embedding_model_name.strip() or default_model_name
    classes = [str(value) for value in getattr(classifier, "classes_", classes)]
    runtime_model = EmbeddingPairClassifier(
        task_name=spec.task_name,
        model_name_or_path=model_name,
        classifier=classifier,
        classes_=classes,
    )
    training_summary = {
        "actual_epochs": 1,
        "best_epoch": 1,
        "early_stopped": False,
        "feature_backend": "embedding_pair",
    }

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    joblib.dump(
        {
            "model": runtime_model,
            "labels": classes,
            "task_spec": spec.__dict__,
            "embedding_model_name": model_name,
        },
        model_path,
    )
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {
            "task": spec.task_name,
            "epoch_stats": [{"epoch": 1}],
            "training_summary": training_summary,
            "calibration": calibration_summary,
        },
    )
    _write_task_metrics(
        output_dir,
        spec.artifact_name,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
        calibration=calibration_summary,
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "trainer": _resolved_trainer(spec),
        "feature_backend": "embedding_pair",
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": metrics_test["overall"],
        "eval": metrics_eval["overall"],
        "actual_epochs": 1,
        "best_epoch": 1,
        "early_stopped": False,
        "embedding_model_name": model_name,
        "calibration": calibration_summary,
    }


def _train_ordinal_threshold(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    print(f"[task:{spec.task_name}] ordinal_threshold training")
    family = spec.family
    train_df, test_df, eval_df = _load_task_splits(prepared_dir, family, spec.task_name)
    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    label_order = list(spec.label_order or spec.labels)
    label_to_index = {label: idx for idx, label in enumerate(label_order)}
    train_x = _encode_features(train_df, family)
    train_y = train_df["label"].astype(str).map(label_to_index).astype(int).to_numpy()
    eval_x = _encode_features(eval_df, family)
    eval_y_idx = (
        eval_df["label"].astype(str).map(label_to_index).astype(int).to_numpy()
        if not eval_df.empty
        else []
    )
    vectorizer = _build_vectorizer(train_cfg)
    x_train = vectorizer.fit_transform(train_x)
    x_eval = vectorizer.transform(eval_x) if eval_x else None

    boundary_models: list[Any] = []
    boundary_stats: list[dict[str, Any]] = []
    boundary_weight_factor = 1.5  # upweight rows within 1 class of the boundary
    for boundary_idx, left_label in enumerate(label_order[:-1]):
        boundary_train_y = (train_y > boundary_idx).astype(int)
        classifier = LogisticRegression(
            solver="liblinear",
            max_iter=1000,
            class_weight="balanced",
            random_state=int(train_cfg["seed"]),
        )
        base_weight = compute_sample_weight(class_weight="balanced", y=boundary_train_y)
        boundary_near = np.isin(train_y, [boundary_idx, boundary_idx + 1])
        sample_weight = np.where(
            boundary_near,
            base_weight * boundary_weight_factor,
            base_weight,
        ).astype(np.float64)
        classifier.fit(x_train, boundary_train_y, sample_weight=sample_weight)

        calibration_summary = None
        if x_eval is not None and len(eval_y_idx) > 0:
            eval_arr = np.asarray(eval_y_idx, dtype=np.intp)
            boundary_eval_y = (eval_arr > boundary_idx).astype(int)
            if len(set(boundary_eval_y.tolist())) > 1:
                classifier, calibration_summary = _maybe_calibrate_matrix_classifier(
                    classifier,
                    calibration_x=x_eval,
                    calibration_y=boundary_eval_y.tolist(),
                    train_cfg=train_cfg,
                )
        boundary_models.append(classifier)
        boundary_stats.append(
            {
                "boundary": f"{left_label}|>{label_order[boundary_idx + 1]}",
                "index": boundary_idx,
                "positive_rows": int(boundary_train_y.sum()),
                "negative_rows": int(len(boundary_train_y) - boundary_train_y.sum()),
                "calibration": calibration_summary,
            }
        )

    runtime_model = CumulativeOrdinalClassifier(
        task_name=spec.task_name,
        vectorizer=vectorizer,
        boundary_models=boundary_models,
        label_order=label_order,
        classes_=[_composite_label(spec.task_name, label) for label in label_order],
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    metrics_test, _ = _evaluate(
        runtime_model, test_df, family=family, split_name="test", predict_batch_size=batch_size
    )
    metrics_eval, _ = _evaluate(
        runtime_model, eval_df, family=family, split_name="eval", predict_batch_size=batch_size
    )
    if not test_df.empty:
        pred_raw = [label.split("::", 1)[1] for label in _predict_batched(runtime_model, _encode_features(test_df, family), batch_size=batch_size, desc=f"Predict task:{spec.task_name}:test")]
        metrics_test["overall"].update(
            _ordinal_metric_block(
                true_labels=test_df["label"].astype(str).tolist(),
                pred_labels=pred_raw,
                label_order=label_order,
            )
        )
    if not eval_df.empty:
        pred_raw = [label.split("::", 1)[1] for label in _predict_batched(runtime_model, _encode_features(eval_df, family), batch_size=batch_size, desc=f"Predict task:{spec.task_name}:eval")]
        metrics_eval["overall"].update(
            _ordinal_metric_block(
                true_labels=eval_df["label"].astype(str).tolist(),
                pred_labels=pred_raw,
                label_order=label_order,
            )
        )

    epoch_stats = [{"boundary_index": item["index"], "boundary": item["boundary"]} for item in boundary_stats]
    training_summary = {
        "actual_epochs": 1,
        "best_epoch": 1,
        "early_stopped": False,
        "boundary_count": len(boundary_models),
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    joblib.dump({"model": runtime_model, "task_spec": spec.__dict__}, model_path)
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {
            "task": spec.task_name,
            "epoch_stats": epoch_stats,
            "training_summary": training_summary,
            "boundaries": boundary_stats,
        },
    )
    _write_task_metrics(
        output_dir,
        spec.artifact_name,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "trainer": _resolved_trainer(spec),
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": metrics_test["overall"],
        "eval": metrics_eval["overall"],
        "actual_epochs": training_summary["actual_epochs"],
        "best_epoch": training_summary["best_epoch"],
        "early_stopped": training_summary["early_stopped"],
        "boundary_count": len(boundary_models),
    }


def _train_hierarchical_text(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    print(f"[task:{spec.task_name}] hierarchical_text training")
    family = spec.family
    train_df, test_df, eval_df = _load_task_splits(prepared_dir, family, spec.task_name)
    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    train_x = _encode_memory_type_features(train_df, family)
    eval_x = _encode_memory_type_features(eval_df, family)
    macro_train_y = [_memory_type_macro(label) for label in train_df["label"].astype(str).tolist()]
    macro_eval_y = [_memory_type_macro(label) for label in eval_df["label"].astype(str).tolist()]
    stage1_model, stage1_epochs, stage1_summary = _train_classifier_with_monitoring(
        train_x=train_x,
        train_y=macro_train_y,
        valid_x=eval_x,
        valid_y=macro_eval_y,
        train_cfg=train_cfg,
        run_name=f"{spec.task_name}:macro",
    )
    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    stage1_model, stage1_calibration = _maybe_calibrate_classifier(
        stage1_model,
        calibration_x=eval_x,
        calibration_y=macro_eval_y,
        train_cfg=train_cfg,
        predict_batch_size=batch_size,
    )

    # Focal-style: upweight weak fine-grained classes (plan, knowledge, reasoning_step)
    weak_memory_type_classes = {"plan", "knowledge", "reasoning_step"}
    focal_weight_weak = 1.8

    stage2_models: dict[str, Any] = {}
    stage2_stats: dict[str, Any] = {}
    stage2_calibration: dict[str, Any] = {}
    for macro_name, labels in _MEMORY_TYPE_MACRO_GROUPS.items():
        sub_train = train_df[train_df["label"].astype(str).isin(labels)].reset_index(drop=True)
        if sub_train.empty:
            continue
        sub_eval = eval_df[eval_df["label"].astype(str).isin(labels)].reset_index(drop=True)
        sub_train_x = _encode_memory_type_features(sub_train, family)
        sub_train_y = sub_train["label"].astype(str).tolist()
        sub_eval_x = _encode_memory_type_features(sub_eval, family)
        sub_eval_y = sub_eval["label"].astype(str).tolist()
        stage2_sample_weight = np.array(
            [
                focal_weight_weak if label in weak_memory_type_classes else 1.0
                for label in sub_train_y
            ],
            dtype=np.float64,
        )
        model, epochs, summary = _train_classifier_with_monitoring(
            train_x=sub_train_x,
            train_y=sub_train_y,
            valid_x=sub_eval_x,
            valid_y=sub_eval_y,
            train_cfg=train_cfg,
            run_name=f"{spec.task_name}:{macro_name}",
            sample_weight=stage2_sample_weight,
        )
        model, cal = _maybe_calibrate_classifier(
            model,
            calibration_x=sub_eval_x,
            calibration_y=sub_eval_y,
            train_cfg=train_cfg,
            predict_batch_size=batch_size,
        )
        stage2_models[macro_name] = model
        stage2_stats[macro_name] = {"epoch_stats": epochs, "training_summary": summary}
        stage2_calibration[macro_name] = cal

    runtime_model = HierarchicalTextClassifier(
        task_name=spec.task_name,
        stage1_model=stage1_model,
        stage2_models=stage2_models,
        macro_to_labels=_MEMORY_TYPE_MACRO_GROUPS,
        classes_=[_composite_label(spec.task_name, label) for label in spec.labels],
    )
    metrics_test, _ = _evaluate(
        runtime_model, test_df, family=family, split_name="test", predict_batch_size=batch_size
    )
    metrics_eval, _ = _evaluate(
        runtime_model, eval_df, family=family, split_name="eval", predict_batch_size=batch_size
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    joblib.dump({"model": runtime_model, "task_spec": spec.__dict__}, model_path)
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {
            "task": spec.task_name,
            "stage1": {"epoch_stats": stage1_epochs, "training_summary": stage1_summary, "calibration": stage1_calibration},
            "stage2": stage2_stats,
            "stage2_calibration": stage2_calibration,
        },
    )
    _write_task_metrics(
        output_dir,
        spec.artifact_name,
        metrics_test=metrics_test,
        metrics_eval=metrics_eval,
        calibration={"stage1": stage1_calibration, "stage2": stage2_calibration},
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "trainer": _resolved_trainer(spec),
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": metrics_test["overall"],
        "eval": metrics_eval["overall"],
        "actual_epochs": stage1_summary["actual_epochs"],
        "best_epoch": stage1_summary["best_epoch"],
        "early_stopped": stage1_summary["early_stopped"],
        "calibration": {"stage1": stage1_calibration, "stage2": stage2_calibration},
    }


def _train_token_classification(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train a dedicated Hugging Face token-classification model for a single task."""
    print(f"[task:{spec.task_name}] token_classification training")
    return train_token_task(
        task_name=spec.task_name,
        prepared_dir=prepared_dir,
        output_dir=output_dir,
        train_cfg=train_cfg,
        spec_payload=spec.__dict__,
    )


def _train_task(
    spec: TaskSpec,
    *,
    prepared_dir: Path,
    output_dir: Path,
    train_cfg: dict,
    strict: bool = True,
) -> dict:
    """Dispatch task-level training to the configured trainer."""
    trainers = {
        "classification": _train_classification_task,
        "pair_ranking": _train_pair_ranking,
        "single_regression": _train_single_regression,
        "token_classification": _train_token_classification,
        "embedding_pair": _train_embedding_pair,
        "ordinal_threshold": _train_ordinal_threshold,
        "hierarchical_text": _train_hierarchical_text,
    }
    trainer_name = _resolved_trainer(spec)
    trainer = trainers.get(trainer_name)
    if trainer is None:
        msg = (
            f"Unknown trainer '{trainer_name}' for task '{spec.task_name}' "
            f"(objective={spec.objective}); skipping."
        )
        _logger.warning(msg)
        if strict:
            raise RuntimeError(msg)
        return {}
    try:
        summary = trainer(
            spec, prepared_dir=prepared_dir, output_dir=output_dir, train_cfg=train_cfg
        )
        if not summary:
            msg = (
                f"Task '{spec.task_name}' produced no artifact/summary "
                "(likely missing rows or unsupported input contract)."
            )
            if strict:
                raise RuntimeError(msg)
            _logger.warning(msg)
            return {}
        return summary
    except NotImplementedError as exc:
        _logger.warning("task_skipped", extra={"task": spec.task_name, "reason": str(exc)})
        if strict:
            raise RuntimeError(
                f"Strict mode: unsupported task '{spec.task_name}' ({spec.objective})"
            ) from exc
        return {}
    except Exception as exc:
        if strict:
            raise
        _logger.warning("task_skipped", extra={"task": spec.task_name, "reason": str(exc)})
        return {}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train/eval all custom model families.")
    parser.add_argument(
        "--config", type=Path, default=DEFAULT_CONFIG_PATH, help="Path to model_pipeline.toml"
    )
    parser.add_argument(
        "--families", type=str, default="", help="Override train.families: comma-separated list."
    )
    parser.add_argument("--seed", type=int, default=None, help="Override train.seed.")
    parser.add_argument("--max-iter", type=int, default=None, help="Override train.max_iter.")
    parser.add_argument(
        "--max-features", type=int, default=None, help="Override train.max_features."
    )
    parser.add_argument(
        "--predict-batch-size", type=int, default=None, help="Override train.predict_batch_size."
    )
    parser.add_argument(
        "--prepared-dir", type=Path, default=None, help="Override paths.prepared_dir."
    )
    parser.add_argument(
        "--output-dir", type=Path, default=None, help="Override paths.trained_models_dir."
    )
    parser.add_argument(
        "--tasks", type=str, default="", help="Comma-separated task names to train (task-level)."
    )
    parser.add_argument(
        "--objective-types",
        type=str,
        default="",
        help="Filter task training by objective type (comma-separated).",
    )
    parser.add_argument(
        "--max-seq-length",
        type=int,
        default=None,
        help="Max sequence length for token classification models.",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=None,
        help="Learning rate override for token classification models.",
    )
    parser.add_argument(
        "--token-model-name-or-path",
        type=str,
        default=None,
        help="HF checkpoint used for token-classification tasks.",
    )
    parser.add_argument(
        "--token-num-train-epochs",
        type=int,
        default=None,
        help="Epochs for token-classification tasks.",
    )
    parser.add_argument(
        "--token-per-device-train-batch-size",
        type=int,
        default=None,
        help="Per-device train batch size for token tasks.",
    )
    parser.add_argument(
        "--token-per-device-eval-batch-size",
        type=int,
        default=None,
        help="Per-device eval batch size for token tasks.",
    )
    parser.add_argument(
        "--token-stride",
        type=int,
        default=None,
        help="Sliding-window stride for long token-task examples.",
    )
    parser.add_argument(
        "--token-warmup-ratio",
        type=float,
        default=None,
        help="Warmup ratio for token-task training.",
    )
    parser.add_argument(
        "--token-weight-decay",
        type=float,
        default=None,
        help="Weight decay for token-task training.",
    )
    parser.add_argument(
        "--token-gradient-accumulation-steps",
        type=int,
        default=None,
        help="Gradient accumulation steps for token-task training.",
    )
    parser.add_argument(
        "--calibration-split",
        type=str,
        default=None,
        help="Name of split to use for threshold calibration (default: eval).",
    )
    parser.add_argument(
        "--early-stopping",
        type=str,
        default=None,
        help="Override train.early_stopping (true/false).",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=None,
        help="Override train.early_stopping_patience.",
    )
    parser.add_argument(
        "--early-stopping-metric",
        type=str,
        default=None,
        help="Override train.early_stopping_metric.",
    )
    parser.add_argument(
        "--early-stopping-min-delta",
        type=float,
        default=None,
        help="Override train.early_stopping_min_delta.",
    )
    parser.add_argument(
        "--calibration-method",
        type=str,
        default=None,
        help="Override train.calibration_method.",
    )
    parser.add_argument(
        "--export-thresholds",
        action="store_true",
        help="Export per-task decision thresholds to <task>_thresholds.json.",
    )
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail when configured tasks/objectives are unsupported or missing training rows.",
    )
    strict_group.add_argument(
        "--allow-skips",
        dest="strict",
        action="store_false",
        help="Allow unsupported or missing tasks to be skipped during training.",
    )
    parser.set_defaults(strict=True)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    config = _load_config(args.config)
    strict_mode = bool(getattr(args, "strict", True))

    paths_cfg = dict(config.get("paths", {}))
    train_cfg = dict(config.get("train", {}))
    train_cfg.setdefault("seed", 42)
    train_cfg.setdefault("max_features", 250000)
    train_cfg.setdefault("min_df", 2)
    train_cfg.setdefault("ngram_min", 1)
    train_cfg.setdefault("ngram_max", 2)
    train_cfg.setdefault("max_iter", 25)
    train_cfg.setdefault("alpha", 1e-5)
    train_cfg.setdefault("predict_batch_size", 8192)
    train_cfg.setdefault("families", list(ALL_FAMILIES))
    train_cfg.setdefault("early_stopping", False)
    train_cfg.setdefault("early_stopping_patience", 3)
    train_cfg.setdefault("early_stopping_metric", "macro_f1")
    train_cfg.setdefault("early_stopping_min_delta", 0.0)
    train_cfg.setdefault("calibration_method", "")
    train_cfg.setdefault("calibration_split", "eval")
    token_cfg = dict(train_cfg.get("token", {}))
    token_cfg.setdefault("model_name_or_path", "distilbert-base-multilingual-cased")
    token_cfg.setdefault("num_train_epochs", 3)
    token_cfg.setdefault("per_device_train_batch_size", 8)
    token_cfg.setdefault("per_device_eval_batch_size", 16)
    token_cfg.setdefault("max_seq_length", 256)
    token_cfg.setdefault("stride", 64)
    token_cfg.setdefault("learning_rate", 5e-5)
    token_cfg.setdefault("warmup_ratio", 0.1)
    token_cfg.setdefault("weight_decay", 0.01)
    token_cfg.setdefault("gradient_accumulation_steps", 1)
    train_cfg["strict"] = strict_mode

    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    if args.max_iter is not None:
        train_cfg["max_iter"] = int(args.max_iter)
    if args.max_features is not None:
        train_cfg["max_features"] = int(args.max_features)
    if args.predict_batch_size is not None:
        train_cfg["predict_batch_size"] = int(args.predict_batch_size)
    if args.early_stopping is not None:
        train_cfg["early_stopping"] = str(args.early_stopping).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
    if args.early_stopping_patience is not None:
        train_cfg["early_stopping_patience"] = int(args.early_stopping_patience)
    if args.early_stopping_metric is not None:
        train_cfg["early_stopping_metric"] = str(args.early_stopping_metric).strip()
    if args.early_stopping_min_delta is not None:
        train_cfg["early_stopping_min_delta"] = float(args.early_stopping_min_delta)
    if args.calibration_method is not None:
        train_cfg["calibration_method"] = str(args.calibration_method).strip()
    if args.calibration_split is not None:
        train_cfg["calibration_split"] = str(args.calibration_split).strip()
    if args.max_seq_length is not None:
        token_cfg["max_seq_length"] = int(args.max_seq_length)
    if args.learning_rate is not None:
        token_cfg["learning_rate"] = float(args.learning_rate)
    if args.token_model_name_or_path is not None:
        token_cfg["model_name_or_path"] = str(args.token_model_name_or_path)
    if args.token_num_train_epochs is not None:
        token_cfg["num_train_epochs"] = int(args.token_num_train_epochs)
    if args.token_per_device_train_batch_size is not None:
        token_cfg["per_device_train_batch_size"] = int(args.token_per_device_train_batch_size)
    if args.token_per_device_eval_batch_size is not None:
        token_cfg["per_device_eval_batch_size"] = int(args.token_per_device_eval_batch_size)
    if args.token_stride is not None:
        token_cfg["stride"] = int(args.token_stride)
    if args.token_warmup_ratio is not None:
        token_cfg["warmup_ratio"] = float(args.token_warmup_ratio)
    if args.token_weight_decay is not None:
        token_cfg["weight_decay"] = float(args.token_weight_decay)
    if args.token_gradient_accumulation_steps is not None:
        token_cfg["gradient_accumulation_steps"] = int(args.token_gradient_accumulation_steps)
    train_cfg["token"] = token_cfg

    if args.prepared_dir is not None:
        prepared_dir = args.prepared_dir.resolve()
    else:
        prepared_dir = _resolve_path(
            str(paths_cfg.get("prepared_dir", "packages/models/prepared_data/modelpack")),
            base=REPO_ROOT,
        )

    if args.output_dir is not None:
        output_dir = args.output_dir.resolve()
    else:
        output_dir = _resolve_path(
            str(paths_cfg.get("trained_models_dir", "packages/models/trained_models")),
            base=REPO_ROOT,
        )

    families: list[str]
    if args.families.strip():
        families = [x.strip() for x in args.families.split(",") if x.strip()]
    else:
        raw = train_cfg.get("families", list(ALL_FAMILIES))
        if isinstance(raw, list):
            families = [str(x).strip() for x in raw if str(x).strip()]
        else:
            families = list(ALL_FAMILIES)

    invalid = [f for f in families if f not in ALL_FAMILIES]
    if invalid:
        print(f"Invalid family names: {invalid}. Allowed: {list(ALL_FAMILIES)}", file=sys.stderr)
        return 1
    if not families:
        print("No model families selected for training.", file=sys.stderr)
        return 1

    task_specs_raw = list(config.get("tasks", []))
    preflight = run_preflight_validation(
        task_specs_raw=task_specs_raw,
        prepared_dir=prepared_dir,
        strict=strict_mode,
    )
    for warning in preflight.warnings:
        print(f"Preflight warning: {warning}", file=sys.stderr)
    if preflight.errors:
        for error in preflight.errors:
            print(f"Preflight error: {error}", file=sys.stderr)
        if strict_mode:
            print("Strict mode enabled: aborting due to preflight errors.", file=sys.stderr)
            return 1
        print("Continuing with preflight errors because --allow-skips is active.", file=sys.stderr)

    configured_tasks = [
        {
            "task_name": str(t.get("task_name", "")),
            "family": str(t.get("family", "")),
            "input_type": str(t.get("input_type", "")),
            "objective": str(t.get("objective", "")),
            "enabled": bool(t.get("enabled", True)),
            "artifact_name": str(t.get("artifact_name", "")),
            "metrics": list(t.get("metrics", [])),
            "trainer": str(t.get("trainer", "")),
            "feature_backend": str(t.get("feature_backend", "")),
            "label_order": list(t.get("label_order", [])),
            "embedding_model_name": str(t.get("embedding_model_name", "")),
        }
        for t in task_specs_raw
    ]

    selected_tasks = {x.strip() for x in args.tasks.split(",") if x.strip()} if args.tasks else None
    selected_obj = (
        {x.strip() for x in args.objective_types.split(",") if x.strip()}
        if args.objective_types
        else None
    )

    existing_manifest: dict[str, Any] = {}
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        try:
            existing_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        except Exception:
            existing_manifest = {}

    families_summary: dict[str, dict] = dict(existing_manifest.get("families", {}))
    task_summaries: dict[str, dict] = dict(existing_manifest.get("task_models", {}))
    task_training_status: dict[str, dict] = {
        str(t.get("task_name", "")): {
            "status": "pending",
            "reason": None,
            "family": str(t.get("family", "")),
            "objective": str(t.get("objective", "")),
            "enabled": bool(t.get("enabled", True)),
        }
        for t in task_specs_raw
        if str(t.get("task_name", ""))
    }
    manifest = {
        "manifest_schema_version": 2,
        "config_path": str(args.config.resolve()),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "paths": {"prepared_dir": str(prepared_dir), "trained_models_dir": str(output_dir)},
        "train_settings": train_cfg,
        "build_metadata": _build_metadata(),
        "configured_tasks": configured_tasks,
        "preflight_validation": preflight.as_dict(),
        "families": families_summary,
        "task_training_status": task_training_status,
    }

    train_selected_only = selected_tasks is not None or selected_obj is not None
    families_to_train = [] if train_selected_only else list(families)

    pbar = _progress(total=len(families_to_train), desc="Model training", unit="family")
    try:
        for family in families_to_train:
            pbar.set_description(f"Train family [{family}]")
            summary = _train_one_family(
                family,
                prepared_dir=prepared_dir,
                output_dir=output_dir,
                train_cfg=train_cfg,
            )
            families_summary[family] = summary
            pbar.update(1)
    finally:
        pbar.close()

    if task_specs_raw:
        all_specs = [
            TaskSpec(**{k: t[k] for k in TaskSpec.__dataclass_fields__ if k in t})
            for t in task_specs_raw
        ]

        tasks_to_train: list[TaskSpec] = []
        disabled_selected: list[str] = []
        for spec in all_specs:
            if selected_tasks is not None and spec.task_name not in selected_tasks:
                task_training_status[spec.task_name] = {
                    **task_training_status.get(spec.task_name, {}),
                    "status": "filtered_out",
                    "reason": "Excluded by --tasks filter",
                }
                continue
            if selected_obj is not None and spec.objective not in selected_obj:
                task_training_status[spec.task_name] = {
                    **task_training_status.get(spec.task_name, {}),
                    "status": "filtered_out",
                    "reason": "Excluded by --objective-types filter",
                }
                continue
            if not spec.enabled:
                task_training_status[spec.task_name] = {
                    **task_training_status.get(spec.task_name, {}),
                    "status": "disabled",
                    "reason": "Task disabled in config",
                }
                if selected_tasks is not None or selected_obj is not None:
                    disabled_selected.append(spec.task_name)
                continue
            task_training_status[spec.task_name] = {
                **task_training_status.get(spec.task_name, {}),
                "status": "queued",
                "reason": None,
            }
            tasks_to_train.append(spec)

        if disabled_selected and strict_mode:
            print(
                f"Strict mode: selected tasks are disabled in config: {sorted(disabled_selected)}",
                file=sys.stderr,
            )
            return 1

        pbar_tasks = _progress(total=len(tasks_to_train), desc="Task training", unit="task")
        try:
            for spec in tasks_to_train:
                pbar_tasks.set_description(f"Train task [{spec.task_name}]")
                try:
                    summary = _train_task(
                        spec,
                        prepared_dir=prepared_dir,
                        output_dir=output_dir,
                        train_cfg=train_cfg,
                        strict=strict_mode,
                    )
                except Exception as exc:
                    task_training_status[spec.task_name] = {
                        **task_training_status.get(spec.task_name, {}),
                        "status": "failed",
                        "reason": str(exc),
                    }
                    if strict_mode:
                        print(
                            f"Strict mode: task '{spec.task_name}' failed: {exc}",
                            file=sys.stderr,
                        )
                        return 1
                    _logger.warning(
                        "task_failed", extra={"task": spec.task_name, "error": str(exc)}
                    )
                    pbar_tasks.update(1)
                    continue

                if summary:
                    task_summaries[spec.task_name] = summary
                    task_training_status[spec.task_name] = {
                        **task_training_status.get(spec.task_name, {}),
                        "status": "trained",
                        "reason": None,
                        "model_path": summary.get("model_path"),
                        "train_rows": summary.get("train_rows"),
                    }
                else:
                    task_training_status[spec.task_name] = {
                        **task_training_status.get(spec.task_name, {}),
                        "status": "skipped",
                        "reason": "No summary returned",
                    }
                pbar_tasks.update(1)
        finally:
            pbar_tasks.close()

    manifest["task_models"] = task_summaries

    if args.export_thresholds:
        for task_name, summary in task_summaries.items():
            if str(summary.get("objective", "")) == "token_classification":
                thresholds = {
                    "task_name": task_name,
                    "type": "token_span_metadata",
                    "labels": summary.get("labels", {}),
                    "calibration_split": args.calibration_split or "eval",
                    "note": "Token span models do not use a single scalar threshold.",
                }
            else:
                thresholds = {
                    "task_name": task_name,
                    "default_threshold": 0.5,
                    "calibration_split": args.calibration_split or "eval",
                    "note": "Auto-generated defaults. Tune via offline calibration.",
                }
            _write_json(output_dir / f"{task_name}_thresholds.json", thresholds)

    thresholds_files = {}
    for task_name in task_summaries:
        thresh_path = output_dir / f"{task_name}_thresholds.json"
        if thresh_path.exists():
            thresholds_files[task_name] = str(thresh_path)

    manifest["runtime_thresholds"] = thresholds_files
    manifest["task_training_status"] = task_training_status

    artifact_errors = validate_manifest_artifacts(
        families_summary=families_summary,
        task_summaries=task_summaries,
    )
    manifest["artifact_validation"] = {"ok": len(artifact_errors) == 0, "errors": artifact_errors}
    if artifact_errors:
        for err in artifact_errors:
            print(f"Artifact validation error: {err}", file=sys.stderr)
        if strict_mode:
            print(
                "Strict mode enabled: aborting due to artifact validation errors.", file=sys.stderr
            )
            return 1

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest)

    print("Training complete.")
    print(f"Models written to: {output_dir}")
    return 0


def train_models(config: TrainConfig) -> int:
    """Run training pipeline from a typed config."""
    argv: list[str] = ["--config", str(config.config_path)]
    if config.families:
        argv.extend(["--families", config.families])
    if config.seed is not None:
        argv.extend(["--seed", str(config.seed)])
    if config.max_iter is not None:
        argv.extend(["--max-iter", str(config.max_iter)])
    if config.max_features is not None:
        argv.extend(["--max-features", str(config.max_features)])
    if config.predict_batch_size is not None:
        argv.extend(["--predict-batch-size", str(config.predict_batch_size)])
    if config.early_stopping is not None:
        argv.extend(["--early-stopping", str(config.early_stopping).lower()])
    if config.early_stopping_patience is not None:
        argv.extend(["--early-stopping-patience", str(config.early_stopping_patience)])
    if config.early_stopping_metric is not None:
        argv.extend(["--early-stopping-metric", config.early_stopping_metric])
    if config.early_stopping_min_delta is not None:
        argv.extend(["--early-stopping-min-delta", str(config.early_stopping_min_delta)])
    if config.calibration_method is not None:
        argv.extend(["--calibration-method", config.calibration_method])
    if config.prepared_dir is not None:
        argv.extend(["--prepared-dir", str(config.prepared_dir)])
    if config.output_dir is not None:
        argv.extend(["--output-dir", str(config.output_dir)])
    if config.tasks:
        argv.extend(["--tasks", config.tasks])
    if config.objective_types:
        argv.extend(["--objective-types", config.objective_types])
    if config.max_seq_length is not None:
        argv.extend(["--max-seq-length", str(config.max_seq_length)])
    if config.learning_rate is not None:
        argv.extend(["--learning-rate", str(config.learning_rate)])
    if config.token_model_name_or_path is not None:
        argv.extend(["--token-model-name-or-path", config.token_model_name_or_path])
    if config.token_num_train_epochs is not None:
        argv.extend(["--token-num-train-epochs", str(config.token_num_train_epochs)])
    if config.token_per_device_train_batch_size is not None:
        argv.extend(
            [
                "--token-per-device-train-batch-size",
                str(config.token_per_device_train_batch_size),
            ]
        )
    if config.token_per_device_eval_batch_size is not None:
        argv.extend(
            [
                "--token-per-device-eval-batch-size",
                str(config.token_per_device_eval_batch_size),
            ]
        )
    if config.token_stride is not None:
        argv.extend(["--token-stride", str(config.token_stride)])
    if config.token_warmup_ratio is not None:
        argv.extend(["--token-warmup-ratio", str(config.token_warmup_ratio)])
    if config.token_weight_decay is not None:
        argv.extend(["--token-weight-decay", str(config.token_weight_decay)])
    if config.token_gradient_accumulation_steps is not None:
        argv.extend(
            [
                "--token-gradient-accumulation-steps",
                str(config.token_gradient_accumulation_steps),
            ]
        )
    if config.calibration_split is not None:
        argv.extend(["--calibration-split", config.calibration_split])
    if config.export_thresholds:
        argv.append("--export-thresholds")
    if config.strict:
        argv.append("--strict")
    else:
        argv.append("--allow-skips")
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
