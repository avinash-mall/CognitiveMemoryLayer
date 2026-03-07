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
import importlib.metadata
import json
import logging
import math
import subprocess
import sys
import tomllib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

try:
    import joblib
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: joblib. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: pandas. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import SGDClassifier, SGDRegressor
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
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: scikit-learn. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

from cml.modeling.config import find_repo_root
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


def _single_metadata_tokens(record: object, available_cols: set[str]) -> list[str]:
    tokens: list[str] = []

    if "memory_type" in available_cols:
        memory_type = str(getattr(record, "memory_type", "") or "").strip()
        if memory_type:
            tokens.append(f"memory_type={memory_type}")
    if "namespace" in available_cols:
        namespace = str(getattr(record, "namespace", "") or "").strip()
        if namespace:
            tokens.append(f"namespace={namespace}")
    if "context_tags" in available_cols:
        for tag in _normalize_tag_list(getattr(record, "context_tags", None))[:3]:
            tokens.append(f"context_tag={tag}")

    importance = (
        _safe_float(getattr(record, "importance", None)) if "importance" in available_cols else None
    )
    if importance is not None:
        tokens.append(f"importance_bin={_ratio_bucket(importance)}")
    confidence = (
        _safe_float(getattr(record, "confidence", None)) if "confidence" in available_cols else None
    )
    if confidence is not None:
        tokens.append(f"confidence_bin={_ratio_bucket(confidence)}")
    access_count = (
        _safe_int(getattr(record, "access_count", None))
        if "access_count" in available_cols
        else None
    )
    if access_count is not None:
        tokens.append(f"access_count={_count_bucket(access_count, high=6, medium=2)}")
    age_days = (
        _safe_int(getattr(record, "age_days", None)) if "age_days" in available_cols else None
    )
    if age_days is not None:
        tokens.append(
            f"age_days={_count_bucket(age_days, high=90, medium=21).replace('none', 'fresh')}"
        )
    dependency_count = (
        _safe_int(getattr(record, "dependency_count", None))
        if "dependency_count" in available_cols
        else None
    )
    if dependency_count is not None:
        tokens.append(f"dependency_count={_count_bucket(dependency_count, high=4, medium=1)}")
    support_count = (
        _safe_int(getattr(record, "support_count", None))
        if "support_count" in available_cols
        else None
    )
    if support_count is not None:
        tokens.append(f"support_count={_count_bucket(support_count, high=4, medium=2)}")
    if "mixed_topic" in available_cols:
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
            meta_tokens = _single_metadata_tokens(row, available_cols)
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


def _build_pipeline(train_cfg: dict) -> Pipeline:
    max_features = max(1000, int(train_cfg["max_features"]))
    min_df = max(1, int(train_cfg["min_df"]))
    ngram_min = max(1, int(train_cfg.get("ngram_min", 1)))
    ngram_max = max(ngram_min, int(train_cfg.get("ngram_max", 2)))
    alpha = max(1e-8, float(train_cfg["alpha"]))
    seed = int(train_cfg["seed"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=1,
        tol=None,
        warm_start=True,
        shuffle=True,
        early_stopping=False,
        class_weight="balanced",
        random_state=seed,
    )
    return Pipeline(steps=[("vectorizer", vectorizer), ("classifier", classifier)])


def _fit_with_epoch_stats(
    model: Pipeline, train_x: list[str], train_y: list[str], *, epochs: int, family: str
) -> list[dict]:
    vectorizer: TfidfVectorizer = model.named_steps["vectorizer"]
    classifier: SGDClassifier = model.named_steps["classifier"]

    x_train = vectorizer.fit_transform(train_x)
    classes = sorted(set(train_y))
    epoch_stats: list[dict] = []

    pbar = _progress(total=epochs, desc=f"Epochs {family}", unit="epoch")
    try:
        for epoch in range(1, epochs + 1):
            classifier.fit(x_train, train_y)
            pred = classifier.predict(x_train).tolist()
            acc = float(accuracy_score(train_y, pred))
            macro = float(f1_score(train_y, pred, average="macro", zero_division=0))
            weighted = float(f1_score(train_y, pred, average="weighted", zero_division=0))

            loss_value: float | None = None
            try:
                prob = classifier.predict_proba(x_train)
                loss_value = float(log_loss(train_y, prob, labels=classes))
            except Exception:
                loss_value = None

            stat = {
                "epoch": epoch,
                "train_loss": loss_value,
                "train_accuracy": acc,
                "train_macro_f1": macro,
                "train_weighted_f1": weighted,
            }
            epoch_stats.append(stat)

            if loss_value is None:
                print(
                    f"[{family}] epoch {epoch}/{epochs} | train_acc={acc:.4f} train_macro_f1={macro:.4f}"
                )
            else:
                print(
                    f"[{family}] epoch {epoch}/{epochs} | "
                    f"train_loss={loss_value:.4f} train_acc={acc:.4f} train_macro_f1={macro:.4f}"
                )
            pbar.update(1)
    finally:
        pbar.close()

    return epoch_stats


def _predict_batched(
    model: Pipeline, features: list[str], *, batch_size: int, desc: str
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
    model: Pipeline,
    features: list[str],
    *,
    positive_class: str,
) -> list[float]:
    classifier = getattr(model, "named_steps", {}).get("classifier")
    classes = (
        [str(x) for x in getattr(classifier, "classes_", [])] if classifier is not None else []
    )
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


def _evaluate(
    model: Pipeline, df: pd.DataFrame, *, family: str, split_name: str, predict_batch_size: int
) -> tuple[dict, dict]:
    x = _encode_features(df, family)
    y_true = _encode_targets(df)
    y_pred = _predict_batched(
        model, x, batch_size=predict_batch_size, desc=f"Predict {family}:{split_name}"
    )

    overall_labels = sorted(set(y_true) | set(y_pred))
    overall_metrics = _metric_block(y_true, y_pred, overall_labels)

    cal_error: float | None = None
    try:
        if hasattr(model, "predict_proba"):
            probs = []
            for start in range(0, len(x), predict_batch_size):
                batch = x[start : start + predict_batch_size]
                probs.extend(model.predict_proba(batch).tolist())
            cal_error = _calibration_error(y_true, y_pred, probs)
    except Exception:
        pass
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

    model = _build_pipeline(train_cfg)
    train_x = _encode_features(train_df, family)
    train_y = _encode_targets(train_df)
    epochs = max(2, int(train_cfg["max_iter"]))
    print(f"Fitting {family} on {len(train_df)} rows for {epochs} epochs...")
    epoch_stats = _fit_with_epoch_stats(model, train_x, train_y, epochs=epochs, family=family)

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
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
        },
        "epoch_stats": epoch_stats,
    }

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
    }


def _train_classification_task(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train a dedicated TF-IDF+SGD classifier for a single task subset."""
    print(f"[task:{spec.task_name}] classification training")
    family = spec.family
    train_df = _load_split(prepared_dir, family, "train")
    test_df = _load_split(prepared_dir, family, "test")

    train_df = train_df[train_df["task"] == spec.task_name].reset_index(drop=True)
    test_df = test_df[test_df["task"] == spec.task_name].reset_index(drop=True)

    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    train_x = _encode_features(train_df, family)
    train_y = _encode_targets(train_df)

    model = _build_pipeline(train_cfg)
    epochs = max(2, int(train_cfg["max_iter"]))
    epoch_stats = _fit_with_epoch_stats(
        model, train_x, train_y, epochs=epochs, family=spec.task_name
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    test_metrics: dict = {}
    if not test_df.empty:
        test_x = _encode_features(test_df, family)
        test_y = _encode_targets(test_df)
        test_pred = _predict_batched(
            model, test_x, batch_size=batch_size, desc=f"Predict task:{spec.task_name}:test"
        )
        test_metrics = _metric_block(test_y, test_pred)

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    all_labels = sorted(set(train_y))
    joblib.dump(
        {"model": model, "labels": all_labels, "task_spec": spec.__dict__},
        model_path,
    )
    _write_json(
        output_dir / f"{spec.artifact_name}_epoch_stats.json",
        {"task": spec.task_name, "epoch_stats": epoch_stats},
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": test_metrics,
    }


def _train_pair_ranking(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train a TF-IDF+SGD model for pair ranking using predict_proba as ranking score."""
    print(f"[task:{spec.task_name}] pair_ranking training")
    family = spec.family
    train_df = _load_split(prepared_dir, family, "train")
    test_df = _load_split(prepared_dir, family, "test")

    train_df = train_df[train_df["task"] == spec.task_name].reset_index(drop=True)
    test_df = test_df[test_df["task"] == spec.task_name].reset_index(drop=True)

    if train_df.empty:
        print(f"[task:{spec.task_name}] no training rows after filtering; skipping.")
        return {}

    train_x = _encode_features(train_df, family)
    train_y = _encode_targets(train_df)

    model = _build_pipeline(train_cfg)
    epochs = max(2, int(train_cfg["max_iter"]))
    epoch_stats = _fit_with_epoch_stats(
        model, train_x, train_y, epochs=epochs, family=spec.task_name
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    test_metrics: dict = {}
    if not test_df.empty:
        test_x = _encode_features(test_df, family)
        test_y = _encode_targets(test_df)
        test_pred = _predict_batched(
            model, test_x, batch_size=batch_size, desc=f"Predict task:{spec.task_name}:test"
        )
        test_metrics = _metric_block(test_y, test_pred)
        ranking = _ranking_metrics_from_groups(
            test_df,
            _positive_class_scores(
                model,
                test_x,
                positive_class=_composite_label(spec.task_name, "relevant"),
            ),
            positive_label="relevant",
        )
        if ranking is not None:
            test_metrics.update(ranking)
        else:
            test_metrics["ranking_proxy"] = {
                "note": "Classification metrics used as ranking proxy; true MRR/NDCG require grouped candidate lists",
                "proxy_accuracy": test_metrics.get("accuracy", 0.0),
                "proxy_f1": test_metrics.get("macro_f1", 0.0),
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
        {"task": spec.task_name, "epoch_stats": epoch_stats},
    )

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "test": test_metrics,
    }


def _train_single_regression(
    spec: TaskSpec, *, prepared_dir: Path, output_dir: Path, train_cfg: dict
) -> dict:
    """Train an SGDRegressor on single-text features with a numeric target."""
    import numpy as np

    print(f"[task:{spec.task_name}] single_regression training")
    family = spec.family
    train_path = prepared_dir / f"{family}_train.parquet"
    test_path = prepared_dir / f"{family}_test.parquet"

    if not train_path.exists():
        print(f"[task:{spec.task_name}] missing training data at {train_path}; skipping.")
        return {}

    train_df = pd.read_parquet(train_path)
    train_df = train_df[train_df["task"] == spec.task_name].reset_index(drop=True)

    score_col = "score" if "score" in train_df.columns else "label"
    train_df[score_col] = pd.to_numeric(train_df[score_col], errors="coerce")
    train_df = train_df.dropna(subset=[score_col]).reset_index(drop=True)

    if train_df.empty:
        print(f"[task:{spec.task_name}] no valid regression rows after filtering; skipping.")
        return {}

    train_y = train_df[score_col].values.astype(float)
    train_features = _encode_features(train_df, family)

    max_features = max(1000, int(train_cfg["max_features"]))
    min_df = max(1, int(train_cfg["min_df"]))
    ngram_min = max(1, int(train_cfg.get("ngram_min", 1)))
    ngram_max = max(ngram_min, int(train_cfg.get("ngram_max", 2)))
    alpha = max(1e-8, float(train_cfg["alpha"]))
    seed = int(train_cfg["seed"])

    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    regressor = SGDRegressor(
        loss="squared_error",
        penalty="l2",
        alpha=alpha,
        max_iter=max(2, int(train_cfg["max_iter"])),
        tol=1e-4,
        random_state=seed,
    )
    model = Pipeline(steps=[("vectorizer", vectorizer), ("regressor", regressor)])

    x_train = vectorizer.fit_transform(train_features)
    regressor.fit(x_train, train_y)

    train_pred = regressor.predict(x_train)
    train_mae = float(mean_absolute_error(train_y, train_pred))
    train_rmse = float(np.sqrt(mean_squared_error(train_y, train_pred)))
    print(f"[task:{spec.task_name}] train MAE={train_mae:.4f} RMSE={train_rmse:.4f}")

    test_metrics: dict = {"train_mae": train_mae, "train_rmse": train_rmse}
    if test_path.exists():
        test_df = pd.read_parquet(test_path)
        test_df = test_df[test_df["task"] == spec.task_name].reset_index(drop=True)
        t_score_col = "score" if "score" in test_df.columns else "label"
        test_df[t_score_col] = pd.to_numeric(test_df[t_score_col], errors="coerce")
        test_df = test_df.dropna(subset=[t_score_col]).reset_index(drop=True)
        if not test_df.empty:
            test_y = test_df[t_score_col].values.astype(float)
            x_test = vectorizer.transform(_encode_features(test_df, family))
            test_pred = regressor.predict(x_test)
            test_metrics["test_mae"] = float(mean_absolute_error(test_y, test_pred))
            test_metrics["test_rmse"] = float(np.sqrt(mean_squared_error(test_y, test_pred)))
            print(
                f"[task:{spec.task_name}] test MAE={test_metrics['test_mae']:.4f} "
                f"RMSE={test_metrics['test_rmse']:.4f}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    model_path = output_dir / f"{spec.artifact_name}_model.joblib"
    joblib.dump({"model": model, "task_spec": spec.__dict__}, model_path)

    return {
        "task": spec.task_name,
        "objective": spec.objective,
        "model_path": str(model_path),
        "train_rows": len(train_df),
        "metrics": test_metrics,
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
    """Dispatch task-level training to the appropriate trainer by objective type."""
    trainers = {
        "classification": _train_classification_task,
        "pair_ranking": _train_pair_ranking,
        "single_regression": _train_single_regression,
        "token_classification": _train_token_classification,
    }
    trainer = trainers.get(spec.objective)
    if trainer is None:
        msg = f"Unknown objective '{spec.objective}' for task '{spec.task_name}'; skipping."
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
