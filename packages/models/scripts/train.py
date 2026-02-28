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
import json
import math
import re
import sys
import tomllib
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import joblib
import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
)
from sklearn.pipeline import FeatureUnion, Pipeline

MODELS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODELS_ROOT.parent.parent
DEFAULT_CONFIG_PATH = MODELS_ROOT / "model_pipeline.toml"

FAMILY_SINGLE_TEXT = {"router", "extractor"}
FAMILY_PAIR_TEXT = {"pair"}
ALL_FAMILIES = ("router", "extractor", "pair")
_TOKEN_RE = re.compile(r"[A-Za-z0-9]+")


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


def _effective_family_train_cfg(train_cfg: dict, family: str) -> dict:
    effective = dict(train_cfg)
    family_overrides = effective.pop("family_overrides", {})
    if not isinstance(family_overrides, dict):
        return effective
    family_cfg = family_overrides.get(family, {})
    if not isinstance(family_cfg, dict):
        return effective
    for key, value in family_cfg.items():
        effective[key] = value
    return effective


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


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    if value is None:
        return default
    return bool(value)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _ratio_bucket(value: float) -> str:
    v = max(0.0, min(1.0, value))
    if v < 0.1:
        return "0_10"
    if v < 0.25:
        return "10_25"
    if v < 0.5:
        return "25_50"
    if v < 0.75:
        return "50_75"
    return "75_100"


def _pair_feature_tokens(text_a: str, text_b: str) -> list[str]:
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    set_a = set(tokens_a)
    set_b = set(tokens_b)

    overlap = set_a & set_b
    union_size = max(1, len(set_a | set_b))
    overlap_size = len(overlap)

    jaccard = overlap_size / union_size
    cover_a = overlap_size / max(1, len(set_a))
    cover_b = overlap_size / max(1, len(set_b))
    len_ratio = min(len(tokens_a), len(tokens_b)) / max(1, max(len(tokens_a), len(tokens_b)))

    nums_a = {tok for tok in set_a if tok.isdigit()}
    nums_b = {tok for tok in set_b if tok.isdigit()}
    shared_num = int(bool(nums_a & nums_b))
    num_conflict = int(bool(nums_a or nums_b) and not shared_num)

    return [
        f"pair_jaccard_{_ratio_bucket(jaccard)}",
        f"pair_cover_a_{_ratio_bucket(cover_a)}",
        f"pair_cover_b_{_ratio_bucket(cover_b)}",
        f"pair_len_ratio_{_ratio_bucket(len_ratio)}",
        f"pair_shared_num_{shared_num}",
        f"pair_num_conflict_{num_conflict}",
        f"pair_qmark_a_{int('?' in text_a)}",
        f"pair_qmark_b_{int('?' in text_b)}",
        f"pair_overlap_gt_3_{int(overlap_size >= 3)}",
    ]


def _task_balanced_macro_f1(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0

    true_task, true_label = zip(*[_split_composite(v) for v in y_true], strict=False)
    pred_task, pred_label = zip(*[_split_composite(v) for v in y_pred], strict=False)
    tasks = sorted(set(true_task))

    scores: list[float] = []
    for task in tasks:
        idx = [i for i, t in enumerate(true_task) if t == task]
        task_true = [true_label[i] for i in idx]
        task_pred: list[str] = []
        for i in idx:
            if pred_task[i] == task:
                task_pred.append(pred_label[i])
            else:
                task_pred.append("__wrong_task__")
        task_labels = sorted(set(task_true))
        task_macro = float(
            f1_score(task_true, task_pred, labels=task_labels, average="macro", zero_division=0)
        )
        scores.append(task_macro)

    if not scores:
        return 0.0
    return float(sum(scores) / len(scores))


def _build_sample_weights(train_y: list[str], train_cfg: dict) -> list[float] | None:
    task_weight_power = max(0.0, float(train_cfg.get("task_weight_power", 0.5)))
    label_weight_power = max(0.0, float(train_cfg.get("label_weight_power", 0.0)))
    if task_weight_power <= 0.0 and label_weight_power <= 0.0:
        return None

    task_focus_weights_raw = train_cfg.get("task_focus_weights", {})
    task_focus_weights: dict[str, float] = {}
    if isinstance(task_focus_weights_raw, dict):
        for key, value in task_focus_weights_raw.items():
            task_name = str(key).strip()
            if not task_name:
                continue
            try:
                parsed = float(value)
            except Exception:
                continue
            if parsed > 0:
                task_focus_weights[task_name] = parsed

    sample_weight_min = max(0.01, float(train_cfg.get("sample_weight_min", 0.2)))
    sample_weight_max = max(sample_weight_min, float(train_cfg.get("sample_weight_max", 5.0)))

    task_counts: Counter[str] = Counter()
    label_counts: Counter[str] = Counter(train_y)
    task_for_label: list[str] = []
    for composite in train_y:
        task, _ = _split_composite(composite)
        task_for_label.append(task)
        task_counts[task] += 1

    if not task_counts:
        return None

    raw_weights: list[float] = []
    for composite, task in zip(train_y, task_for_label, strict=False):
        task_count = max(1, task_counts.get(task, 1))
        label_count = max(1, label_counts.get(composite, 1))
        task_term = math.pow(task_count, -task_weight_power) if task_weight_power > 0 else 1.0
        label_term = math.pow(label_count, -label_weight_power) if label_weight_power > 0 else 1.0
        focus_term = task_focus_weights.get(task, 1.0)
        raw_weights.append(task_term * label_term * focus_term)

    mean_weight = sum(raw_weights) / max(1, len(raw_weights))
    if mean_weight <= 0:
        return None

    normalized = [w / mean_weight for w in raw_weights]
    return [float(min(sample_weight_max, max(sample_weight_min, w))) for w in normalized]


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
        return [f"task={t} [text] {x}" for t, x in zip(df["task"], df["text"], strict=False)]
    encoded: list[str] = []
    for t, a, b in zip(df["task"], df["text_a"], df["text_b"], strict=False):
        pair_feats = " ".join(_pair_feature_tokens(a, b))
        encoded.append(f"task={t} [a] {a} [b] {b} [pair] {pair_feats}")
    return encoded


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
    use_char_ngrams = _as_bool(train_cfg.get("use_char_ngrams", True), default=True)
    char_ngram_min = max(2, int(train_cfg.get("char_ngram_min", 3)))
    char_ngram_max = max(char_ngram_min, int(train_cfg.get("char_ngram_max", 5)))
    char_min_df = max(1, int(train_cfg.get("char_min_df", 2)))
    char_max_features = max(1000, int(train_cfg.get("char_max_features", max_features)))
    sgd_average = _as_bool(train_cfg.get("sgd_average", True), default=True)

    word_vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(ngram_min, ngram_max),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    if use_char_ngrams:
        char_vectorizer = TfidfVectorizer(
            lowercase=True,
            strip_accents="unicode",
            analyzer="char_wb",
            ngram_range=(char_ngram_min, char_ngram_max),
            min_df=char_min_df,
            max_features=char_max_features,
            sublinear_tf=True,
        )
        vectorizer = FeatureUnion(
            transformer_list=[
                ("word", word_vectorizer),
                ("char", char_vectorizer),
            ]
        )
    else:
        vectorizer = word_vectorizer

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
        average=sgd_average,
        random_state=seed,
    )
    return Pipeline(steps=[("vectorizer", vectorizer), ("classifier", classifier)])


def _fit_with_epoch_stats(
    model: Pipeline,
    train_x: list[str],
    train_y: list[str],
    *,
    epochs: int,
    family: str,
    train_cfg: dict,
    eval_x: list[str] | None = None,
    eval_y: list[str] | None = None,
) -> list[dict]:
    vectorizer = model.named_steps["vectorizer"]
    classifier: SGDClassifier = model.named_steps["classifier"]

    x_train = vectorizer.fit_transform(train_x)
    classes = sorted(set(train_y))
    sample_weights = _build_sample_weights(train_y, train_cfg)
    x_eval = vectorizer.transform(eval_x) if eval_x is not None else None

    use_early_stopping = _as_bool(train_cfg.get("use_early_stopping", True), default=True)
    early_stopping_patience = max(1, int(train_cfg.get("early_stopping_patience", 4)))
    early_stopping_min_delta = max(0.0, float(train_cfg.get("early_stopping_min_delta", 1e-4)))
    early_stopping_metric = str(
        train_cfg.get("early_stopping_metric", "eval_task_macro_f1")
    ).strip()
    if early_stopping_metric not in {
        "eval_macro_f1",
        "eval_task_macro_f1",
        "eval_focus_macro_f1",
        "eval_accuracy",
    }:
        early_stopping_metric = "eval_task_macro_f1"
    focus_task_weights_raw = train_cfg.get("focus_task_weights", {})
    focus_task_weights: dict[str, float] = {}
    if isinstance(focus_task_weights_raw, dict):
        for key, value in focus_task_weights_raw.items():
            name = str(key).strip()
            if not name:
                continue
            try:
                weight = float(value)
            except Exception:
                continue
            if weight > 0:
                focus_task_weights[name] = weight

    epoch_stats: list[dict] = []
    best_classifier: SGDClassifier | None = None
    best_epoch: int | None = None
    best_metric = float("-inf")
    epochs_since_improvement = 0

    pbar = _progress(total=epochs, desc=f"Epochs {family}", unit="epoch")
    try:
        for epoch in range(1, epochs + 1):
            classifier.fit(x_train, train_y, sample_weight=sample_weights)
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

            eval_acc: float | None = None
            eval_macro: float | None = None
            eval_task_macro: float | None = None
            objective: float | None = None
            if x_eval is not None and eval_y is not None:
                eval_pred = classifier.predict(x_eval).tolist()
                eval_acc = float(accuracy_score(eval_y, eval_pred))
                eval_macro = float(f1_score(eval_y, eval_pred, average="macro", zero_division=0))
                eval_task_macro = _task_balanced_macro_f1(eval_y, eval_pred)
                eval_focus_macro: float | None = None
                if focus_task_weights:
                    focus_scores: list[tuple[float, float]] = []
                    true_task, true_label = zip(
                        *[_split_composite(v) for v in eval_y], strict=False
                    )
                    pred_task, pred_label = zip(
                        *[_split_composite(v) for v in eval_pred], strict=False
                    )
                    for task_name, weight in focus_task_weights.items():
                        idx = [i for i, t in enumerate(true_task) if t == task_name]
                        if not idx:
                            continue
                        task_true = [true_label[i] for i in idx]
                        task_pred: list[str] = []
                        for i in idx:
                            if pred_task[i] == task_name:
                                task_pred.append(pred_label[i])
                            else:
                                task_pred.append("__wrong_task__")
                        task_labels = sorted(set(task_true))
                        score = float(
                            f1_score(
                                task_true,
                                task_pred,
                                labels=task_labels,
                                average="macro",
                                zero_division=0,
                            )
                        )
                        focus_scores.append((weight, score))
                    if focus_scores:
                        total_weight = sum(w for w, _ in focus_scores)
                        eval_focus_macro = sum(w * s for w, s in focus_scores) / max(
                            1e-12, total_weight
                        )

                if early_stopping_metric == "eval_accuracy":
                    objective = eval_acc
                elif early_stopping_metric == "eval_focus_macro_f1":
                    objective = (
                        eval_focus_macro if eval_focus_macro is not None else eval_task_macro
                    )
                elif early_stopping_metric == "eval_macro_f1":
                    objective = eval_macro
                else:
                    objective = eval_task_macro

                stat["eval_accuracy"] = eval_acc
                stat["eval_macro_f1"] = eval_macro
                stat["eval_task_macro_f1"] = eval_task_macro
                stat["eval_focus_macro_f1"] = eval_focus_macro
                stat["selection_metric"] = objective

                improved = objective > (best_metric + early_stopping_min_delta)
                if improved:
                    best_metric = objective
                    best_epoch = epoch
                    best_classifier = copy.deepcopy(classifier)
                    epochs_since_improvement = 0
                else:
                    epochs_since_improvement += 1

            epoch_stats.append(stat)

            if loss_value is None and objective is None:
                print(
                    f"[{family}] epoch {epoch}/{epochs} | train_acc={acc:.4f} train_macro_f1={macro:.4f}"
                )
            elif objective is None:
                print(
                    f"[{family}] epoch {epoch}/{epochs} | "
                    f"train_loss={loss_value:.4f} train_acc={acc:.4f} train_macro_f1={macro:.4f}"
                )
            elif loss_value is None:
                print(
                    f"[{family}] epoch {epoch}/{epochs} | "
                    f"train_acc={acc:.4f} train_macro_f1={macro:.4f} "
                    f"eval_acc={eval_acc:.4f} eval_macro_f1={eval_macro:.4f} "
                    f"eval_task_macro_f1={eval_task_macro:.4f}"
                )
            else:
                print(
                    f"[{family}] epoch {epoch}/{epochs} | "
                    f"train_loss={loss_value:.4f} train_acc={acc:.4f} train_macro_f1={macro:.4f} "
                    f"eval_acc={eval_acc:.4f} eval_macro_f1={eval_macro:.4f} "
                    f"eval_task_macro_f1={eval_task_macro:.4f}"
                )

            if (
                use_early_stopping
                and eval_y is not None
                and objective is not None
                and epochs_since_improvement >= early_stopping_patience
            ):
                print(
                    f"[{family}] early stopping at epoch {epoch} "
                    f"(best_epoch={best_epoch}, {early_stopping_metric}={best_metric:.4f})"
                )
                pbar.update(1)
                break

            pbar.update(1)
    finally:
        pbar.close()

    if best_classifier is not None:
        model.set_params(classifier=best_classifier)

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
    eval_x = _encode_features(eval_df, family)
    eval_y = _encode_targets(eval_df)
    epochs = max(2, int(train_cfg["max_iter"]))
    print(f"Fitting {family} on {len(train_df)} rows for {epochs} epochs...")
    epoch_stats = _fit_with_epoch_stats(
        model,
        train_x,
        train_y,
        epochs=epochs,
        family=family,
        train_cfg=train_cfg,
        eval_x=eval_x,
        eval_y=eval_y,
    )

    batch_size = max(64, int(train_cfg["predict_batch_size"]))
    metrics_test, reports_test = _evaluate(
        model, test_df, family=family, split_name="test", predict_batch_size=batch_size
    )
    metrics_eval, reports_eval = _evaluate(
        model, eval_df, family=family, split_name="eval", predict_batch_size=batch_size
    )

    all_labels = sorted(set(train_y) | set(_encode_targets(test_df)) | set(eval_y))
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
            "use_char_ngrams": _as_bool(train_cfg.get("use_char_ngrams", True), default=True),
            "char_ngram_min": int(train_cfg.get("char_ngram_min", 3)),
            "char_ngram_max": int(train_cfg.get("char_ngram_max", 5)),
            "char_min_df": int(train_cfg.get("char_min_df", 2)),
            "char_max_features": int(
                train_cfg.get("char_max_features", int(train_cfg["max_features"]))
            ),
            "sgd_average": _as_bool(train_cfg.get("sgd_average", True), default=True),
            "use_early_stopping": _as_bool(train_cfg.get("use_early_stopping", True), default=True),
            "early_stopping_metric": str(
                train_cfg.get("early_stopping_metric", "eval_task_macro_f1")
            ),
            "early_stopping_patience": int(train_cfg.get("early_stopping_patience", 4)),
            "early_stopping_min_delta": float(train_cfg.get("early_stopping_min_delta", 1e-4)),
            "task_weight_power": float(train_cfg.get("task_weight_power", 0.5)),
            "label_weight_power": float(train_cfg.get("label_weight_power", 0.0)),
            "sample_weight_min": float(train_cfg.get("sample_weight_min", 0.2)),
            "sample_weight_max": float(train_cfg.get("sample_weight_max", 5.0)),
            "task_focus_weights": train_cfg.get("task_focus_weights", {}),
            "focus_task_weights": train_cfg.get("focus_task_weights", {}),
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


def parse_args() -> argparse.Namespace:
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)

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
    train_cfg.setdefault("use_char_ngrams", True)
    train_cfg.setdefault("char_ngram_min", 3)
    train_cfg.setdefault("char_ngram_max", 5)
    train_cfg.setdefault("char_min_df", 2)
    train_cfg.setdefault("char_max_features", train_cfg["max_features"])
    train_cfg.setdefault("sgd_average", True)
    train_cfg.setdefault("use_early_stopping", True)
    train_cfg.setdefault("early_stopping_metric", "eval_task_macro_f1")
    train_cfg.setdefault("early_stopping_patience", 4)
    train_cfg.setdefault("early_stopping_min_delta", 1e-4)
    train_cfg.setdefault("task_weight_power", 0.5)
    train_cfg.setdefault("label_weight_power", 0.0)
    train_cfg.setdefault("sample_weight_min", 0.2)
    train_cfg.setdefault("sample_weight_max", 5.0)
    train_cfg.setdefault("task_focus_weights", {})
    train_cfg.setdefault("focus_task_weights", {})
    train_cfg.setdefault("family_overrides", {})
    train_cfg.setdefault("families", list(ALL_FAMILIES))

    if args.seed is not None:
        train_cfg["seed"] = int(args.seed)
    if args.max_iter is not None:
        train_cfg["max_iter"] = int(args.max_iter)
    if args.max_features is not None:
        train_cfg["max_features"] = int(args.max_features)
    if args.predict_batch_size is not None:
        train_cfg["predict_batch_size"] = int(args.predict_batch_size)

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

    families_summary: dict[str, dict] = {}
    manifest = {
        "config_path": str(args.config.resolve()),
        "trained_at_utc": datetime.now(UTC).isoformat(),
        "paths": {"prepared_dir": str(prepared_dir), "trained_models_dir": str(output_dir)},
        "train_settings": train_cfg,
        "families": families_summary,
    }

    pbar = _progress(total=len(families), desc="Model training", unit="family")
    try:
        for family in families:
            pbar.set_description(f"Train family [{family}]")
            family_train_cfg = _effective_family_train_cfg(train_cfg, family)
            summary = _train_one_family(
                family,
                prepared_dir=prepared_dir,
                output_dir=output_dir,
                train_cfg=family_train_cfg,
            )
            summary["effective_train_config"] = family_train_cfg
            families_summary[family] = summary
            pbar.update(1)
    finally:
        pbar.close()

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest)

    print("Training complete.")
    print(f"Models written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
