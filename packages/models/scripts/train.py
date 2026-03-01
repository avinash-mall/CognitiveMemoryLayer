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
import json
import sys
import tomllib
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
from sklearn.pipeline import Pipeline

MODELS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODELS_ROOT.parent.parent
DEFAULT_CONFIG_PATH = MODELS_ROOT / "model_pipeline.toml"

FAMILY_SINGLE_TEXT = {"router", "extractor"}
FAMILY_PAIR_TEXT = {"pair"}
ALL_FAMILIES = ("router", "extractor", "pair")


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

    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json(output_dir / "manifest.json", manifest)

    print("Training complete.")
    print(f"Models written to: {output_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
