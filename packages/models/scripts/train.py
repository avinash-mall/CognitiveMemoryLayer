"""
Train, test, and eval script for the 15-way memory-type classifier.

Reads:
  - packages/models/prepared/train.parquet
  - packages/models/prepared/test.parquet
  - packages/models/prepared/eval.parquet

Writes:
  - packages/models/artifacts/classifier/model.joblib
  - packages/models/artifacts/classifier/label_map.json
  - packages/models/artifacts/classifier/metrics_test.json
  - packages/models/artifacts/classifier/metrics_eval.json
  - packages/models/artifacts/classifier/classification_report_test.json
  - packages/models/artifacts/classifier/classification_report_eval.json
"""

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, Tuple

import joblib
import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    _tqdm = None
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.pipeline import Pipeline


MODELS_ROOT = Path(__file__).resolve().parent.parent
PREPARED_DIR = MODELS_ROOT / "prepared"
ARTIFACT_DIR = MODELS_ROOT / "artifacts" / "classifier"


class _NoopProgress:
    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")

    def update(self, n: int = 1) -> None:
        return

    def set_description(self, desc: str) -> None:
        return

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs) -> None:
        return

    def close(self) -> None:
        return


def _progress(*, total: int, desc: str, unit: str):
    if _tqdm is None:
        return _NoopProgress(total=total, desc=desc, unit=unit)
    return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def load_split(path: Path, split_name: str) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{split_name} split missing: {path}")
    df = pd.read_parquet(path)
    required = {"text", "label"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{split_name} split missing required columns: {sorted(missing)}")
    df = df.dropna(subset=["text", "label"]).copy()
    df["text"] = df["text"].astype(str).str.strip()
    df["label"] = df["label"].astype(str).str.strip()
    df = df[(df["text"] != "") & (df["label"] != "")]
    if df.empty:
        raise ValueError(f"{split_name} split is empty after cleanup: {path}")
    return df


def build_pipeline(
    *,
    max_features: int,
    min_df: int,
    max_iter: int,
    alpha: float,
    random_seed: int,
) -> Pipeline:
    vectorizer = TfidfVectorizer(
        lowercase=True,
        strip_accents="unicode",
        ngram_range=(1, 2),
        min_df=min_df,
        max_features=max_features,
        sublinear_tf=True,
    )
    classifier = SGDClassifier(
        loss="log_loss",
        penalty="l2",
        alpha=alpha,
        max_iter=max_iter,
        early_stopping=True,
        n_iter_no_change=3,
        class_weight="balanced",
        random_state=random_seed,
    )
    return Pipeline(
        steps=[
            ("vectorizer", vectorizer),
            ("classifier", classifier),
        ]
    )


def evaluate_split(
    model: Pipeline,
    df: pd.DataFrame,
    *,
    split_name: str,
    predict_batch_size: int,
) -> Tuple[Dict, Dict]:
    y_true = df["label"].tolist()
    texts = df["text"].tolist()
    y_pred = []
    pred_pbar = _progress(total=len(texts), desc=f"Predict {split_name}", unit="row")
    try:
        for start in range(0, len(texts), predict_batch_size):
            batch = texts[start : start + predict_batch_size]
            y_pred.extend(model.predict(batch).tolist())
            pred_pbar.update(len(batch))
    finally:
        pred_pbar.close()

    labels_in_split = sorted(set(y_true))
    report = classification_report(
        y_true,
        y_pred,
        labels=labels_in_split,
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "split": split_name,
        "rows": int(len(df)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "macro_f1": float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "weighted_f1": float(f1_score(y_true, y_pred, average="weighted", zero_division=0)),
        "micro_f1": float(f1_score(y_true, y_pred, average="micro", zero_division=0)),
        "labels": labels_in_split,
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=labels_in_split).tolist(),
    }
    return metrics, report


def write_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train/test/eval memory-type classifier from prepared parquet splits."
    )
    parser.add_argument(
        "--train-path",
        type=Path,
        default=PREPARED_DIR / "train.parquet",
        help="Path to train parquet.",
    )
    parser.add_argument(
        "--test-path",
        type=Path,
        default=PREPARED_DIR / "test.parquet",
        help="Path to test parquet.",
    )
    parser.add_argument(
        "--eval-path",
        type=Path,
        default=PREPARED_DIR / "eval.parquet",
        help="Path to eval parquet.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ARTIFACT_DIR,
        help="Directory where model and reports are written.",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=200_000,
        help="Max TF-IDF features.",
    )
    parser.add_argument(
        "--min-df",
        type=int,
        default=2,
        help="Minimum document frequency for TF-IDF tokens.",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=20,
        help="Max SGD iterations.",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=1e-5,
        help="SGD regularization strength.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed.",
    )
    parser.add_argument(
        "--predict-batch-size",
        type=int,
        default=4096,
        help="Batch size used for prediction progress on test/eval.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    predict_batch_size = max(64, int(args.predict_batch_size))
    stage_pbar = _progress(total=7, desc="Train/Eval pipeline", unit="stage")
    try:
        stage_pbar.set_description("Load train")
        train_df = load_split(args.train_path, "train")
        stage_pbar.update(1)
        stage_pbar.set_postfix({"rows": len(train_df)})

        stage_pbar.set_description("Load test")
        test_df = load_split(args.test_path, "test")
        stage_pbar.update(1)
        stage_pbar.set_postfix({"rows": len(test_df)})

        stage_pbar.set_description("Load eval")
        eval_df = load_split(args.eval_path, "eval")
        stage_pbar.update(1)
        stage_pbar.set_postfix({"rows": len(eval_df)})

        stage_pbar.set_description("Build model")
        model = build_pipeline(
            max_features=max(500, args.max_features),
            min_df=max(1, args.min_df),
            max_iter=max(2, args.max_iter),
            alpha=max(1e-8, args.alpha),
            random_seed=args.seed,
        )
        stage_pbar.update(1)

        stage_pbar.set_description("Train model")
        print(f"Training on {len(train_df)} rows...")
        model.fit(train_df["text"].tolist(), train_df["label"].tolist())
        stage_pbar.update(1)

        stage_pbar.set_description("Evaluate test")
        metrics_test, report_test = evaluate_split(
            model,
            test_df,
            split_name="test",
            predict_batch_size=predict_batch_size,
        )
        stage_pbar.update(1)

        stage_pbar.set_description("Evaluate eval")
        metrics_eval, report_eval = evaluate_split(
            model,
            eval_df,
            split_name="eval",
            predict_batch_size=predict_batch_size,
        )
        stage_pbar.update(1)
    finally:
        stage_pbar.close()

    trained_at = datetime.now(timezone.utc).isoformat()
    all_labels = sorted(set(train_df["label"]).union(test_df["label"]).union(eval_df["label"]))
    label_to_id = {label: i for i, label in enumerate(all_labels)}
    id_to_label = {str(i): label for label, i in label_to_id.items()}

    metadata = {
        "trained_at_utc": trained_at,
        "train_rows": int(len(train_df)),
        "test_rows": int(len(test_df)),
        "eval_rows": int(len(eval_df)),
        "labels": all_labels,
        "label_to_id": label_to_id,
        "vectorizer": {
            "type": "TfidfVectorizer",
            "max_features": int(max(500, args.max_features)),
            "min_df": int(max(1, args.min_df)),
            "ngram_range": [1, 2],
        },
        "classifier": {
            "type": "SGDClassifier",
            "loss": "log_loss",
            "alpha": float(max(1e-8, args.alpha)),
            "max_iter": int(max(2, args.max_iter)),
            "class_weight": "balanced",
            "seed": int(args.seed),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    model_path = args.output_dir / "model.joblib"
    joblib.dump(
        {
            "model": model,
            "metadata": metadata,
            "label_to_id": label_to_id,
            "id_to_label": id_to_label,
        },
        model_path,
    )
    write_json(args.output_dir / "label_map.json", {"label_to_id": label_to_id, "id_to_label": id_to_label})
    write_json(args.output_dir / "metrics_test.json", metrics_test)
    write_json(args.output_dir / "metrics_eval.json", metrics_eval)
    write_json(args.output_dir / "classification_report_test.json", report_test)
    write_json(args.output_dir / "classification_report_eval.json", report_eval)
    write_json(args.output_dir / "training_metadata.json", metadata)

    print("Model written to:", model_path)
    print("Test metrics:", json.dumps(metrics_test, indent=2))
    print("Eval metrics:", json.dumps(metrics_eval, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
