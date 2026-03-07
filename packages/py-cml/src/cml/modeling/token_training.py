"""Helpers for token/span task preparation, validation, and training."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from cml.token_runtime import HFTokenSpanPredictor

TOKEN_REQUIRED_COLUMNS = ("text", "task", "spans", "source", "language")


def token_task_split_path(prepared_dir: Path, task_name: str, split_name: str) -> Path:
    return prepared_dir / f"{task_name}_{split_name}.parquet"


def load_token_task_split(prepared_dir: Path, task_name: str, split_name: str) -> pd.DataFrame:
    path = token_task_split_path(prepared_dir, task_name, split_name)
    if not path.exists():
        raise FileNotFoundError(f"Missing prepared token split: {path}")
    df = pd.read_parquet(path)
    missing_cols = [col for col in TOKEN_REQUIRED_COLUMNS if col not in df.columns]
    if missing_cols:
        raise ValueError(f"{path.name} missing required columns: {missing_cols}")
    rows: list[dict[str, Any]] = []
    for item in df.to_dict(orient="records"):
        text = str(item.get("text", "") or "").strip()
        task = str(item.get("task", "") or "").strip()
        if not text or not task:
            continue
        spans = normalize_spans(item.get("spans"))
        rows.append(
            {
                "text": text,
                "task": task,
                "spans": spans,
                "source": str(item.get("source", "") or ""),
                "language": str(item.get("language", "en") or "en"),
            }
        )
    return pd.DataFrame(rows, columns=list(TOKEN_REQUIRED_COLUMNS))


def normalize_spans(value: Any) -> list[dict[str, Any]]:
    if value is None:
        return []
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return []
        try:
            value = json.loads(text)
        except Exception:
            return []
    elif not isinstance(value, list) and hasattr(value, "tolist"):
        try:
            value = value.tolist()
        except Exception:
            if isinstance(value, (tuple, set)):
                value = list(value)

    spans: list[dict[str, Any]] = []
    if isinstance(value, (list, tuple)):
        for item in value:
            if isinstance(item, dict):
                start = item.get("start")
                end = item.get("end")
                label = item.get("label")
            elif isinstance(item, (list, tuple)) and len(item) >= 3:
                start, end, label = item[0], item[1], item[2]
            else:
                continue
            try:
                s = int(start)  # type: ignore[arg-type]
                e = int(end)  # type: ignore[arg-type]
            except Exception:
                continue
            lab = str(label or "").strip()
            if e <= s or not lab:
                continue
            spans.append({"start": s, "end": e, "label": lab})
    spans.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
    return spans


def span_signature(spans: list[dict[str, Any]]) -> str:
    labels = sorted(str(item.get("label", "")).strip() for item in spans if item.get("label"))
    if not labels:
        return "__empty__"
    return "|".join(labels)


def span_metrics(
    gold_rows: list[list[dict[str, Any]]],
    pred_rows: list[list[tuple[int, int, str]]],
) -> dict[str, float]:
    gold_total = 0
    pred_total = 0
    true_positive = 0
    exact_matches = 0

    for gold, pred in zip(gold_rows, pred_rows, strict=False):
        gold_set = {
            (int(item["start"]), int(item["end"]), str(item["label"]))
            for item in normalize_spans(gold)
        }
        pred_set = {(int(s), int(e), str(label)) for s, e, label in pred}
        gold_total += len(gold_set)
        pred_total += len(pred_set)
        true_positive += len(gold_set & pred_set)
        if gold_set == pred_set:
            exact_matches += 1

    precision = true_positive / max(1, pred_total)
    recall = true_positive / max(1, gold_total)
    f1 = 0.0 if precision + recall == 0 else (2.0 * precision * recall) / (precision + recall)
    exact = exact_matches / max(1, len(gold_rows))
    return {
        "span_precision": float(precision),
        "span_recall": float(recall),
        "span_f1": float(f1),
        "span_exact_match": float(exact),
    }


@dataclass
class _TokenFeature:
    input_ids: list[int]
    attention_mask: list[int]
    labels: list[int]
    token_type_ids: list[int] | None = None


def _token_cfg(train_cfg: dict[str, Any]) -> dict[str, Any]:
    cfg = dict(train_cfg.get("token", {}))
    cfg.setdefault("model_name_or_path", "distilbert-base-multilingual-cased")
    cfg.setdefault("num_train_epochs", 3)
    cfg.setdefault("per_device_train_batch_size", 8)
    cfg.setdefault("per_device_eval_batch_size", 16)
    cfg.setdefault("max_seq_length", 256)
    cfg.setdefault("stride", 64)
    cfg.setdefault("learning_rate", 5e-5)
    cfg.setdefault("warmup_ratio", 0.1)
    cfg.setdefault("weight_decay", 0.01)
    cfg.setdefault("gradient_accumulation_steps", 1)
    return cfg


def _build_bio_maps(rows: list[list[dict[str, Any]]]) -> tuple[dict[str, int], dict[int, str]]:
    labels = sorted(
        {
            str(item.get("label", "")).strip()
            for row in rows
            for item in normalize_spans(row)
            if str(item.get("label", "")).strip()
        }
    )
    bio = ["O"]
    for label in labels:
        bio.append(f"B-{label}")
        bio.append(f"I-{label}")
    label_to_id = {label: idx for idx, label in enumerate(bio)}
    id_to_label = {idx: label for label, idx in label_to_id.items()}
    return label_to_id, id_to_label


def _assign_token_labels(
    offsets: list[tuple[int, int]],
    spans: list[dict[str, Any]],
    label_to_id: dict[str, int],
) -> list[int]:
    labels: list[int] = []
    for start, end in offsets:
        start = int(start)
        end = int(end)
        if end <= start:
            labels.append(-100)
            continue
        assigned = "O"
        best_overlap = 0
        for span in spans:
            span_start = int(span["start"])
            span_end = int(span["end"])
            if end <= span_start or start >= span_end:
                continue
            overlap = min(end, span_end) - max(start, span_start)
            if overlap <= 0 or overlap < best_overlap:
                continue
            prefix = "B" if start <= span_start < end or start == span_start else "I"
            assigned = f"{prefix}-{span['label']}"
            best_overlap = overlap
        labels.append(label_to_id.get(assigned, label_to_id["O"]))
    return labels


def _tokenize_examples(
    texts: list[str],
    spans_rows: list[list[dict[str, Any]]],
    *,
    tokenizer: Any,
    label_to_id: dict[str, int],
    max_seq_length: int,
    stride: int,
) -> list[_TokenFeature]:
    features: list[_TokenFeature] = []
    for text, spans in zip(texts, spans_rows, strict=False):
        encoded = tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max_seq_length,
            stride=stride,
            padding=False,
        )
        chunks = len(encoded["input_ids"])
        for idx in range(chunks):
            offsets = [(int(a), int(b)) for a, b in encoded["offset_mapping"][idx]]
            labels = _assign_token_labels(offsets, spans, label_to_id)
            feature = _TokenFeature(
                input_ids=list(encoded["input_ids"][idx]),
                attention_mask=list(encoded["attention_mask"][idx]),
                labels=labels,
                token_type_ids=(
                    list(encoded["token_type_ids"][idx]) if "token_type_ids" in encoded else None
                ),
            )
            features.append(feature)
    return features


def _collate_token_features(features: list[_TokenFeature], *, tokenizer: Any) -> dict[str, Any]:
    from transformers import DataCollatorForTokenClassification

    collator = DataCollatorForTokenClassification(tokenizer=tokenizer)
    rows: list[dict[str, Any]] = []
    for feat in features:
        item = {
            "input_ids": feat.input_ids,
            "attention_mask": feat.attention_mask,
            "labels": feat.labels,
        }
        if feat.token_type_ids is not None:
            item["token_type_ids"] = feat.token_type_ids
        rows.append(item)
    return collator(rows)


def _predict_spans_for_texts(
    *,
    model: Any,
    tokenizer: Any,
    texts: list[str],
    id_to_label: dict[int, str],
    max_seq_length: int,
    stride: int,
) -> list[list[tuple[int, int, str]]]:
    wrapper = HFTokenSpanPredictor(
        task_name="runtime",
        model_dir="",
        id_to_label=id_to_label,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    wrapper._model = model
    wrapper._tokenizer = tokenizer
    try:
        import torch
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise ImportError("torch is required for token prediction") from exc
    first_param = next(model.parameters(), None)
    wrapper._device = first_param.device if first_param is not None else torch.device("cpu")
    return wrapper.predict(texts)


def train_token_task(
    *,
    task_name: str,
    prepared_dir: Path,
    output_dir: Path,
    train_cfg: dict[str, Any],
    spec_payload: dict[str, Any],
) -> dict[str, Any]:
    try:
        import torch
        from torch.utils.data import DataLoader
        from transformers import AutoModelForTokenClassification, AutoTokenizer
    except ImportError as exc:  # pragma: no cover - optional dependency
        raise ImportError(
            "Token training requires torch and transformers. "
            'Install with: pip install "cognitive-memory-layer[modeling]"'
        ) from exc

    token_cfg = _token_cfg(train_cfg)
    train_df = load_token_task_split(prepared_dir, task_name, "train")
    test_df = load_token_task_split(prepared_dir, task_name, "test")
    eval_df = load_token_task_split(prepared_dir, task_name, "eval")

    label_rows: list[list[dict[str, Any]]] = []
    for df in (train_df, test_df, eval_df):
        label_rows.extend(df["spans"].tolist())
    label_to_id, id_to_label = _build_bio_maps(label_rows)

    model_name = str(token_cfg["model_name_or_path"])
    max_seq_length = max(32, int(token_cfg["max_seq_length"]))
    stride = max(0, int(token_cfg["stride"]))
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)

    train_features = _tokenize_examples(
        train_df["text"].astype(str).tolist(),
        train_df["spans"].tolist(),
        tokenizer=tokenizer,
        label_to_id=label_to_id,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    if not train_features:
        raise RuntimeError(f"[task:{task_name}] token split produced no train features")

    train_loader: Any = DataLoader(
        train_features,  # type: ignore[arg-type]
        batch_size=max(1, int(token_cfg["per_device_train_batch_size"])),
        shuffle=True,
        collate_fn=lambda batch: _collate_token_features(batch, tokenizer=tokenizer),
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForTokenClassification.from_pretrained(
        model_name,
        num_labels=len(label_to_id),
        id2label=id_to_label,
        label2id=label_to_id,
    )
    model.to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(token_cfg["learning_rate"]),
        weight_decay=float(token_cfg["weight_decay"]),
    )
    gradient_accumulation = max(1, int(token_cfg["gradient_accumulation_steps"]))
    epochs = max(1, int(token_cfg["num_train_epochs"]))
    total_steps = max(1, math.ceil(len(train_loader) / gradient_accumulation) * epochs)
    warmup_steps = int(total_steps * float(token_cfg["warmup_ratio"]))
    from transformers import get_linear_schedule_with_warmup

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=warmup_steps,
        num_training_steps=total_steps,
    )

    epoch_stats: list[dict[str, Any]] = []
    model.train()
    for epoch in range(1, epochs + 1):
        total_loss = 0.0
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(train_loader, start=1):
            batch = {k: v.to(device) for k, v in batch.items()}
            loss = model(**batch).loss / gradient_accumulation
            loss.backward()
            total_loss += float(loss.item()) * gradient_accumulation
            if step % gradient_accumulation == 0 or step == len(train_loader):
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
        epoch_stats.append(
            {
                "epoch": epoch,
                "train_loss": float(total_loss / max(1, len(train_loader))),
            }
        )

    model.eval()
    test_pred = _predict_spans_for_texts(
        model=model,
        tokenizer=tokenizer,
        texts=test_df["text"].astype(str).tolist(),
        id_to_label=id_to_label,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    eval_pred = _predict_spans_for_texts(
        model=model,
        tokenizer=tokenizer,
        texts=eval_df["text"].astype(str).tolist(),
        id_to_label=id_to_label,
        max_seq_length=max_seq_length,
        stride=stride,
    )

    test_metrics = span_metrics(test_df["spans"].tolist(), test_pred)
    test_metrics["rows"] = len(test_df)
    eval_metrics = span_metrics(eval_df["spans"].tolist(), eval_pred)
    eval_metrics["rows"] = len(eval_df)

    hf_model_dir = output_dir / f"{task_name}_hf"
    hf_model_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(str(hf_model_dir))
    tokenizer.save_pretrained(str(hf_model_dir))

    runtime_model = HFTokenSpanPredictor(
        task_name=task_name,
        model_dir=str(hf_model_dir),
        id_to_label=id_to_label,
        max_seq_length=max_seq_length,
        stride=stride,
    )
    model_path = output_dir / f"{task_name}_model.joblib"
    import joblib

    joblib.dump(
        {
            "model": runtime_model,
            "task_spec": dict(spec_payload),
            "labels": {str(k): v for k, v in id_to_label.items()},
            "hf_model_dir": str(hf_model_dir),
        },
        model_path,
    )

    return {
        "task": task_name,
        "objective": "token_classification",
        "model_path": str(model_path),
        "hf_model_dir": str(hf_model_dir),
        "train_rows": len(train_df),
        "test": test_metrics,
        "eval": eval_metrics,
        "labels": {str(k): v for k, v in id_to_label.items()},
        "epoch_stats": epoch_stats,
    }
