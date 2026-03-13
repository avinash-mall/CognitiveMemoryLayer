"""Runtime-friendly wrappers for non-standard task-model artifacts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from cml.modeling.memory_type_features import append_memory_type_feature_tokens
from cml.modeling.pair_features import build_pair_dense_features, parse_serialized_pair_feature

_PAIR_GROUP_TOP1_TASKS = {
    "retrieval_constraint_relevance_pair",
    "memory_rerank_pair",
    "reconsolidation_candidate_pair",
}


def _task_name_from_feature(feature: str) -> str:
    text = str(feature or "")
    if not text.startswith("task="):
        return ""
    tail = text.removeprefix("task=")
    return tail.split(" ", 1)[0].strip()


def build_task_conditional_calibration_features(task_probs: np.ndarray) -> np.ndarray:
    probs = np.asarray(task_probs, dtype=np.float64)
    if probs.ndim != 2:
        raise ValueError("Task calibration features require a 2D probability matrix.")
    row_sums = probs.sum(axis=1, keepdims=True)
    normalized = probs / np.maximum(row_sums, 1e-12)
    clipped = np.clip(normalized, 1e-8, 1.0)
    return np.log(clipped)


def _predict_proba_for_model(model: Any, features: list[str]) -> np.ndarray:
    if hasattr(model, "predict_proba"):
        probs = np.asarray(model.predict_proba(features), dtype=np.float64)
        if probs.ndim == 2:
            return probs
    if hasattr(model, "decision_function"):
        raw = np.asarray(model.decision_function(features), dtype=np.float64)
        if raw.ndim == 1:
            pos = 1.0 / (1.0 + np.exp(-raw))
            return np.column_stack([1.0 - pos, pos])
        exp = np.exp(raw - raw.max(axis=1, keepdims=True))
        denom = exp.sum(axis=1, keepdims=True)
        return exp / np.maximum(denom, 1e-12)
    preds = [str(x) for x in model.predict(features)]
    classes = [str(x) for x in getattr(model, "classes_", [])]
    if not classes:
        classes = sorted(set(preds))
    out = np.zeros((len(features), len(classes)), dtype=np.float64)
    index = {name: idx for idx, name in enumerate(classes)}
    for row_idx, pred in enumerate(preds):
        out[row_idx, index.get(pred, 0)] = 1.0
    return out


@dataclass
class EmbeddingPairClassifier:
    """Lazy sentence-transformer wrapper around a dense sklearn classifier."""

    task_name: str
    model_name_or_path: str
    classifier: Any
    classes_: list[str]
    batch_size: int = 32

    _encoder: Any = field(default=None, init=False, repr=False)

    def _ensure_encoder(self) -> Any:
        if self._encoder is not None:
            return self._encoder
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - depends on optional extras
            raise ImportError(
                "Embedding pair runtime requires sentence-transformers. "
                'Install with: pip install "cognitive-memory-layer[modeling]"'
            ) from exc
        self._encoder = SentenceTransformer(self.model_name_or_path)
        return self._encoder

    def _dense_matrix(self, features: list[str]) -> np.ndarray:
        encoder = self._ensure_encoder()
        pairs: list[tuple[str, str]] = []
        unique_texts: dict[str, None] = {}
        for feature in features:
            _, text_a, text_b = parse_serialized_pair_feature(str(feature))
            pairs.append((text_a, text_b))
            if text_a:
                unique_texts.setdefault(text_a, None)
            if text_b:
                unique_texts.setdefault(text_b, None)
        ordered_texts = list(unique_texts.keys())
        encoded = encoder.encode(
            ordered_texts,
            batch_size=max(1, int(self.batch_size)),
            convert_to_numpy=True,
            show_progress_bar=False,
            normalize_embeddings=True,
        )
        vectors = {
            text: np.asarray(vector, dtype=np.float32)
            for text, vector in zip(ordered_texts, encoded, strict=False)
        }
        rows = [
            build_pair_dense_features(
                vectors[text_a],
                vectors[text_b],
                text_a=text_a,
                text_b=text_b,
            )
            for text_a, text_b in pairs
        ]
        return np.vstack(rows).astype(np.float32, copy=False)

    def predict_proba(self, features: list[str]) -> np.ndarray:
        dense = self._dense_matrix(features)
        return np.asarray(self.classifier.predict_proba(dense), dtype=np.float64)

    def predict(self, features: list[str]) -> np.ndarray:
        probs = self.predict_proba(features)
        relevant_label = f"{self.task_name}::relevant"
        not_relevant_label = f"{self.task_name}::not_relevant"
        if (
            self.task_name in _PAIR_GROUP_TOP1_TASKS
            and relevant_label in self.classes_
            and not_relevant_label in self.classes_
        ):
            pos_idx = self.classes_.index(relevant_label)
            neg_idx = self.classes_.index(not_relevant_label)
            indices = np.full(len(features), neg_idx, dtype=int)
            groups: dict[tuple[str, str], list[int]] = {}
            for row_idx, feature in enumerate(features):
                task_name, text_a, _ = parse_serialized_pair_feature(str(feature))
                key = ((task_name or self.task_name).strip(), text_a.strip())
                groups.setdefault(key, []).append(row_idx)
            for rows in groups.values():
                if len(rows) <= 1:
                    row_idx = rows[0]
                    indices[row_idx] = int(probs[row_idx].argmax())
                    continue
                best_idx = max(rows, key=lambda row_idx: float(probs[row_idx, pos_idx]))
                indices[best_idx] = pos_idx
        else:
            indices = probs.argmax(axis=1)
        return np.asarray([self.classes_[idx] for idx in indices], dtype=object)

@dataclass
class CumulativeOrdinalClassifier:
    """Ordinal classifier assembled from calibrated cumulative binary boundaries."""

    task_name: str
    vectorizer: Any
    boundary_models: list[Any]
    label_order: list[str]
    classes_: list[str]

    def _boundary_probs(self, features: list[str]) -> np.ndarray:
        if not self.boundary_models:
            raise ValueError("At least one boundary model is required.")
        matrix = self.vectorizer.transform(features)
        columns: list[np.ndarray] = []
        for model in self.boundary_models:
            probs = np.asarray(model.predict_proba(matrix), dtype=np.float64)
            if probs.ndim != 2 or probs.shape[1] < 2:
                raise ValueError("Boundary model must expose binary predict_proba outputs.")
            columns.append(probs[:, 1])
        raw = np.column_stack(columns)
        # Enforce monotonic cumulative probabilities: P(y > k) must decrease as k grows.
        return np.minimum.accumulate(raw, axis=1)

    def predict_proba(self, features: list[str]) -> np.ndarray:
        cumulative = self._boundary_probs(features)
        rows = len(features)
        classes = len(self.label_order)
        probs = np.zeros((rows, classes), dtype=np.float64)
        probs[:, 0] = 1.0 - cumulative[:, 0]
        for idx in range(1, classes - 1):
            probs[:, idx] = cumulative[:, idx - 1] - cumulative[:, idx]
        probs[:, classes - 1] = cumulative[:, classes - 2]
        np.clip(probs, 0.0, 1.0, out=probs)
        row_sums = probs.sum(axis=1, keepdims=True)
        return probs / np.maximum(row_sums, 1e-12)

    def predict(self, features: list[str]) -> np.ndarray:
        probs = self.predict_proba(features)
        indices = probs.argmax(axis=1)
        return np.asarray([self.classes_[idx] for idx in indices], dtype=object)


@dataclass
class TaskConditionalCalibratedClassifier:
    """Apply task-specific calibration on top of a shared family classifier."""

    base_model: Any
    classes_: list[str]
    task_label_indices: dict[str, list[int]]
    calibrators: dict[str, Any]
    calibrator_classes: dict[str, list[int]]

    def predict_proba(self, features: list[str]) -> np.ndarray:
        raw = np.asarray(self.base_model.predict_proba(features), dtype=np.float64)
        out = np.zeros_like(raw)
        tasks = [_task_name_from_feature(str(feature)) for feature in features]
        task_rows: dict[str, list[int]] = {}
        for row_idx, task in enumerate(tasks):
            task_rows.setdefault(task, []).append(row_idx)

        for task, rows in task_rows.items():
            indices = self.task_label_indices.get(task)
            if not indices:
                out[rows] = raw[rows]
                continue
            task_raw = raw[np.ix_(rows, indices)]
            row_sums = task_raw.sum(axis=1, keepdims=True)
            normalized = task_raw / np.maximum(row_sums, 1e-12)
            calibrator = self.calibrators.get(task)
            if calibrator is None:
                calibrated = normalized
            else:
                cal_features = build_task_conditional_calibration_features(normalized)
                cal_raw = np.asarray(calibrator.predict_proba(cal_features), dtype=np.float64)
                calibrated = np.zeros_like(normalized)
                class_ids = self.calibrator_classes.get(task, [])
                for src_idx, class_id in enumerate(class_ids):
                    if 0 <= int(class_id) < calibrated.shape[1]:
                        calibrated[:, int(class_id)] = cal_raw[:, src_idx]
                empty = calibrated.sum(axis=1) <= 0
                if np.any(empty):
                    calibrated[empty] = normalized[empty]
            normalized_cal = calibrated / np.maximum(calibrated.sum(axis=1, keepdims=True), 1e-12)
            out[np.ix_(rows, indices)] = normalized_cal

        missing = out.sum(axis=1) <= 0
        if np.any(missing):
            out[missing] = raw[missing]
        return out / np.maximum(out.sum(axis=1, keepdims=True), 1e-12)

    def predict(self, features: list[str]) -> np.ndarray:
        probs = self.predict_proba(features)
        indices = probs.argmax(axis=1)
        return np.asarray([self.classes_[idx] for idx in indices], dtype=object)


@dataclass
class HierarchicalTextClassifier:
    """Two-stage text classifier with macro and fine label probabilities."""

    task_name: str
    stage1_model: Any
    stage2_models: dict[str, Any]
    macro_to_labels: dict[str, list[str]]
    classes_: list[str]

    def _enriched_features(self, features: list[str]) -> list[str]:
        return [append_memory_type_feature_tokens(str(feature)) for feature in features]

    def predict_proba(self, features: list[str]) -> np.ndarray:
        enriched = self._enriched_features(features)
        stage1_probs = _predict_proba_for_model(self.stage1_model, enriched)
        stage1_classes = [str(x) for x in getattr(self.stage1_model, "classes_", [])]
        macro_index = {name: idx for idx, name in enumerate(stage1_classes)}
        class_index = {name: idx for idx, name in enumerate(self.classes_)}
        out = np.zeros((len(features), len(self.classes_)), dtype=np.float64)

        stage2_cache: dict[str, tuple[list[str], np.ndarray]] = {}
        for macro_name, model in self.stage2_models.items():
            probs = _predict_proba_for_model(model, enriched)
            labels = [str(x) for x in getattr(model, "classes_", [])]
            stage2_cache[macro_name] = (labels, probs)

        for macro_name, labels in self.macro_to_labels.items():
            macro_prob = (
                stage1_probs[:, macro_index[macro_name]]
                if macro_name in macro_index
                else np.zeros(len(features), dtype=np.float64)
            )
            raw_labels, fine_probs = stage2_cache[macro_name]
            fine_index = {name: idx for idx, name in enumerate(raw_labels)}
            for label in labels:
                composite = f"{self.task_name}::{label}"
                out[:, class_index[composite]] = macro_prob * fine_probs[:, fine_index[label]]

        row_sums = out.sum(axis=1, keepdims=True)
        empty_rows = row_sums.squeeze(axis=1) <= 0
        if np.any(empty_rows):
            pred = np.asarray(self.stage1_model.predict(enriched), dtype=object)
            for row_idx, macro_name in enumerate(pred):
                if not empty_rows[row_idx]:
                    continue
                labels = self.macro_to_labels.get(str(macro_name), [])
                if labels:
                    composite = f"{self.task_name}::{labels[0]}"
                    out[row_idx, class_index[composite]] = 1.0
        row_sums = out.sum(axis=1, keepdims=True)
        return out / np.maximum(row_sums, 1e-12)

    def predict(self, features: list[str]) -> np.ndarray:
        probs = self.predict_proba(features)
        indices = probs.argmax(axis=1)
        return np.asarray([self.classes_[idx] for idx in indices], dtype=object)
