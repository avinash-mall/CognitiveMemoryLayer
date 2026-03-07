"""Runtime wrapper for local Hugging Face token/span models."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_PII_FALLBACK_PATTERNS: tuple[tuple[str, re.Pattern[str]], ...] = (
    (
        "EMAIL",
        re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}\b", flags=re.IGNORECASE),
    ),
    (
        "PHONE",
        re.compile(
            r"(?<!\w)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\w)",
            flags=re.IGNORECASE,
        ),
    ),
    ("SSN", re.compile(r"\b\d{3}-\d{2}-\d{4}\b", flags=re.IGNORECASE)),
    (
        "CREDIT_CARD",
        re.compile(r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b", flags=re.IGNORECASE),
    ),
    (
        "IP_ADDRESS",
        re.compile(r"\b\d{1,3}(?:\.\d{1,3}){3}\b", flags=re.IGNORECASE),
    ),
    (
        "SECRET",
        re.compile(
            r"\b(?:sk-[A-Za-z0-9_-]+|tok_[A-Za-z0-9_-]+|AKIA[0-9A-Z]{16}|gh[pousr]_[A-Za-z0-9]+)\b",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "SECRET",
        re.compile(
            r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]+?-----END [A-Z ]*PRIVATE KEY-----",
            flags=re.IGNORECASE,
        ),
    ),
)


@dataclass
class HFTokenSpanPredictor:
    """Lazy token-classification wrapper that returns char-offset spans."""

    task_name: str
    model_dir: str
    id_to_label: dict[int, str]
    max_seq_length: int = 256
    stride: int = 64

    _tokenizer: Any = field(default=None, init=False)
    _model: Any = field(default=None, init=False)
    _device: Any = field(default=None, init=False)

    def __post_init__(self) -> None:
        pass

    def _ensure_loaded(self) -> None:
        if self._tokenizer is not None and self._model is not None:
            return
        try:
            import torch
            from transformers import AutoModelForTokenClassification, AutoTokenizer
        except Exception as exc:  # pragma: no cover - depends on optional deps
            raise ImportError(
                "Token runtime requires transformers and torch. "
                'Install with: pip install "cognitive-memory-layer[modeling]"'
            ) from exc

        model_path = Path(self.model_dir)
        if not model_path.exists():
            raise FileNotFoundError(f"Token model directory not found: {model_path}")

        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._tokenizer = AutoTokenizer.from_pretrained(str(model_path), use_fast=True)
        self._model = AutoModelForTokenClassification.from_pretrained(str(model_path))
        self._model.to(self._device)
        self._model.eval()

    def predict(self, texts: list[str]) -> list[list[tuple[int, int, str]]]:
        self._ensure_loaded()
        return [self._predict_one(text) for text in texts]

    def _predict_one(self, text: str) -> list[tuple[int, int, str]]:
        if not text.strip():
            return []

        import torch

        assert self._tokenizer is not None
        assert self._model is not None
        assert self._device is not None

        encoded = self._tokenizer(
            text,
            return_offsets_mapping=True,
            return_overflowing_tokens=True,
            truncation=True,
            max_length=max(32, int(self.max_seq_length)),
            stride=max(0, int(self.stride)),
            padding=False,
        )

        spans: list[tuple[int, int, str]] = []
        total_chunks = len(encoded["input_ids"])
        for idx in range(total_chunks):
            inputs = {}
            for key in ("input_ids", "attention_mask", "token_type_ids"):
                if key in encoded:
                    inputs[key] = torch.tensor([encoded[key][idx]], device=self._device)
            with torch.no_grad():
                logits = self._model(**inputs).logits[0]
            pred_ids = logits.argmax(dim=-1).tolist()
            offsets = encoded["offset_mapping"][idx]
            spans.extend(self._decode_chunk(pred_ids, offsets, text))

        merged = self._merge_spans(spans)
        if self.task_name == "pii_span_detection":
            merged = self._merge_spans(merged + self._regex_pii_spans(text))
        return merged

    def _decode_chunk(
        self,
        pred_ids: list[int],
        offsets: list[tuple[int, int]],
        text: str,
    ) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        current_label = ""
        current_start: int | None = None
        current_end: int | None = None

        def flush() -> None:
            nonlocal current_label, current_start, current_end
            if (
                current_start is not None
                and current_end is not None
                and current_end > current_start
            ):
                span_text = text[current_start:current_end].strip()
                if span_text:
                    spans.append((current_start, current_end, current_label))
            current_label = ""
            current_start = None
            current_end = None

        for pred_id, offset in zip(pred_ids, offsets, strict=False):
            start, end = int(offset[0]), int(offset[1])
            if end <= start:
                flush()
                continue
            raw_label = str(self.id_to_label.get(int(pred_id), "O"))
            if raw_label == "O":
                flush()
                continue
            prefix = "B"
            label = raw_label
            if "-" in raw_label:
                prefix, label = raw_label.split("-", 1)
            prefix = prefix.upper()
            if prefix == "B" or current_label != label:
                flush()
                current_label = label
                current_start = start
                current_end = end
            else:
                current_end = max(current_end or end, end)
        flush()
        return spans

    @staticmethod
    def _merge_spans(spans: list[tuple[int, int, str]]) -> list[tuple[int, int, str]]:
        if not spans:
            return []
        unique = sorted(
            set((int(s), int(e), str(lbl)) for s, e, lbl in spans),
            key=lambda x: (x[0], x[1], x[2]),
        )
        merged: list[tuple[int, int, str]] = []
        for start, end, label in unique:
            if not merged:
                merged.append((start, end, label))
                continue
            prev_start, prev_end, prev_label = merged[-1]
            if label == prev_label and start <= prev_end:
                merged[-1] = (prev_start, max(prev_end, end), prev_label)
                continue
            merged.append((start, end, label))
        return merged

    @staticmethod
    def _regex_pii_spans(text: str) -> list[tuple[int, int, str]]:
        spans: list[tuple[int, int, str]] = []
        for label, pattern in _PII_FALLBACK_PATTERNS:
            for match in pattern.finditer(text):
                spans.append((match.start(), match.end(), label))
        return spans
