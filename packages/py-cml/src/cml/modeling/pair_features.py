"""Shared helpers for embedding-backed pair models."""

from __future__ import annotations

import hashlib
import re
from collections import Counter
from math import log
from pathlib import Path

import numpy as np

EMBEDDING_CACHE_FILENAME = "pair_text_embeddings.parquet"
# Unicode-aware token pattern: Latin words, CJK individual characters, other scripts
_TOKEN_RE = re.compile(
    r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?"  # Latin/digits
    r"|[\u4e00-\u9fff\u3400-\u4dbf]"  # CJK Unified Ideographs
    r"|[\u3040-\u309f\u30a0-\u30ff]+"  # Hiragana + Katakana runs
    r"|[\uac00-\ud7af]+"  # Korean Hangul syllables
    r"|[\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff]+"  # Arabic
    r"|[\u0900-\u097f]+"  # Devanagari (Hindi)
    r"|[\u0400-\u04ff]+"  # Cyrillic (Russian)
    r"|[\u00c0-\u024f]+"  # Latin Extended (French, German, Turkish, Vietnamese accented)
)
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_NEGATION_TOKENS = {
    # English
    "no", "not", "never", "none", "nothing", "neither", "nor",
    "cannot", "cant", "can't", "dont", "don't", "won't", "without",
    # Chinese
    "不", "没", "没有", "无", "非", "未", "别",
    # Spanish
    "nunca", "nada", "ninguno", "ninguna", "ni", "sin", "jamás",
    # Arabic
    "لا", "ليس", "لم", "لن", "غير", "بدون", "ما",
    # Hindi
    "नहीं", "न", "मत", "बिना", "कभी",
    # Portuguese
    "não", "nenhum", "nenhuma", "nem", "sem",
    # French
    "ne", "pas", "jamais", "rien", "aucun", "aucune", "sans",
    # Japanese
    "ない", "ず", "ぬ", "なし",
    # Russian
    "не", "нет", "никогда", "ничего", "без", "ни",
    # German
    "nicht", "kein", "keine", "nie", "niemals", "ohne", "weder", "noch",
    # Korean
    "않", "못", "안", "없",
    # Turkish
    "değil", "yok", "hiç", "asla",
    # Indonesian
    "tidak", "bukan", "tanpa", "tak", "belum",
    # Vietnamese
    "không", "chưa", "chẳng",
    # Italian
    "non", "mai", "niente", "nessuno", "nessuna", "senza",
}


def hash_text(text: str) -> str:
    """Stable short hash used for embedding cache joins."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]


def pair_embedding_cache_path(prepared_dir: Path) -> Path:
    return prepared_dir / EMBEDDING_CACHE_FILENAME


def parse_serialized_pair_feature(feature: str) -> tuple[str, str, str]:
    """Parse `task=<task> [a] <text_a> [b] <text_b>`."""
    text = str(feature or "")
    if " [a] " not in text or " [b] " not in text:
        return "", "", ""
    task_part, remainder = text.split(" [a] ", 1)
    text_a, text_b = remainder.split(" [b] ", 1)
    task = task_part.removeprefix("task=").strip()
    return task, text_a.strip(), text_b.strip()


def _tokenize(text: str) -> list[str]:
    return [match.group(0).casefold() for match in _TOKEN_RE.finditer(str(text or ""))]


def build_pair_lexical_features(text_a: str, text_b: str) -> np.ndarray:
    """Construct low-cost lexical interaction features from raw pair texts."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    set_a = set(tokens_a)
    set_b = set(tokens_b)
    union = set_a | set_b
    overlap = float(len(set_a & set_b) / max(1, len(union)))

    len_a = len(tokens_a)
    len_b = len(tokens_b)
    max_len = max(1, len_a, len_b)
    len_ratio = float(min(len_a, len_b) / max_len)
    len_delta = float(abs(len_a - len_b) / max_len)

    nums_a = set(_NUMBER_RE.findall(str(text_a or "")))
    nums_b = set(_NUMBER_RE.findall(str(text_b or "")))
    if not nums_a and not nums_b:
        numeric_overlap = 0.0
    else:
        numeric_overlap = float(len(nums_a & nums_b) / max(1, len(nums_a | nums_b)))

    neg_a = any(token in _NEGATION_TOKENS for token in tokens_a)
    neg_b = any(token in _NEGATION_TOKENS for token in tokens_b)
    negation_mismatch = 1.0 if neg_a != neg_b else 0.0

    # BM25-like scalar: term-frequency weighted overlap (stronger relevance signal)
    if not tokens_b:
        bm25_like = 0.0
    else:
        cnt_b = Counter(tokens_b)
        set_a = set(tokens_a)
        bm25_like = sum(
            1.0 / (1.0 + log(1.0 + cnt_b[t])) for t in set_a if t in cnt_b
        ) / max(1, len(tokens_b))
        bm25_like = min(1.0, float(bm25_like))

    return np.asarray(
        [
            overlap,
            len_ratio,
            len_delta,
            numeric_overlap,
            negation_mismatch,
            bm25_like,
        ],
        dtype=np.float32,
    )


def build_pair_dense_features(
    embedding_a: object,
    embedding_b: object,
    *,
    text_a: str = "",
    text_b: str = "",
) -> np.ndarray:
    """Construct similarity + interaction features from embeddings and raw text."""
    vec_a = np.asarray(embedding_a, dtype=np.float32)
    vec_b = np.asarray(embedding_b, dtype=np.float32)
    if vec_a.ndim != 1 or vec_b.ndim != 1 or vec_a.size == 0 or vec_a.shape != vec_b.shape:
        raise ValueError("Embedding vectors must be non-empty 1D arrays with matching shapes.")

    dot = float(np.dot(vec_a, vec_b))
    norm = float(np.linalg.norm(vec_a) * np.linalg.norm(vec_b))
    cosine = dot / norm if norm > 0 else 0.0
    euclidean = float(np.linalg.norm(vec_a - vec_b))
    abs_diff = np.abs(vec_a - vec_b)
    product = vec_a * vec_b
    lexical = build_pair_lexical_features(text_a, text_b)
    return np.concatenate(
        [
            np.array([cosine, euclidean], dtype=np.float32),
            lexical,
            abs_diff.astype(np.float32, copy=False),
            product.astype(np.float32, copy=False),
        ]
    )
