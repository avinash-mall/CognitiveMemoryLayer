"""
Unified preparation script for all custom model families.

Families prepared:
1) router      (single-text multi-task)
2) extractor   (single-text multi-task)
3) pair        (text-pair multi-task)

Configuration:
  packages/models/model_pipeline.toml
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import re
import shutil
import sys
import threading
import time
import tomllib
import warnings
from collections import defaultdict
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast
from urllib.parse import urlparse

import numpy as np

try:
    import pandas as pd
except ImportError as exc:
    raise ImportError(
        "Modeling dependency missing: pandas. Install with: "
        'pip install "cognitive-memory-layer[modeling]"'
    ) from exc

try:
    import httpx
except Exception:  # pragma: no cover - optional dependency
    httpx = None  # type: ignore[assignment]

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

try:
    from datasets import load_dataset as _hf_load_dataset
except Exception:  # pragma: no cover - optional dependency
    _hf_load_dataset = None

try:
    from . import multilingual_prompts as _multilingual_prompts
except ImportError:
    import multilingual_prompts as _multilingual_prompts  # type: ignore[no-redef]

from cml.modeling.config import find_repo_root
from cml.modeling.memory_type_features import (
    MEMORY_TYPE_FEATURE_COLUMNS,
    derive_memory_type_feature_columns,
)
from cml.modeling.pair_features import hash_text, pair_embedding_cache_path
from cml.modeling.types import PrepareConfig

# Some model outputs may trigger parser fallback paths in third-party code that emit
# SyntaxWarning ("invalid escape sequence ...") for raw backslashes in free-form text.
# These warnings are noisy and non-actionable during long synth runs.
warnings.filterwarnings(
    "ignore",
    category=SyntaxWarning,
    message=r"invalid escape sequence .*",
)


REPO_ROOT = find_repo_root(Path(__file__).resolve()) or Path.cwd()
MODELS_ROOT = REPO_ROOT / "packages" / "models"
DEFAULT_CONFIG_PATH = MODELS_ROOT / "model_pipeline.toml"
DEFAULT_PAIR_EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
FEVER_TRAIN_URL = "https://fever.ai/download/fever/train.jsonl"

LOCAL_BOOTSTRAP_SPLITS = ("train", "test", "eval")
ROUTER_TASKS = (
    "memory_type",
    "query_intent",
    "query_domain",
    "constraint_dimension",
    "context_tag",
    "salience_bin",
    "importance_bin",
    "confidence_bin",
    "decay_profile",
)
EXTRACTOR_TASKS = (
    "constraint_type",
    "constraint_scope",
    "constraint_stability",
    "fact_type",
    "pii_presence",
)
PAIR_TASKS = ("conflict_detection", "constraint_rerank", "scope_match", "supersession")

ROUTER_TASK_LABELS: dict[str, list[str]] = {
    "memory_type": [
        "episodic_event",
        "semantic_fact",
        "preference",
        "constraint",
        "procedure",
        "hypothesis",
        "task_state",
        "conversation",
        "message",
        "tool_result",
        "reasoning_step",
        "scratch",
        "knowledge",
        "observation",
        "plan",
    ],
    "query_intent": ["constraint_check", "tool_query", "planning", "factual", "conversation"],
    "query_domain": ["general", "food", "travel", "finance", "health", "work", "tech", "social"],
    "constraint_dimension": ["policy", "goal", "value", "causal", "state", "other"],
    "context_tag": ["general", "food", "travel", "finance", "health", "work", "tech", "social"],
    "salience_bin": ["low", "medium", "high"],
    "importance_bin": ["low", "medium", "high"],
    "confidence_bin": ["low", "medium", "high"],
    "decay_profile": ["very_fast", "fast", "medium", "slow", "very_slow"],
}

EXTRACTOR_TASK_LABELS: dict[str, list[str]] = {
    "constraint_type": [
        "policy",
        "goal",
        "value",
        "causal",
        "state",
        "preference",
        "constraint_other",
        "none",
    ],
    "constraint_scope": [
        "none",
        "general",
        "food",
        "travel",
        "finance",
        "health",
        "work",
        "tech",
        "social",
    ],
    "constraint_stability": ["stable", "semi_stable", "volatile"],
    "fact_type": ["none", "other_fact", "preference", "identity", "location", "occupation"],
    "pii_presence": ["pii", "no_pii"],
}

PAIR_TASK_LABELS: dict[str, list[str]] = {
    "conflict_detection": ["conflict", "no_conflict"],
    "constraint_rerank": ["relevant", "not_relevant"],
    "scope_match": ["match", "no_match"],
    "supersession": ["supersedes", "no_supersedes"],
}

NEW_PAIR_TASK_LABELS: dict[str, list[str]] = {
    "novelty_pair": ["duplicate", "novel", "changed"],
    "schema_match_pair": ["match", "no_match"],
    "retrieval_constraint_relevance_pair": ["relevant", "not_relevant"],
    "reconsolidation_candidate_pair": ["relevant", "not_relevant"],
    "memory_rerank_pair": ["relevant", "not_relevant"],
}

NEW_SINGLE_TASK_LABELS: dict[str, list[str]] = {
    "consolidation_gist_quality": ["accept", "reject"],
    "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
}

ALL_PAIR_TASKS = PAIR_TASKS + tuple(NEW_PAIR_TASK_LABELS.keys())
ALL_ROUTER_TASKS = ROUTER_TASKS + tuple(NEW_SINGLE_TASK_LABELS.keys())
TOKEN_TASKS = ("fact_extraction_structured", "pii_span_detection")
ORDINAL_ROUTER_TASKS = {"salience_bin", "importance_bin", "confidence_bin", "decay_profile"}
PAIR_RANKING_TASKS = {
    "retrieval_constraint_relevance_pair",
    "memory_rerank_pair",
    "reconsolidation_candidate_pair",
}
REBUILT_ROUTER_TASKS = ORDINAL_ROUTER_TASKS | {
    "consolidation_gist_quality",
    "forgetting_action_policy",
}
REBUILT_PAIR_TASKS = {"schema_match_pair"}
_TEMPLATE_SOURCE_PREFIXES = ("template:", "template_hardened:")
_TASK_TEMPLATE_CAPS = {
    "consolidation_gist_quality": 0.5,
    "forgetting_action_policy": 0.5,
    "schema_match_pair": 0.1,
}
_TASK_REQUIRED_SOURCE_TARGET_RATIOS = {"schema_match_pair": {"hf:fever": 0.2}}
_REQUIRED_SOURCE_PREFIXES = {"schema_match_pair": ("hf:fever",)}
_ADVERSARIAL_FIXTURE_PATHS = {
    "consolidation_gist_quality": MODELS_ROOT / "adversarial" / "adversarial_gist_quality.jsonl",
    "forgetting_action_policy": MODELS_ROOT / "adversarial" / "adversarial_forgetting_policy.jsonl",
    "schema_match_pair": MODELS_ROOT / "adversarial" / "adversarial_schema_match_pair.jsonl",
}
_MIN_ADVERSARIAL_ROWS = 500
_ADVERSARIAL_TRAIN_FRACTION = 0.5
_ADVERSARIAL_HELDOUT_SUFFIX = "_heldout"
FACT_SPAN_LABELS = [
    "preference",
    "identity",
    "location",
    "occupation",
    "attribute",
    "goal",
    "value",
    "state",
    "causal",
    "policy",
]

_FACT_HEURISTIC_PATTERNS: tuple[tuple[str, tuple[re.Pattern[str], ...]], ...] = (
    (
        "preference",
        (
            re.compile(
                r"\b(?:i|we)\s+(?:really\s+|also\s+|just\s+|still\s+)*(?:prefer|like|love|enjoy|hate|dislike)\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "identity",
        (
            re.compile(
                r"\b(?:my name is|call me)\s+(?P<value>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "location",
        (
            re.compile(
                r"\bi live in\s+(?P<value>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bi(?: am|'m)\s+(?:from|based in)\s+(?P<value>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bi moved to\s+(?P<value>[A-Z][a-zA-Z]+(?:\s+[A-Z][a-zA-Z]+)*)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "occupation",
        (
            re.compile(
                r"\bi work as\s+(?:a|an)?\s*(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bi(?: am|'m)\s+(?:a|an)\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "attribute",
        (
            re.compile(
                r"\bi(?: am|'m)\s+(?P<value>(?:allergic to|left-handed|right-handed|vegetarian|vegan|detail-oriented|introverted|extroverted)[^.!?;\n]*)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "goal",
        (
            re.compile(
                r"\b(?:my goal is to|i want to|i need to|i'm trying to|i am trying to)\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "value",
        (
            re.compile(
                r"\b(?:i value|i care about)\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"(?P<value>[A-Za-z][^.!?;\n]{2,})\s+matters to me\b",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "state",
        (
            re.compile(
                r"\bi(?: am|'m)\s+(?:currently\s+)?(?P<value>(?:stressed|tired|exhausted|excited|worried|anxious|calm|busy|overwhelmed)[^.!?;\n]*)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\bi feel\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "causal",
        (
            re.compile(
                r"\bbecause\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
            re.compile(
                r"\b(?P<value>[^.!?;\n]+(?:triggers|causes)[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
    (
        "policy",
        (
            re.compile(
                r"\b(?:never|always|please don't|do not)\s+(?P<value>[^.!?;\n]+)",
                flags=re.IGNORECASE,
            ),
        ),
    ),
)


def _enabled_task_label_map(
    task_specs_raw: list[dict],
    *,
    family: str,
    base_labels: dict[str, list[str]],
    extra_labels: dict[str, list[str]] | None = None,
) -> dict[str, list[str]]:
    enabled = dict(base_labels)
    if not extra_labels:
        return enabled

    enabled_extra = {
        str(spec.get("task_name", "")).strip()
        for spec in task_specs_raw
        if bool(spec.get("enabled", True)) and str(spec.get("family", "")).strip() == family
    }
    for task_name, labels in extra_labels.items():
        if task_name in enabled_extra:
            enabled[task_name] = labels
    return enabled


def _enabled_regression_tasks(task_specs_raw: list[dict], *, family: str) -> set[str]:
    return {
        str(spec.get("task_name", "")).strip()
        for spec in task_specs_raw
        if bool(spec.get("enabled", True))
        and str(spec.get("family", "")).strip() == family
        and str(spec.get("objective", "")).strip() == "single_regression"
        and str(spec.get("task_name", "")).strip()
    }


def _enabled_embedding_pair_tasks(task_specs_raw: list[dict]) -> tuple[set[str], str | None]:
    task_names: set[str] = set()
    model_names: set[str] = set()
    for spec in task_specs_raw:
        if not bool(spec.get("enabled", True)):
            continue
        if str(spec.get("family", "")).strip() != "pair":
            continue
        trainer = str(spec.get("trainer", "")).strip()
        feature_backend = str(spec.get("feature_backend", "")).strip()
        if trainer != "embedding_pair" and feature_backend != "embedding_pair":
            continue
        task_name = str(spec.get("task_name", "")).strip()
        if not task_name:
            continue
        task_names.add(task_name)
        model_names.add(
            str(spec.get("embedding_model_name", "")).strip()
            or DEFAULT_PAIR_EMBEDDING_MODEL_NAME
        )
    if not task_names:
        return set(), None
    if len(model_names) > 1:
        raise ValueError(
            "Embedding-pair tasks must share one embedding_model_name during prepare."
        )
    return task_names, next(iter(model_names))


def _normalize_pair_task_label(task: str, label: str) -> str:
    cleaned_task = str(task).strip()
    cleaned_label = str(label).strip()
    if cleaned_task == "novelty_pair" and cleaned_label in {"contradiction", "temporal_change"}:
        return "changed"
    return cleaned_label


def _download_to_path(url: str, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    if httpx is not None:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120.0) as response:
            response.raise_for_status()
            with open(tmp_path, "wb") as handle:
                for chunk in response.iter_bytes():
                    if chunk:
                        handle.write(chunk)
    else:  # pragma: no cover - httpx is expected in the modeling environment
        from urllib.request import urlopen

        with urlopen(url, timeout=120) as response, open(tmp_path, "wb") as handle:
            shutil.copyfileobj(response, handle)
    tmp_path.replace(path)


def _fever_evidence_text(record: dict[str, Any]) -> str:
    raw_evidence = record.get("evidence")
    if not isinstance(raw_evidence, list):
        return ""
    snippets: list[str] = []
    seen: set[str] = set()
    for evidence_set in raw_evidence:
        if not isinstance(evidence_set, list):
            continue
        for item in evidence_set:
            if not isinstance(item, (list, tuple)) or len(item) < 4:
                continue
            title = str(item[2] or "").replace("_", " ").strip()
            if not title:
                continue
            sentence_id = item[3]
            try:
                sentence_index = int(sentence_id)
            except Exception:
                sentence_index = None
            if sentence_index is not None and sentence_index >= 0:
                snippet = f"{title} sentence {sentence_index}"
            else:
                snippet = title
            if snippet not in seen:
                seen.add(snippet)
                snippets.append(snippet)
            if len(snippets) >= 3:
                return "; ".join(snippets)
    return "; ".join(snippets)


def _load_fever_rows_from_jsonl(path: Path, *, limit: int) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            if len(rows) >= limit:
                break
            raw = raw.strip()
            if not raw:
                continue
            try:
                record = json.loads(raw)
            except json.JSONDecodeError:
                continue
            claim = _clean(record.get("claim", ""), 700)
            label = str(record.get("label", "")).strip()
            evidence_text = _clean(_fever_evidence_text(record), 700)
            if not claim or not label or not evidence_text:
                continue
            rows.append(
                {
                    "claim": claim,
                    "label": label,
                    "evidence_sentence": evidence_text,
                }
            )
    return rows


def _load_fever_dataset(cache_dir: Path | None, *, limit: int) -> list[dict[str, Any]]:
    root = cache_dir or (MODELS_ROOT / "datasets")
    data_path = root / "fever_raw" / "train.jsonl"
    if not data_path.exists():
        _download_to_path(FEVER_TRAIN_URL, data_path)
    return _load_fever_rows_from_jsonl(data_path, limit=limit)


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


def _warn(msg: str) -> None:
    print(f"[prepare] {msg}", file=sys.stderr)


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
    required_sections = ("paths", "prepare", "synthetic_llm")
    missing = [name for name in required_sections if name not in cfg]
    if missing:
        raise ValueError(f"Config missing required sections: {', '.join(missing)}")
    return cfg


def _load_dotenv(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key and key not in os.environ:
            os.environ[key] = value


def _clean(text: object, limit: int = 1200) -> str:
    if text is None:
        return ""
    value = str(text).replace("\x00", " ")
    value = re.sub(r"\s+", " ", value).strip()
    if len(value) > limit:
        value = value[:limit].rstrip()
    return value


def _norm_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def _template_stem_from_source(source: object) -> str | None:
    """Extract template stem from source e.g. template:schema_match_pair:5 -> template:schema_match_pair."""
    s = str(source or "").strip()
    if not s:
        return None
    if s.startswith("template_hardened:"):
        parts = s.split(":", 2)
        return f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else s
    if s.startswith("template:"):
        parts = s.split(":", 2)
        return f"{parts[0]}:{parts[1]}" if len(parts) >= 2 else s
    return None


def _shingle_set(text: str, k: int = 3) -> set[int]:
    """Word k-gram hashes for MinHash-style grouping."""
    normalized = _norm_key(_clean(text, 2000))
    if not normalized:
        return set()
    words = normalized.split()
    if len(words) < k:
        return {hash(normalized)}
    out: set[int] = set()
    for i in range(len(words) - k + 1):
        shingle = " ".join(words[i : i + k])
        out.add(hash(shingle) & 0x7FFFFFFF)
    return out


def _jaccard(a: set[int], b: set[int]) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


def _near_dup_union_find(
    parent: dict[int, int],
    shingles: dict[int, set[int]],
    jaccard_threshold: float,
) -> tuple[Callable[[int], int], Callable[[int, int], None]]:
    def find(x: int) -> int:
        if x not in parent:
            parent[x] = x
        if parent[x] != x:
            parent[x] = find(parent[x])
        return parent[x]

    def union(x: int, y: int) -> None:
        px, py = find(x), find(y)
        if px != py and _jaccard(shingles[px], shingles[py]) >= jaccard_threshold:
            parent[px] = py

    return find, union


def _stable_group_id(prefix: str, *parts: object) -> str:
    payload = "||".join(_norm_key(_clean(part, 400)) for part in parts)
    digest = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:20]
    return f"{prefix}:{digest}"


def _is_template_source(source: object) -> bool:
    return str(source or "").startswith(_TEMPLATE_SOURCE_PREFIXES)


def _source_bucket(source: object) -> str:
    text = str(source or "").strip()
    if not text:
        return "unknown"
    if text.startswith("template_hardened:"):
        return "template_hardened"
    if text.startswith("template:"):
        return "template"
    if text.startswith("structured:"):
        return "structured"
    if text.startswith("llm:"):
        return "llm"
    if text.startswith("hf:"):
        return "hf"
    if text.startswith("derived:"):
        return "derived"
    if text.startswith("prepared:"):
        return "prepared"
    return text.split(":", 1)[0]


# Surrogate code points (U+D800-U+DFFF) are invalid in UTF-8 and break pandas/pyarrow.
def _sanitize_unicode(s: str) -> str:
    return s.encode("utf-8", errors="replace").decode("utf-8")


def _sanitize_row_dicts(rows: list[dict]) -> list[dict]:
    out: list[dict] = []
    for row in rows:
        out.append({k: _sanitize_unicode(v) if isinstance(v, str) else v for k, v in row.items()})
    return out


def _sample_df(df: pd.DataFrame, limit: int, seed: int) -> pd.DataFrame:
    if len(df) <= limit:
        return df.copy()
    return df.sample(n=limit, random_state=seed).reset_index(drop=True)


def _safe_feature_names(dataset: object, field: str) -> list[str]:
    try:
        feature = dataset.features[field]  # type: ignore[attr-defined]
        names = getattr(feature, "names", None)
        if isinstance(names, list):
            return [str(x) for x in names]
    except Exception:
        pass
    return []


def _resolve_class_name(value: object, names: list[str]) -> str:
    if not names:
        return _clean(value, 80)
    try:
        idx = int(value)  # type: ignore[call-overload]
    except Exception:
        return _clean(value, 80)
    if 0 <= idx < len(names):
        return names[idx]
    return ""


def _infer_hf_dataset_id(link: str) -> str:
    marker = "huggingface.co/datasets/"
    if marker not in link:
        return ""
    tail = link.split(marker, 1)[1].strip().strip("/")
    if not tail:
        return ""
    return tail.split("?", 1)[0].strip("/")


class _SingleTaskStore:
    def __init__(self, max_per_task_label: int) -> None:
        self.max_per_task_label = max_per_task_label
        self.rows: list[dict] = []
        self._seen: set[tuple[str, str, str]] = set()
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def count(self, task: str, label: str) -> int:
        return int(self._counts.get((task, label), 0))

    def add(
        self,
        task: str,
        text: object,
        label: str,
        source: str,
        language: str = "en",
        extras: dict[str, Any] | None = None,
    ) -> bool:
        cleaned = _clean(text, 1000)
        if not cleaned or not task or not label:
            return False
        key = (task, _norm_key(cleaned), label)
        if key in self._seen:
            return False
        task_label = (task, label)
        if self._counts[task_label] >= self.max_per_task_label:
            return False
        self._seen.add(key)
        self._counts[task_label] += 1
        row: dict[str, Any] = {
            "text": cleaned,
            "task": task,
            "label": label,
            "source": source,
            "language": str(language),
        }
        if extras:
            for key_name, value in extras.items():
                if value is not None:
                    row[key_name] = value
        self.rows.append(row)
        return True


class _PairTaskStore:
    def __init__(self, max_per_task_label: int) -> None:
        self.max_per_task_label = max_per_task_label
        self.rows: list[dict] = []
        self._seen: set[tuple[str, str, str, str]] = set()
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def count(self, task: str, label: str) -> int:
        return int(self._counts.get((task, label), 0))

    def add(
        self,
        task: str,
        text_a: object,
        text_b: object,
        label: str,
        source: str,
        language: str = "en",
        extras: dict[str, Any] | None = None,
    ) -> bool:
        a = _clean(text_a, 700)
        b = _clean(text_b, 700)
        label = _normalize_pair_task_label(task, label)
        if not a or not b or not task or not label:
            return False
        key = (task, _norm_key(a), _norm_key(b), label)
        if key in self._seen:
            return False
        task_label = (task, label)
        if self._counts[task_label] >= self.max_per_task_label:
            return False
        self._seen.add(key)
        self._counts[task_label] += 1
        row: dict[str, Any] = {
            "text_a": a,
            "text_b": b,
            "task": task,
            "label": label,
            "source": source,
            "language": str(language),
        }
        if extras:
            for key_name, value in extras.items():
                if value is not None:
                    row[key_name] = value
        self.rows.append(row)
        return True


class _RegressionTaskStore:
    """Store for regression tasks with numeric score target."""

    def __init__(self, max_per_task: int) -> None:
        self.max_per_task = max_per_task
        self.rows: list[dict] = []
        self._seen: set[tuple[str, str]] = set()
        self._counts: dict[str, int] = defaultdict(int)

    def count(self, task: str) -> int:
        return int(self._counts.get(task, 0))

    def add(
        self,
        task: str,
        text: object,
        score: float,
        source: str,
        language: str = "en",
        extras: dict[str, Any] | None = None,
    ) -> bool:
        cleaned = _clean(text, 1000)
        if not cleaned or not task:
            return False
        key = (task, _norm_key(cleaned))
        if key in self._seen:
            return False
        if self._counts[task] >= self.max_per_task:
            return False
        self._seen.add(key)
        self._counts[task] += 1
        row: dict[str, Any] = {
            "text": cleaned,
            "task": task,
            "label": "",
            "score": float(score),
            "source": source,
            "language": str(language),
        }
        if extras:
            for key_name, value in extras.items():
                if value is not None:
                    row[key_name] = value
        self.rows.append(row)
        return True


class _TokenTaskStore:
    """Store for token classification tasks with span annotations."""

    def __init__(self, max_per_task: int) -> None:
        self.max_per_task = max_per_task
        self.rows: list[dict] = []
        self._counts: dict[str, int] = defaultdict(int)
        self._seen: set[tuple[str, str, str]] = set()

    def count(self, task: str) -> int:
        return int(self._counts.get(task, 0))

    def add(
        self,
        task: str,
        text: str,
        spans: list[dict],
        source: str,
        language: str = "en",
    ) -> bool:
        """Add a token-classification example.

        Args:
            task: Task name.
            text: Input text.
            spans: list of {"start": int, "end": int, "label": str}.
            source: Data source identifier.
            language: ISO 639-1 language code (default en).
        """
        cleaned = _clean(text, 2000)
        if not cleaned or not task:
            return False
        normalized_spans = _normalize_token_spans(spans, text=cleaned)
        if not normalized_spans:
            return False
        key = (task, _norm_key(cleaned), json.dumps(normalized_spans, sort_keys=True))
        if key in self._seen:
            return False
        if self._counts[task] >= self.max_per_task:
            return False
        self._seen.add(key)
        self._counts[task] += 1
        self.rows.append(
            {
                "text": cleaned,
                "task": task,
                "spans": normalized_spans,
                "source": source,
                "language": str(language),
            }
        )
        return True


def _seed_single_store_from_df(rows: _SingleTaskStore, df: pd.DataFrame) -> None:
    if df.empty:
        return
    base_cols = {"text", "task", "label", "source", "language"}
    for item in df.itertuples(index=False):
        extras = {col: getattr(item, col, None) for col in df.columns if col not in base_cols}
        rows.add(
            str(getattr(item, "task", "")),
            getattr(item, "text", ""),
            str(getattr(item, "label", "")),
            str(getattr(item, "source", "prepared:existing")),
            language=str(getattr(item, "language", "en")),
            extras=extras,
        )


def _seed_pair_store_from_df(rows: _PairTaskStore, df: pd.DataFrame) -> None:
    if df.empty:
        return
    base_cols = {"text_a", "text_b", "task", "label", "source", "language"}
    for item in df.itertuples(index=False):
        extras = {col: getattr(item, col, None) for col in df.columns if col not in base_cols}
        rows.add(
            str(getattr(item, "task", "")),
            getattr(item, "text_a", ""),
            getattr(item, "text_b", ""),
            str(getattr(item, "label", "")),
            str(getattr(item, "source", "prepared:existing")),
            language=str(getattr(item, "language", "en")),
            extras=extras,
        )


def _seed_regression_store_from_df(rows: _RegressionTaskStore, df: pd.DataFrame) -> None:
    if df.empty or "score" not in df.columns:
        return
    base_cols = {"text", "task", "label", "score", "source", "language"}
    for item in df.itertuples(index=False):
        try:
            score = float(item.score)
        except Exception:
            continue
        extras = {col: getattr(item, col, None) for col in df.columns if col not in base_cols}
        rows.add(
            str(getattr(item, "task", "")),
            getattr(item, "text", ""),
            score,
            str(getattr(item, "source", "prepared:existing")),
            language=str(getattr(item, "language", "en")),
            extras=extras,
        )


def _enabled_token_tasks(task_specs_raw: list[dict], *, family: str) -> set[str]:
    return {
        str(spec.get("task_name", "")).strip()
        for spec in task_specs_raw
        if bool(spec.get("enabled", True))
        and str(spec.get("family", "")).strip() == family
        and str(spec.get("objective", "")).strip() == "token_classification"
        and str(spec.get("task_name", "")).strip()
    }


def _normalize_token_spans(
    spans: list[dict] | Any,
    *,
    text: str,
    uppercase_labels: bool = False,
    max_spans: int | None = None,
) -> list[dict[str, Any]]:
    if spans is None:
        return []
    if isinstance(spans, str):
        try:
            spans = json.loads(spans)
        except Exception:
            return []
    out: list[dict[str, Any]] = []
    for item in spans if isinstance(spans, list) else []:
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
        raw_label = str(label or "").strip()
        if uppercase_labels:
            raw_label = raw_label.upper()
        if not raw_label or s < 0 or e <= s or e > len(text):
            continue
        if not text[s:e].strip():
            continue
        out.append({"start": s, "end": e, "label": raw_label})
        if max_spans is not None and len(out) >= max_spans:
            break
    out.sort(key=lambda item: (int(item["start"]), int(item["end"]), str(item["label"])))
    return out


def _seed_token_store_from_df(rows: _TokenTaskStore, df: pd.DataFrame) -> None:
    if df.empty:
        return
    for item in df.itertuples(index=False):
        rows.add(
            str(getattr(item, "task", "")),
            str(getattr(item, "text", "")),
            getattr(item, "spans", []),
            str(getattr(item, "source", "prepared:token")),
            language=str(getattr(item, "language", "en")),
        )


def _load_existing_token_df(prepared_dir: Path, task_name: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split in LOCAL_BOOTSTRAP_SPLITS:
        path = prepared_dir / f"{task_name}_{split}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            _warn(f"Failed reading existing token split {path.name}: {exc}")
            continue
        required = {"text", "task", "spans"}
        if not required.issubset(df.columns):
            _warn(
                f"Existing token split missing required columns ({path.name}): {sorted(required)}"
            )
            continue
        if "source" not in df.columns:
            df = df.copy()
            df["source"] = f"prepared:{task_name}:{split}"
        if "language" not in df.columns:
            df = df.copy()
            df["language"] = "en"
        df = df.copy()
        df["spans"] = [
            _normalize_token_spans(value, text=str(text or ""))
            for value, text in zip(df["spans"].tolist(), df["text"].tolist(), strict=False)
        ]
        parts.append(df[["text", "task", "spans", "source", "language"]])
    if not parts:
        return pd.DataFrame(columns=["text", "task", "spans", "source", "language"])
    return pd.concat(parts, ignore_index=True, sort=False)


def _existing_token_split_counts(prepared_dir: Path, task_name: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in LOCAL_BOOTSTRAP_SPLITS:
        path = prepared_dir / f"{task_name}_{split}.parquet"
        if not path.exists():
            out[split] = 0
            continue
        try:
            out[split] = len(pd.read_parquet(path))
        except Exception:
            out[split] = 0
    return out


def _missing_token_tasks(
    token_frames: dict[str, pd.DataFrame],
    *,
    task_names: set[str],
    target_per_task: int,
) -> tuple[dict[str, int], int]:
    missing: dict[str, int] = {}
    total = 0
    for task_name in sorted(task_names):
        df = token_frames.get(task_name)
        have = 0 if df is None or df.empty else len(df)
        miss = max(0, int(target_per_task) - have)
        missing[task_name] = miss
        total += miss
    return missing, total


def _token_signature(spans: list[dict]) -> str:
    labels = sorted(str(item.get("label", "")).strip() for item in spans if item.get("label"))
    if not labels:
        return "__empty__"
    return "|".join(labels)


def _split_token_rows(
    df: pd.DataFrame, seed: int, ratios: dict[str, float]
) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []
    keyed = df.copy()
    keyed["__signature"] = [
        _token_signature(value if isinstance(value, list) else [])
        for value in keyed["spans"].tolist()
    ]
    for (_task, _signature), group in keyed.groupby(["task", "__signature"], sort=False):
        idx = list(group.index)
        rng.shuffle(idx)
        n_train, n_test, n_eval = _split_counts(len(idx), ratios)
        train_parts.append(keyed.loc[idx[:n_train]])
        test_parts.append(keyed.loc[idx[n_train : n_train + n_test]])
        eval_parts.append(keyed.loc[idx[n_train + n_test : n_train + n_test + n_eval]])
    out = {
        "train": pd.concat(train_parts, ignore_index=True)
        if train_parts
        else keyed.iloc[0:0].copy(),
        "test": pd.concat(test_parts, ignore_index=True) if test_parts else keyed.iloc[0:0].copy(),
        "eval": pd.concat(eval_parts, ignore_index=True) if eval_parts else keyed.iloc[0:0].copy(),
    }
    for split_name, split_df in out.items():
        shuffled = split_df.sample(frac=1, random_state=seed).reset_index(drop=True)
        out[split_name] = shuffled.drop(columns=["__signature"], errors="ignore")
    return out


def _write_token_task_splits(
    *,
    df: pd.DataFrame,
    task_name: str,
    out_dir: Path,
    seed: int,
    ratios: dict[str, float],
) -> dict[str, int]:
    splits = _split_token_rows(df, seed, ratios)
    counts: dict[str, int] = {}
    for split_name, split_df in splits.items():
        path = out_dir / f"{task_name}_{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        counts[split_name] = len(split_df)
    return counts


def _token_summary(df: pd.DataFrame) -> dict[str, Any]:
    source_top = df["source"].value_counts().head(20).to_dict() if "source" in df.columns else {}
    source_mix: dict[str, dict[str, int]] = {}
    label_counts: dict[str, int] = {}
    language_counts = (
        df["language"].astype(str).value_counts().to_dict() if "language" in df.columns else {}
    )
    if not df.empty:
        for _, row in df.iterrows():
            task = str(row.get("task", "") or "")
            source = str(row.get("source", "") or "")
            source_mix.setdefault(task, {})
            source_mix[task][source] = source_mix[task].get(source, 0) + 1
            for span in _normalize_token_spans(row.get("spans"), text=str(row.get("text", ""))):
                label = str(span["label"])
                label_counts[label] = label_counts.get(label, 0) + 1
    return {
        "rows": len(df),
        "task_counts": df["task"].value_counts().to_dict() if "task" in df.columns else {},
        "span_label_counts": label_counts,
        "language_counts": language_counts,
        "source_top_20": source_top,
        "source_mix_per_task": source_mix,
    }


def _build_secret_token_examples(*, target: int) -> list[tuple[str, list[dict], str]]:
    categories = [
        ("SECRET", "api_key", "sk-live-{idx:08x}"),
        ("SECRET", "bearer", "Bearer tok_{idx:08x}{idx:08x}"),
        ("SECRET", "password", "P@ssword-{idx:06d}!"),
        ("SECRET", "connection", "Server=tcp:db{idx:04d};Password=Secret{idx:04d}!"),
        ("SECRET", "ssh", "-----BEGIN PRIVATE KEY----- KEY{idx:08x} -----END PRIVATE KEY-----"),
    ]
    examples: list[tuple[str, list[dict], str]] = []
    for idx in range(target):
        label, kind, secret_template = categories[idx % len(categories)]
        secret_value = secret_template.format(idx=idx)
        text = (
            f"Internal note {idx}: the {kind} credential is {secret_value}. "
            f"Do not share this outside the team."
        )
        start = text.index(secret_value)
        end = start + len(secret_value)
        examples.append(
            (
                text,
                [{"start": start, "end": end, "label": label}],
                f"template:pii_span_detection:{kind}",
            )
        )
    return examples


def _pii_template_example(label: str, idx: int) -> tuple[str, list[dict], str]:
    normalized = str(label or "SECRET").strip().upper() or "SECRET"
    if "EMAIL" in normalized:
        value = f"user{idx}@example.com"
        text = f"Reach me at {value} about account ticket {idx}."
    elif "PHONE" in normalized:
        value = f"+1-555-{(idx // 100) % 10000:04d}-{idx % 10000:04d}"
        text = f"My backup phone number is {value}."
    elif normalized in {"PERSON", "NAME", "FULL_NAME"} or "NAME" in normalized:
        value = f"Jordan Example {idx}"
        text = f"The contact name on file is {value}."
    elif "ADDRESS" in normalized:
        value = f"{100 + (idx % 900)} Example Street Apt {idx}"
        text = f"Ship the replacement card to {value}."
    elif normalized in {"LOCATION", "CITY"} or "LOC" in normalized:
        value = f"District {idx} City"
        text = f"The incident was reported from {value}."
    elif "DATE" in normalized:
        value = f"2026-03-{1 + (idx % 28):02d}"
        text = f"The appointment date is {value}."
    elif "IP" in normalized:
        value = f"10.{(idx // 65536) % 255}.{(idx // 256) % 255}.{idx % 255}"
        text = f"The login came from IP {value}."
    elif "URL" in normalized:
        value = f"https://example.com/private/{idx}"
        text = f"Use secure URL {value} for the download."
    elif "USER" in normalized:
        value = f"user_{idx}"
        text = f"The temporary username is {value}."
    elif "ID" in normalized:
        value = f"ID-{idx:08d}"
        text = f"The identity document number is {value}."
    else:
        value = f"sk-live-{idx:08x}"
        text = f"Security note {idx}: the credential is {value}."
        normalized = "SECRET"
    start = text.index(value)
    end = start + len(value)
    return text, [{"start": start, "end": end, "label": normalized}], f"template:pii:{normalized}"


_ENGLISH_FACT_VALUE_TEMPLATES: dict[str, str] = {
    "preference": "vegetarian food batch {idx}",
    "identity": "Alex Carter {idx}",
    "location": "Paris {idx}",
    "occupation": "product manager level {idx}",
    "attribute": "detail oriented in sprint {idx}",
    "goal": "finish milestone {idx} before quarter end",
    "value": "honesty in team review {idx}",
    "state": "stressed about deadline batch {idx}",
    "causal": "sugar triggers migraine cycle {idx}",
    "policy": "do not schedule meetings after {hour} PM in block {idx}",
}

_LOCALIZED_FACT_VALUE_TEMPLATES: dict[str, dict[str, str]] = {
    "es": {
        "preference": "comida vegetariana lote {idx}",
        "goal": "terminar el hito {idx} antes del fin del trimestre",
        "value": "la honestidad en la revision del equipo {idx}",
        "state": "estresado por la fecha limite lote {idx}",
    },
    "fr": {
        "preference": "repas vegetarien lot {idx}",
        "goal": "terminer le jalon {idx} avant la fin du trimestre",
        "value": "l'honnetete dans la revue d'equipe {idx}",
        "state": "stresse par l'echeance lot {idx}",
    },
    "pt": {
        "preference": "refeicao vegetariana lote {idx}",
        "goal": "terminar o marco {idx} antes do fim do trimestre",
        "value": "a honestidade na revisao da equipe {idx}",
        "state": "estressado com o prazo lote {idx}",
    },
    "de": {
        "preference": "vegetarisches essen stapel {idx}",
        "goal": "meilenstein {idx} vor quartalsende abschliessen",
        "value": "ehrlichkeit in der teamprufung {idx}",
        "state": "wegen der frist belastet stapel {idx}",
    },
    "it": {
        "preference": "pasto vegetariano lotto {idx}",
        "goal": "completare la milestone {idx} prima della fine del trimestre",
        "value": "l'onesta nella revisione del team {idx}",
        "state": "stressato per la scadenza lotto {idx}",
    },
}

_FACT_TEMPLATE_SENTENCES: dict[str, dict[str, str]] = {
    "en": {
        "preference": "I prefer [[[{value}]]] when cooking at home.",
        "identity": "My name is [[[{value}]]].",
        "location": "I live in [[[{value}]]].",
        "occupation": "I work as a [[[{value}]]].",
        "attribute": "I am [[[{value}]]].",
        "goal": "My goal is to [[[{value}]]].",
        "value": "I value [[[{value}]]].",
        "state": "I am currently [[[{value}]]].",
        "causal": "I avoid desserts because [[[{value}]]].",
        "policy": "Please [[[{value}]]].",
    },
    "es": {
        "preference": "Yo prefiero [[[{value}]]] cuando cocino en casa.",
        "identity": "Me llamo [[[{value}]]].",
        "location": "Vivo en [[[{value}]]].",
        "occupation": "Trabajo como [[[{value}]]].",
        "attribute": "Soy [[[{value}]]].",
        "goal": "Mi objetivo es [[[{value}]]].",
        "value": "Valoro [[[{value}]]].",
        "state": "Ahora estoy [[[{value}]]].",
        "causal": "Evito los postres porque [[[{value}]]].",
        "policy": "Por favor [[[{value}]]].",
    },
    "fr": {
        "preference": "Je prefere [[[{value}]]] quand je cuisine chez moi.",
        "identity": "Je m'appelle [[[{value}]]].",
        "location": "J'habite a [[[{value}]]].",
        "occupation": "Je travaille comme [[[{value}]]].",
        "attribute": "Je suis [[[{value}]]].",
        "goal": "Mon objectif est de [[[{value}]]].",
        "value": "J'accorde de la valeur a [[[{value}]]].",
        "state": "En ce moment je suis [[[{value}]]].",
        "causal": "J'evite les desserts parce que [[[{value}]]].",
        "policy": "S'il te plait [[[{value}]]].",
    },
    "pt": {
        "preference": "Eu prefiro [[[{value}]]] quando cozinho em casa.",
        "identity": "Meu nome e [[[{value}]]].",
        "location": "Eu moro em [[[{value}]]].",
        "occupation": "Eu trabalho como [[[{value}]]].",
        "attribute": "Eu sou [[[{value}]]].",
        "goal": "Meu objetivo e [[[{value}]]].",
        "value": "Eu valorizo [[[{value}]]].",
        "state": "Agora estou [[[{value}]]].",
        "causal": "Eu evito sobremesas porque [[[{value}]]].",
        "policy": "Por favor [[[{value}]]].",
    },
    "de": {
        "preference": "Ich bevorzuge [[[{value}]]], wenn ich zu Hause koche.",
        "identity": "Ich heisse [[[{value}]]].",
        "location": "Ich wohne in [[[{value}]]].",
        "occupation": "Ich arbeite als [[[{value}]]].",
        "attribute": "Ich bin [[[{value}]]].",
        "goal": "Mein Ziel ist es, [[[{value}]]].",
        "value": "Ich schatze [[[{value}]]].",
        "state": "Im Moment bin ich [[[{value}]]].",
        "causal": "Ich vermeide Desserts, weil [[[{value}]]].",
        "policy": "Bitte [[[{value}]]].",
    },
    "it": {
        "preference": "Preferisco [[[{value}]]] quando cucino a casa.",
        "identity": "Mi chiamo [[[{value}]]].",
        "location": "Vivo a [[[{value}]]].",
        "occupation": "Lavoro come [[[{value}]]].",
        "attribute": "Sono [[[{value}]]].",
        "goal": "Il mio obiettivo e [[[{value}]]].",
        "value": "Do valore a [[[{value}]]].",
        "state": "In questo momento sono [[[{value}]]].",
        "causal": "Evito i dolci perche [[[{value}]]].",
        "policy": "Per favore [[[{value}]]].",
    },
    "zh": {
        "preference": "我更喜欢[[[{value}]]]。",
        "identity": "我叫[[[{value}]]]。",
        "location": "我住在[[[{value}]]]。",
        "occupation": "我的工作是[[[{value}]]]。",
        "attribute": "我[[[{value}]]]。",
        "goal": "我的目标是[[[{value}]]]。",
        "value": "我重视[[[{value}]]]。",
        "state": "我现在[[[{value}]]]。",
        "causal": "我避开甜点,因为[[[{value}]]]。",
        "policy": "请[[[{value}]]]。",
    },
    "ar": {
        "preference": "انا افضل [[[{value}]]] عندما اطبخ في المنزل.",
        "identity": "اسمي [[[{value}]]].",
        "location": "انا اعيش في [[[{value}]]].",
        "occupation": "اعمل كـ [[[{value}]]].",
        "attribute": "انا [[[{value}]]].",
        "goal": "هدفي هو [[[{value}]]].",
        "value": "انا اقدر [[[{value}]]].",
        "state": "انا حاليا [[[{value}]]].",
        "causal": "اتجنب الحلويات لان [[[{value}]]].",
        "policy": "من فضلك [[[{value}]]].",
    },
    "hi": {
        "preference": "मैं घर पर पकाते समय [[[{value}]]] पसंद करता हूँ।",
        "identity": "मेरा नाम [[[{value}]]] है।",
        "location": "मैं [[[{value}]]] में रहता हूँ।",
        "occupation": "मैं [[[{value}]]] के रूप में काम करता हूँ।",
        "attribute": "मैं [[[{value}]]] हूँ।",
        "goal": "मेरा लक्ष्य [[[{value}]]] है।",
        "value": "मैं [[[{value}]]] को महत्व देता हूँ।",
        "state": "मैं अभी [[[{value}]]] हूँ।",
        "causal": "मैं मिठाई से बचता हूँ क्योंकि [[[{value}]]]।",
        "policy": "कृपया [[[{value}]]]।",
    },
    "ja": {
        "preference": "家で料理するときは[[[{value}]]]が好きです。",
        "identity": "私の名前は[[[{value}]]]です。",
        "location": "[[[{value}]]]に住んでいます。",
        "occupation": "私は[[[{value}]]]として働いています。",
        "attribute": "私は[[[{value}]]]です。",
        "goal": "私の目標は[[[{value}]]]ことです。",
        "value": "私は[[[{value}]]]を大切にします。",
        "state": "今は[[[{value}]]]です。",
        "causal": "[[[{value}]]]ので甘い物を避けます。",
        "policy": "どうか[[[{value}]]]。",
    },
    "ru": {
        "preference": "Я предпочитаю [[[{value}]]], когда готовлю дома.",
        "identity": "Меня зовут [[[{value}]]].",
        "location": "Я живу в [[[{value}]]].",
        "occupation": "Я работаю [[[{value}]]].",
        "attribute": "Я [[[{value}]]].",
        "goal": "Моя цель - [[[{value}]]].",
        "value": "Я ценю [[[{value}]]].",
        "state": "Сейчас я [[[{value}]]].",
        "causal": "Я избегаю десертов, потому что [[[{value}]]].",
        "policy": "Пожалуйста, [[[{value}]]].",
    },
    "ko": {
        "preference": "집에서 요리할 때는 [[[{value}]]]를 더 좋아합니다.",
        "identity": "제 이름은 [[[{value}]]]입니다.",
        "location": "저는 [[[{value}]]]에 살아요.",
        "occupation": "저는 [[[{value}]]]로 일합니다.",
        "attribute": "저는 [[[{value}]]]입니다.",
        "goal": "제 목표는 [[[{value}]]]입니다.",
        "value": "저는 [[[{value}]]]를 중요하게 생각합니다.",
        "state": "저는 지금 [[[{value}]]] 상태입니다.",
        "causal": "[[[{value}]]] 때문에 디저트를 피합니다.",
        "policy": "제발 [[[{value}]]].",
    },
    "tr": {
        "preference": "Evde yemek yaparken [[[{value}]]] tercih ederim.",
        "identity": "Benim adim [[[{value}]]].",
        "location": "[[[{value}]]] sehrinde yasiyorum.",
        "occupation": "[[[{value}]]] olarak calisiyorum.",
        "attribute": "Ben [[[{value}]]].",
        "goal": "Hedefim [[[{value}]]].",
        "value": "[[[{value}]]] benim icin onemli.",
        "state": "Su anda [[[{value}]]].",
        "causal": "[[[{value}]]] oldugu icin tatlidan kacinirim.",
        "policy": "Lutfen [[[{value}]]].",
    },
    "id": {
        "preference": "Saat memasak di rumah, saya lebih suka [[[{value}]]].",
        "identity": "Nama saya [[[{value}]]].",
        "location": "Saya tinggal di [[[{value}]]].",
        "occupation": "Saya bekerja sebagai [[[{value}]]].",
        "attribute": "Saya [[[{value}]]].",
        "goal": "Tujuan saya adalah [[[{value}]]].",
        "value": "Saya menghargai [[[{value}]]].",
        "state": "Sekarang saya sedang [[[{value}]]].",
        "causal": "Saya menghindari makanan penutup karena [[[{value}]]].",
        "policy": "Tolong [[[{value}]]].",
    },
    "vi": {
        "preference": "Khi nau an o nha, toi thich [[[{value}]]].",
        "identity": "Ten toi la [[[{value}]]].",
        "location": "Toi song o [[[{value}]]].",
        "occupation": "Toi lam viec nhu mot [[[{value}]]].",
        "attribute": "Toi la nguoi [[[{value}]]].",
        "goal": "Muc tieu cua toi la [[[{value}]]].",
        "value": "Toi coi trong [[[{value}]]].",
        "state": "Luc nay toi dang [[[{value}]]].",
        "causal": "Toi tranh do ngot vi [[[{value}]]].",
        "policy": "Xin hay [[[{value}]]].",
    },
}


def _fact_supported_languages(*, use_multilingual: bool) -> list[_multilingual_prompts.Language]:
    if not use_multilingual:
        return [lang for lang in _multilingual_prompts.SUPPORTED_LANGUAGES if lang.code == "en"]
    return list(_multilingual_prompts.SUPPORTED_LANGUAGES)


def _fact_value_template(label: str, idx: int, *, lang_code: str) -> str:
    hour = 6 + (idx % 5)
    localized = _LOCALIZED_FACT_VALUE_TEMPLATES.get(lang_code, {})
    template = localized.get(label) or _ENGLISH_FACT_VALUE_TEMPLATES.get(label, "fact span {idx}")
    return template.format(idx=idx, hour=hour)


def _render_marked_fact_template(text: str, *, label: str) -> tuple[str, list[dict], str]:
    start_marker = text.find("[[[")
    end_marker = text.find("]]]", start_marker + 3)
    if start_marker < 0 or end_marker < 0:
        raise ValueError(f"Fact template is missing span markers for label '{label}'")
    value = text[start_marker + 3 : end_marker]
    clean_text = text[:start_marker] + value + text[end_marker + 3 :]
    start = start_marker
    end = start + len(value)
    return clean_text, [{"start": start, "end": end, "label": label}], value


def _fact_template_example(
    label: str,
    idx: int,
    *,
    lang_code: str = "en",
) -> tuple[str, list[dict], str]:
    templates = _FACT_TEMPLATE_SENTENCES.get(lang_code) or _FACT_TEMPLATE_SENTENCES["en"]
    template = templates.get(label) or _FACT_TEMPLATE_SENTENCES["en"].get(label)
    if template is None:
        template = "I note [[[{value}]]]."
    value = _fact_value_template(label, idx, lang_code=lang_code)
    text, spans, _ = _render_marked_fact_template(template.format(value=value), label=label)
    return (
        text,
        spans,
        f"template:fact_extraction_structured:{label}:{lang_code}",
    )


def _clean_fact_heuristic_value(value: str) -> str:
    return value.strip().strip(".,!?;:()[]{}\"'`")


def _unique_fact_span(text: str, value: str) -> tuple[int, int] | None:
    needle = _clean_fact_heuristic_value(value)
    if not needle:
        return None
    matches = list(re.finditer(re.escape(needle), text, flags=re.IGNORECASE))
    if len(matches) != 1:
        return None
    match = matches[0]
    return match.start(), match.end()


def _heuristic_fact_spans(text: str, *, max_spans: int) -> list[dict[str, Any]]:
    spans: list[dict[str, Any]] = []
    occupied: list[tuple[int, int]] = []
    lowered = text.lower()
    if not lowered:
        return spans

    for label, patterns in _FACT_HEURISTIC_PATTERNS:
        for pattern in patterns:
            match = pattern.search(text)
            if match is None:
                continue
            span = _unique_fact_span(text, match.group("value"))
            if span is None:
                continue
            start, end = span
            if any(not (end <= s or start >= e) for s, e in occupied):
                continue
            occupied.append((start, end))
            spans.append({"start": start, "end": end, "label": label})
            break
        if len(spans) >= max_spans:
            break
    return spans


def _extract_pii_spans_from_example(
    example: dict[str, Any],
    *,
    max_spans: int,
) -> tuple[str, list[dict[str, Any]], str]:
    text = _clean(example.get("source_text", example.get("source", "")), 2000)
    language = _clean(example.get("language", "en"), 16) or "en"
    if not text:
        return "", [], language
    spans: list[dict[str, Any]] = []
    privacy_mask = example.get("privacy_mask")
    if isinstance(privacy_mask, list) and privacy_mask:
        spans = _normalize_token_spans(
            [
                {
                    "start": item.get("start"),
                    "end": item.get("end"),
                    "label": str(item.get("label", "")).upper(),
                }
                for item in privacy_mask
            ],
            text=text,
            uppercase_labels=True,
            max_spans=max_spans,
        )
    if not spans:
        raw = example.get("span_labels")
        try:
            decoded = json.loads(raw) if isinstance(raw, str) else raw
        except Exception:
            decoded = None
        if isinstance(decoded, list):
            parsed = []
            for item in decoded:
                if not isinstance(item, (list, tuple)) or len(item) < 3:
                    continue
                if str(item[2]).upper() == "O":
                    continue
                parsed.append({"start": item[0], "end": item[1], "label": str(item[2]).upper()})
            spans = _normalize_token_spans(
                parsed,
                text=text,
                uppercase_labels=True,
                max_spans=max_spans,
            )
    return text, spans, language


def _build_pii_token_rows(
    *,
    registry: _HFRegistry,
    token_cfg: dict[str, Any],
    existing_df: pd.DataFrame | None = None,
) -> list[dict]:
    target = int(token_cfg["target_examples_per_task"])
    cap = max(int(token_cfg["max_examples_per_task"]), target)
    max_spans = int(token_cfg["max_spans_per_example"])
    rows = _TokenTaskStore(cap)
    if existing_df is not None and not existing_df.empty:
        _seed_token_store_from_df(rows, existing_df)
    if rows.count("pii_span_detection") >= target:
        return rows.rows

    # Secret-shaped credentials should always be part of the training set.
    secret_target = min(4000, target)
    for text, spans, source in _build_secret_token_examples(target=secret_target):
        rows.add("pii_span_detection", text, spans, source, language="en")

    ds = registry.get("pii_masking")
    if ds is not None:
        for ex in _iter_dataset_rows(
            ds, limit=registry.limit("pii_masking"), desc="Token rows [pii_masking]"
        ):
            if not isinstance(ex, dict):
                continue
            text, spans, language = _extract_pii_spans_from_example(ex, max_spans=max_spans)
            if text and spans:
                rows.add("pii_span_detection", text, spans, "hf:pii_masking", language=language)
            if rows.count("pii_span_detection") >= target:
                break

    observed_labels = sorted(
        {
            str(span.get("label", "")).strip().upper()
            for row in rows.rows
            for span in row.get("spans", [])
            if str(span.get("label", "")).strip()
        }
    )
    if not observed_labels:
        observed_labels = ["SECRET", "EMAIL", "PHONE", "PERSON", "ADDRESS"]

    fill_index = 0
    while rows.count("pii_span_detection") < target:
        label = observed_labels[fill_index % len(observed_labels)]
        text, spans, source = _pii_template_example(label, fill_index)
        rows.add("pii_span_detection", text, spans, source, language="en")
        fill_index += 1
    return rows.rows


def _build_fact_token_rows(
    *,
    single_pools: dict[str, list[str]],
    token_cfg: dict[str, Any],
    seed: int,
    use_multilingual: bool,
    existing_df: pd.DataFrame | None = None,
) -> list[dict]:
    target = int(token_cfg["target_examples_per_task"])
    cap = max(int(token_cfg["max_examples_per_task"]), target)
    max_spans = int(token_cfg["max_spans_per_example"])
    rows = _TokenTaskStore(cap)
    if existing_df is not None and not existing_df.empty:
        _seed_token_store_from_df(rows, existing_df)
    if rows.count("fact_extraction_structured") >= target:
        return rows.rows

    candidate_texts: list[str] = []
    for values in single_pools.values():
        candidate_texts.extend(values)
    candidate_texts = list(
        dict.fromkeys([_clean(text, 2000) for text in candidate_texts if _clean(text, 2000)])
    )
    supported_languages = _fact_supported_languages(use_multilingual=use_multilingual)
    language_weights = [float(lang.weight) for lang in supported_languages]
    rng = random.Random(seed)

    heuristic_target = min(target // 4, 4000)
    heuristic_scan_limit = min(len(candidate_texts), 5000)
    for text in candidate_texts[:heuristic_scan_limit]:
        spans = _heuristic_fact_spans(text, max_spans=max_spans)
        if spans:
            rows.add("fact_extraction_structured", text, spans, "heuristic:write_time_facts")
        if rows.count("fact_extraction_structured") >= heuristic_target:
            break

    per_label_target = max(1, target // max(1, len(FACT_SPAN_LABELS)))
    label_index = {
        (label, language.code): 0 for label in FACT_SPAN_LABELS for language in supported_languages
    }
    label_counts = {label: 0 for label in FACT_SPAN_LABELS}
    label_language_counts = {
        (label, language.code): 0 for label in FACT_SPAN_LABELS for language in supported_languages
    }
    for row in rows.rows:
        language = str(row.get("language", "en") or "en")
        for span in row.get("spans", []):
            label = str(span.get("label", "")).lower()
            if label in label_counts:
                label_counts[label] += 1
                if (label, language) in label_language_counts:
                    label_language_counts[(label, language)] += 1

    if use_multilingual:
        per_label_language_target = 25
        for lang in supported_languages:
            for label in FACT_SPAN_LABELS:
                attempts = 0
                max_attempts = max(per_label_language_target * 8, 256)
                while (
                    label_language_counts[(label, lang.code)] < per_label_language_target
                    and rows.count("fact_extraction_structured") < cap
                    and attempts < max_attempts
                ):
                    text, spans, source = _fact_template_example(
                        label,
                        label_index[(label, lang.code)],
                        lang_code=lang.code,
                    )
                    label_index[(label, lang.code)] += 1
                    attempts += 1
                    if rows.add(
                        "fact_extraction_structured", text, spans, source, language=lang.code
                    ):
                        label_counts[label] += 1
                        label_language_counts[(label, lang.code)] += 1
                if label_language_counts[(label, lang.code)] < per_label_language_target:
                    raise RuntimeError(
                        "Unable to reach multilingual template coverage for "
                        f"{label}:{lang.code} "
                        f"(have={label_language_counts[(label, lang.code)]} "
                        f"target={per_label_language_target})"
                    )

    for label in FACT_SPAN_LABELS:
        attempts = 0
        max_attempts = max(per_label_target * 4, 1024)
        while (
            label_counts[label] < per_label_target
            and rows.count("fact_extraction_structured") < cap
            and attempts < max_attempts
        ):
            lang = cast(
                "_multilingual_prompts.Language",
                rng.choices(supported_languages, weights=language_weights, k=1)[0],
            )
            text, spans, source = _fact_template_example(
                label,
                label_index[(label, lang.code)],
                lang_code=lang.code,
            )
            label_index[(label, lang.code)] += 1
            attempts += 1
            if rows.add("fact_extraction_structured", text, spans, source, language=lang.code):
                label_counts[label] += 1
        if label_counts[label] < per_label_target:
            raise RuntimeError(
                f"Unable to reach template coverage for {label}: "
                f"have={label_counts[label]} target={per_label_target}"
            )

    cycle = 0
    while rows.count("fact_extraction_structured") < target:
        label = FACT_SPAN_LABELS[cycle % len(FACT_SPAN_LABELS)]
        lang = cast(
            "_multilingual_prompts.Language",
            rng.choices(supported_languages, weights=language_weights, k=1)[0],
        )
        text, spans, source = _fact_template_example(
            label,
            label_index[(label, lang.code)],
            lang_code=lang.code,
        )
        label_index[(label, lang.code)] += 1
        rows.add("fact_extraction_structured", text, spans, source, language=lang.code)
        cycle += 1

    return rows.rows


class _HFRegistry:
    def __init__(
        self, *, datasets_cfg: list[dict], cache_dir: Path | None, prepare_cfg: dict
    ) -> None:
        self.datasets_cfg = {str(d["name"]): d for d in datasets_cfg if d.get("enabled", True)}
        self.cache_dir = cache_dir
        self.prepare_cfg = prepare_cfg
        self.enforce_requirements = bool(prepare_cfg.get("require_datasets_package", True))
        self._cache: dict[str, object | None] = {}
        self.status: dict[str, dict] = {}

        for name, dcfg in self.datasets_cfg.items():
            self.status[name] = {
                "name": name,
                "link": dcfg.get("link", ""),
                "dataset_id": dcfg.get("dataset_id", ""),
                "required": bool(dcfg.get("required", False)),
                "enabled": bool(dcfg.get("enabled", True)),
                "loaded": False,
                "error": "",
                "target": dcfg.get("target", ""),
            }

    def _split_spec(self, dcfg: dict) -> str:
        split = str(dcfg.get("split", "train"))
        configured = int(dcfg.get("max_rows", self.prepare_cfg["max_rows_per_source"]))
        global_cap = int(self.prepare_cfg.get("max_rows_per_source", configured))
        max_rows = max(1, min(configured, global_cap))
        if "[" in split and "]" in split:
            return split
        return f"{split}[:{max_rows}]"

    def limit(self, name: str) -> int:
        dcfg = self.datasets_cfg[name]
        configured = int(dcfg.get("max_rows", self.prepare_cfg["max_rows_per_source"]))
        global_cap = int(self.prepare_cfg.get("max_rows_per_source", configured))
        return max(1, min(configured, global_cap))

    def get(self, name: str) -> object | None:
        if name in self._cache:
            return self._cache[name]
        if name not in self.datasets_cfg:
            _warn(f"Dataset config not found: {name}")
            self._cache[name] = None
            return None

        dcfg = self.datasets_cfg[name]
        required = bool(dcfg.get("required", False))
        kind = str(dcfg.get("kind", ""))
        dataset_id = str(dcfg.get("dataset_id", "")).strip()
        if not dataset_id:
            dataset_id = _infer_hf_dataset_id(str(dcfg.get("link", "")))
        config_name = str(dcfg.get("config", "")).strip()
        split_spec = self._split_spec(dcfg)

        if kind != "huggingface":
            msg = f"Unsupported dataset kind for {name}: {kind}"
            if required and self.enforce_requirements:
                raise ValueError(msg)
            _warn(msg)
            self.status[name]["error"] = msg
            self._cache[name] = None
            return None

        if not dataset_id:
            msg = f"Missing dataset_id for {name}; provide `dataset_id` or a Hugging Face `link`."
            if required and self.enforce_requirements:
                raise ValueError(msg)
            _warn(msg)
            self.status[name]["error"] = msg
            self._cache[name] = None
            return None

        if name == "fever":
            print(f"Loading dataset `{name}` from {FEVER_TRAIN_URL}")
            try:
                ds = _load_fever_dataset(self.cache_dir, limit=self.limit(name))
                self.status[name]["loaded"] = True
                self.status[name]["rows"] = len(ds)
                self._cache[name] = ds
                return ds
            except Exception as exc:
                msg = str(exc)
                self.status[name]["error"] = msg
                if required and self.enforce_requirements:
                    raise
                _warn(f"Failed to load optional dataset `{name}`: {msg}")
                self._cache[name] = None
                return None

        if _hf_load_dataset is None:
            msg = "python package `datasets` is not installed"
            if required and self.enforce_requirements:
                raise RuntimeError(msg)
            _warn(f"{name}: {msg}")
            self.status[name]["error"] = msg
            self._cache[name] = None
            return None

        print(f"Loading dataset `{name}` from {dcfg.get('link', dataset_id)}")
        try:
            if config_name:
                ds = _hf_load_dataset(
                    dataset_id,
                    config_name,
                    split=split_spec,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
            else:
                ds = _hf_load_dataset(
                    dataset_id,
                    split=split_spec,
                    cache_dir=str(self.cache_dir) if self.cache_dir else None,
                )
            self.status[name]["loaded"] = True
            try:
                self.status[name]["rows"] = int(ds.num_rows)
            except Exception:
                self.status[name]["rows"] = None
            self._cache[name] = ds
            return ds
        except Exception as exc:
            msg = str(exc)
            self.status[name]["error"] = msg
            if required and self.enforce_requirements:
                raise
            _warn(f"Failed to load optional dataset `{name}`: {msg}")
            self._cache[name] = None
            return None

    def ensure_required(self) -> None:
        for name, dcfg in self.datasets_cfg.items():
            if bool(dcfg.get("required", False)):
                _ = self.get(name)


def _parse_json_content(content: str) -> dict[str, object] | None:
    text = content.strip()
    if not text:
        return None
    fenced = re.search(r"```(?:json)?\s*(\{.*\})\s*```", text, flags=re.DOTALL)
    candidate = fenced.group(1) if fenced else text
    try:
        payload = json.loads(candidate)
        if isinstance(payload, dict):
            return payload
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, flags=re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(0))
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _decode_json_string(raw: str) -> str:
    try:
        return str(json.loads(f'"{raw}"'))
    except Exception:
        # Best-effort fallback for partially escaped fragments.
        return (
            str(raw).replace("\\n", " ").replace("\\t", " ").replace('\\"', '"').replace("\\/", "/")
        )


def _extract_json_string_fields(
    *, content: str, field: str, limit: int, max_items: int | None = None
) -> list[str]:
    pattern = re.compile(rf'"{re.escape(field)}"\s*:\s*"((?:\\.|[^"\\])*)"', flags=re.DOTALL)
    out: list[str] = []
    seen: set[str] = set()
    for match in pattern.finditer(content):
        text = _clean(_decode_json_string(match.group(1)), limit)
        if not text:
            continue
        key = _norm_key(text)
        if key in seen:
            continue
        seen.add(key)
        out.append(text)
        if max_items is not None and len(out) >= max_items:
            break
    return out


def _match_model_metadata(model_name: str, payload: dict[str, Any] | None) -> dict[str, Any] | None:
    if not payload or not isinstance(payload, dict):
        return None
    items = payload.get("data")
    if not isinstance(items, list) or not items:
        return None
    wanted = str(model_name or "").strip().lower()
    if not wanted:
        return next((item for item in items if isinstance(item, dict)), None)

    def _names(item: dict[str, Any]) -> list[str]:
        names = []
        for key in ("id", "root", "parent"):
            value = str(item.get(key, "") or "").strip()
            if value:
                names.append(value)
        return names

    exact = next(
        (
            item
            for item in items
            if isinstance(item, dict)
            and any(name.lower() == wanted for name in _names(item))
        ),
        None,
    )
    if exact is not None:
        return exact

    partial = next(
        (
            item
            for item in items
            if isinstance(item, dict)
            and any(wanted in name.lower() or name.lower() in wanted for name in _names(item))
        ),
        None,
    )
    if partial is not None:
        return partial
    if len(items) == 1 and isinstance(items[0], dict):
        return items[0]
    return None


def _thinking_disable_policy(
    *,
    provider: str,
    base_url: str,
    model_name: str,
    model_metadata: dict[str, Any] | None,
) -> dict[str, Any]:
    model_id = (
        str((model_metadata or {}).get("id", "") or (model_metadata or {}).get("root", "") or model_name)
        .strip()
    )
    owner = str((model_metadata or {}).get("owned_by", "") or "").strip().lower()
    lowered = model_id.lower()
    provider_lower = str(provider or "").strip().lower()
    host = str(urlparse(base_url).netloc or "").strip().lower()

    policy: dict[str, Any] = {
        "model_id": model_id or model_name,
        "owned_by": owner,
        "chat_template_kwargs": None,
        "assistant_prefill": None,
        "reasoning_effort": None,
        "omit_sampling_controls": False,
        "warning": "",
        "reason": "",
    }

    # vLLM / compatible backends safely ignore unknown chat_template kwargs.
    if provider_lower in {"vllm", "openai_compatible", "sglang"} or owner == "vllm":
        policy["chat_template_kwargs"] = {
            "enable_thinking": False,
            "thinking": False,
        }

    # Some reasoning-tuned families still ignore template kwargs.
    if any(token in lowered for token in ("gpt-oss", "gpt_oss", "gptoss")):
        policy["assistant_prefill"] = "<think></think>\n"

    is_openai = provider_lower == "openai" or "api.openai.com" in host or owner == "openai"
    is_gemini_openai = "generativelanguage.googleapis.com" in host or lowered.startswith("gemini-")
    is_deepseek = "api.deepseek.com" in host or lowered.startswith("deepseek-")

    if is_openai:
        if lowered.startswith(("gpt-5-pro", "gpt-5.2-pro", "o3-pro")):
            policy["warning"] = (
                f"{model_id or model_name} cannot be switched to no-thinking in Chat Completions; "
                "these higher-effort variants require reasoning or use Responses-only flows."
            )
        elif lowered.startswith(("gpt-5.2-codex", "gpt-5.1-codex-max")):
            policy["warning"] = (
                f"{model_id or model_name} is a Codex-only reasoning variant; "
                "Chat Completions cannot force no-thinking for this model."
            )
        elif lowered.startswith(("gpt-5.1", "gpt-5.2")):
            policy["reasoning_effort"] = "none"
        elif lowered.startswith(("gpt-5", "gpt-5-mini", "gpt-5-nano")):
            policy["reasoning_effort"] = "minimal"
            policy["omit_sampling_controls"] = True
        elif lowered.startswith(("o3", "o4-mini", "o3-mini", "codex-mini-latest")) or lowered.startswith("gpt-oss"):
            policy["reasoning_effort"] = "low"

    if is_gemini_openai:
        if lowered.startswith("gemini-2.5") and "pro" not in lowered:
            policy["reasoning_effort"] = "none"
        elif lowered.startswith("gemini-2.5-pro") or lowered.startswith("gemini-3"):
            policy["warning"] = (
                f"{model_id or model_name} does not support fully disabling thinking; "
                "Google documents thinking-off only for Gemini 2.5 non-Pro models."
            )

    if is_deepseek:
        if lowered.startswith("deepseek-chat"):
            # Official non-thinking mode according to DeepSeek docs.
            pass
        elif lowered.startswith("deepseek-reasoner"):
            policy["warning"] = (
                "deepseek-reasoner is the thinking-mode model. "
                "Use deepseek-chat if you need non-thinking behavior on the official DeepSeek API."
            )

    reasons: list[str] = []
    if policy["chat_template_kwargs"] is not None:
        reasons.append("chat_template_kwargs")
    if policy["assistant_prefill"]:
        reasons.append("assistant_prefill")
    if policy["reasoning_effort"]:
        reasons.append(f"reasoning_effort={policy['reasoning_effort']}")
    if policy["omit_sampling_controls"]:
        reasons.append("omit_sampling_controls")
    if any(token in lowered for token in ("qwen", "deepseek", "glm", "granite", "holo", "gpt-oss")):
        reasons.append(f"model_family={model_id or model_name}")
    policy["reason"] = ", ".join(reasons)
    return policy


def _looks_like_thinking_output(content: str) -> bool:
    text = str(content or "").strip()
    if not text:
        return False
    lowered = text[:400].lower()
    if lowered.startswith("<think") or "<think>" in lowered or "</think>" in lowered:
        return True
    if text.startswith("{") or text.startswith("```json"):
        return False
    markers = (
        "okay, let's",
        "let's tackle",
        "let me think",
        "i need to",
        "the user wants me to",
        "first, i",
        "first i",
        "i should",
        "reasoning:",
        "thought:",
    )
    return any(marker in lowered for marker in markers)


class _LLMGenerator:
    def __init__(self, cfg: dict) -> None:
        if httpx is None:
            raise RuntimeError("`httpx` is required for LLM synthetic generation.")

        provider_env = str(cfg.get("provider_env", "LLM_EVAL__PROVIDER"))
        model_env = str(cfg.get("model_env", "LLM_EVAL__MODEL"))
        base_url_env = str(cfg.get("base_url_env", "LLM_EVAL__BASE_URL"))
        api_key_env = str(cfg.get("api_key_env", "OPENAI_API_KEY"))

        self.provider = (
            (str(cfg.get("provider", "")).strip() or os.getenv(provider_env, "")).strip().lower()
        )
        self.model = (str(cfg.get("model", "")).strip() or os.getenv(model_env, "")).strip()
        self.base_url = (
            str(cfg.get("base_url", "")).strip() or os.getenv(base_url_env, "")
        ).strip()
        self.api_key = os.getenv(api_key_env, "").strip()

        self.temperature = float(cfg.get("temperature", 1.3))
        self.top_p = float(cfg.get("top_p", 0.95))
        self.max_tokens = int(cfg.get("max_tokens", 2048))
        self.batch_size = max(1, int(cfg.get("batch_size", 32)))
        self.concurrency = max(1, int(cfg.get("concurrency", 8)))
        self.timeout_seconds = max(10, int(cfg.get("timeout_seconds", 120)))
        self.max_retries = max(1, int(cfg.get("max_retries", 3)))
        self.log_stats_every_seconds = max(1.0, float(cfg.get("log_stats_every_seconds", 10.0)))
        self.log_zero_progress_every = max(1, int(cfg.get("log_zero_progress_every", 5)))
        self.parse_failure_log_every = max(1, int(cfg.get("parse_failure_log_every", 20)))
        self.log_request_failures = bool(cfg.get("log_request_failures", True))

        if not self.model:
            raise ValueError("LLM model missing. Set LLM_EVAL__MODEL or synthetic_llm.model.")
        if not self.base_url:
            raise ValueError(
                "LLM base URL missing. Set LLM_EVAL__BASE_URL or synthetic_llm.base_url."
            )

        supported = {"", "ollama", "openai", "openai_compatible", "vllm", "sglang"}
        if self.provider not in supported:
            _warn(
                f"LLM provider is `{self.provider}`; expected one of "
                "{ollama, openai, openai_compatible, vllm, sglang}."
            )

        self.url = self.base_url.rstrip("/") + "/chat/completions"
        self.models_url = self.base_url.rstrip("/") + "/models"
        limits = httpx.Limits(
            max_connections=max(64, self.concurrency * 8),
            max_keepalive_connections=max(32, self.concurrency * 4),
        )
        self.client = httpx.Client(timeout=self.timeout_seconds, limits=limits)
        self.model_metadata = self._fetch_model_metadata()
        self.thinking_disable = _thinking_disable_policy(
            provider=self.provider,
            base_url=self.base_url,
            model_name=self.model,
            model_metadata=self.model_metadata,
        )
        if self.thinking_disable.get("reason"):
            _warn(
                "LLM thinking suppression enabled: "
                f"model={self.thinking_disable.get('model_id') or self.model} "
                f"owner={self.thinking_disable.get('owned_by') or 'unknown'} "
                f"strategy={self.thinking_disable.get('reason')}"
            )
        if self.thinking_disable.get("warning"):
            _warn(str(self.thinking_disable["warning"]))

        self._stats_lock = threading.Lock()
        self._started = time.perf_counter()
        self._last_report = self._started
        self._requests_total = 0
        self._requests_failed = 0
        self._retries_total = 0
        self._generated_total = 0
        self._accepted_total = 0
        self._parse_fail_total = 0
        self._parse_recovered_total = 0
        self._finish_reason_counts: dict[str, int] = defaultdict(int)

    def _fetch_model_metadata(self) -> dict[str, Any] | None:
        if self.provider not in {"vllm", "openai_compatible", "sglang", "openai"}:
            return None
        try:
            resp = self.client.get(self.models_url)
            resp.raise_for_status()
            payload = resp.json()
        except Exception as exc:
            _warn(f"Unable to fetch {self.models_url}: {exc}")
            return None
        matched = _match_model_metadata(self.model, payload if isinstance(payload, dict) else None)
        if matched is not None:
            return matched
        return None

    def stats_snapshot(self) -> dict[str, Any]:
        with self._stats_lock:
            return {
                "requests_total": self._requests_total,
                "requests_failed": self._requests_failed,
                "retries_total": self._retries_total,
                "generated_total": self._generated_total,
                "accepted_total": self._accepted_total,
                "parse_fail_total": self._parse_fail_total,
                "parse_recovered_total": self._parse_recovered_total,
                "finish_reason_counts": dict(self._finish_reason_counts),
            }

    def _record_request(self, *, success: bool, retries: int, finish_reason: str = "") -> None:
        with self._stats_lock:
            self._requests_total += 1
            self._retries_total += max(0, retries)
            if not success:
                self._requests_failed += 1
            if finish_reason:
                self._finish_reason_counts[finish_reason] += 1

    def record_batch_result(self, *, generated: int, accepted: int) -> None:
        with self._stats_lock:
            self._generated_total += max(0, int(generated))
            self._accepted_total += max(0, int(accepted))

    def record_parse_failure(self, *, task: str, label: str, content: str) -> None:
        with self._stats_lock:
            self._parse_fail_total += 1
            count = self._parse_fail_total
        if count == 1 or count % self.parse_failure_log_every == 0:
            snippet = _clean(content, 220)
            _warn(f"LLM parse failure #{count} for {task}::{label}. sample_response={snippet!r}")

    def record_parse_recovery(self, *, recovered_count: int) -> None:
        with self._stats_lock:
            self._parse_recovered_total += max(0, int(recovered_count))

    def maybe_report(self, *, context: str = "", force: bool = False) -> None:
        now = time.perf_counter()
        with self._stats_lock:
            elapsed = max(1e-6, now - self._started)
            interval = now - self._last_report
            if not force and interval < self.log_stats_every_seconds:
                return
            self._last_report = now
            req_total = self._requests_total
            req_failed = self._requests_failed
            retries = self._retries_total
            generated = self._generated_total
            accepted = self._accepted_total
            parse_fails = self._parse_fail_total
            parse_recovered = self._parse_recovered_total
            finish_reasons = dict(self._finish_reason_counts)

        req_per_s = req_total / elapsed
        gen_per_s = generated / elapsed
        acc_per_s = accepted / elapsed
        accept_ratio = (accepted / generated) if generated > 0 else 0.0
        scope = f" [{context}]" if context else ""
        finish_reason_text = ""
        if finish_reasons:
            top = sorted(finish_reasons.items(), key=lambda kv: kv[1], reverse=True)[:3]
            finish_reason_text = " finish_reason=" + ",".join(f"{k}:{v}" for k, v in top)
        _warn(
            f"LLM stats{scope}: req={req_total} fail={req_failed} retries={retries} "
            f"parse_fail={parse_fails} req/s={req_per_s:.2f} gen={generated} "
            f"acc={accepted} gen/s={gen_per_s:.2f} acc/s={acc_per_s:.2f} "
            f"acc/gen={accept_ratio:.2%} parse_recovered={parse_recovered}{finish_reason_text}"
        )

    def _request(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        force_prefill_thinking_closed: bool = False,
    ) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        messages: list[dict[str, str]] = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]
        assistant_prefill = None
        if force_prefill_thinking_closed:
            assistant_prefill = str(
                self.thinking_disable.get("assistant_prefill") or "<think></think>\n"
            )
        elif self.thinking_disable.get("assistant_prefill"):
            assistant_prefill = str(self.thinking_disable["assistant_prefill"])
        if assistant_prefill:
            messages.append({"role": "assistant", "content": assistant_prefill})

        payload = {
            "model": self.model,
            "messages": messages,
            "max_tokens": self.max_tokens,
        }
        if not bool(self.thinking_disable.get("omit_sampling_controls")):
            payload["temperature"] = self.temperature
            payload["top_p"] = self.top_p
        if self.provider in {"openai", "openai_compatible", "vllm", "sglang"}:
            payload["response_format"] = {"type": "json_object"}
        if self.thinking_disable.get("reasoning_effort"):
            payload["reasoning_effort"] = self.thinking_disable["reasoning_effort"]
        chat_template_kwargs = self.thinking_disable.get("chat_template_kwargs")
        if (
            chat_template_kwargs
            and isinstance(chat_template_kwargs, dict)
            and self.provider in {"openai_compatible", "vllm", "sglang"}
        ):
            payload["extra_body"] = {"chat_template_kwargs": chat_template_kwargs}

        last_err = ""
        for attempt in range(1, self.max_retries + 1):
            try:
                resp = self.client.post(self.url, headers=headers, json=payload)
                resp.raise_for_status()
                data = resp.json()
                choice = data["choices"][0]
                content = choice["message"]["content"]
                finish_reason = str(choice.get("finish_reason", "")).strip().lower()
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty LLM response content.")
                self._record_request(
                    success=True,
                    retries=attempt - 1,
                    finish_reason=finish_reason,
                )
                self.maybe_report()
                return content
            except Exception as exc:
                last_err = str(exc)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(6, attempt * 1.5))
        self._record_request(success=False, retries=self.max_retries - 1)
        self.maybe_report()
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_err}")

    def generate_single(
        self,
        *,
        task: str,
        label: str,
        seed_text: str,
        n: int,
        language: _multilingual_prompts.Language | None = None,
    ) -> list[str]:
        if language:
            system = _multilingual_prompts.system_prompt_single(language)
            user = _multilingual_prompts.user_prompt_single(task, label, n, seed_text, language)
        else:
            system = (
                "Generate synthetic classification data. "
                "Return STRICT JSON only, no markdown fences. "
                "Do not explain, think aloud, or prepend commentary."
            )
            user = (
                f"Task: {task}\nTarget label: {label}\nCount: {n}\n"
                f"Seed example from related dataset (do not copy): {seed_text}\n\n"
                'Return exactly: {"samples":[{"text":"..."}]}\n'
                "Each sample must match target label exactly, be diverse, "
                "and be one concise sentence (8-22 words)."
            )
        raw = self._request(system, user)
        payload = _parse_json_content(raw)
        if payload is None and _looks_like_thinking_output(raw):
            _warn(f"Detected reasoning output for {task}::{label}; retrying with forced no-thinking.")
            raw = self._request(system, user, force_prefill_thinking_closed=True)
            payload = _parse_json_content(raw)
        if not payload:
            recovered = _extract_json_string_fields(
                content=raw,
                field="text",
                limit=1000,
                max_items=max(1, n * 2),
            )
            if recovered:
                self.record_parse_recovery(recovered_count=len(recovered))
                return recovered
            self.record_parse_failure(task=task, label=label, content=raw)
            return []
        out: list[str] = []
        samples = payload.get("samples", [])
        if isinstance(samples, list):
            for item in samples:
                if isinstance(item, dict):
                    text = _clean(item.get("text", ""), 1000)
                else:
                    text = _clean(item, 1000)
                if text:
                    out.append(text)
        return out

    def generate_pair(
        self,
        *,
        task: str,
        label: str,
        seed_a: str,
        seed_b: str,
        n: int,
        language: _multilingual_prompts.Language | None = None,
    ) -> list[tuple[str, str]]:
        if language:
            system = _multilingual_prompts.system_prompt_pair(language)
            user = _multilingual_prompts.user_prompt_pair(task, label, n, seed_a, seed_b, language)
        else:
            system = (
                "Generate synthetic text-pair classification data. "
                "Return STRICT JSON only, no markdown fences. "
                "Do not explain, think aloud, or prepend commentary."
            )
            user = (
                f"Task: {task}\nTarget label: {label}\nCount: {n}\n"
                f"Seed pair from related dataset (do not copy): A={seed_a} | B={seed_b}\n\n"
                'Return exactly: {"samples":[{"text_a":"...","text_b":"..."}]}\n'
                "Every pair must match target label exactly and be diverse. "
                "Each field should be one concise sentence (6-20 words)."
            )
        raw = self._request(system, user)
        payload = _parse_json_content(raw)
        if payload is None and _looks_like_thinking_output(raw):
            _warn(f"Detected reasoning output for {task}::{label}; retrying with forced no-thinking.")
            raw = self._request(system, user, force_prefill_thinking_closed=True)
            payload = _parse_json_content(raw)
        if not payload:
            recovered_a = _extract_json_string_fields(
                content=raw,
                field="text_a",
                limit=700,
                max_items=max(1, n * 2),
            )
            recovered_b = _extract_json_string_fields(
                content=raw,
                field="text_b",
                limit=700,
                max_items=max(1, n * 2),
            )
            recovered_pairs = list(zip(recovered_a, recovered_b, strict=False))
            if recovered_pairs:
                self.record_parse_recovery(recovered_count=len(recovered_pairs))
                return recovered_pairs
            self.record_parse_failure(task=task, label=label, content=raw)
            return []
        out: list[tuple[str, str]] = []
        samples = payload.get("samples", [])
        if isinstance(samples, list):
            for item in samples:
                if not isinstance(item, dict):
                    continue
                a = _clean(item.get("text_a", ""), 700)
                b = _clean(item.get("text_b", ""), 700)
                if a and b:
                    out.append((a, b))
        return out


def _iter_dataset_rows(dataset: object, *, limit: int, desc: str):
    total = limit
    if hasattr(dataset, "num_rows"):
        try:
            total = min(limit, int(getattr(dataset, "num_rows", 0)))
        except Exception:
            total = limit
    pbar = _progress(total=total, desc=desc, unit="row")
    count = 0
    try:
        for row in dataset:  # type: ignore[attr-defined]
            if count >= limit:
                break
            yield row
            count += 1
            pbar.update(1)
    finally:
        pbar.close()


def _load_local_bootstrap_rows(
    bootstrap_dir: Path, max_rows_per_source: int, seed: int
) -> list[dict]:
    rows: list[dict] = []
    for i, split in enumerate(LOCAL_BOOTSTRAP_SPLITS):
        path = bootstrap_dir / f"{split}.parquet"
        if not path.exists():
            continue
        df = pd.read_parquet(path)
        if not {"text", "label"}.issubset(df.columns):
            continue
        sampled = _sample_df(df[["text", "label"]], max_rows_per_source, seed + i)
        for item in sampled.itertuples(index=False):
            text = _clean(getattr(item, "text", ""))
            label = _clean(getattr(item, "label", ""))
            if text and label:
                rows.append({"text": text, "label": label, "source": f"local:{split}"})
    return rows


def _extract_text_candidates(ex: dict, limit: int = 3) -> list[str]:
    keys = (
        "text",
        "utt",
        "query",
        "question",
        "content",
        "title",
        "norm",
        "moral_action",
        "immoral_action",
        "situation",
        "source_text",
        "target_text",
        "source",
        "target",
        "premise",
        "hypothesis",
        "sentence1",
        "sentence2",
    )
    out: list[str] = []
    for key in keys:
        if key in ex:
            text = _clean(ex.get(key, ""), 1000)
            if text:
                out.append(text)
        if len(out) >= limit:
            return out

    # Common conversational datasets store message lists; keep only first turns.
    messages = ex.get("messages")
    if isinstance(messages, list):
        for msg in messages[:6]:
            if not isinstance(msg, dict):
                continue
            text = _clean(msg.get("content", ""), 1000)
            if text:
                out.append(text)
            if len(out) >= limit:
                return out

    # Preference / dialogue datasets frequently use chosen/rejected style fields.
    for key in ("chosen", "rejected"):
        text = _clean(ex.get(key, ""), 1000)
        if text:
            out.append(text)
        if len(out) >= limit:
            return out
    return out


def _extract_pair_candidates(ex: dict) -> list[tuple[str, str]]:
    pairs: list[tuple[str, str]] = []
    for a_key, b_key in (("premise", "hypothesis"), ("sentence1", "sentence2"), ("query", "text")):
        a = _clean(ex.get(a_key, ""), 700)
        b = _clean(ex.get(b_key, ""), 700)
        if a and b:
            pairs.append((a, b))

    query = _clean(ex.get("query", ""), 700)
    passages = ex.get("passages", {})
    if query and isinstance(passages, dict):
        ptexts = passages.get("passage_text", [])
        if isinstance(ptexts, list):
            for text in ptexts[:2]:
                pt = _clean(text, 700)
                if pt:
                    pairs.append((query, pt))

    chosen = _clean(ex.get("chosen", ""), 700)
    rejected = _clean(ex.get("rejected", ""), 700)
    if chosen and rejected:
        pairs.append((chosen, rejected))

    messages = ex.get("messages")
    if isinstance(messages, list):
        turns = []
        for msg in messages[:8]:
            if isinstance(msg, dict):
                content = _clean(msg.get("content", ""), 700)
                if content:
                    turns.append(content)
        for i in range(0, len(turns) - 1, 2):
            pairs.append((turns[i], turns[i + 1]))
    return pairs


def _scan_local_raw_seed_texts(
    raw_dir: Path, max_rows_per_file: int
) -> tuple[list[str], list[tuple[str, str]]]:
    if not raw_dir.exists():
        return [], []

    texts: list[str] = []
    pairs: list[tuple[str, str]] = []

    files = sorted([p for p in raw_dir.iterdir() if p.is_file()])
    pbar = _progress(total=len(files), desc="Local raw seed scan", unit="file")
    try:
        for file_path in files:
            pbar.set_description(f"Seed scan [{file_path.name}]")
            suffix = file_path.suffix.lower()
            try:
                if suffix == ".parquet":
                    df = pd.read_parquet(file_path).head(max_rows_per_file)
                    for row in df.to_dict(orient="records"):
                        texts.extend(_extract_text_candidates(row))
                        pairs.extend(_extract_pair_candidates(row))
                elif suffix == ".csv":
                    df = pd.read_csv(file_path, nrows=max_rows_per_file, low_memory=False)
                    for row in df.to_dict(orient="records"):
                        texts.extend(_extract_text_candidates(row))
                        pairs.extend(_extract_pair_candidates(row))
                elif suffix == ".jsonl":
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= max_rows_per_file:
                                break
                            line = line.strip()
                            if not line:
                                continue
                            try:
                                row = json.loads(line)
                            except Exception:
                                continue
                            if isinstance(row, dict):
                                texts.extend(_extract_text_candidates(row))
                                pairs.extend(_extract_pair_candidates(row))
                elif suffix in {".txt", ".en"}:
                    with open(file_path, encoding="utf-8", errors="ignore") as f:
                        for i, line in enumerate(f):
                            if i >= max_rows_per_file:
                                break
                            text = _clean(line, 1000)
                            if text:
                                texts.append(text)
            except Exception as exc:
                _warn(f"Skipping local raw file {file_path.name}: {exc}")
            pbar.update(1)
    finally:
        pbar.close()

    return list(dict.fromkeys(texts)), list(dict.fromkeys(pairs))


def _collect_seed_pools(
    *,
    local_rows: list[dict],
    local_raw_texts: list[str],
    local_raw_pairs: list[tuple[str, str]],
    registry: _HFRegistry,
    datasets_cfg: list[dict],
    max_rows_per_source: int,
) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    single_pools: dict[str, list[str]] = {"router": [], "extractor": [], "pair": []}
    pair_pool: list[tuple[str, str]] = []

    for item in local_rows:
        text = _clean(item.get("text", ""), 1000)
        if text:
            for fam in single_pools:
                single_pools[fam].append(text)

    for text in local_raw_texts:
        for fam in single_pools:
            single_pools[fam].append(text)
    pair_pool.extend(local_raw_pairs)

    enabled = [d for d in datasets_cfg if d.get("enabled", True)]
    pbar = _progress(total=len(enabled), desc="Remote seed pools", unit="dataset")
    for dcfg in enabled:
        name = str(dcfg.get("name", "")).strip()
        target = str(dcfg.get("target", "")).strip().lower()
        if not name:
            pbar.update(1)
            continue
        pbar.set_description(f"Remote seeds [{name}]")
        ds = registry.get(name)
        if ds is None:
            pbar.update(1)
            continue

        limit = min(max_rows_per_source, registry.limit(name), 10000)
        for ex in _iter_dataset_rows(ds, limit=limit, desc=f"Seed rows [{name}]"):
            if not isinstance(ex, dict):
                continue
            texts = _extract_text_candidates(ex)
            if target in single_pools:
                single_pools[target].extend(texts)
            else:
                for fam in single_pools:
                    single_pools[fam].extend(texts)
            if target == "pair":
                pair_pool.extend(_extract_pair_candidates(ex))
        pbar.update(1)
    pbar.close()

    for fam in single_pools:
        single_pools[fam] = list(dict.fromkeys([x for x in single_pools[fam] if x]))
    pair_pool = list(dict.fromkeys([(a, b) for a, b in pair_pool if a and b]))
    return single_pools, pair_pool


def _add_existing_router_memory_rows(rows: _SingleTaskStore, local_rows: list[dict]) -> None:
    valid = set(ROUTER_TASK_LABELS["memory_type"])
    for item in local_rows:
        text = _clean(item.get("text", ""), 1000)
        label = _clean(item.get("label", ""), 80)
        if text and label in valid:
            rows.add("memory_type", text, label, str(item.get("source", "local")))


def _add_existing_router_domain_rows(rows: _SingleTaskStore, registry: _HFRegistry) -> None:
    # banking77 is fully banking/finance intent data and aligns directly with finance domain.
    ds = registry.get("banking77")
    if ds is None:
        return
    for ex in _iter_dataset_rows(
        ds, limit=registry.limit("banking77"), desc="Existing rows [banking77 domain]"
    ):
        if not isinstance(ex, dict):
            continue
        text = _clean(ex.get("text", ""), 1000)
        if text:
            rows.add("query_domain", text, "finance", "hf:banking77")


def _add_existing_extractor_pii_rows(rows: _SingleTaskStore, registry: _HFRegistry) -> None:
    ds = registry.get("pii_masking")
    if ds is None:
        return
    for ex in _iter_dataset_rows(
        ds, limit=registry.limit("pii_masking"), desc="Existing rows [pii_masking]"
    ):
        if not isinstance(ex, dict):
            continue
        src = _clean(ex.get("source_text", ex.get("source", "")), 1000)
        tgt = _clean(ex.get("target_text", ex.get("target", "")), 1000)
        if src:
            rows.add("pii_presence", src, "pii", "hf:pii_masking")
        if tgt:
            rows.add("pii_presence", tgt, "no_pii", "hf:pii_masking")


def _add_existing_pair_rows(rows: _PairTaskStore, registry: _HFRegistry) -> None:
    for dataset_name in ("snli", "multi_nli", "anli"):
        ds = registry.get(dataset_name)
        if ds is None:
            continue
        label_names = _safe_feature_names(ds, "label")
        for ex in _iter_dataset_rows(
            ds, limit=registry.limit(dataset_name), desc=f"Existing rows [{dataset_name}]"
        ):
            if not isinstance(ex, dict):
                continue
            label_name = _nli_label_name(ex.get("label", -1), label_names)
            is_conflict = label_name == "contradiction"
            is_scope_match = label_name in {"contradiction", "entailment"}
            is_supersedes = label_name == "contradiction"

            rows.add(
                "conflict_detection",
                ex.get("premise", ""),
                ex.get("hypothesis", ""),
                "conflict" if is_conflict else "no_conflict",
                f"hf:{dataset_name}",
            )
            rows.add(
                "scope_match",
                ex.get("premise", ""),
                ex.get("hypothesis", ""),
                "match" if is_scope_match else "no_match",
                f"hf:{dataset_name}",
            )
            rows.add(
                "supersession",
                ex.get("premise", ""),
                ex.get("hypothesis", ""),
                "supersedes" if is_supersedes else "no_supersedes",
                f"hf:{dataset_name}",
            )
            rows.add(
                "novelty_pair",
                ex.get("premise", ""),
                ex.get("hypothesis", ""),
                "changed" if is_conflict else "novel",
                f"hf:{dataset_name}",
            )

    ds_mm = registry.get("ms_marco")
    if ds_mm is not None:
        for idx, ex in enumerate(
            _iter_dataset_rows(
                ds_mm, limit=registry.limit("ms_marco"), desc="Existing rows [ms_marco]"
            )
        ):
            if not isinstance(ex, dict):
                continue
            query = _clean(ex.get("query", ""), 700)
            passages = ex.get("passages", {})
            if not query or not isinstance(passages, dict):
                continue
            selected = passages.get("is_selected", [])
            ptexts = passages.get("passage_text", [])
            if not isinstance(selected, list) or not isinstance(ptexts, list):
                continue

            positive = ""
            negative = ""
            for flag, ptxt in zip(selected, ptexts, strict=False):
                passage = _clean(ptxt, 700)
                try:
                    f = int(flag)
                except Exception:
                    f = 0
                if f == 1 and passage and not positive:
                    positive = passage
                if f == 0 and passage and not negative:
                    negative = passage
                if positive and negative:
                    break
            group_id = f"hf:ms_marco:{idx}"
            if positive:
                rows.add("constraint_rerank", query, positive, "relevant", "hf:ms_marco")
                rows.add("scope_match", query, positive, "match", "hf:ms_marco")
                rows.add(
                    "retrieval_constraint_relevance_pair",
                    query,
                    positive,
                    "relevant",
                    "hf:ms_marco",
                    extras={"group_id": group_id},
                )
                rows.add(
                    "memory_rerank_pair",
                    query,
                    positive,
                    "relevant",
                    "hf:ms_marco",
                    extras={"group_id": group_id},
                )
                rows.add(
                    "reconsolidation_candidate_pair",
                    query,
                    positive,
                    "relevant",
                    "hf:ms_marco",
                    extras={"group_id": group_id},
                )
            if negative:
                rows.add("constraint_rerank", query, negative, "not_relevant", "hf:ms_marco")
                rows.add("scope_match", query, negative, "no_match", "hf:ms_marco")
                rows.add(
                    "retrieval_constraint_relevance_pair",
                    query,
                    negative,
                    "not_relevant",
                    "hf:ms_marco",
                    extras={"group_id": group_id},
                )
                rows.add(
                    "memory_rerank_pair",
                    query,
                    negative,
                    "not_relevant",
                    "hf:ms_marco",
                    extras={"group_id": group_id},
                )
                rows.add(
                    "reconsolidation_candidate_pair",
                    query,
                    negative,
                    "not_relevant",
                    "hf:ms_marco",
                    extras={"group_id": group_id},
                )

    ds_quora = registry.get("quora_duplicates")
    if ds_quora is not None:
        for idx, ex in enumerate(
            _iter_dataset_rows(
                ds_quora,
                limit=registry.limit("quora_duplicates"),
                desc="Existing rows [quora_duplicates]",
            )
        ):
            if not isinstance(ex, dict):
                continue
            raw = ex.get("label", 0)
            try:
                relevant = int(raw) == 1
            except Exception:
                relevant = _clean(raw, 40).lower() in {"1", "true", "duplicate", "relevant"}
            group_id = f"hf:quora_duplicates:{idx}"
            rows.add(
                "constraint_rerank",
                ex.get("sentence1", ""),
                ex.get("sentence2", ""),
                "relevant" if relevant else "not_relevant",
                "hf:quora_duplicates",
            )
            rows.add(
                "scope_match",
                ex.get("sentence1", ""),
                ex.get("sentence2", ""),
                "match" if relevant else "no_match",
                "hf:quora_duplicates",
            )
            rows.add(
                "retrieval_constraint_relevance_pair",
                ex.get("sentence1", ""),
                ex.get("sentence2", ""),
                "relevant" if relevant else "not_relevant",
                "hf:quora_duplicates",
                extras={"group_id": group_id},
            )
            rows.add(
                "memory_rerank_pair",
                ex.get("sentence1", ""),
                ex.get("sentence2", ""),
                "relevant" if relevant else "not_relevant",
                "hf:quora_duplicates",
                extras={"group_id": group_id},
            )
            rows.add(
                "reconsolidation_candidate_pair",
                ex.get("sentence1", ""),
                ex.get("sentence2", ""),
                "relevant" if relevant else "not_relevant",
                "hf:quora_duplicates",
                extras={"group_id": group_id},
            )


def _add_changed_rows(
    rows: _PairTaskStore, *, target_per_task_label: int, rng: random.Random
) -> None:
    while rows.count("novelty_pair", "changed") < target_per_task_label:
        idx = rows.count("novelty_pair", "changed")
        topic = idx % 5
        if topic in {0, 3}:
            old = f"I live in City {idx}."
            new = f"I moved to City {idx + 1} last month."
        elif topic == 1:
            old = f"I work as Role {idx}."
            new = f"I changed jobs and now work as Role {idx + 1}."
        elif topic == 2:
            old = f"I wake up at {6 + (idx % 5)} AM every day."
            new = f"I changed my routine and now wake up at {7 + (idx % 5)} AM."
        else:
            old = f"My monthly budget is ${1000 + idx}."
            new = f"My monthly budget increased to ${1200 + idx} this quarter."
        if idx % 2 == 1:
            old = f"{old} I also avoid shellfish."
            new = f"{new} I no longer avoid shellfish."
        rows.add(
            "novelty_pair",
            old,
            new,
            "changed",
            "template:changed",
            extras={"group_id": f"template:novelty_pair:{idx}"},
        )


def _fill_pair_tasks_without_llm(
    rows: _PairTaskStore,
    *,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    rng: random.Random,
) -> None:
    if "novelty_pair" in task_labels:
        _fill_template_novelty_rows(rows, target_per_task_label=target_per_task_label)
        _add_changed_rows(rows, target_per_task_label=target_per_task_label, rng=rng)
    if "schema_match_pair" in task_labels:
        _fill_template_schema_rows(rows, target_per_task_label=target_per_task_label)
    for task in (
        "retrieval_constraint_relevance_pair",
        "memory_rerank_pair",
        "reconsolidation_candidate_pair",
    ):
        if task in task_labels:
            _fill_template_relevance_task(
                rows,
                task=task,
                target_per_task_label=target_per_task_label,
            )


def _add_paws_novelty_rows(rows: _PairTaskStore, registry: _HFRegistry) -> None:
    ds = registry.get("paws")
    if ds is None:
        return
    for ex in _iter_dataset_rows(ds, limit=registry.limit("paws"), desc="Novelty rows [paws]"):
        if not isinstance(ex, dict):
            continue
        try:
            lbl = int(ex.get("label", -1))
        except Exception:
            continue
        if lbl == 1:
            mapped = "duplicate"
        elif lbl == 0:
            mapped = "novel"
        else:
            continue
        rows.add(
            "novelty_pair",
            ex.get("sentence1", ""),
            ex.get("sentence2", ""),
            mapped,
            "hf:paws",
        )


def _add_glue_novelty_rows(rows: _PairTaskStore, registry: _HFRegistry) -> None:
    ds = registry.get("glue_qqp")
    if ds is None:
        return
    for ex in _iter_dataset_rows(
        ds, limit=registry.limit("glue_qqp"), desc="Novelty rows [glue_qqp]"
    ):
        if not isinstance(ex, dict):
            continue
        try:
            lbl = int(ex.get("label", -1))
        except Exception:
            continue
        if lbl == 1:
            mapped = "duplicate"
        elif lbl == 0:
            mapped = "novel"
        else:
            continue
        rows.add(
            "novelty_pair",
            ex.get("question1", ex.get("sentence1", "")),
            ex.get("question2", ex.get("sentence2", "")),
            mapped,
            "hf:glue_qqp",
        )


def _add_fever_schema_match_rows(
    rows: _PairTaskStore,
    registry: _HFRegistry,
    *,
    target_per_task_label: int | None = None,
) -> None:
    ds = registry.get("fever")
    if ds is None:
        return
    fever_target_per_label = None
    if target_per_task_label is not None:
        fever_target_per_label = max(
            1,
            round(
                target_per_task_label
                * float(
                    _TASK_REQUIRED_SOURCE_TARGET_RATIOS.get("schema_match_pair", {}).get(
                        "hf:fever", 1.0
                    )
                )
            ),
        )
    fever_counts = {"match": 0, "no_match": 0}
    for idx, ex in enumerate(
        _iter_dataset_rows(
            ds, limit=registry.limit("fever"), desc="Schema/reconsolidation rows [fever]"
        )
    ):
        if not isinstance(ex, dict):
            continue
        claim = _clean(ex.get("claim", ""), 700)
        evidence = _clean(
            ex.get("evidence", ex.get("evidence_sentence", ex.get("evidences", ""))), 700
        )
        if not claim or not evidence:
            continue
        raw_label = str(ex.get("label", ex.get("gold_label", ""))).strip().lower()
        if "support" in raw_label or "entail" in raw_label:
            schema_label = "match"
            recon_label = "relevant"
        elif (
            "refute" in raw_label
            or "contradict" in raw_label
            or "not enough" in raw_label
            or "nei" in raw_label
        ):
            schema_label = "no_match"
            recon_label = "not_relevant"
        else:
            continue
        if fever_target_per_label is not None and fever_counts[schema_label] >= fever_target_per_label:
            continue
        group_id = f"hf:fever:{idx}"
        added_schema = rows.add(
            "schema_match_pair",
            claim,
            evidence,
            schema_label,
            "hf:fever",
            extras={"group_id": group_id},
        )
        if added_schema:
            fever_counts[schema_label] += 1
        rows.add(
            "reconsolidation_candidate_pair",
            claim,
            evidence,
            recon_label,
            "hf:fever",
            extras={"group_id": group_id},
        )


def _nli_label_name(raw_label: object, label_names: list[str]) -> str:
    if label_names:
        name = _resolve_class_name(raw_label, label_names).strip().lower()
        if name:
            if "entailment" in name:
                return "entailment"
            if "neutral" in name:
                return "neutral"
            if "contradiction" in name or "conflict" in name:
                return "contradiction"
    try:
        idx = int(raw_label)  # type: ignore[call-overload]
    except Exception:
        text = _clean(raw_label, 80).lower()
        if "entailment" in text:
            return "entailment"
        if "neutral" in text:
            return "neutral"
        if "contradiction" in text or "conflict" in text:
            return "contradiction"
        return ""
    if idx == 0:
        return "entailment"
    if idx == 1:
        return "neutral"
    if idx == 2:
        return "contradiction"
    return ""


def _pick_seed_text(single_pools: dict[str, list[str]], family: str, rng: random.Random) -> str:
    pool = single_pools.get(family, [])
    if pool:
        return rng.choice(pool)

    fallback: list[str] = []
    for key in ("router", "extractor", "pair"):
        fallback.extend(single_pools.get(key, []))
    if fallback:
        return rng.choice(fallback)

    return "A user and assistant discuss plans, preferences, and factual details."


def _pick_seed_pair(
    pair_pool: list[tuple[str, str]],
    single_pools: dict[str, list[str]],
    rng: random.Random,
) -> tuple[str, str]:
    if pair_pool:
        return rng.choice(pair_pool)

    base_pool = single_pools.get("pair", [])
    if len(base_pool) >= 2:
        a, b = rng.sample(base_pool, 2)
        return a, b
    if len(base_pool) == 1:
        return base_pool[0], base_pool[0]
    seed = _pick_seed_text(single_pools, "router", rng)
    return seed, seed


_TEMPLATE_TOPIC_PACKS: tuple[dict[str, str], ...] = (
    {
        "topic": "food",
        "query": "What should I cook for dinner",
        "relevant": "User prefers vegetarian meals and avoids pork",
        "irrelevant": "User tracks cloud-infrastructure costs every week",
        "gist": "The user consistently chooses vegetarian meals and avoids pork",
        "fact": "User preference: vegetarian meals without pork",
    },
    {
        "topic": "travel",
        "query": "Which hotel should I book for my trip",
        "relevant": "User prefers quiet hotels near train stations",
        "irrelevant": "User is studying graph databases for work",
        "gist": "The user wants quiet hotels close to transit",
        "fact": "Travel preference: quiet hotels near rail stations",
    },
    {
        "topic": "finance",
        "query": "How should I plan this month",
        "relevant": "User keeps an emergency fund and avoids high-interest debt",
        "irrelevant": "User likes hiking before sunrise on weekends",
        "gist": "The user prioritizes an emergency fund and avoids high-interest debt",
        "fact": "Finance policy: keep emergency savings and avoid high-interest debt",
    },
    {
        "topic": "health",
        "query": "What should I order at the restaurant",
        "relevant": "User has a shellfish allergy and prefers low-sodium meals",
        "irrelevant": "User manages a release schedule for mobile apps",
        "gist": "The user must avoid shellfish and usually chooses low-sodium meals",
        "fact": "Health constraint: avoid shellfish and favor low-sodium meals",
    },
    {
        "topic": "work",
        "query": "How should I organize this project",
        "relevant": "User prefers written project plans and weekly status summaries",
        "irrelevant": "User likes citrus desserts after dinner",
        "gist": "The user prefers written plans and weekly status summaries",
        "fact": "Work preference: written plans with weekly status summaries",
    },
    {
        "topic": "tech",
        "query": "What setup should I use for this task",
        "relevant": "User prefers Python tooling and reproducible command-line workflows",
        "irrelevant": "User books aisle seats on long flights",
        "gist": "The user prefers Python-based tooling and reproducible CLI workflows",
        "fact": "Tech preference: Python tooling with reproducible CLI workflows",
    },
    {
        "topic": "social",
        "query": "What gift would fit best",
        "relevant": "User prefers practical gifts and handwritten notes",
        "irrelevant": "User monitors spending in a budgeting spreadsheet",
        "gist": "The user values practical gifts with a personal note",
        "fact": "Social preference: practical gifts and handwritten notes",
    },
)


def _topic_pack(idx: int) -> dict[str, str]:
    return _TEMPLATE_TOPIC_PACKS[idx % len(_TEMPLATE_TOPIC_PACKS)]


def _other_topic_pack(idx: int) -> dict[str, str]:
    return _TEMPLATE_TOPIC_PACKS[(idx + 3) % len(_TEMPLATE_TOPIC_PACKS)]


def _seed_fragment(single_pools: dict[str, list[str]], *, rng: random.Random, idx: int) -> str:
    seed = _pick_seed_text(single_pools, "router", rng)
    fragment = _clean(seed.split(".")[0], 120)
    if fragment:
        return fragment
    pack = _topic_pack(idx)
    return f"{pack['topic'].title()} preference reference {idx}"


def _cycle_choice(options: list[str] | tuple[str, ...], idx: int, *, offset: int = 0) -> str:
    values = list(options)
    if not values:
        return ""
    return values[(idx + offset) % len(values)]


def _structured_row_extras(
    *,
    topic: str,
    memory_type: str,
    importance: float,
    confidence: float,
    access_count: int,
    age_days: int,
    dependency_count: int,
    support_count: int | None = None,
    mixed_topic: bool | None = None,
    context_tags: list[str] | None = None,
    group_id: str | None = None,
) -> dict[str, Any]:
    extras: dict[str, Any] = {
        "memory_type": memory_type,
        "namespace": topic,
        "context_tags": list(context_tags or [topic]),
        "importance": round(max(0.0, min(1.0, importance)), 4),
        "confidence": round(max(0.0, min(1.0, confidence)), 4),
        "access_count": int(max(0, access_count)),
        "age_days": int(max(0, age_days)),
        "dependency_count": int(max(0, dependency_count)),
    }
    if support_count is not None:
        extras["support_count"] = int(max(1, support_count))
    if mixed_topic is not None:
        extras["mixed_topic"] = bool(mixed_topic)
    if group_id is not None:
        extras["group_id"] = str(group_id)
    return extras


def _count_single_rows_with_source_prefix(
    rows: _SingleTaskStore,
    *,
    task: str,
    label: str,
    prefix: str,
) -> int:
    return sum(
        1
        for row in rows.rows
        if str(row.get("task", "")) == task
        and str(row.get("label", "")) == label
        and str(row.get("source", "")).startswith(prefix)
    )


def _hardening_target(target_per_task_label: int) -> int:
    return min(max(1, target_per_task_label // 5), 2500, max(1, target_per_task_label // 2))


def _router_hardening_shell(idx: int, *, single_pools: dict[str, list[str]], rng: random.Random) -> str:
    pack = _topic_pack(idx)
    seed = _seed_fragment(single_pools, rng=rng, idx=idx)
    shell_idx = idx % 3
    if shell_idx == 0:
        return (
            f"Cluster review {idx}: {pack['gist']}. "
            f"Support notes mention {seed.lower()} and a follow-up action item."
        )
    if shell_idx == 1:
        return (
            f"Session draft {idx}: {pack['fact']}. "
            f"Supporting line references {seed.lower()} for later consolidation."
        )
    return (
        f"Memory brief {idx}: {pack['relevant']}. "
        f"Additional note records {seed.lower()} in the same working set."
    )


def _inject_hardened_router_rows(
    *,
    rows: _SingleTaskStore,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    single_pools: dict[str, list[str]],
    rng: random.Random,
) -> None:
    hard_target = _hardening_target(target_per_task_label)

    if "consolidation_gist_quality" in task_labels:
        for label in ("accept", "reject"):
            while (
                _count_single_rows_with_source_prefix(
                    rows,
                    task="consolidation_gist_quality",
                    label=label,
                    prefix=f"template_hardened:consolidation_gist_quality:{label}",
                )
                < hard_target
                and rows.count("consolidation_gist_quality", label) < target_per_task_label
            ):
                idx = rows.count("consolidation_gist_quality", label)
                pack = _topic_pack(idx)
                other = _other_topic_pack(idx)
                text = _router_hardening_shell(idx, single_pools=single_pools, rng=rng)
                if label == "accept":
                    group_id = f"template_hardened:consolidation_gist_quality:{idx}"
                    extras = _structured_row_extras(
                        topic=pack["topic"],
                        memory_type="semantic_fact",
                        importance=0.66 + (0.03 * (idx % 4)),
                        confidence=0.78 + (0.03 * (idx % 3)),
                        access_count=2 + (idx % 3),
                        age_days=9 + (idx % 24),
                        dependency_count=1 + (idx % 2),
                        support_count=4 + (idx % 3),
                        mixed_topic=False,
                        context_tags=[pack["topic"]],
                        group_id=group_id,
                    )
                else:
                    group_id = f"template_hardened:consolidation_gist_quality:{idx}"
                    extras = _structured_row_extras(
                        topic=pack["topic"],
                        memory_type="semantic_fact",
                        importance=0.38 + (0.03 * (idx % 3)),
                        confidence=0.46 + (0.04 * (idx % 2)),
                        access_count=1 + (idx % 2),
                        age_days=34 + (idx % 60),
                        dependency_count=idx % 2,
                        support_count=1 + (idx % 2),
                        mixed_topic=True,
                        context_tags=[pack["topic"], other["topic"]],
                        group_id=group_id,
                    )
                rows.add(
                    "consolidation_gist_quality",
                    text,
                    label,
                    f"template_hardened:consolidation_gist_quality:{label}",
                    extras=extras,
                )

    if "forgetting_action_policy" in task_labels:
        policy_profiles: dict[str, dict[str, Any]] = {
            "keep": {
                "importance": 0.86,
                "confidence": 0.88,
                "access_count": 7,
                "age_days": 5,
                "dependency_count": 2,
                "support_count": 5,
                "mixed_topic": False,
                "memory_type": "constraint",
            },
            "decay": {
                "importance": 0.52,
                "confidence": 0.73,
                "access_count": 3,
                "age_days": 48,
                "dependency_count": 1,
                "support_count": 3,
                "mixed_topic": False,
                "memory_type": "episodic_event",
            },
            "silence": {
                "importance": 0.28,
                "confidence": 0.6,
                "access_count": 1,
                "age_days": 132,
                "dependency_count": 0,
                "support_count": 1,
                "mixed_topic": True,
                "memory_type": "episodic_event",
            },
            "compress": {
                "importance": 0.49,
                "confidence": 0.77,
                "access_count": 2,
                "age_days": 104,
                "dependency_count": 4,
                "support_count": 5,
                "mixed_topic": False,
                "memory_type": "semantic_fact",
            },
            "delete": {
                "importance": 0.11,
                "confidence": 0.39,
                "access_count": 0,
                "age_days": 366,
                "dependency_count": 0,
                "support_count": 1,
                "mixed_topic": True,
                "memory_type": "scratch",
            },
        }
        for label in task_labels["forgetting_action_policy"]:
            profile = policy_profiles.get(label)
            if profile is None:
                continue
            while (
                _count_single_rows_with_source_prefix(
                    rows,
                    task="forgetting_action_policy",
                    label=label,
                    prefix=f"template_hardened:forgetting_action_policy:{label}",
                )
                < hard_target
                and rows.count("forgetting_action_policy", label) < target_per_task_label
            ):
                idx = rows.count("forgetting_action_policy", label)
                pack = _topic_pack(idx)
                other = _other_topic_pack(idx)
                text = (
                    f"Retention review {idx}: {pack['fact']}. "
                    f"Reference note keeps {pack['gist'].lower()} near {other['topic']} follow-up details."
                )
                group_id = f"template_hardened:forgetting_action_policy:{idx}"
                context_tags = [pack["topic"]]
                if bool(profile["mixed_topic"]):
                    context_tags.append(other["topic"])
                rows.add(
                    "forgetting_action_policy",
                    text,
                    label,
                    f"template_hardened:forgetting_action_policy:{label}",
                    extras=_structured_row_extras(
                        topic=pack["topic"],
                        memory_type=str(profile["memory_type"]),
                        importance=float(profile["importance"]) + (0.01 * (idx % 2)),
                        confidence=float(profile["confidence"]) + (0.01 * (idx % 2)),
                        access_count=int(profile["access_count"]) + (idx % 2),
                        age_days=int(profile["age_days"]) + (idx % 18),
                        dependency_count=int(profile["dependency_count"]) + (idx % 2),
                        support_count=int(profile["support_count"]) + (idx % 2),
                        mixed_topic=bool(profile["mixed_topic"]),
                        context_tags=context_tags,
                        group_id=group_id,
                    ),
                )


def _fill_structured_router_quality_rows(
    *,
    rows: _SingleTaskStore,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    single_pools: dict[str, list[str]],
    rng: random.Random,
) -> None:
    gist_accept_clauses = (
        "the recap keeps the anchor detail and trims only obvious repetition",
        "the recap stays close to the original decision boundary",
        "the recap preserves the main action detail even with a side reference",
    )
    gist_reject_clauses = (
        "the recap keeps the topic but blurs one important boundary",
        "the recap folds in a nearby thread and softens a concrete detail",
        "the recap sounds plausible but smooths away one action cue",
    )
    gist_boundary_clauses = (
        "the recap stays readable but leaves one supporting detail open",
        "the recap keeps most of the thread while a side reference drifts a little",
        "the recap is concise although one boundary now feels softer",
    )
    if "consolidation_gist_quality" in task_labels:
        for label in ("accept", "reject"):
            while rows.count("consolidation_gist_quality", label) < target_per_task_label:
                idx = rows.count("consolidation_gist_quality", label)
                pack = _topic_pack(idx)
                other = _other_topic_pack(idx)
                seed = _seed_fragment(single_pools, rng=rng, idx=idx)
                boundary = idx % 6 == 0
                group_id = f"structured:consolidation_gist_quality:{idx}"
                accept = label == "accept"
                memory_types = (
                    ["semantic_fact", "preference", "constraint"]
                    if accept
                    else ["semantic_fact", "episodic_event", "preference"]
                )
                memory_type = _cycle_choice(memory_types, idx, offset=1 if boundary else 0)
                importance = (0.56 if accept else 0.49) + (0.03 * (idx % 5))
                confidence = (0.6 if accept else 0.55) + (0.035 * ((idx + 1) % 4))
                access_count = 1 + (idx % 4)
                age_days = (10 if accept else 18) + (idx % (54 if accept else 72))
                dependency_count = (1 if accept else 0) + (idx % 3)
                support_count = (2 if accept else 1) + (idx % 3)
                mixed_topic = bool((idx + (0 if accept else 1)) % (6 if accept else 3) == 0)
                if boundary:
                    importance += -0.05 if accept else 0.04
                    confidence += -0.04 if accept else 0.03
                    access_count = max(1, access_count + (0 if accept else 1))
                    support_count = max(1, support_count + (0 if accept else 1))
                    mixed_topic = True
                clause_pool = gist_boundary_clauses if boundary else (
                    gist_accept_clauses if accept else gist_reject_clauses
                )
                clause = _cycle_choice(clause_pool, idx, offset=1 if mixed_topic else 0)
                text = (
                    f"Cluster digest {idx}: {pack['gist']}. Review notes say {clause}. "
                    f"One supporting line keeps {seed.lower()} beside {other['topic']} follow-up context."
                )
                context_tags = [pack["topic"]]
                if mixed_topic:
                    context_tags.append(other["topic"])
                rows.add(
                    "consolidation_gist_quality",
                    text,
                    label,
                    f"structured:consolidation_gist_quality:{label}",
                    extras=_structured_row_extras(
                        topic=pack["topic"],
                        memory_type=memory_type,
                        importance=importance,
                        confidence=confidence,
                        access_count=access_count,
                        age_days=age_days,
                        dependency_count=dependency_count,
                        support_count=support_count,
                        mixed_topic=mixed_topic,
                        context_tags=context_tags,
                        group_id=group_id,
                    ),
                )

    if "forgetting_action_policy" in task_labels:
        reuse_high = (
            "keeps resurfacing in active reviews",
            "shows up whenever the same plan is revisited",
            "still shapes live decisions in the working set",
        )
        reuse_mid = (
            "returns in periodic reviews",
            "still helps in a few follow-up checks",
            "comes back when the topic cycles around again",
        )
        reuse_low = (
            "rarely changes the next answer",
            "mostly sits in the background of later notes",
            "only appears when an old thread gets reopened",
        )
        overlap_low = (
            "has only light overlap with nearby notes",
            "does not duplicate much of the surrounding context",
            "keeps a fairly distinct role in the current bundle",
        )
        overlap_mid = (
            "shares some wording with neighboring summaries",
            "partly overlaps with a nearby recap",
            "echoes one or two related notes in the same bundle",
        )
        overlap_high = (
            "appears in several near-duplicate summaries",
            "is repeated across overlapping notes",
            "keeps resurfacing through multiple paraphrased recaps",
        )
        dependency_high = (
            "still feeds several downstream decisions",
            "connects to multiple active follow-up notes",
            "anchors a few dependent reminders",
        )
        dependency_mid = (
            "still touches one or two dependent notes",
            "connects to a small chain of follow-up items",
            "keeps a modest link to later reminders",
        )
        dependency_low = (
            "hardly links to any active follow-up",
            "has almost no downstream dependency left",
            "does not drive much else in the current graph",
        )
        resolution_live = (
            "the underlying issue is still live",
            "the original need remains open",
            "the note still points at an unresolved commitment",
        )
        resolution_mixed = (
            "parts of the issue look settled while some context still matters",
            "the original need is fading but not fully gone",
            "the note is partly resolved yet still useful in narrow cases",
        )
        resolution_done = (
            "the original prompt is already resolved",
            "the temporary issue has already closed out",
            "the note no longer points at an active decision",
        )
        for label in task_labels["forgetting_action_policy"]:
            while rows.count("forgetting_action_policy", label) < target_per_task_label:
                idx = rows.count("forgetting_action_policy", label)
                pack = _topic_pack(idx)
                other = _other_topic_pack(idx)
                seed = _seed_fragment(single_pools, rng=rng, idx=idx)
                group_id = f"structured:forgetting_action_policy:{idx}"
                boundary = idx % 7 == 0
                if label == "keep":
                    memory_type = _cycle_choice(
                        ["constraint", "preference", "semantic_fact"],
                        idx,
                    )
                    importance = 0.6 + (0.045 * (idx % 4))
                    confidence = 0.62 + (0.05 * ((idx + 1) % 3))
                    access_count = 2 + (idx % 4)
                    age_days = 8 + (idx % 58)
                    dependency_count = 1 + (idx % 3)
                    support_count = 2 + (idx % 3)
                    mixed_topic = idx % 6 == 0
                    if boundary:
                        importance -= 0.08
                        confidence -= 0.05
                        access_count = max(1, access_count - 1)
                        age_days += 24
                        mixed_topic = True
                    clauses = (
                        _cycle_choice(reuse_mid if boundary else reuse_high, idx),
                        _cycle_choice(overlap_mid if boundary else overlap_low, idx),
                        _cycle_choice(dependency_mid if boundary else dependency_high, idx),
                        _cycle_choice(resolution_mixed if boundary else resolution_live, idx),
                    )
                elif label == "decay":
                    memory_type = _cycle_choice(
                        ["episodic_event", "observation", "preference"],
                        idx,
                    )
                    importance = 0.38 + (0.06 * (idx % 4))
                    confidence = 0.5 + (0.05 * ((idx + 1) % 3))
                    access_count = 2 + (idx % 3)
                    age_days = 28 + (idx % 92)
                    dependency_count = idx % 3
                    support_count = 1 + (idx % 3)
                    mixed_topic = idx % 4 == 0
                    if boundary:
                        importance += 0.06
                        confidence += 0.03
                        age_days = max(8, age_days - 18)
                    clauses = (
                        _cycle_choice(reuse_high if boundary else reuse_mid, idx),
                        _cycle_choice(overlap_mid, idx),
                        _cycle_choice(dependency_high if boundary else dependency_mid, idx),
                        _cycle_choice(resolution_live if boundary else resolution_mixed, idx),
                    )
                elif label == "silence":
                    memory_type = _cycle_choice(
                        ["episodic_event", "observation", "scratch"],
                        idx,
                    )
                    importance = 0.24 + (0.05 * (idx % 4))
                    confidence = 0.4 + (0.05 * ((idx + 2) % 3))
                    access_count = idx % 2
                    age_days = 54 + (idx % 150)
                    dependency_count = idx % 3
                    support_count = 1 + (idx % 3)
                    mixed_topic = True
                    if boundary:
                        importance += 0.07
                        confidence += 0.04
                        access_count = 1
                        age_days = max(18, age_days - 20)
                    clauses = (
                        _cycle_choice(reuse_mid if boundary else reuse_low, idx),
                        _cycle_choice(overlap_high if boundary else overlap_mid, idx),
                        _cycle_choice(dependency_mid if boundary else dependency_low, idx),
                        _cycle_choice(resolution_mixed, idx),
                    )
                elif label == "compress":
                    memory_type = _cycle_choice(
                        ["semantic_fact", "constraint", "observation"],
                        idx,
                    )
                    importance = 0.42 + (0.06 * (idx % 4))
                    confidence = 0.56 + (0.05 * ((idx + 1) % 3))
                    access_count = 1 + (idx % 3)
                    age_days = 24 + (idx % 108)
                    dependency_count = 1 + (idx % 4)
                    support_count = 3 + (idx % 3)
                    mixed_topic = idx % 5 == 0
                    if boundary:
                        importance += 0.05
                        access_count += 1
                        age_days = max(10, age_days - 16)
                    clauses = (
                        _cycle_choice(reuse_high if boundary else reuse_mid, idx),
                        _cycle_choice(overlap_mid if boundary else overlap_high, idx),
                        _cycle_choice(dependency_mid if boundary else dependency_high, idx),
                        _cycle_choice(resolution_live if boundary else resolution_mixed, idx),
                    )
                else:
                    memory_type = _cycle_choice(
                        ["scratch", "observation", "episodic_event"],
                        idx,
                    )
                    importance = 0.08 + (0.04 * (idx % 4))
                    confidence = 0.2 + (0.05 * ((idx + 1) % 3))
                    access_count = 0
                    age_days = 140 + (idx % 280)
                    dependency_count = idx % 2
                    support_count = 1 + (idx % 2)
                    mixed_topic = idx % 2 == 0
                    if boundary:
                        importance += 0.06
                        confidence += 0.05
                        age_days = max(50, age_days - 48)
                    clauses = (
                        _cycle_choice(reuse_mid if boundary else reuse_low, idx),
                        _cycle_choice(overlap_mid if boundary else overlap_low, idx),
                        _cycle_choice(dependency_mid if boundary else dependency_low, idx),
                        _cycle_choice(resolution_mixed if boundary else resolution_done, idx),
                    )
                text = (
                    f"Retention candidate {idx}: {pack['fact']}. "
                    f"Review notes say the detail {clauses[0]}, {clauses[1]}, {clauses[2]}, "
                    f"and {clauses[3]}. A nearby {other['topic']} thread still mentions {seed.lower()}."
                )
                context_tags = [pack["topic"]]
                if mixed_topic:
                    context_tags.append(other["topic"])
                rows.add(
                    "forgetting_action_policy",
                    text,
                    label,
                    f"structured:forgetting_action_policy:{label}",
                    extras=_structured_row_extras(
                        topic=pack["topic"],
                        memory_type=memory_type,
                        importance=importance,
                        confidence=confidence,
                        access_count=access_count,
                        age_days=age_days,
                        dependency_count=dependency_count,
                        support_count=support_count,
                        mixed_topic=mixed_topic,
                        context_tags=context_tags,
                        group_id=group_id,
                    ),
                )


def _fill_structured_router_ordinal_rows(
    *,
    rows: _SingleTaskStore,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    single_pools: dict[str, list[str]],
    rng: random.Random,
) -> None:
    task_profiles: dict[str, dict[str, dict[str, Any]]] = {
        "salience_bin": {
            "low": {"memory_types": ["scratch", "observation", "episodic_event"], "importance": 0.28, "confidence": 0.48, "access_count": 0, "age_days": 110, "dependency_count": 0, "support_count": 1, "mixed_topic": True},
            "medium": {"memory_types": ["episodic_event", "semantic_fact", "observation"], "importance": 0.46, "confidence": 0.6, "access_count": 2, "age_days": 42, "dependency_count": 1, "support_count": 2, "mixed_topic": False},
            "high": {"memory_types": ["constraint", "semantic_fact", "preference"], "importance": 0.62, "confidence": 0.72, "access_count": 3, "age_days": 18, "dependency_count": 2, "support_count": 3, "mixed_topic": False},
        },
        "importance_bin": {
            "low": {"memory_types": ["scratch", "observation", "episodic_event"], "importance": 0.22, "confidence": 0.42, "access_count": 0, "age_days": 130, "dependency_count": 0, "support_count": 1, "mixed_topic": True},
            "medium": {"memory_types": ["semantic_fact", "observation", "preference"], "importance": 0.5, "confidence": 0.58, "access_count": 2, "age_days": 44, "dependency_count": 1, "support_count": 2, "mixed_topic": False},
            "high": {"memory_types": ["constraint", "preference", "semantic_fact"], "importance": 0.78, "confidence": 0.72, "access_count": 3, "age_days": 18, "dependency_count": 2, "support_count": 3, "mixed_topic": False},
        },
        "confidence_bin": {
            "low": {"memory_types": ["observation", "episodic_event", "semantic_fact"], "importance": 0.42, "confidence": 0.26, "access_count": 1, "age_days": 38, "dependency_count": 0, "support_count": 1, "mixed_topic": True},
            "medium": {"memory_types": ["semantic_fact", "observation", "preference"], "importance": 0.48, "confidence": 0.55, "access_count": 2, "age_days": 28, "dependency_count": 1, "support_count": 2, "mixed_topic": False},
            "high": {"memory_types": ["knowledge", "constraint", "semantic_fact"], "importance": 0.56, "confidence": 0.78, "access_count": 3, "age_days": 18, "dependency_count": 2, "support_count": 3, "mixed_topic": False},
        },
        "decay_profile": {
            "very_fast": {"memory_types": ["scratch", "episodic_event", "observation"], "importance": 0.24, "confidence": 0.38, "access_count": 0, "age_days": 170, "dependency_count": 0, "support_count": 1, "mixed_topic": True},
            "fast": {"memory_types": ["episodic_event", "observation", "semantic_fact"], "importance": 0.34, "confidence": 0.48, "access_count": 1, "age_days": 95, "dependency_count": 0, "support_count": 1, "mixed_topic": True},
            "medium": {"memory_types": ["observation", "semantic_fact", "preference"], "importance": 0.46, "confidence": 0.58, "access_count": 2, "age_days": 56, "dependency_count": 1, "support_count": 2, "mixed_topic": False},
            "slow": {"memory_types": ["semantic_fact", "preference", "constraint"], "importance": 0.58, "confidence": 0.68, "access_count": 3, "age_days": 30, "dependency_count": 2, "support_count": 3, "mixed_topic": False},
            "very_slow": {"memory_types": ["constraint", "semantic_fact", "preference"], "importance": 0.69, "confidence": 0.76, "access_count": 4, "age_days": 14, "dependency_count": 3, "support_count": 4, "mixed_topic": False},
        },
    }
    review_clauses = (
        "still comes up when the same thread returns",
        "shares the workspace with one nearby follow-up",
        "mostly stays in the background until the topic repeats",
        "keeps a small footprint in current review passes",
    )
    support_clauses = (
        "reviewers still keep one supporting line around",
        "a nearby summary keeps a reference in circulation",
        "one follow-up note still echoes the detail",
        "the working set keeps a compact reminder attached",
    )
    for task in sorted(ORDINAL_ROUTER_TASKS & set(task_labels)):
        profile_map = task_profiles[task]
        for label in task_labels[task]:
            base = profile_map[label]
            while rows.count(task, label) < target_per_task_label:
                idx = rows.count(task, label)
                pack = _topic_pack(idx)
                other = _other_topic_pack(idx)
                seed = _seed_fragment(single_pools, rng=rng, idx=idx)
                boundary = idx % 6 == 0
                group_id = f"structured:{task}:{idx}"
                importance = float(base["importance"]) + (0.035 * (idx % 3))
                confidence = float(base["confidence"]) + (0.04 * ((idx + 1) % 3))
                access_count = int(base["access_count"]) + (idx % 3)
                age_days = int(base["age_days"]) + (idx % 30)
                dependency_count = int(base["dependency_count"]) + (idx % 2)
                support_count = int(base["support_count"]) + (idx % 2)
                mixed_topic = bool(base["mixed_topic"])
                memory_type = _cycle_choice(
                    list(base["memory_types"]),
                    idx,
                    offset=1 if boundary else 0,
                )
                if boundary:
                    if label in {"low", "very_fast"}:
                        importance += 0.08
                        confidence += 0.05
                        access_count += 1
                        age_days = max(6, age_days - 20)
                    elif label in {"high", "very_slow"}:
                        importance -= 0.07
                        confidence -= 0.05
                        access_count = max(1, access_count - 1)
                        age_days += 12
                    else:
                        importance += 0.02
                        confidence += 0.01
                        mixed_topic = True
                    mixed_topic = True if label not in {"high", "very_slow"} else mixed_topic
                context_tags = [pack["topic"]]
                if mixed_topic:
                    context_tags.append(other["topic"])
                review_clause = _cycle_choice(
                    review_clauses,
                    idx,
                    offset=(1 if mixed_topic else 0),
                )
                support_clause = _cycle_choice(
                    support_clauses,
                    idx,
                    offset=(1 if boundary else 0),
                )
                text_variants = [
                    f"Memory note {idx}: {pack['fact']}. Review notes say it {review_clause}.",
                    f"Memory note {idx}: {pack['relevant']}. The current thread still mentions {seed.lower()} while {support_clause}.",
                    f"Memory note {idx}: {pack['gist']}. A nearby {other['topic']} thread stays visible and {support_clause}.",
                ]
                if boundary:
                    text_variants.append(
                        f"Memory note {idx}: {pack['gist']}. Reviewers note a small overlap with {other['topic']} while {seed.lower()} remains attached."
                    )
                rows.add(
                    task,
                    text_variants[idx % len(text_variants)],
                    label,
                    f"structured:{task}:{label}",
                    extras=_structured_row_extras(
                        topic=pack["topic"],
                        memory_type=memory_type,
                        importance=importance,
                        confidence=confidence,
                        access_count=access_count,
                        age_days=age_days,
                        dependency_count=dependency_count,
                        support_count=support_count,
                        mixed_topic=mixed_topic,
                        context_tags=context_tags,
                        group_id=group_id,
                    ),
                )


def _apply_router_feature_enrichment(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty or "text" not in df.columns:
        return df
    enriched = df.copy()
    feature_rows = [
        derive_memory_type_feature_columns(text) for text in enriched["text"].astype(str).tolist()
    ]
    feature_df = pd.DataFrame(feature_rows)
    for column in MEMORY_TYPE_FEATURE_COLUMNS:
        enriched[column] = feature_df[column]
    return enriched


def _seed_router_regression_from_existing(
    rows: _RegressionTaskStore,
    existing_df: pd.DataFrame | None,
    *,
    target: int,
) -> None:
    if existing_df is None or existing_df.empty:
        return
    score_map = {"low": 0.2, "medium": 0.55, "high": 0.85}
    subset = existing_df[existing_df.get("task", "").astype(str) == "importance_bin"]
    for item in subset.itertuples(index=False):
        label = str(getattr(item, "label", "")).strip().lower()
        score = score_map.get(label)
        if score is None:
            continue
        rows.add(
            "write_importance_regression",
            getattr(item, "text", ""),
            score,
            f"derived:{getattr(item, 'source', 'prepared:router')}",
            language=str(getattr(item, "language", "en")),
            extras={
                "memory_type": "semantic_fact" if label == "high" else "episodic_event",
                "namespace": "general",
                "context_tags": ["general"],
                "importance": score,
                "confidence": 0.8 if label == "high" else 0.6,
                "access_count": 4 if label == "high" else 1,
                "age_days": 3 if label == "high" else 45,
                "dependency_count": 2 if label == "high" else 0,
            },
        )
        if rows.count("write_importance_regression") >= target:
            break


def _fill_router_tasks_without_llm(
    *,
    rows: _SingleTaskStore,
    regression_rows: _RegressionTaskStore,
    task_labels: dict[str, list[str]],
    regression_tasks: set[str],
    target_per_task_label: int,
    single_pools: dict[str, list[str]],
    rng: random.Random,
) -> None:
    _fill_structured_router_quality_rows(
        rows=rows,
        task_labels=task_labels,
        target_per_task_label=target_per_task_label,
        single_pools=single_pools,
        rng=rng,
    )
    _fill_structured_router_ordinal_rows(
        rows=rows,
        task_labels=task_labels,
        target_per_task_label=target_per_task_label,
        single_pools=single_pools,
        rng=rng,
    )

    if "write_importance_regression" in regression_tasks:
        while regression_rows.count("write_importance_regression") < target_per_task_label:
            idx = regression_rows.count("write_importance_regression")
            pack = _topic_pack(idx)
            band = idx % 5
            if band == 0:
                score = 0.94
                memory_type = "constraint"
                text = f"High-priority policy {idx}: {pack['relevant']} and this should always guide decisions."
                importance = 0.95
                access_count = 8
                age_days = 2
                dependency_count = 3
            elif band == 1:
                score = 0.76
                memory_type = "preference"
                text = f"Stable preference {idx}: {pack['relevant']} for future recommendations."
                importance = 0.78
                access_count = 5
                age_days = 7
                dependency_count = 1
            elif band == 2:
                score = 0.52
                memory_type = "semantic_fact"
                text = f"Useful fact {idx}: {pack['fact']}."
                importance = 0.55
                access_count = 2
                age_days = 21
                dependency_count = 1
            elif band == 3:
                score = 0.27
                memory_type = "episodic_event"
                text = f"Minor event {idx}: the user briefly mentioned {pack['topic']} logistics."
                importance = 0.3
                access_count = 1
                age_days = 75
                dependency_count = 0
            else:
                score = 0.08
                memory_type = "scratch"
                text = f"Ephemeral scratch note {idx}: temporary reminder about {pack['topic']}."
                importance = 0.1
                access_count = 0
                age_days = 180
                dependency_count = 0
            regression_rows.add(
                "write_importance_regression",
                text,
                score,
                "template:write_importance_regression",
                extras=_structured_row_extras(
                    topic=pack["topic"],
                    memory_type=memory_type,
                    importance=importance,
                    confidence=0.92 if score > 0.7 else 0.62 if score > 0.3 else 0.38,
                    access_count=access_count,
                    age_days=age_days,
                    dependency_count=dependency_count,
                ),
            )


def _add_existing_pair_task_mappings(
    rows: _PairTaskStore, existing_df: pd.DataFrame | None
) -> None:
    if existing_df is None or existing_df.empty:
        return
    for idx, item in enumerate(existing_df.itertuples(index=False)):
        task = str(getattr(item, "task", "")).strip()
        label = str(getattr(item, "label", "")).strip()
        text_a = getattr(item, "text_a", "")
        text_b = getattr(item, "text_b", "")
        source = str(getattr(item, "source", "prepared:pair"))
        if task == "constraint_rerank":
            extras = {"group_id": f"prepared:rerank:{idx}"}
            rows.add(
                "retrieval_constraint_relevance_pair",
                text_a,
                text_b,
                label,
                f"derived:{source}",
                extras=extras,
            )
            rows.add(
                "memory_rerank_pair",
                text_a,
                text_b,
                label,
                f"derived:{source}",
                extras=extras,
            )
            rows.add(
                "reconsolidation_candidate_pair",
                text_a,
                text_b,
                label,
                f"derived:{source}",
                extras=extras,
            )
        elif (task == "conflict_detection" and label == "conflict") or (task == "supersession" and label == "supersedes"):
            rows.add(
                "novelty_pair",
                text_a,
                text_b,
                "changed",
                f"derived:{source}",
                extras={"group_id": _stable_group_id("derived:novelty_pair", source, text_a, text_b)},
            )


def _derive_schema_from_pair_rows(rows: _PairTaskStore) -> None:
    for item in list(rows.rows):
        task = str(item.get("task", "")).strip()
        if task != "scope_match":
            continue
        label = str(item.get("label", "")).strip()
        mapped = "match" if label == "match" else "no_match"
        source = str(item.get("source", "derived:scope_match"))
        text_a = item.get("text_a", "")
        text_b = item.get("text_b", "")
        group_id = str(item.get("group_id", "")).strip() or _stable_group_id(
            "derived:schema_match_pair", source, text_a, text_b
        )
        rows.add(
            "schema_match_pair",
            text_a,
            text_b,
            mapped,
            f"derived:{source}",
            extras={"group_id": group_id},
        )


def _fill_template_relevance_task(
    rows: _PairTaskStore, *, task: str, target_per_task_label: int
) -> None:
    while rows.count(task, "relevant") < target_per_task_label:
        idx = rows.count(task, "relevant")
        pack = _topic_pack(idx)
        query = f"{pack['query']} for scenario {idx}?"
        memory = f"{pack['relevant']}; case {idx}."
        rows.add(
            task,
            query,
            memory,
            "relevant",
            f"template:{task}:relevant",
            extras={"group_id": f"template:{task}:{idx}"},
        )
    while rows.count(task, "not_relevant") < target_per_task_label:
        idx = rows.count(task, "not_relevant")
        pack = _topic_pack(idx)
        other = _other_topic_pack(idx)
        query = f"{pack['query']} for scenario {idx}?"
        memory = f"{other['irrelevant']}; distractor {idx}."
        rows.add(
            task,
            query,
            memory,
            "not_relevant",
            f"template:{task}:not_relevant",
            extras={"group_id": f"template:{task}:{idx}"},
        )


def _fill_template_schema_rows(rows: _PairTaskStore, *, target_per_task_label: int) -> None:
    template_cap = max(1, round(target_per_task_label * _TASK_TEMPLATE_CAPS["schema_match_pair"]))
    while (
        rows.count("schema_match_pair", "match") < target_per_task_label
        and sum(
            1
            for row in rows.rows
            if str(row.get("task", "")) == "schema_match_pair"
            and str(row.get("label", "")) == "match"
            and _is_template_source(row.get("source"))
        )
        < template_cap
    ):
        idx = rows.count("schema_match_pair", "match")
        pack = _topic_pack(idx)
        rows.add(
            "schema_match_pair",
            f"{pack['gist']} Summary {idx}.",
            f"{pack['fact']} Fact {idx}.",
            "match",
            "template:schema_match_pair:match",
            extras={"group_id": f"template:schema_match_pair:{idx}"},
        )
    while (
        rows.count("schema_match_pair", "no_match") < target_per_task_label
        and sum(
            1
            for row in rows.rows
            if str(row.get("task", "")) == "schema_match_pair"
            and str(row.get("label", "")) == "no_match"
            and _is_template_source(row.get("source"))
        )
        < template_cap
    ):
        idx = rows.count("schema_match_pair", "no_match")
        pack = _topic_pack(idx)
        other = _other_topic_pack(idx)
        rows.add(
            "schema_match_pair",
            f"{pack['gist']} Summary {idx}.",
            f"{other['fact']} Fact {idx}.",
            "no_match",
            "template:schema_match_pair:no_match",
            extras={"group_id": f"template:schema_match_pair:{idx}"},
        )


def _fill_template_novelty_rows(rows: _PairTaskStore, *, target_per_task_label: int) -> None:
    while rows.count("novelty_pair", "duplicate") < target_per_task_label:
        idx = rows.count("novelty_pair", "duplicate")
        pack = _topic_pack(idx)
        rows.add(
            "novelty_pair",
            f"{pack['relevant']} in profile {idx}.",
            f"In profile {idx}, {pack['relevant'].lower()}.",
            "duplicate",
            "template:novelty_pair:duplicate",
            extras={"group_id": f"template:novelty_pair:{idx}"},
        )
    while rows.count("novelty_pair", "novel") < target_per_task_label:
        idx = rows.count("novelty_pair", "novel")
        pack = _topic_pack(idx)
        other = _other_topic_pack(idx)
        rows.add(
            "novelty_pair",
            f"{pack['relevant']} in profile {idx}.",
            f"New detail {idx}: {other['relevant'].lower()}.",
            "novel",
            "template:novelty_pair:novel",
            extras={"group_id": f"template:novelty_pair:{idx}"},
        )
    while rows.count("novelty_pair", "changed") < target_per_task_label:
        idx = rows.count("novelty_pair", "changed")
        pack = _topic_pack(idx)
        if idx % 2 == 0:
            text_a = f"{pack['relevant']} in preference record {idx}."
            text_b = (
                f"Preference record {idx} now reverses that detail: "
                f"{pack['irrelevant'].lower()}."
            )
        else:
            text_a = f"{pack['relevant']} in profile {idx}."
            text_b = f"Updated profile {idx}: {pack['fact'].lower()} after a recent change."
        rows.add(
            "novelty_pair",
            text_a,
            text_b,
            "changed",
            "template:novelty_pair:changed",
            extras={"group_id": f"template:novelty_pair:{idx}"},
        )


def _label_guidance(task: str, label: str) -> str:
    guidance = {
        "memory_type": {
            "episodic_event": "A specific personal event anchored in time/place.",
            "semantic_fact": "A factual statement about the world or domain knowledge.",
            "preference": "A stable like/dislike or personal choice.",
            "constraint": "A hard/soft rule, policy, must/never condition.",
            "procedure": "Step-by-step instructions or process knowledge.",
            "hypothesis": "A tentative explanation or guess.",
            "task_state": "Current progress/status of a task.",
            "conversation": "General conversational turn with little durable content.",
            "message": "Message-like communication content.",
            "tool_result": "Output or observation from a tool/API/query.",
            "reasoning_step": "Intermediate reasoning or derivation step.",
            "scratch": "Temporary short-term note.",
            "knowledge": "Domain information suitable for long-term memory.",
            "observation": "Observed condition or signal.",
            "plan": "Future actions or strategy.",
        },
        "query_intent": {
            "constraint_check": "Asks if an action is allowed or policy-compliant.",
            "tool_query": "Requests tool/API/command execution info.",
            "planning": "Requests next steps, plans, or strategy.",
            "factual": "Asks for facts or objective information.",
            "conversation": "General social chat not mainly factual/tooling.",
        },
        "query_domain": {
            "general": "General domain with no specific topical anchor.",
            "food": "Food, meals, diet, restaurant, nutrition domain.",
            "travel": "Travel, transport, trips, flights, hotels domain.",
            "finance": "Money, banking, budgeting, investment, payment domain.",
            "health": "Health, wellness, medication, symptoms, exercise domain.",
            "work": "Professional work, project, task, office domain.",
            "tech": "Technology, software, code, device, systems domain.",
            "social": "Family, friends, relationships, social events domain.",
        },
        "constraint_dimension": {
            "policy": "Rule-like must/should/forbidden framing.",
            "goal": "Objective to achieve.",
            "value": "Principle or preference criterion.",
            "causal": "Cause/effect dependency.",
            "state": "Condition/state-based constraint.",
            "other": "Constraint that does not fit other dimensions.",
        },
        "context_tag": {
            "general": "General context.",
            "food": "Meals, diet, restaurants, cooking.",
            "travel": "Trips, flights, hotels, commute.",
            "finance": "Money, banking, payments, budgeting.",
            "health": "Medical, wellbeing, exercise, sleep.",
            "work": "Professional tasks and projects.",
            "tech": "Software, code, tools, systems.",
            "social": "Family, friends, social events.",
        },
        "salience_bin": {
            "low": "Minor detail.",
            "medium": "Useful but not critical detail.",
            "high": "Critical detail likely needed later.",
        },
        "importance_bin": {
            "low": "Low downstream impact.",
            "medium": "Moderate downstream impact.",
            "high": "High downstream impact.",
        },
        "confidence_bin": {
            "low": "Uncertain or speculative content.",
            "medium": "Reasonably likely but not certain.",
            "high": "Clear and reliable statement.",
        },
        "decay_profile": {
            "very_fast": "Very short-lived memory.",
            "fast": "Short-lived memory.",
            "medium": "Moderately persistent memory.",
            "slow": "Long-lived memory.",
            "very_slow": "Very persistent memory.",
        },
        "constraint_type": {
            "policy": "Rule, must, should, forbidden behavior.",
            "goal": "Target outcome or objective.",
            "value": "Preference or principle.",
            "causal": "Cause/effect dependency.",
            "state": "State/condition requirement.",
            "preference": "Personal preference statement.",
            "constraint_other": "Constraint-like but not in other categories.",
            "none": "No actionable constraint.",
        },
        "constraint_scope": {
            "none": "No explicit scope.",
            "general": "General scope.",
            "food": "Food/diet scope.",
            "travel": "Travel scope.",
            "finance": "Finance scope.",
            "health": "Health scope.",
            "work": "Work scope.",
            "tech": "Tech scope.",
            "social": "Social scope.",
        },
        "constraint_stability": {
            "stable": "Long-lived rule/principle/value with low expected change.",
            "semi_stable": "Medium-term preference/goal likely to change occasionally.",
            "volatile": "Short-lived state/situation likely to change quickly.",
        },
        "fact_type": {
            "none": "No clear durable fact.",
            "other_fact": "Fact not in named categories.",
            "preference": "Fact about likes/dislikes.",
            "identity": "Personal identity fact.",
            "location": "Location-related fact.",
            "occupation": "Job/profession fact.",
        },
        "pii_presence": {
            "pii": "Contains sensitive personal identifiers.",
            "no_pii": "Contains no sensitive personal identifiers.",
        },
        "conflict_detection": {
            "conflict": "Two texts contradict or are incompatible.",
            "no_conflict": "Two texts are compatible or non-contradictory.",
        },
        "constraint_rerank": {
            "relevant": "Second text is relevant to first text.",
            "not_relevant": "Second text is not relevant to first text.",
        },
        "scope_match": {
            "match": "Two texts share the same topic/domain scope.",
            "no_match": "Two texts are about different domains/topics.",
        },
        "supersession": {
            "supersedes": "Second text updates or replaces the first statement.",
            "no_supersedes": "Second text does not replace the first statement.",
        },
        "novelty_pair": {
            "duplicate": "Two texts express the same fact with trivial rephrasing.",
            "novel": "Second text adds genuinely new information.",
            "changed": "Second text changes or updates the first, including contradictions.",
        },
        "schema_match_pair": {
            "match": "Gist and existing fact key are semantically compatible.",
            "no_match": "Gist and existing fact key are about different topics.",
        },
        "retrieval_constraint_relevance_pair": {
            "relevant": "Query is relevant to the constraint text.",
            "not_relevant": "Query is not relevant to the constraint text.",
        },
        "reconsolidation_candidate_pair": {
            "relevant": "Memory is a relevant reconsolidation candidate for the turn.",
            "not_relevant": "Memory is not relevant for reconsolidation.",
        },
        "memory_rerank_pair": {
            "relevant": "Memory is relevant to the query for reranking.",
            "not_relevant": "Memory is not relevant to the query.",
        },
        "consolidation_gist_quality": {
            "accept": "Gist accurately summarizes the episode cluster.",
            "reject": "Gist is too generic, inaccurate, or low quality.",
        },
        "forgetting_action_policy": {
            "keep": "Memory should be kept as-is.",
            "decay": "Memory importance should be reduced.",
            "silence": "Memory should be silenced from retrieval.",
            "compress": "Memory should be compressed/summarized.",
            "delete": "Memory should be permanently removed.",
        },
    }
    return guidance.get(task, {}).get(label, "")


def _finish_reason_delta(before: dict[str, Any], after: dict[str, Any], reason: str) -> int:
    before_counts = before.get("finish_reason_counts", {})
    after_counts = after.get("finish_reason_counts", {})
    before_value = int(before_counts.get(reason, 0)) if isinstance(before_counts, dict) else 0
    after_value = int(after_counts.get(reason, 0)) if isinstance(after_counts, dict) else 0
    return max(0, after_value - before_value)


def _llm_round_strategy(
    *,
    label_batch_size: int,
    max_batch_size: int,
    round_generated: int,
    round_accepted: int,
    round_errors: int,
    attempts_without_progress: int,
    parse_fail_delta: int,
    finish_length_delta: int,
    finish_stop_delta: int,
    use_multilingual: bool,
) -> tuple[int, bool, bool, str | None]:
    new_batch_size = max(1, int(label_batch_size))
    new_use_multilingual = bool(use_multilingual)
    adjust_reasons: list[str] = []

    if round_generated == 0 and round_errors == 0 and new_batch_size > 1:
        new_batch_size = max(1, new_batch_size // 2)
        adjust_reasons.append("no parseable outputs")
    elif finish_length_delta > max(1, finish_stop_delta) and new_batch_size > 1:
        new_batch_size = max(1, new_batch_size // 2)
        adjust_reasons.append(
            f"truncated outputs (length={finish_length_delta}, stop={finish_stop_delta})"
        )
    elif parse_fail_delta > 0 and round_accepted == 0 and new_batch_size > 1:
        new_batch_size = max(1, new_batch_size // 2)
        adjust_reasons.append(f"parse failures ({parse_fail_delta})")
    elif (
        round_accepted > 0
        and new_batch_size < max_batch_size
        and parse_fail_delta == 0
        and finish_length_delta == 0
    ):
        new_batch_size = min(max_batch_size, new_batch_size * 2)
        if new_batch_size != label_batch_size:
            adjust_reasons.append("restoring batch size after clean round")

    if new_use_multilingual and round_accepted == 0 and attempts_without_progress >= 3:
        new_use_multilingual = False
        if new_batch_size > 4:
            new_batch_size = 4
        adjust_reasons.append("disabled multilingual after repeated stalled rounds")

    abort_stalled_label = (
        attempts_without_progress >= 8
        and round_accepted == 0
        and new_batch_size == 1
        and not new_use_multilingual
    )
    reason = "; ".join(adjust_reasons) if adjust_reasons else None
    return new_batch_size, new_use_multilingual, abort_stalled_label, reason


def _fill_single_task_with_llm(
    *,
    rows: _SingleTaskStore,
    llm: _LLMGenerator,
    family: str,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    single_pools: dict[str, list[str]],
    rng: random.Random,
    max_attempts_per_label: int,
    use_multilingual: bool = True,
) -> None:
    total_missing = 0
    for task, labels in task_labels.items():
        for label in labels:
            total_missing += max(0, target_per_task_label - rows.count(task, label))

    pbar = _progress(total=total_missing, desc=f"LLM synth [{family}]", unit="sample")
    try:
        with ThreadPoolExecutor(max_workers=llm.concurrency) as executor:
            for task, labels in task_labels.items():
                for label in labels:
                    missing = max(0, target_per_task_label - rows.count(task, label))
                    if missing <= 0:
                        continue

                    _warn(
                        f"LLM fill start [{family}] {task}::{label} "
                        f"missing={missing} batch_size={llm.batch_size} "
                        f"concurrency={llm.concurrency}."
                    )
                    attempts = 0
                    rounds = 0
                    label_batch_size = llm.batch_size
                    label_use_multilingual = use_multilingual
                    while missing > 0 and attempts < max_attempts_per_label:
                        rounds += 1
                        stats_before = llm.stats_snapshot()
                        jobs = min(
                            llm.concurrency,
                            max(1, (missing + label_batch_size - 1) // label_batch_size),
                        )
                        futures = []
                        for _ in range(jobs):
                            seed_text = _pick_seed_text(single_pools, family, rng)
                            guidance = _label_guidance(task, label)
                            batch = min(label_batch_size, missing)
                            prompt_seed = (
                                f"{seed_text}\n\nLabel guidance: {guidance}"
                                if guidance
                                else seed_text
                            )
                            lang = (
                                _multilingual_prompts.pick_language(rng)
                                if label_use_multilingual
                                else None
                            )

                            def _run_single(
                                _task=task,
                                _label=label,
                                _seed=prompt_seed,
                                _n=batch,
                                _lang=lang,
                            ):
                                out = llm.generate_single(
                                    task=_task,
                                    label=_label,
                                    seed_text=_seed,
                                    n=_n,
                                    language=_lang,
                                )
                                return (out, _lang.code if _lang else "en")

                            futures.append(executor.submit(_run_single))

                        round_generated = 0
                        round_accepted = 0
                        round_errors = 0
                        for future in as_completed(futures):
                            try:
                                generated, lang_code = future.result()
                            except Exception as exc:
                                round_errors += 1
                                if llm.log_request_failures:
                                    _warn(f"LLM request error for {task}::{label}: {exc}")
                                continue

                            round_generated += len(generated)
                            accepted = 0
                            for text in generated:
                                if rows.count(task, label) >= target_per_task_label:
                                    break
                                if rows.add(
                                    task,
                                    text,
                                    label,
                                    f"llm:{task}:{label}",
                                    language=lang_code,
                                    extras={
                                        "group_id": _stable_group_id("llm", task, label, text),
                                    },
                                ):
                                    accepted += 1
                            if accepted > 0:
                                round_accepted += accepted
                                pbar.update(accepted)

                        llm.record_batch_result(generated=round_generated, accepted=round_accepted)
                        stats_after = llm.stats_snapshot()
                        missing = max(0, target_per_task_label - rows.count(task, label))
                        if round_accepted <= 0:
                            attempts += 1
                        else:
                            attempts = 0

                        (
                            label_batch_size,
                            label_use_multilingual,
                            abort_stalled_label,
                            adjust_reason,
                        ) = _llm_round_strategy(
                            label_batch_size=label_batch_size,
                            max_batch_size=llm.batch_size,
                            round_generated=round_generated,
                            round_accepted=round_accepted,
                            round_errors=round_errors,
                            attempts_without_progress=attempts,
                            parse_fail_delta=int(stats_after["parse_fail_total"]) - int(stats_before["parse_fail_total"]),
                            finish_length_delta=_finish_reason_delta(stats_before, stats_after, "length"),
                            finish_stop_delta=_finish_reason_delta(stats_before, stats_after, "stop"),
                            use_multilingual=label_use_multilingual,
                        )
                        if adjust_reason:
                            _warn(
                                f"LLM strategy update for {task}::{label}: "
                                f"batch_size={label_batch_size} multilingual={label_use_multilingual}. "
                                f"reason={adjust_reason}"
                            )

                        should_log_round = (
                            rounds == 1
                            or missing == 0
                            or round_accepted > 0
                            or attempts == 1
                            or attempts % llm.log_zero_progress_every == 0
                        )
                        if should_log_round:
                            _warn(
                                f"LLM fill [{family}] {task}::{label} "
                                f"round={rounds} jobs={jobs} generated={round_generated} "
                                f"accepted={round_accepted} errors={round_errors} "
                                f"missing={missing} attempts_without_progress={attempts} "
                                f"batch_size={label_batch_size} multilingual={label_use_multilingual}."
                            )
                        llm.maybe_report(context=f"{family}:{task}::{label}")
                        if abort_stalled_label:
                            _warn(
                                f"LLM abandoning stalled label fill for {task}::{label}. "
                                f"missing={missing}, current={rows.count(task, label)}."
                            )
                            break

                    llm.maybe_report(context=f"{family}:{task}::{label}", force=True)
                    if missing > 0:
                        _warn(
                            f"LLM underfilled {task}::{label}. "
                            f"missing={missing}, current={rows.count(task, label)}."
                        )
                    else:
                        _warn(
                            f"LLM fill complete [{family}] {task}::{label}. "
                            f"current={rows.count(task, label)}."
                        )
    finally:
        pbar.close()


def _fill_pair_task_with_llm(
    *,
    rows: _PairTaskStore,
    llm: _LLMGenerator,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    pair_pool: list[tuple[str, str]],
    single_pools: dict[str, list[str]],
    rng: random.Random,
    max_attempts_per_label: int,
    use_multilingual: bool = True,
) -> None:
    total_missing = 0
    for task, labels in task_labels.items():
        for label in labels:
            total_missing += max(0, target_per_task_label - rows.count(task, label))

    pbar = _progress(total=total_missing, desc="LLM synth [pair]", unit="sample")
    try:
        with ThreadPoolExecutor(max_workers=llm.concurrency) as executor:
            for task, labels in task_labels.items():
                for label in labels:
                    missing = max(0, target_per_task_label - rows.count(task, label))
                    if missing <= 0:
                        continue

                    _warn(
                        f"LLM fill start [pair] {task}::{label} "
                        f"missing={missing} batch_size={llm.batch_size} "
                        f"concurrency={llm.concurrency}."
                    )
                    attempts = 0
                    rounds = 0
                    label_batch_size = llm.batch_size
                    label_use_multilingual = use_multilingual
                    while missing > 0 and attempts < max_attempts_per_label:
                        rounds += 1
                        stats_before = llm.stats_snapshot()
                        jobs = min(
                            llm.concurrency,
                            max(1, (missing + label_batch_size - 1) // label_batch_size),
                        )
                        futures = []
                        for _ in range(jobs):
                            seed_a, seed_b = _pick_seed_pair(pair_pool, single_pools, rng)
                            guidance = _label_guidance(task, label)
                            batch = min(label_batch_size, missing)
                            if guidance:
                                seed_a = f"{seed_a}\n\nLabel guidance: {guidance}"
                            lang = (
                                _multilingual_prompts.pick_language(rng)
                                if label_use_multilingual
                                else None
                            )

                            def _run_pair(
                                _task=task,
                                _label=label,
                                _seed_a=seed_a,
                                _seed_b=seed_b,
                                _n=batch,
                                _lang=lang,
                            ):
                                out = llm.generate_pair(
                                    task=_task,
                                    label=_label,
                                    seed_a=_seed_a,
                                    seed_b=_seed_b,
                                    n=_n,
                                    language=_lang,
                                )
                                return (out, _lang.code if _lang else "en")

                            futures.append(executor.submit(_run_pair))

                        round_generated = 0
                        round_accepted = 0
                        round_errors = 0
                        for future in as_completed(futures):
                            try:
                                generated, lang_code = future.result()
                            except Exception as exc:
                                round_errors += 1
                                if llm.log_request_failures:
                                    _warn(f"LLM request error for {task}::{label}: {exc}")
                                continue

                            round_generated += len(generated)
                            accepted = 0
                            for text_a, text_b in generated:
                                if rows.count(task, label) >= target_per_task_label:
                                    break
                                if rows.add(
                                    task,
                                    text_a,
                                    text_b,
                                    label,
                                    f"llm:{task}:{label}",
                                    language=lang_code,
                                    extras={
                                        "group_id": _stable_group_id(
                                            "llm",
                                            task,
                                            label,
                                            text_a,
                                            text_b,
                                        )
                                    },
                                ):
                                    accepted += 1
                            if accepted > 0:
                                round_accepted += accepted
                                pbar.update(accepted)

                        llm.record_batch_result(generated=round_generated, accepted=round_accepted)
                        stats_after = llm.stats_snapshot()
                        missing = max(0, target_per_task_label - rows.count(task, label))
                        if round_accepted <= 0:
                            attempts += 1
                        else:
                            attempts = 0

                        (
                            label_batch_size,
                            label_use_multilingual,
                            abort_stalled_label,
                            adjust_reason,
                        ) = _llm_round_strategy(
                            label_batch_size=label_batch_size,
                            max_batch_size=llm.batch_size,
                            round_generated=round_generated,
                            round_accepted=round_accepted,
                            round_errors=round_errors,
                            attempts_without_progress=attempts,
                            parse_fail_delta=int(stats_after["parse_fail_total"]) - int(stats_before["parse_fail_total"]),
                            finish_length_delta=_finish_reason_delta(stats_before, stats_after, "length"),
                            finish_stop_delta=_finish_reason_delta(stats_before, stats_after, "stop"),
                            use_multilingual=label_use_multilingual,
                        )
                        if adjust_reason:
                            _warn(
                                f"LLM strategy update for {task}::{label}: "
                                f"batch_size={label_batch_size} multilingual={label_use_multilingual}. "
                                f"reason={adjust_reason}"
                            )

                        should_log_round = (
                            rounds == 1
                            or missing == 0
                            or round_accepted > 0
                            or attempts == 1
                            or attempts % llm.log_zero_progress_every == 0
                        )
                        if should_log_round:
                            _warn(
                                f"LLM fill [pair] {task}::{label} "
                                f"round={rounds} jobs={jobs} generated={round_generated} "
                                f"accepted={round_accepted} errors={round_errors} "
                                f"missing={missing} attempts_without_progress={attempts} "
                                f"batch_size={label_batch_size} multilingual={label_use_multilingual}."
                            )
                        llm.maybe_report(context=f"pair:{task}::{label}")
                        if abort_stalled_label:
                            _warn(
                                f"LLM abandoning stalled label fill for {task}::{label}. "
                                f"missing={missing}, current={rows.count(task, label)}."
                            )
                            break

                    llm.maybe_report(context=f"pair:{task}::{label}", force=True)
                    if missing > 0:
                        _warn(
                            f"LLM underfilled {task}::{label}. "
                            f"missing={missing}, current={rows.count(task, label)}."
                        )
                    else:
                        _warn(
                            f"LLM fill complete [pair] {task}::{label}. "
                            f"current={rows.count(task, label)}."
                        )
    finally:
        pbar.close()


def _build_router_rows(
    *,
    local_rows: list[dict],
    registry: _HFRegistry,
    prepare_cfg: dict,
    synthetic_cfg: dict,
    single_pools: dict[str, list[str]],
    llm: _LLMGenerator | None,
    task_labels: dict[str, list[str]],
    regression_tasks: set[str],
    existing_df: pd.DataFrame | None = None,
    use_multilingual: bool = True,
) -> list[dict]:
    cap = max(int(prepare_cfg["max_per_task_label"]), int(prepare_cfg["target_per_task_label"]))
    rows = _SingleTaskStore(cap)
    regression_rows = _RegressionTaskStore(cap)
    rng = random.Random(int(prepare_cfg["seed"]))
    if existing_df is not None and not existing_df.empty:
        _seed_single_store_from_df(rows, existing_df)
        _seed_regression_store_from_df(regression_rows, existing_df)

    target = int(prepare_cfg["target_per_task_label"])
    missing_total = sum(
        max(0, target - rows.count(task, label))
        for task, labels in task_labels.items()
        for label in labels
    )
    missing_regression_total = sum(
        max(0, target - regression_rows.count(task_name)) for task_name in regression_tasks
    )
    if missing_total <= 0 and missing_regression_total <= 0:
        return [*rows.rows, *regression_rows.rows]

    missing_memory_type = any(
        rows.count("memory_type", label) < target for label in ROUTER_TASK_LABELS["memory_type"]
    )
    missing_query_domain = any(
        rows.count("query_domain", label) < target for label in ROUTER_TASK_LABELS["query_domain"]
    )
    if missing_memory_type:
        _add_existing_router_memory_rows(rows, local_rows)
    if missing_query_domain:
        _add_existing_router_domain_rows(rows, registry)
    if "write_importance_regression" in regression_tasks:
        _seed_router_regression_from_existing(regression_rows, existing_df, target=target)

    _inject_hardened_router_rows(
        rows=rows,
        task_labels=task_labels,
        target_per_task_label=target,
        single_pools=single_pools,
        rng=rng,
    )

    _fill_router_tasks_without_llm(
        rows=rows,
        regression_rows=regression_rows,
        task_labels=task_labels,
        regression_tasks=regression_tasks,
        target_per_task_label=target,
        single_pools=single_pools,
        rng=rng,
    )

    llm_task_labels = {
        task: labels for task, labels in task_labels.items() if task not in ORDINAL_ROUTER_TASKS
    }
    if llm is not None and llm_task_labels:
        _fill_single_task_with_llm(
            rows=rows,
            llm=llm,
            family="router",
            task_labels=llm_task_labels,
            target_per_task_label=target,
            single_pools=single_pools,
            rng=rng,
            max_attempts_per_label=int(synthetic_cfg.get("max_attempts_per_label", 80)),
            use_multilingual=use_multilingual,
        )
    return [*rows.rows, *regression_rows.rows]


def _build_extractor_rows(
    *,
    registry: _HFRegistry,
    prepare_cfg: dict,
    synthetic_cfg: dict,
    single_pools: dict[str, list[str]],
    llm: _LLMGenerator | None,
    existing_df: pd.DataFrame | None = None,
    use_multilingual: bool = True,
) -> list[dict]:
    cap = max(int(prepare_cfg["max_per_task_label"]), int(prepare_cfg["target_per_task_label"]))
    rows = _SingleTaskStore(cap)
    rng = random.Random(int(prepare_cfg["seed"]) + 1)
    if existing_df is not None and not existing_df.empty:
        _seed_single_store_from_df(rows, existing_df)

    target = int(prepare_cfg["target_per_task_label"])
    missing_total = sum(
        max(0, target - rows.count(task, label))
        for task, labels in EXTRACTOR_TASK_LABELS.items()
        for label in labels
    )
    if missing_total <= 0:
        return rows.rows

    missing_pii = any(
        rows.count("pii_presence", label) < target
        for label in EXTRACTOR_TASK_LABELS["pii_presence"]
    )
    if missing_pii:
        _add_existing_extractor_pii_rows(rows, registry)

    if llm is not None:
        _fill_single_task_with_llm(
            rows=rows,
            llm=llm,
            family="extractor",
            task_labels=EXTRACTOR_TASK_LABELS,
            target_per_task_label=target,
            single_pools=single_pools,
            rng=rng,
            max_attempts_per_label=int(synthetic_cfg.get("max_attempts_per_label", 80)),
            use_multilingual=use_multilingual,
        )
    return rows.rows


def _build_pair_rows(
    *,
    registry: _HFRegistry,
    prepare_cfg: dict,
    synthetic_cfg: dict,
    single_pools: dict[str, list[str]],
    pair_pool: list[tuple[str, str]],
    llm: _LLMGenerator | None,
    task_labels: dict[str, list[str]],
    existing_df: pd.DataFrame | None = None,
    use_multilingual: bool = True,
) -> list[dict]:
    cap = max(int(prepare_cfg["max_per_task_label"]), int(prepare_cfg["target_per_task_label"]))
    rows = _PairTaskStore(cap)
    rng = random.Random(int(prepare_cfg["seed"]) + 2)
    if existing_df is not None and not existing_df.empty:
        _seed_pair_store_from_df(rows, existing_df)

    target = int(prepare_cfg["target_per_task_label"])
    missing_total = sum(
        max(0, target - rows.count(task, label))
        for task, labels in task_labels.items()
        for label in labels
    )
    if missing_total <= 0:
        return rows.rows

    _add_existing_pair_task_mappings(rows, existing_df)
    _add_existing_pair_rows(rows, registry)
    _add_paws_novelty_rows(rows, registry)
    _add_glue_novelty_rows(rows, registry)
    _add_fever_schema_match_rows(rows, registry, target_per_task_label=target)
    _derive_schema_from_pair_rows(rows)
    _fill_pair_tasks_without_llm(
        rows,
        task_labels=task_labels,
        target_per_task_label=target,
        rng=rng,
    )
    llm_task_labels = {
        task: labels for task, labels in task_labels.items() if task != "schema_match_pair"
    }
    if llm is not None and llm_task_labels:
        _fill_pair_task_with_llm(
            rows=rows,
            llm=llm,
            task_labels=llm_task_labels,
            target_per_task_label=target,
            pair_pool=pair_pool,
            single_pools=single_pools,
            rng=rng,
            max_attempts_per_label=int(synthetic_cfg.get("max_attempts_per_label", 80)),
            use_multilingual=use_multilingual,
        )
    return rows.rows


def _split_counts(n: int, ratios: dict[str, float]) -> tuple[int, int, int]:
    if n <= 0:
        return 0, 0, 0
    if n == 1:
        return 1, 0, 0
    if n == 2:
        return 1, 1, 0

    n_train = int(n * ratios["train"])
    n_test = int(n * ratios["test"])
    n_eval = n - n_train - n_test

    counts = [n_train, n_test, n_eval]
    minimums = [1, 1, 1]
    for i in range(3):
        while counts[i] < minimums[i]:
            donor = max(range(3), key=lambda idx: counts[idx])
            if counts[donor] <= minimums[donor]:
                break
            counts[donor] -= 1
            counts[i] += 1
    return counts[0], counts[1], counts[2]


def _effective_group_ids(df: pd.DataFrame) -> pd.Series:
    if df.empty:
        return pd.Series(dtype=object)
    if "group_id" in df.columns:
        raw = df["group_id"].fillna("").astype(str).str.strip()
    else:
        raw = pd.Series([""] * len(df), index=df.index, dtype=object)

    if "text" in df.columns:
        fallback = [
            _stable_group_id(
                "row",
                task,
                label,
                text,
            )
            for task, label, text in zip(
                df["task"].astype(str),
                df.get("label", pd.Series([""] * len(df), index=df.index)).astype(str),
                df["text"].astype(str),
                strict=False,
            )
        ]
        has_text = True
    else:
        fallback = [
            _stable_group_id(
                "row",
                task,
                label,
                text_a,
                text_b,
            )
            for task, label, text_a, text_b in zip(
                df["task"].astype(str),
                df.get("label", pd.Series([""] * len(df), index=df.index)).astype(str),
                df["text_a"].astype(str),
                df["text_b"].astype(str),
                strict=False,
            )
        ]
        has_text = True
    fallback_series = pd.Series(fallback, index=df.index, dtype=object)
    # Prefer existing group_id when present
    fallback_series = raw.where(raw != "", fallback_series)

    # Template-stem grouping: same template stem -> same group
    if "source" in df.columns:
        stems = df["source"].map(_template_stem_from_source)
        mask = stems.notna() & (stems.astype(str).str.len() > 0)
        if mask.any():
            fallback_series = fallback_series.where(~mask, stems)

    # HF rows: group by source_document_id when available
    if "source_document_id" in df.columns:
        hf_mask = df["source"].fillna("").astype(str).str.startswith("hf:")
        doc_ids = df["source_document_id"].fillna("").astype(str)
        use_doc = hf_mask & (doc_ids != "")
        if use_doc.any():
            fallback_series = fallback_series.where(
                ~use_doc,
                "hf_doc:" + doc_ids,
            )
    elif "source" in df.columns:
        # Fallback: treat existing group_id as document id for hf: rows (already unique per row)
        pass

    # Near-duplicate clustering: merge rows with Jaccard > 0.8 on text shingles (positional indices)
    near_dup_jaccard_threshold = 0.8
    near_dup_max_group_size = 500  # skip clustering for huge task/label groups
    if has_text and ("text" in df.columns or "text_a" in df.columns):
        for (_task, _label), idx_group in df.groupby(["task", "label"], sort=False).indices.items():
            positions = list(idx_group)
            if len(positions) > near_dup_max_group_size or len(positions) < 2:
                continue
            if "text" in df.columns:
                shingles = {pos: _shingle_set(str(df.iloc[pos]["text"])) for pos in positions}
            else:
                shingles = {
                    pos: _shingle_set(
                        str(df.iloc[pos]["text_a"]) + " " + str(df.iloc[pos]["text_b"])
                    )
                    for pos in positions
                }
            parent: dict[int, int] = {}
            find, union = _near_dup_union_find(parent, shingles, near_dup_jaccard_threshold)
            for i in range(len(positions)):
                for j in range(i + 1, len(positions)):
                    union(positions[i], positions[j])
            reps = {pos: find(pos) for pos in positions}
            for pos in positions:
                rep = reps[pos]
                if rep != pos:
                    fallback_series.iloc[pos] = fallback_series.iloc[rep]

    return fallback_series


def _split_integrity_summary(split_frames: dict[str, pd.DataFrame]) -> dict[str, Any]:
    provided_sets: dict[str, set[str]] = {}
    rows_without_group_id: dict[str, int] = {}
    unique_group_ids: dict[str, int] = {}
    for split_name, frame in split_frames.items():
        if frame.empty or "group_id" not in frame.columns:
            provided_sets[split_name] = set()
            rows_without_group_id[split_name] = len(frame)
            unique_group_ids[split_name] = 0
            continue
        series = frame["group_id"].fillna("").astype(str).str.strip()
        provided = {value for value in series.tolist() if value}
        provided_sets[split_name] = provided
        rows_without_group_id[split_name] = int((series == "").sum())
        unique_group_ids[split_name] = len(provided)

    overlap_counts: dict[str, int] = {}
    overlap_samples: dict[str, list[str]] = {}
    split_names = list(split_frames.keys())
    ok = True
    for idx, left in enumerate(split_names):
        for right in split_names[idx + 1 :]:
            key = f"{left}_{right}"
            overlap = sorted(provided_sets[left] & provided_sets[right])
            overlap_counts[key] = len(overlap)
            if overlap:
                ok = False
                overlap_samples[key] = overlap[:10]
    return {
        "ok": ok,
        "rows_without_group_id": rows_without_group_id,
        "unique_group_ids": unique_group_ids,
        "overlap_counts": overlap_counts,
        "overlap_samples": overlap_samples,
    }


def _source_diagnostics(df: pd.DataFrame) -> dict[str, dict[str, Any]]:
    if df.empty or not {"task", "source"}.issubset(df.columns):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for task, group in df.groupby("task", sort=False):
        bucket_counts: dict[str, int] = {}
        for source, count in group["source"].astype(str).map(_source_bucket).value_counts().items():
            bucket_counts[str(source)] = int(count)
        template_rows = int(group["source"].map(_is_template_source).sum())
        out[str(task)] = {
            "rows": len(group),
            "template_rows": template_rows,
            "non_template_rows": len(group) - template_rows,
            "template_ratio": round(template_rows / max(1, len(group)), 4),
            "source_buckets": bucket_counts,
        }
    return out


def _validate_source_coverage(df: pd.DataFrame, *, split_name: str) -> dict[str, dict[str, Any]]:
    diagnostics = _source_diagnostics(df)
    for task, max_template_ratio in _TASK_TEMPLATE_CAPS.items():
        task_diag = diagnostics.get(task)
        if task_diag is None:
            continue
        if float(task_diag["template_ratio"]) > float(max_template_ratio) + 1e-9:
            raise ValueError(
                f"{split_name} split template ratio too high for {task}: "
                f"{task_diag['template_ratio']:.4f} > {max_template_ratio:.4f}"
            )
    if {"task", "source"}.issubset(df.columns):
        source_series = df["source"].fillna("").astype(str)
        task_series = df["task"].fillna("").astype(str)
        for task, prefixes in _REQUIRED_SOURCE_PREFIXES.items():
            task_mask = task_series == task
            if not task_mask.any():
                continue
            for prefix in prefixes:
                if not source_series[task_mask].str.startswith(prefix, na=False).any():
                    raise ValueError(
                        f"{split_name} split is missing required source prefix for {task}: {prefix}"
                    )
    return diagnostics


def _count_jsonl_rows(path: Path) -> int:
    if not path.exists():
        return 0
    count = 0
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            if raw.strip():
                count += 1
    return count


def _validate_adversarial_fixtures() -> dict[str, dict[str, Any]]:
    summary: dict[str, dict[str, Any]] = {}
    for task_name, path in _ADVERSARIAL_FIXTURE_PATHS.items():
        rows = _count_jsonl_rows(path)
        summary[task_name] = {"path": str(path), "rows": rows}
        if rows < _MIN_ADVERSARIAL_ROWS:
            raise ValueError(
                f"Adversarial fixture for {task_name} must contain at least "
                f"{_MIN_ADVERSARIAL_ROWS} rows: {path}"
            )
    return summary


def _load_adversarial_fixture_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with open(path, encoding="utf-8") as handle:
        for raw in handle:
            text = raw.strip()
            if not text:
                continue
            rows.append(json.loads(text))
    return rows


def _strip_adversarial_text_prefix(text: str, *, task_name: str) -> str:
    """Strip 'Adversarial digest N:' or 'Adversarial retention case N:' so model does not key on prefix."""
    if task_name == "consolidation_gist_quality":
        return re.sub(r"^Adversarial digest \d+:\s*", "", text, count=1).strip() or text
    if task_name == "forgetting_action_policy":
        return re.sub(r"^Adversarial retention case \d+:\s*", "", text, count=1).strip() or text
    return text


def _inject_adversarial_into_router_df(
    router_df: pd.DataFrame,
    *,
    seed: int,
    task_names: set[str],
) -> pd.DataFrame:
    """Append a fraction of adversarial rows into router_df. Write heldout to JSONL. Return updated df."""
    out = router_df
    for task_name in task_names:
        path = _ADVERSARIAL_FIXTURE_PATHS.get(task_name)
        if path is None or not path.exists():
            continue
        rows = _load_adversarial_fixture_rows(path)
        if len(rows) < 2:
            continue
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        n_train = max(1, int(len(rows) * _ADVERSARIAL_TRAIN_FRACTION))
        train_idx = set(indices[:n_train])
        eval_idx = [i for i in indices[n_train:]]
        _base_cols = {"text", "task", "label", "source", "language", "group_id"}
        extra_cols = [k for k in rows[0] if k not in {"text", "label"}]
        new_records: list[dict[str, Any]] = []
        for i in train_idx:
            item = rows[i]
            text = _strip_adversarial_text_prefix(str(item.get("text", "")), task_name=task_name)
            source = f"adversarial_train:{task_name}:{i}"
            group_id = source
            record: dict[str, Any] = {
                "text": text,
                "task": task_name,
                "label": str(item.get("label", "")).strip(),
                "source": source,
                "group_id": group_id,
            }
            for col in extra_cols:
                if col in item and col not in record:
                    record[col] = item[col]
            new_records.append(record)
        if new_records:
            extra = pd.DataFrame(new_records)
            if "language" not in extra.columns:
                extra["language"] = "en"
            out = pd.concat([out, extra], ignore_index=True)
            out = out.drop_duplicates(subset=["task", "group_id"], keep="first").reset_index(drop=True)
        heldout_path = path.parent / f"{path.stem}{_ADVERSARIAL_HELDOUT_SUFFIX}.jsonl"
        heldout_records = [rows[i] for i in eval_idx]
        with open(heldout_path, "w", encoding="utf-8") as f:
            for rec in heldout_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def _inject_adversarial_into_pair_df(
    pair_df: pd.DataFrame,
    *,
    seed: int,
    task_names: set[str],
) -> pd.DataFrame:
    """Append a fraction of adversarial rows into pair_df. Write heldout to JSONL. Return updated df."""
    out = pair_df
    for task_name in task_names:
        path = _ADVERSARIAL_FIXTURE_PATHS.get(task_name)
        if path is None or not path.exists():
            continue
        rows = _load_adversarial_fixture_rows(path)
        if len(rows) < 2:
            continue
        if "text_a" not in rows[0] or "text_b" not in rows[0]:
            continue
        rng = random.Random(seed)
        indices = list(range(len(rows)))
        rng.shuffle(indices)
        n_train = max(1, int(len(rows) * _ADVERSARIAL_TRAIN_FRACTION))
        train_idx = set(indices[:n_train])
        eval_idx = [i for i in indices[n_train:]]
        extra_cols = [k for k in rows[0] if k not in {"text_a", "text_b", "label"}]
        new_records = []
        for i in train_idx:
            item = rows[i]
            source = f"adversarial_train:{task_name}:{i}"
            record: dict[str, Any] = {
                "text_a": str(item.get("text_a", "")),
                "text_b": str(item.get("text_b", "")),
                "task": task_name,
                "label": str(item.get("label", "")).strip(),
                "source": source,
                "group_id": source,
            }
            for col in extra_cols:
                if col in item and col not in record:
                    record[col] = item[col]
            new_records.append(record)
        if new_records:
            extra = pd.DataFrame(new_records)
            if "language" not in extra.columns:
                extra["language"] = "en"
            out = pd.concat([out, extra], ignore_index=True)
            out = out.drop_duplicates(
                subset=["task", "text_a", "text_b", "label"], keep="first"
            ).reset_index(drop=True)
        heldout_path = path.parent / f"{path.stem}{_ADVERSARIAL_HELDOUT_SUFFIX}.jsonl"
        heldout_records = [rows[i] for i in eval_idx]
        with open(heldout_path, "w", encoding="utf-8") as f:
            for rec in heldout_records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return out


def _split_by_task_label(
    df: pd.DataFrame, seed: int, ratios: dict[str, float]
) -> dict[str, pd.DataFrame]:
    if df.empty:
        return {"train": df.copy(), "test": df.copy(), "eval": df.copy()}

    working = df.copy()
    working["_effective_group_id"] = _effective_group_ids(working)
    working["_task_label_key"] = (
        working["task"].astype(str) + "::" + working.get("label", "").astype(str)
    )

    target_counts: dict[str, dict[str, int]] = {"train": {}, "test": {}, "eval": {}}
    for key, count in working["_task_label_key"].value_counts().items():
        n_train, n_test, n_eval = _split_counts(int(count), ratios)
        target_counts["train"][str(key)] = n_train
        target_counts["test"][str(key)] = n_test
        target_counts["eval"][str(key)] = n_eval

    total_train, total_test, total_eval = _split_counts(len(working), ratios)
    target_rows = {"train": total_train, "test": total_test, "eval": total_eval}
    current_counts: dict[str, dict[str, int]] = {
        "train": defaultdict(int),
        "test": defaultdict(int),
        "eval": defaultdict(int),
    }
    current_rows = {"train": 0, "test": 0, "eval": 0}
    assignments: dict[str, str] = {}

    group_payloads: list[dict[str, Any]] = []
    for group_id, group in working.groupby("_effective_group_id", sort=False):
        payload = {
            str(key): int(value)
            for key, value in group["_task_label_key"].value_counts().to_dict().items()
        }
        task_sources = {
            (str(task), str(source))
            for task, source in zip(
                group["task"].astype(str),
                group["source"].fillna("").astype(str),
                strict=False,
            )
        }
        group_payloads.append(
            {
                "group_id": str(group_id),
                "row_indices": list(group.index),
                "counts": payload,
                "size": len(group),
                "task_sources": task_sources,
            }
        )
    group_payloads.sort(key=lambda item: (-int(item["size"]), str(item["group_id"])))

    def _assign_group(payload: dict[str, Any], split_name: str) -> None:
        group_id = str(payload["group_id"])
        assignments[group_id] = split_name
        current_rows[split_name] += int(payload["size"])
        for key, value in dict(payload["counts"]).items():
            current_counts[split_name][str(key)] = int(current_counts[split_name].get(str(key), 0)) + int(value)

    preassigned_group_ids: set[str] = set()
    for task, prefixes in _REQUIRED_SOURCE_PREFIXES.items():
        for prefix in prefixes:
            candidate = next(
                (
                    payload
                    for payload in group_payloads
                    if str(payload["group_id"]) not in preassigned_group_ids
                    and any(
                        group_task == task and source.startswith(prefix)
                        for group_task, source in payload["task_sources"]
                    )
                ),
                None,
            )
            if candidate is None:
                continue
            _assign_group(candidate, "train")
            preassigned_group_ids.add(str(candidate["group_id"]))

    required_splits = [name for name, ratio in ratios.items() if ratio > 0]
    remaining_payloads = [
        payload for payload in group_payloads if str(payload["group_id"]) not in preassigned_group_ids
    ]
    reserved_splits = [
        name for name in required_splits if current_rows[name] == 0
    ][: min(len(required_splits), len(remaining_payloads))]

    def _assignment_penalty(split_name: str, counts: dict[str, int], group_size: int) -> tuple[float, float]:
        penalty = 0.0
        for key, value in counts.items():
            current = int(current_counts[split_name].get(key, 0))
            target = int(target_counts[split_name].get(key, 0))
            projected = current + int(value)
            penalty += abs(projected - target) - abs(current - target)
            if target > 0 and projected > target:
                penalty += 0.35 * (projected - target)
        current_total = current_rows[split_name]
        projected_total = current_total + group_size
        total_target = int(target_rows[split_name])
        penalty += 0.15 * (
            abs(projected_total - total_target) - abs(current_total - total_target)
        )
        return penalty, float(projected_total)

    for idx, payload in enumerate(remaining_payloads):
        group_id = str(payload["group_id"])
        counts = cast("dict[str, int]", payload["counts"])
        group_size = int(payload["size"])
        if idx < len(reserved_splits):
            split_name = reserved_splits[idx]
        else:
            split_name = min(
                ("train", "test", "eval"),
                key=lambda name: _assignment_penalty(name, counts, group_size),
            )
        _assign_group(payload, split_name)

    out: dict[str, pd.DataFrame] = {}
    for split_name in ("train", "test", "eval"):
        selected = [
            idx
            for payload in group_payloads
            if assignments[str(payload["group_id"])] == split_name
            for idx in list(payload["row_indices"])
        ]
        frame = working.loc[selected].drop(columns=["_effective_group_id", "_task_label_key"])
        out[split_name] = frame.sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def _write_splits(
    df: pd.DataFrame, *, prefix: str, out_dir: Path, seed: int, ratios: dict[str, float]
) -> tuple[dict[str, int], dict[str, Any], dict[str, dict[str, Any]]]:
    splits = _split_by_task_label(df, seed, ratios)
    split_integrity = _split_integrity_summary(splits)
    if not split_integrity["ok"]:
        raise ValueError(f"{prefix} split integrity failure: {split_integrity['overlap_counts']}")
    train_source_diagnostics = _validate_source_coverage(splits["train"], split_name=f"{prefix}:train")
    counts: dict[str, int] = {}
    for split_name, split_df in splits.items():
        path = out_dir / f"{prefix}_{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        counts[split_name] = len(split_df)
    return counts, split_integrity, train_source_diagnostics


def _pair_embedding_cache_summary(prepared_dir: Path) -> dict[str, Any] | None:
    path = pair_embedding_cache_path(prepared_dir)
    if not path.exists():
        return None
    try:
        df = pd.read_parquet(path)
    except Exception:
        return {"path": str(path), "rows": 0}
    model_name = ""
    if "embedding_model_name" in df.columns and not df.empty:
        names = [str(value).strip() for value in df["embedding_model_name"].astype(str).tolist()]
        model_name = next((name for name in names if name), "")
    return {
        "path": str(path),
        "rows": len(df),
        "embedding_model_name": model_name or DEFAULT_PAIR_EMBEDDING_MODEL_NAME,
    }


def _write_pair_embedding_cache(
    *,
    df: pd.DataFrame,
    task_names: set[str],
    model_name: str,
    out_dir: Path,
) -> dict[str, Any]:
    subset = df[df["task"].astype(str).isin(task_names)].copy()
    if subset.empty:
        raise ValueError("No pair rows available for configured embedding_pair tasks.")

    ordered_texts: dict[str, None] = {}
    for text in subset["text_a"].astype(str).tolist():
        cleaned = text.strip()
        if cleaned:
            ordered_texts.setdefault(cleaned, None)
    for text in subset["text_b"].astype(str).tolist():
        cleaned = text.strip()
        if cleaned:
            ordered_texts.setdefault(cleaned, None)
    if not ordered_texts:
        raise ValueError("No text rows available for embedding cache generation.")

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:
        raise ImportError(
            "Embedding pair prepare requires sentence-transformers. "
            'Install with: pip install "cognitive-memory-layer[modeling]"'
        ) from exc

    encoder = SentenceTransformer(model_name)
    texts = list(ordered_texts.keys())
    embeddings = encoder.encode(
        texts,
        batch_size=64,
        convert_to_numpy=True,
        show_progress_bar=False,
        normalize_embeddings=True,
    )
    payload = pd.DataFrame(
        [
            {
                "text_hash": hash_text(text),
                "text": text,
                "embedding": embeddings[idx].astype("float32").tolist(),
                "embedding_model_name": model_name,
            }
            for idx, text in enumerate(texts)
        ]
    )
    cache_path = pair_embedding_cache_path(out_dir)
    payload.to_parquet(cache_path, index=False)
    return {
        "path": str(cache_path),
        "rows": len(payload),
        "tasks": sorted(task_names),
        "embedding_model_name": model_name,
    }


def _load_pair_embedding_vectors(cache_path: Path) -> dict[str, np.ndarray]:
    if not cache_path.exists():
        return {}
    cache_df = pd.read_parquet(cache_path)
    vectors: dict[str, np.ndarray] = {}
    for item in cache_df.itertuples(index=False):
        text = _clean(getattr(item, "text", ""), 700)
        if not text:
            continue
        vector = np.asarray(getattr(item, "embedding", []), dtype=np.float32)
        if vector.ndim != 1 or vector.size == 0:
            continue
        vectors[text] = vector
    return vectors


def _augment_pair_hard_negatives(
    df: pd.DataFrame,
    *,
    cache_path: Path,
    task_names: set[str],
) -> tuple[pd.DataFrame, dict[str, int]]:
    if df.empty or not task_names:
        return df, {}
    vectors = _load_pair_embedding_vectors(cache_path)
    if not vectors:
        return df, {}

    extra_rows: list[dict[str, Any]] = []
    added_by_task: dict[str, int] = {}
    for task_name in sorted(task_names):
        task_df = df[df["task"].astype(str) == task_name].copy()
        positive_df = task_df[task_df["label"].astype(str) == "relevant"].copy()
        candidate_df = task_df[task_df["label"].astype(str) == "not_relevant"].copy()
        if positive_df.empty or candidate_df.empty:
            continue

        positive_groups = positive_df.drop_duplicates(subset=["group_id"], keep="first").reset_index(drop=True)
        candidate_df = candidate_df.drop_duplicates(subset=["text_b", "group_id"], keep="first").reset_index(drop=True)
        candidate_vectors = [
            vectors.get(text)
            for text in candidate_df["text_b"].astype(str).tolist()
        ]
        if not candidate_vectors or any(vector is None for vector in candidate_vectors):
            candidate_rows = [
                idx for idx, vector in enumerate(candidate_vectors) if vector is not None
            ]
            if not candidate_rows:
                continue
            candidate_df = candidate_df.iloc[candidate_rows].reset_index(drop=True)
            candidate_vectors = [candidate_vectors[idx] for idx in candidate_rows]
        non_none_vectors = [v for v in candidate_vectors if v is not None]
        candidate_matrix = np.vstack(non_none_vectors).astype(np.float32, copy=False)
        candidate_groups = candidate_df["group_id"].fillna("").astype(str).to_numpy(dtype=object)
        candidate_texts = candidate_df["text_b"].astype(str).to_numpy(dtype=object)

        query_rows: list[int] = []
        query_vectors: list[np.ndarray] = []
        for idx, row in positive_groups.iterrows():
            vector = vectors.get(str(row["text_a"]))
            if vector is None:
                continue
            query_rows.append(idx)
            query_vectors.append(vector)
        if not query_vectors:
            continue

        added = 0
        query_matrix = np.vstack(query_vectors).astype(np.float32, copy=False)
        batch_size = 256
        for start in range(0, len(query_rows), batch_size):
            batch_indices = query_rows[start : start + batch_size]
            batch_matrix = query_matrix[start : start + batch_size]
            scores = batch_matrix @ candidate_matrix.T
            for batch_row, pos_idx in enumerate(batch_indices):
                row = positive_groups.iloc[pos_idx]
                group_id = str(row.get("group_id", "")).strip()
                if group_id:
                    scores[batch_row, candidate_groups == group_id] = -np.inf
                scores[batch_row, candidate_texts == str(row.get("text_b", ""))] = -np.inf
                top_k = 3
                best_indices = np.argsort(scores[batch_row])[-top_k:][::-1]
                for best_idx in best_indices:
                    best_idx = int(best_idx)
                    best_score = float(scores[batch_row, best_idx])
                    if not np.isfinite(best_score):
                        continue
                    extra_rows.append(
                        {
                            "task": task_name,
                            "label": "not_relevant",
                            "text_a": str(row.get("text_a", "")),
                            "text_b": str(candidate_texts[best_idx]),
                            "source": f"derived_hard_negative:{task_name}",
                            "language": str(row.get("language", "en") or "en"),
                            "group_id": group_id or _stable_group_id(
                                "hard_negative",
                                task_name,
                                row.get("text_a", ""),
                                candidate_texts[best_idx],
                            ),
                        }
                    )
                    added += 1
        if added > 0:
            added_by_task[task_name] = added

    if not extra_rows:
        return df, {}
    augmented = pd.concat([df, pd.DataFrame(extra_rows)], ignore_index=True, sort=False)
    augmented = augmented.drop_duplicates(subset=["task", "label", "text_a", "text_b"], keep="first")
    return augmented.reset_index(drop=True), added_by_task


def _summary(df: pd.DataFrame) -> dict:
    task_counts = df["task"].value_counts().to_dict() if "task" in df.columns else {}
    source_top = df["source"].value_counts().head(20).to_dict() if "source" in df.columns else {}
    task_label_counts = (
        df.groupby(["task", "label"]).size().sort_values(ascending=False).to_dict()
        if {"task", "label"}.issubset(df.columns)
        else {}
    )

    source_mix: dict[str, dict[str, int]] = {}
    if {"task", "source"}.issubset(df.columns):
        for (task, source), count in df.groupby(["task", "source"]).size().items():
            source_mix.setdefault(str(task), {})[str(source)] = int(count)

    synthetic_ratio: dict[str, float] = {}
    if {"task", "label", "source"}.issubset(df.columns):
        for (task, label), group in df.groupby(["task", "label"]):
            key = f"{task}::{label}"
            total = len(group)
            synth = len(
                group[
                    group["source"].astype(str).str.startswith(
                        ("llm:", "template:", "template_hardened:"), na=False
                    )
                ]
            )
            synthetic_ratio[key] = round(synth / max(1, total), 4)

    return {
        "rows": len(df),
        "task_counts": task_counts,
        "task_label_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in task_label_counts.items()},
        "source_top_20": source_top,
        "source_mix_per_task": source_mix,
        "synthetic_ratio_per_label": synthetic_ratio,
        "source_diagnostics_per_task": _source_diagnostics(df),
        "group_id_rows": int(df["group_id"].fillna("").astype(str).str.strip().ne("").sum())
        if "group_id" in df.columns
        else 0,
    }


def _load_existing_family_df(prepared_dir: Path, family: str) -> pd.DataFrame:
    if family in {"router", "extractor"}:
        required = ["text", "task", "label"]
        base_cols = ["text", "task", "label", "source"]
    elif family == "pair":
        required = ["text_a", "text_b", "task", "label"]
        base_cols = ["text_a", "text_b", "task", "label", "source"]
    else:
        raise ValueError(f"Unknown family: {family}")

    parts: list[pd.DataFrame] = []
    for split in LOCAL_BOOTSTRAP_SPLITS:
        path = prepared_dir / f"{family}_{split}.parquet"
        if not path.exists():
            continue
        try:
            df = pd.read_parquet(path)
        except Exception as exc:
            _warn(f"Failed reading existing split {path.name}: {exc}")
            continue
        if not set(required).issubset(df.columns):
            _warn(f"Existing split missing required columns ({path.name}): {required}")
            continue
        if "source" not in df.columns:
            df = df.copy()
            df["source"] = f"prepared:{family}:{split}"
        for col in base_cols:
            if col not in df.columns:
                df[col] = "" if col != "source" else f"prepared:{family}:{split}"
        parts.append(df.copy())

    if not parts:
        return pd.DataFrame(columns=base_cols)
    return pd.concat(parts, ignore_index=True, sort=False)


def _existing_split_counts(prepared_dir: Path, family: str) -> dict[str, int]:
    out: dict[str, int] = {}
    for split in LOCAL_BOOTSTRAP_SPLITS:
        path = prepared_dir / f"{family}_{split}.parquet"
        if not path.exists():
            out[split] = 0
            continue
        try:
            out[split] = len(pd.read_parquet(path))
        except Exception:
            out[split] = 0
    return out


def _existing_split_integrity(prepared_dir: Path, family: str) -> dict[str, Any]:
    split_frames: dict[str, pd.DataFrame] = {}
    for split in LOCAL_BOOTSTRAP_SPLITS:
        path = prepared_dir / f"{family}_{split}.parquet"
        if not path.exists():
            split_frames[split] = pd.DataFrame()
            continue
        try:
            split_frames[split] = pd.read_parquet(path)
        except Exception:
            split_frames[split] = pd.DataFrame()
    return _split_integrity_summary(split_frames)


def _existing_train_source_diagnostics(prepared_dir: Path, family: str) -> dict[str, dict[str, Any]]:
    path = prepared_dir / f"{family}_train.parquet"
    if not path.exists():
        return {}
    try:
        return _source_diagnostics(pd.read_parquet(path))
    except Exception:
        return {}


def _missing_task_labels(
    df: pd.DataFrame,
    *,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
) -> tuple[dict[str, int], int]:
    if df.empty:
        counts: dict[tuple[str, str], int] = {}
    else:
        grouped = df.groupby(["task", "label"]).size().to_dict()
        counts = {(str(k[0]), str(k[1])): int(v) for k, v in grouped.items()}

    missing: dict[str, int] = {}
    total = 0
    for task, labels in task_labels.items():
        for label in labels:
            have = int(counts.get((task, label), 0))
            miss = max(0, int(target_per_task_label) - have)
            missing[f"{task}::{label}"] = miss
            total += miss
    return missing, total


def _missing_regression_tasks(
    df: pd.DataFrame,
    *,
    task_names: set[str],
    target_per_task: int,
) -> tuple[dict[str, int], int]:
    if df.empty or "task" not in df.columns:
        counts: dict[str, int] = {}
    else:
        subset = df[df["task"].astype(str).isin(task_names)]
        if "score" in subset.columns:
            valid = subset[pd.to_numeric(subset["score"], errors="coerce").notna()]
        else:
            valid = subset.iloc[0:0]
        counts = valid.groupby("task").size().to_dict() if not valid.empty else {}

    missing: dict[str, int] = {}
    total = 0
    for task_name in sorted(task_names):
        have = int(counts.get(task_name, 0))
        miss = max(0, int(target_per_task) - have)
        missing[task_name] = miss
        total += miss
    return missing, total


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for all custom model families.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model_pipeline.toml",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override prepare.seed from config.")
    parser.add_argument(
        "--max-rows-per-source",
        type=int,
        default=None,
        help="Override prepare.max_rows_per_source.",
    )
    parser.add_argument(
        "--max-per-task-label", type=int, default=None, help="Override prepare.max_per_task_label."
    )
    parser.add_argument(
        "--target-per-task-label",
        type=int,
        default=None,
        help="Override prepare.target_per_task_label (minimum rows per task/label).",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=None,
        help="Override synthetic_llm.temperature.",
    )
    parser.add_argument(
        "--llm-concurrency",
        type=int,
        default=None,
        help="Override synthetic_llm.concurrency.",
    )
    parser.add_argument(
        "--disable-download",
        action="store_true",
        help="Do not pre-fetch required remote datasets before processing.",
    )
    parser.add_argument(
        "--allow-missing-datasets-package",
        action="store_true",
        help="Allow running with local bootstrap data only when `datasets` package is missing.",
    )
    parser.add_argument(
        "--force-full",
        action="store_true",
        help="Ignore existing prepared splits and rebuild all families from scratch.",
    )
    parser.add_argument(
        "--no-multilingual",
        action="store_true",
        help="Disable multilingual synthetic generation (English only).",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    _load_dotenv(REPO_ROOT / ".env")
    config = _load_config(args.config)

    paths_cfg = dict(config.get("paths", {}))
    prepare_cfg = dict(config.get("prepare", {}))
    synthetic_cfg = dict(config.get("synthetic_llm", {}))
    multilingual_cfg = dict(config.get("multilingual", {}))
    datasets_cfg = list(config.get("datasets", []))
    task_specs_raw = list(config.get("tasks", []))
    token_prepare_cfg = dict(prepare_cfg.get("token", {}))

    if args.seed is not None:
        prepare_cfg["seed"] = int(args.seed)
    if args.max_rows_per_source is not None:
        prepare_cfg["max_rows_per_source"] = int(args.max_rows_per_source)
    if args.max_per_task_label is not None:
        prepare_cfg["max_per_task_label"] = int(args.max_per_task_label)
    if args.target_per_task_label is not None:
        prepare_cfg["target_per_task_label"] = int(args.target_per_task_label)
    if args.llm_temperature is not None:
        synthetic_cfg["temperature"] = float(args.llm_temperature)
    if args.llm_concurrency is not None:
        synthetic_cfg["concurrency"] = int(args.llm_concurrency)
    if args.allow_missing_datasets_package:
        prepare_cfg["require_datasets_package"] = False

    use_multilingual = bool(multilingual_cfg.get("enabled", True)) and not getattr(
        args, "no_multilingual", False
    )

    prepare_cfg.setdefault("seed", 42)
    prepare_cfg.setdefault("train_ratio", 0.8)
    prepare_cfg.setdefault("test_ratio", 0.1)
    prepare_cfg.setdefault("eval_ratio", 0.1)
    prepare_cfg.setdefault("max_rows_per_source", 60000)
    prepare_cfg.setdefault("target_per_task_label", 10000)
    prepare_cfg.setdefault("max_per_task_label", 50000)
    prepare_cfg.setdefault("auto_download_missing", True)
    prepare_cfg.setdefault("require_datasets_package", True)
    prepare_cfg["max_per_task_label"] = max(
        int(prepare_cfg["max_per_task_label"]),
        int(prepare_cfg["target_per_task_label"]),
    )
    token_prepare_cfg.setdefault("target_examples_per_task", 40000)
    token_prepare_cfg.setdefault("max_examples_per_task", 80000)
    token_prepare_cfg.setdefault("max_spans_per_example", 16)
    prepare_cfg["token"] = token_prepare_cfg

    synthetic_cfg.setdefault("provider_env", "LLM_EVAL__PROVIDER")
    synthetic_cfg.setdefault("model_env", "LLM_EVAL__MODEL")
    synthetic_cfg.setdefault("base_url_env", "LLM_EVAL__BASE_URL")
    synthetic_cfg.setdefault("api_key_env", "OPENAI_API_KEY")
    synthetic_cfg.setdefault("provider", "ollama")
    synthetic_cfg.setdefault("model", "")
    synthetic_cfg.setdefault("base_url", "")
    synthetic_cfg.setdefault("temperature", 1.35)
    synthetic_cfg.setdefault("top_p", 0.95)
    synthetic_cfg.setdefault("max_tokens", 1024)
    synthetic_cfg.setdefault("batch_size", 24)
    synthetic_cfg.setdefault("concurrency", 8)
    synthetic_cfg.setdefault("timeout_seconds", 120)
    synthetic_cfg.setdefault("max_retries", 3)
    synthetic_cfg.setdefault("max_attempts_per_label", 80)
    synthetic_cfg.setdefault("log_stats_every_seconds", 10.0)
    synthetic_cfg.setdefault("log_zero_progress_every", 5)
    synthetic_cfg.setdefault("parse_failure_log_every", 20)
    synthetic_cfg.setdefault("log_request_failures", True)
    synthetic_cfg.setdefault("local_raw_scan_rows_per_file", 300)

    if int(prepare_cfg["max_rows_per_source"]) <= 0:
        print("prepare.max_rows_per_source must be > 0.", file=sys.stderr)
        return 1
    if int(prepare_cfg["target_per_task_label"]) <= 0:
        print("prepare.target_per_task_label must be > 0.", file=sys.stderr)
        return 1
    if int(prepare_cfg["max_per_task_label"]) <= 0:
        print("prepare.max_per_task_label must be > 0.", file=sys.stderr)
        return 1
    if int(synthetic_cfg["concurrency"]) <= 0:
        print("synthetic_llm.concurrency must be > 0.", file=sys.stderr)
        return 1
    if int(synthetic_cfg["max_attempts_per_label"]) <= 0:
        print("synthetic_llm.max_attempts_per_label must be > 0.", file=sys.stderr)
        return 1
    if int(token_prepare_cfg["target_examples_per_task"]) <= 0:
        print("prepare.token.target_examples_per_task must be > 0.", file=sys.stderr)
        return 1
    if int(token_prepare_cfg["max_examples_per_task"]) < int(
        token_prepare_cfg["target_examples_per_task"]
    ):
        print(
            "prepare.token.max_examples_per_task must be >= target_examples_per_task.",
            file=sys.stderr,
        )
        return 1
    if int(token_prepare_cfg["max_spans_per_example"]) <= 0:
        print("prepare.token.max_spans_per_example must be > 0.", file=sys.stderr)
        return 1

    prepared_dir = _resolve_path(
        str(paths_cfg.get("prepared_dir", "packages/models/prepared_data/modelpack")),
        base=REPO_ROOT,
    )
    bootstrap_dir = _resolve_path(
        str(paths_cfg.get("bootstrap_prepared_dir", "packages/models/prepared_data")),
        base=REPO_ROOT,
    )
    cache_dir_raw = str(paths_cfg.get("datasets_cache_dir", "packages/models/datasets")).strip()
    cache_dir = _resolve_path(cache_dir_raw, base=REPO_ROOT) if cache_dir_raw else None

    if not bootstrap_dir.exists():
        alt_bootstrap = (REPO_ROOT / "packages/models/prepared_data").resolve()
        if alt_bootstrap.exists():
            _warn(
                f"bootstrap_prepared_dir not found ({bootstrap_dir}); using {alt_bootstrap} instead."
            )
            bootstrap_dir = alt_bootstrap

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    target_per_task_label = int(prepare_cfg["target_per_task_label"])
    target_token_examples = int(token_prepare_cfg["target_examples_per_task"])

    if args.force_full:
        print("Force-full mode: rebuilding outputs while reusing prior prepared rows as seed data.")
        router_seed_df = _load_existing_family_df(prepared_dir, "router")
        extractor_seed_df = _load_existing_family_df(prepared_dir, "extractor")
        pair_seed_df = _load_existing_family_df(prepared_dir, "pair")
        router_existing_df = (
            router_seed_df[~router_seed_df["task"].astype(str).isin(REBUILT_ROUTER_TASKS)].copy()
            if not router_seed_df.empty
            else pd.DataFrame(columns=["text", "task", "label", "source"])
        )
        extractor_existing_df = (
            extractor_seed_df.copy()
            if not extractor_seed_df.empty
            else pd.DataFrame(columns=["text", "task", "label", "source"])
        )
        pair_existing_df = (
            pair_seed_df[~pair_seed_df["task"].astype(str).isin(REBUILT_PAIR_TASKS)].copy()
            if not pair_seed_df.empty
            else pd.DataFrame(columns=["text_a", "text_b", "task", "label", "source"])
        )
        token_existing_dfs = {
            task_name: _load_existing_token_df(prepared_dir, task_name) for task_name in TOKEN_TASKS
        }
    else:
        router_existing_df = _load_existing_family_df(prepared_dir, "router")
        extractor_existing_df = _load_existing_family_df(prepared_dir, "extractor")
        pair_existing_df = _load_existing_family_df(prepared_dir, "pair")
        token_existing_dfs = {
            task_name: _load_existing_token_df(prepared_dir, task_name) for task_name in TOKEN_TASKS
        }

    router_task_labels = _enabled_task_label_map(
        task_specs_raw,
        family="router",
        base_labels=ROUTER_TASK_LABELS,
        extra_labels=NEW_SINGLE_TASK_LABELS,
    )
    pair_task_labels = _enabled_task_label_map(
        task_specs_raw,
        family="pair",
        base_labels=PAIR_TASK_LABELS,
        extra_labels=NEW_PAIR_TASK_LABELS,
    )
    embedding_pair_tasks, embedding_pair_model_name = _enabled_embedding_pair_tasks(task_specs_raw)
    router_regression_tasks = _enabled_regression_tasks(task_specs_raw, family="router")
    token_tasks = _enabled_token_tasks(task_specs_raw, family="extractor")

    router_missing, router_missing_total = _missing_task_labels(
        router_existing_df,
        task_labels=router_task_labels,
        target_per_task_label=target_per_task_label,
    )
    router_regression_missing, router_regression_missing_total = _missing_regression_tasks(
        router_existing_df,
        task_names=router_regression_tasks,
        target_per_task=target_per_task_label,
    )
    extractor_missing, extractor_missing_total = _missing_task_labels(
        extractor_existing_df,
        task_labels=EXTRACTOR_TASK_LABELS,
        target_per_task_label=target_per_task_label,
    )
    pair_missing, pair_missing_total = _missing_task_labels(
        pair_existing_df,
        task_labels=pair_task_labels,
        target_per_task_label=target_per_task_label,
    )
    token_missing, token_missing_total = _missing_token_tasks(
        token_existing_dfs,
        task_names=token_tasks,
        target_per_task=target_token_examples,
    )

    needs_router = (
        args.force_full or router_missing_total > 0 or router_regression_missing_total > 0
    )
    needs_extractor = args.force_full or extractor_missing_total > 0
    needs_pair = args.force_full or pair_missing_total > 0
    needs_tokens = args.force_full or token_missing_total > 0

    print(
        "Existing coverage missing counts: "
        f"router={router_missing_total + router_regression_missing_total}, "
        f"extractor={extractor_missing_total}, pair={pair_missing_total}, "
        f"token={token_missing_total}"
    )

    ratios = {
        "train": float(prepare_cfg["train_ratio"]),
        "test": float(prepare_cfg["test_ratio"]),
        "eval": float(prepare_cfg["eval_ratio"]),
    }
    if min(ratios.values()) < 0:
        print("Split ratios must be non-negative.", file=sys.stderr)
        return 1
    if sum(ratios.values()) <= 0:
        print("At least one split ratio must be positive.", file=sys.stderr)
        return 1
    total_ratio = sum(ratios.values())
    ratios = {k: v / total_ratio for k, v in ratios.items()}
    adversarial_fixture_summary = _validate_adversarial_fixtures()

    if not (needs_router or needs_extractor or needs_pair or needs_tokens):
        embedding_cache_summary = _pair_embedding_cache_summary(prepared_dir)
        if embedding_pair_tasks and (
            embedding_cache_summary is None or embedding_cache_summary.get("rows", 0) <= 0
        ):
            embedding_cache_summary = _write_pair_embedding_cache(
                df=pair_existing_df,
                task_names=embedding_pair_tasks,
                model_name=embedding_pair_model_name or DEFAULT_PAIR_EMBEDDING_MODEL_NAME,
                out_dir=prepared_dir,
            )
        manifest = {
            "config_path": str(args.config.resolve()),
            "seed": int(prepare_cfg["seed"]),
            "incremental": {
                "mode": "missing_only",
                "skipped_all_families": True,
                "force_full": False,
                "missing_counts": {
                    "router": router_missing_total + router_regression_missing_total,
                    "extractor": extractor_missing_total,
                    "pair": pair_missing_total,
                    "token": token_missing_total,
                },
            },
            "paths": {
                "prepared_dir": str(prepared_dir),
                "bootstrap_prepared_dir": str(bootstrap_dir),
                "datasets_cache_dir": str(cache_dir) if cache_dir else "",
            },
            "datasets": {},
            "configured_tasks": {
                "router": sorted(router_task_labels.keys()) + sorted(router_regression_tasks),
                "extractor": list(EXTRACTOR_TASK_LABELS.keys()),
                "pair": sorted(pair_task_labels.keys()),
            },
            "configured_token_tasks": sorted(token_tasks),
            "observed_tasks": {
                "router": sorted(router_existing_df["task"].astype(str).unique().tolist())
                if "task" in router_existing_df.columns
                else [],
                "extractor": sorted(extractor_existing_df["task"].astype(str).unique().tolist())
                if "task" in extractor_existing_df.columns
                else [],
                "pair": sorted(pair_existing_df["task"].astype(str).unique().tolist())
                if "task" in pair_existing_df.columns
                else [],
            },
            "observed_token_tasks": sorted(
                task_name for task_name, df in token_existing_dfs.items() if not df.empty
            ),
            "router": {
                **_summary(router_existing_df),
                "splits": _existing_split_counts(prepared_dir, "router"),
                "split_integrity": _existing_split_integrity(prepared_dir, "router"),
                "train_source_diagnostics": _existing_train_source_diagnostics(prepared_dir, "router"),
                "tasks": sorted(router_task_labels.keys()) + sorted(router_regression_tasks),
                "updated": False,
            },
            "extractor": {
                **_summary(extractor_existing_df),
                "splits": _existing_split_counts(prepared_dir, "extractor"),
                "split_integrity": _existing_split_integrity(prepared_dir, "extractor"),
                "train_source_diagnostics": _existing_train_source_diagnostics(prepared_dir, "extractor"),
                "tasks": list(EXTRACTOR_TASKS),
                "updated": False,
            },
            "pair": {
                **_summary(pair_existing_df),
                "splits": _existing_split_counts(prepared_dir, "pair"),
                "split_integrity": _existing_split_integrity(prepared_dir, "pair"),
                "train_source_diagnostics": _existing_train_source_diagnostics(prepared_dir, "pair"),
                "tasks": sorted(pair_task_labels.keys()),
                "updated": False,
            },
            "pair_embedding_cache": embedding_cache_summary,
            "pair_hard_negative_augmentation": {},
            "adversarial_fixtures": adversarial_fixture_summary,
            "token_task_splits": {
                task_name: _existing_token_split_counts(prepared_dir, task_name)
                for task_name in sorted(token_tasks)
            },
            "token_tasks": {
                task_name: {
                    **_token_summary(token_existing_dfs.get(task_name, pd.DataFrame())),
                    "splits": _existing_token_split_counts(prepared_dir, task_name),
                    "updated": False,
                }
                for task_name in sorted(token_tasks)
            },
        }
        manifest_path = prepared_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)
        print("All prepared families already satisfy target_per_task_label; no update needed.")
        print(f"Prepared dir: {prepared_dir}")
        print(f"Manifest: {manifest_path}")
        return 0

    # Load only datasets relevant to missing families (+ seed pools).
    active_targets = {"seed"}
    if needs_router:
        active_targets.add("router")
    if needs_extractor:
        active_targets.add("extractor")
    if needs_pair:
        active_targets.add("pair")
    if needs_tokens:
        active_targets.add("extractor")
    datasets_cfg_active = [
        d for d in datasets_cfg if str(d.get("target", "")).strip().lower() in active_targets
    ]
    if token_tasks:
        slow_optional = {"docred", "re_tacred"}
        datasets_cfg_active = [
            d for d in datasets_cfg_active if str(d.get("name", "")).strip() not in slow_optional
        ]

    has_enabled_remote = any(bool(d.get("enabled", True)) for d in datasets_cfg_active)
    if (
        has_enabled_remote
        and _hf_load_dataset is None
        and bool(prepare_cfg["require_datasets_package"])
    ):
        _warn("Missing `datasets` package; continuing with existing/bootstrap/template rows only.")

    registry = _HFRegistry(
        datasets_cfg=datasets_cfg_active, cache_dir=cache_dir, prepare_cfg=prepare_cfg
    )
    if bool(prepare_cfg["auto_download_missing"]) and not args.disable_download:
        try:
            registry.ensure_required()
        except Exception as exc:
            _warn(f"Dataset prefetch failed; continuing with available rows only: {exc}")

    print("Loading local bootstrap rows...")
    local_rows = _load_local_bootstrap_rows(
        bootstrap_dir=bootstrap_dir,
        max_rows_per_source=int(prepare_cfg["max_rows_per_source"]),
        seed=int(prepare_cfg["seed"]),
    )
    print(f"Local bootstrap rows: {len(local_rows)}")

    print("Scanning local raw dataset files for seed conversations...")
    local_raw_texts: list[str] = []
    local_raw_pairs: list[tuple[str, str]] = []
    if cache_dir:
        local_raw_texts, local_raw_pairs = _scan_local_raw_seed_texts(
            raw_dir=cache_dir,
            max_rows_per_file=int(synthetic_cfg["local_raw_scan_rows_per_file"]),
        )
    print(
        f"Local raw seed texts: {len(local_raw_texts)} | local raw seed pairs: {len(local_raw_pairs)}"
    )

    print("Collecting remote/local seed pools...")
    single_pools, pair_pool = _collect_seed_pools(
        local_rows=local_rows,
        local_raw_texts=local_raw_texts,
        local_raw_pairs=local_raw_pairs,
        registry=registry,
        datasets_cfg=datasets_cfg_active,
        max_rows_per_source=int(prepare_cfg["max_rows_per_source"]),
    )
    pool_sizes = {k: len(v) for k, v in single_pools.items()}
    print(f"Seed pool sizes: single={pool_sizes}, pair={len(pair_pool)}")

    llm: _LLMGenerator | None = None
    try:
        llm = _LLMGenerator(synthetic_cfg)
    except Exception as exc:
        _warn(f"Continuing without synthetic LLM fill: {exc}")
    else:
        print(
            "Using synthetic LLM provider="
            f"{llm.provider or 'unknown'} model={llm.model} base_url={llm.base_url} "
            f"temperature={llm.temperature} batch_size={llm.batch_size} "
            f"concurrency={llm.concurrency} timeout={llm.timeout_seconds}s "
            f"multilingual={use_multilingual}"
        )
    manifest_warnings: list[str] = []
    if use_multilingual and "fact_extraction_structured" in token_tasks and llm is None:
        manifest_warnings.append(
            "fact_extraction_structured multilingual token prep used deterministic templates only; "
            "token-specific LLM fill was unavailable."
        )
    embedding_cache_summary = _pair_embedding_cache_summary(prepared_dir)
    router_split_integrity = _existing_split_integrity(prepared_dir, "router")
    extractor_split_integrity = _existing_split_integrity(prepared_dir, "extractor")
    pair_split_integrity = _existing_split_integrity(prepared_dir, "pair")
    router_train_source_diagnostics = _existing_train_source_diagnostics(prepared_dir, "router")
    extractor_train_source_diagnostics = _existing_train_source_diagnostics(prepared_dir, "extractor")
    pair_train_source_diagnostics = _existing_train_source_diagnostics(prepared_dir, "pair")
    pair_hard_negative_augmentation: dict[str, int] = {}
    try:
        if needs_router:
            print("Preparing router dataset (missing-only mode)...")
            router_rows = _build_router_rows(
                local_rows=local_rows,
                registry=registry,
                prepare_cfg=prepare_cfg,
                synthetic_cfg=synthetic_cfg,
                single_pools=single_pools,
                llm=llm,
                task_labels=router_task_labels,
                regression_tasks=router_regression_tasks,
                existing_df=router_existing_df,
                use_multilingual=use_multilingual,
            )
            router_df = pd.DataFrame(_sanitize_row_dicts(router_rows))
            if router_df.empty:
                print("Router dataset is empty; cannot continue.", file=sys.stderr)
                return 1
            router_adversarial_tasks = set(_ADVERSARIAL_FIXTURE_PATHS.keys()) & set(router_task_labels.keys()) - {"schema_match_pair"}
            if router_adversarial_tasks:
                router_df = _inject_adversarial_into_router_df(
                    router_df,
                    seed=int(prepare_cfg["seed"]),
                    task_names=router_adversarial_tasks,
                )
            router_df = _apply_router_feature_enrichment(router_df)
            (
                router_splits,
                router_split_integrity,
                router_train_source_diagnostics,
            ) = _write_splits(
                router_df,
                prefix="router",
                out_dir=prepared_dir,
                seed=int(prepare_cfg["seed"]),
                ratios=ratios,
            )
        else:
            router_df = router_existing_df
            router_splits = _existing_split_counts(prepared_dir, "router")

        if needs_extractor:
            print("Preparing extractor dataset (missing-only mode)...")
            extractor_rows = _build_extractor_rows(
                registry=registry,
                prepare_cfg=prepare_cfg,
                synthetic_cfg=synthetic_cfg,
                single_pools=single_pools,
                llm=llm,
                existing_df=extractor_existing_df,
                use_multilingual=use_multilingual,
            )
            extractor_df = pd.DataFrame(_sanitize_row_dicts(extractor_rows))
            if extractor_df.empty:
                print("Extractor dataset is empty; cannot continue.", file=sys.stderr)
                return 1
            (
                extractor_splits,
                extractor_split_integrity,
                extractor_train_source_diagnostics,
            ) = _write_splits(
                extractor_df,
                prefix="extractor",
                out_dir=prepared_dir,
                seed=int(prepare_cfg["seed"]),
                ratios=ratios,
            )
        else:
            extractor_df = extractor_existing_df
            extractor_splits = _existing_split_counts(prepared_dir, "extractor")

        if needs_pair:
            print("Preparing pair dataset (missing-only mode)...")
            pair_rows = _build_pair_rows(
                registry=registry,
                prepare_cfg=prepare_cfg,
                synthetic_cfg=synthetic_cfg,
                single_pools=single_pools,
                pair_pool=pair_pool,
                llm=llm,
                task_labels=pair_task_labels,
                existing_df=pair_existing_df,
                use_multilingual=use_multilingual,
            )
            print(f"Building pair DataFrame ({len(pair_rows)} rows)...")
            pair_df = pd.DataFrame(_sanitize_row_dicts(pair_rows))
            if pair_df.empty:
                print("Pair dataset is empty; cannot continue.", file=sys.stderr)
                return 1
            pair_adversarial_tasks = set(_ADVERSARIAL_FIXTURE_PATHS.keys()) & set(pair_task_labels.keys()) & {"schema_match_pair"}
            if pair_adversarial_tasks:
                pair_df = _inject_adversarial_into_pair_df(
                    pair_df,
                    seed=int(prepare_cfg["seed"]),
                    task_names=pair_adversarial_tasks,
                )
            if embedding_pair_tasks:
                print("Writing pair embedding cache...")
                embedding_cache_summary = _write_pair_embedding_cache(
                    df=pair_df,
                    task_names=embedding_pair_tasks,
                    model_name=embedding_pair_model_name or DEFAULT_PAIR_EMBEDDING_MODEL_NAME,
                    out_dir=prepared_dir,
                )
                pair_df, pair_hard_negative_augmentation = _augment_pair_hard_negatives(
                    pair_df,
                    cache_path=pair_embedding_cache_path(prepared_dir),
                    task_names=embedding_pair_tasks & PAIR_RANKING_TASKS,
                )
            print("Writing pair splits...")
            (
                pair_splits,
                pair_split_integrity,
                pair_train_source_diagnostics,
            ) = _write_splits(
                pair_df,
                prefix="pair",
                out_dir=prepared_dir,
                seed=int(prepare_cfg["seed"]),
                ratios=ratios,
            )
        else:
            pair_df = pair_existing_df
            pair_splits = _existing_split_counts(prepared_dir, "pair")

        if embedding_pair_tasks and (
            (not needs_pair)
            and (embedding_cache_summary is None or embedding_cache_summary.get("rows", 0) <= 0)
        ):
            print("Writing pair embedding cache...")
            embedding_cache_summary = _write_pair_embedding_cache(
                df=pair_df,
                task_names=embedding_pair_tasks,
                model_name=embedding_pair_model_name or DEFAULT_PAIR_EMBEDDING_MODEL_NAME,
                out_dir=prepared_dir,
            )

        token_task_splits: dict[str, dict[str, int]] = {}
        token_task_dfs: dict[str, pd.DataFrame] = {}
        if needs_tokens:
            for task_name in sorted(token_tasks):
                print(f"Preparing token dataset [{task_name}] (missing-only mode)...")
                if task_name == "pii_span_detection":
                    token_rows = _build_pii_token_rows(
                        registry=registry,
                        token_cfg=token_prepare_cfg,
                        existing_df=token_existing_dfs.get(task_name),
                    )
                elif task_name == "fact_extraction_structured":
                    token_rows = _build_fact_token_rows(
                        single_pools=single_pools,
                        token_cfg=token_prepare_cfg,
                        seed=int(prepare_cfg["seed"]),
                        use_multilingual=use_multilingual,
                        existing_df=token_existing_dfs.get(task_name),
                    )
                else:
                    token_rows = list(
                        token_existing_dfs.get(task_name, pd.DataFrame()).to_dict("records")
                    )
                token_df = pd.DataFrame(_sanitize_row_dicts(token_rows))
                if token_df.empty:
                    print(f"Token dataset {task_name} is empty; cannot continue.", file=sys.stderr)
                    return 1
                token_task_dfs[task_name] = token_df
                token_task_splits[task_name] = _write_token_task_splits(
                    df=token_df,
                    task_name=task_name,
                    out_dir=prepared_dir,
                    seed=int(prepare_cfg["seed"]),
                    ratios=ratios,
                )
        else:
            for task_name in sorted(token_tasks):
                token_task_dfs[task_name] = token_existing_dfs.get(task_name, pd.DataFrame())
                token_task_splits[task_name] = _existing_token_split_counts(prepared_dir, task_name)

        manifest = {
            "config_path": str(args.config.resolve()),
            "seed": int(prepare_cfg["seed"]),
            "warnings": manifest_warnings,
            "incremental": {
                "mode": "missing_only",
                "force_full": bool(args.force_full),
                "missing_counts": {
                    "router": router_missing_total + router_regression_missing_total,
                    "extractor": extractor_missing_total,
                    "pair": pair_missing_total,
                    "token": token_missing_total,
                },
                "updated": {
                    "router": needs_router,
                    "extractor": needs_extractor,
                    "pair": needs_pair,
                    "token": needs_tokens,
                },
            },
            "paths": {
                "prepared_dir": str(prepared_dir),
                "bootstrap_prepared_dir": str(bootstrap_dir),
                "datasets_cache_dir": str(cache_dir) if cache_dir else "",
            },
            "prepare_settings": {
                k: prepare_cfg[k]
                for k in [
                    "max_rows_per_source",
                    "target_per_task_label",
                    "max_per_task_label",
                    "train_ratio",
                    "test_ratio",
                    "eval_ratio",
                    "auto_download_missing",
                ]
            }
            | {
                "llm_temperature": synthetic_cfg["temperature"],
                "token": token_prepare_cfg,
            },
            "synthetic_llm": (
                {
                    "enabled": True,
                    "provider": llm.provider,
                    "model": llm.model,
                    "base_url": llm.base_url,
                    "temperature": llm.temperature,
                    "top_p": llm.top_p,
                    "batch_size": llm.batch_size,
                    "concurrency": llm.concurrency,
                    "max_tokens": llm.max_tokens,
                    "max_attempts_per_label": int(synthetic_cfg["max_attempts_per_label"]),
                    "log_stats_every_seconds": float(synthetic_cfg["log_stats_every_seconds"]),
                    "log_zero_progress_every": int(synthetic_cfg["log_zero_progress_every"]),
                    "parse_failure_log_every": int(synthetic_cfg["parse_failure_log_every"]),
                    "log_request_failures": bool(synthetic_cfg["log_request_failures"]),
                    "local_raw_scan_rows_per_file": int(
                        synthetic_cfg["local_raw_scan_rows_per_file"]
                    ),
                }
                if llm is not None
                else {
                    "enabled": False,
                    "provider": synthetic_cfg.get("provider", ""),
                    "model": synthetic_cfg.get("model", ""),
                    "base_url": synthetic_cfg.get("base_url", ""),
                    "reason": "Synthetic LLM unavailable; deterministic/public data only.",
                }
            ),
            "datasets": registry.status,
            "configured_tasks": {
                "router": sorted(router_task_labels.keys()) + sorted(router_regression_tasks),
                "extractor": list(EXTRACTOR_TASK_LABELS.keys()),
                "pair": sorted(pair_task_labels.keys()),
            },
            "configured_token_tasks": sorted(token_tasks),
            "observed_tasks": {
                "router": sorted(router_df["task"].astype(str).unique().tolist()),
                "extractor": sorted(extractor_df["task"].astype(str).unique().tolist()),
                "pair": sorted(pair_df["task"].astype(str).unique().tolist()),
            },
            "observed_token_tasks": sorted(
                task_name for task_name, df in token_task_dfs.items() if not df.empty
            ),
            "router": {
                **_summary(router_df),
                "splits": router_splits,
                "split_integrity": router_split_integrity,
                "train_source_diagnostics": router_train_source_diagnostics,
                "tasks": sorted(router_task_labels.keys()) + sorted(router_regression_tasks),
                "updated": needs_router,
                "missing_before": router_missing
                | {
                    f"{task_name}::score_rows": count
                    for task_name, count in router_regression_missing.items()
                },
            },
            "extractor": {
                **_summary(extractor_df),
                "splits": extractor_splits,
                "split_integrity": extractor_split_integrity,
                "train_source_diagnostics": extractor_train_source_diagnostics,
                "tasks": list(EXTRACTOR_TASKS),
                "updated": needs_extractor,
                "missing_before": extractor_missing,
            },
            "pair": {
                **_summary(pair_df),
                "splits": pair_splits,
                "split_integrity": pair_split_integrity,
                "train_source_diagnostics": pair_train_source_diagnostics,
                "tasks": sorted(pair_task_labels.keys()),
                "updated": needs_pair,
                "missing_before": pair_missing,
            },
            "pair_embedding_cache": embedding_cache_summary,
            "pair_hard_negative_augmentation": pair_hard_negative_augmentation,
            "adversarial_fixtures": adversarial_fixture_summary,
            "token_task_splits": token_task_splits,
            "token_tasks": {
                task_name: {
                    **_token_summary(token_task_dfs.get(task_name, pd.DataFrame())),
                    "splits": token_task_splits.get(task_name, {}),
                    "updated": needs_tokens,
                    "missing_before": token_missing.get(task_name, 0),
                    "warnings": (
                        manifest_warnings if task_name == "fact_extraction_structured" else []
                    ),
                }
                for task_name in sorted(token_tasks)
            },
        }

        manifest_path = prepared_dir / "manifest.json"
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2)

        print("Preparation complete.")
        print(f"Prepared dir: {prepared_dir}")
        print(f"Manifest: {manifest_path}")
        return 0
    finally:
        if llm is not None:
            try:
                llm.maybe_report(context="final", force=True)
            except Exception:
                pass

            # Close client in a daemon thread so we never block on connection pool shutdown
            # (httpx.Client.close() can hang after many requests). Process exit will clean up.
            def _close_llm_client() -> None:
                try:
                    llm.client.close()
                except Exception:
                    pass

            t = threading.Thread(target=_close_llm_client, daemon=True)
            t.start()


def prepare_data(config: PrepareConfig) -> int:
    """Run preparation pipeline from a typed config."""
    argv: list[str] = ["--config", str(config.config_path)]
    if config.seed is not None:
        argv.extend(["--seed", str(config.seed)])
    if config.max_rows_per_source is not None:
        argv.extend(["--max-rows-per-source", str(config.max_rows_per_source)])
    if config.max_per_task_label is not None:
        argv.extend(["--max-per-task-label", str(config.max_per_task_label)])
    if config.target_per_task_label is not None:
        argv.extend(["--target-per-task-label", str(config.target_per_task_label)])
    if config.llm_temperature is not None:
        argv.extend(["--llm-temperature", str(config.llm_temperature)])
    if config.llm_concurrency is not None:
        argv.extend(["--llm-concurrency", str(config.llm_concurrency)])
    if config.disable_download:
        argv.append("--disable-download")
    if config.allow_missing_datasets_package:
        argv.append("--allow-missing-datasets-package")
    if config.force_full:
        argv.append("--force-full")
    if config.no_multilingual:
        argv.append("--no-multilingual")
    return main(argv)


if __name__ == "__main__":
    sys.exit(main())
