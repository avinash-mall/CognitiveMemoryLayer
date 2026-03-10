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
import json
import os
import random
import re
import sys
import threading
import time
import tomllib
import warnings
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, cast

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
    "novelty_pair": ["duplicate", "novel", "temporal_change", "contradiction"],
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
        limits = httpx.Limits(
            max_connections=max(64, self.concurrency * 8),
            max_keepalive_connections=max(32, self.concurrency * 4),
        )
        self.client = httpx.Client(timeout=self.timeout_seconds, limits=limits)

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

    def _request(self, system_prompt: str, user_prompt: str) -> str:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": self.temperature,
            "top_p": self.top_p,
            "max_tokens": self.max_tokens,
        }

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
                "Return STRICT JSON only, no markdown fences."
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
                "Return STRICT JSON only, no markdown fences."
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
                "contradiction" if is_conflict else "novel",
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


def _add_temporal_change_rows(
    rows: _PairTaskStore, *, target_per_task_label: int, rng: random.Random
) -> None:
    while rows.count("novelty_pair", "temporal_change") < target_per_task_label:
        idx = rows.count("novelty_pair", "temporal_change")
        topic = idx % 5
        if topic == 0:
            old = f"I live in City {idx}."
            new = f"I moved to City {idx + 1} last month."
        elif topic == 1:
            old = f"I work as Role {idx}."
            new = f"I changed jobs and now work as Role {idx + 1}."
        elif topic == 2:
            old = f"I wake up at {6 + (idx % 5)} AM every day."
            new = f"I changed my routine and now wake up at {7 + (idx % 5)} AM."
        elif topic == 3:
            old = f"My monthly budget is ${1000 + idx}."
            new = f"My monthly budget increased to ${1200 + idx} this quarter."
        else:
            old = f"I drink {1 + (idx % 2)} cups of coffee each morning."
            new = f"I stopped that habit and now drink tea instead as of week {idx}."
        rows.add("novelty_pair", old, new, "temporal_change", "template:temporal_change")


def _fill_pair_tasks_without_llm(
    rows: _PairTaskStore,
    *,
    task_labels: dict[str, list[str]],
    target_per_task_label: int,
    rng: random.Random,
) -> None:
    if "novelty_pair" in task_labels:
        _fill_template_novelty_rows(rows, target_per_task_label=target_per_task_label)
        _add_temporal_change_rows(rows, target_per_task_label=target_per_task_label, rng=rng)
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


def _add_fever_schema_match_rows(rows: _PairTaskStore, registry: _HFRegistry) -> None:
    ds = registry.get("fever")
    if ds is None:
        return
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
        rows.add("schema_match_pair", claim, evidence, schema_label, "hf:fever")
        rows.add(
            "reconsolidation_candidate_pair",
            claim,
            evidence,
            recon_label,
            "hf:fever",
            extras={"group_id": f"hf:fever:{idx}"},
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
) -> dict[str, Any]:
    extras: dict[str, Any] = {
        "memory_type": memory_type,
        "namespace": topic,
        "context_tags": [topic],
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
    return extras


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
    if "consolidation_gist_quality" in task_labels:
        while rows.count("consolidation_gist_quality", "accept") < target_per_task_label:
            idx = rows.count("consolidation_gist_quality", "accept")
            pack = _topic_pack(idx)
            seed = _seed_fragment(single_pools, rng=rng, idx=idx)
            text = (
                f"{pack['gist']} across {3 + (idx % 4)} related conversations; "
                f"anchor detail {idx}: {seed.lower()}."
            )
            rows.add(
                "consolidation_gist_quality",
                text,
                "accept",
                "template:consolidation_gist_quality:accept",
                extras=_structured_row_extras(
                    topic=pack["topic"],
                    memory_type="semantic_fact",
                    importance=0.74,
                    confidence=0.88,
                    access_count=4 + (idx % 4),
                    age_days=10 + (idx % 20),
                    dependency_count=1 + (idx % 3),
                    support_count=3 + (idx % 4),
                    mixed_topic=False,
                ),
            )
        while rows.count("consolidation_gist_quality", "reject") < target_per_task_label:
            idx = rows.count("consolidation_gist_quality", "reject")
            pack = _topic_pack(idx)
            text = (
                f"Various things happened in multiple conversations around {pack['topic']} "
                f"and some details changed later in note {idx}."
            )
            rows.add(
                "consolidation_gist_quality",
                text,
                "reject",
                "template:consolidation_gist_quality:reject",
                extras=_structured_row_extras(
                    topic=pack["topic"],
                    memory_type="episodic_event",
                    importance=0.28,
                    confidence=0.42,
                    access_count=0,
                    age_days=45 + (idx % 120),
                    dependency_count=0,
                    support_count=1 + (idx % 2),
                    mixed_topic=True,
                ),
            )

    if "forgetting_action_policy" in task_labels:
        policy_profiles: dict[str, dict[str, Any]] = {
            "keep": {
                "memory_type": "constraint",
                "importance": 0.95,
                "confidence": 0.94,
                "access_count": 9,
                "age_days": 2,
                "dependency_count": 3,
                "template": "Critical reminder {idx}: {relevant}; keep this active for all future decisions.",
            },
            "decay": {
                "memory_type": "episodic_event",
                "importance": 0.48,
                "confidence": 0.72,
                "access_count": 2,
                "age_days": 45,
                "dependency_count": 1,
                "template": "Routine note {idx}: {relevant}; it still matters a bit but should fade over time.",
            },
            "silence": {
                "memory_type": "episodic_event",
                "importance": 0.26,
                "confidence": 0.58,
                "access_count": 0,
                "age_days": 130,
                "dependency_count": 0,
                "template": "Low-signal observation {idx}: {relevant}; keep it stored but avoid surfacing it by default.",
            },
            "compress": {
                "memory_type": "semantic_fact",
                "importance": 0.39,
                "confidence": 0.81,
                "access_count": 1,
                "age_days": 95,
                "dependency_count": 5,
                "template": "Redundant cluster note {idx}: {relevant}; preserve the theme but compress the details.",
            },
            "delete": {
                "memory_type": "episodic_event",
                "importance": 0.08,
                "confidence": 0.34,
                "access_count": 0,
                "age_days": 365,
                "dependency_count": 0,
                "template": "Disposable scratch note {idx}: {relevant}; it is stale and no longer useful.",
            },
        }
        for label in task_labels["forgetting_action_policy"]:
            profile = policy_profiles.get(label)
            if profile is None:
                continue
            while rows.count("forgetting_action_policy", label) < target_per_task_label:
                idx = rows.count("forgetting_action_policy", label)
                pack = _topic_pack(idx)
                text = str(profile["template"]).format(idx=idx, relevant=pack["relevant"].lower())
                rows.add(
                    "forgetting_action_policy",
                    text,
                    label,
                    f"template:forgetting_action_policy:{label}",
                    extras=_structured_row_extras(
                        topic=pack["topic"],
                        memory_type=str(profile["memory_type"]),
                        importance=float(profile["importance"]),
                        confidence=float(profile["confidence"]),
                        access_count=int(profile["access_count"]) + (idx % 2),
                        age_days=int(profile["age_days"]) + (idx % 15),
                        dependency_count=int(profile["dependency_count"]) + (idx % 2),
                    ),
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
        elif task == "scope_match":
            mapped = "match" if label == "match" else "no_match"
            rows.add("schema_match_pair", text_a, text_b, mapped, f"derived:{source}")
        elif task == "conflict_detection" and label == "conflict":
            rows.add("novelty_pair", text_a, text_b, "contradiction", f"derived:{source}")
        elif task == "supersession" and label == "supersedes":
            rows.add("novelty_pair", text_a, text_b, "temporal_change", f"derived:{source}")


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
    while rows.count("schema_match_pair", "match") < target_per_task_label:
        idx = rows.count("schema_match_pair", "match")
        pack = _topic_pack(idx)
        rows.add(
            "schema_match_pair",
            f"{pack['gist']} Summary {idx}.",
            f"{pack['fact']} Fact {idx}.",
            "match",
            "template:schema_match_pair:match",
        )
    while rows.count("schema_match_pair", "no_match") < target_per_task_label:
        idx = rows.count("schema_match_pair", "no_match")
        pack = _topic_pack(idx)
        other = _other_topic_pack(idx)
        rows.add(
            "schema_match_pair",
            f"{pack['gist']} Summary {idx}.",
            f"{other['fact']} Fact {idx}.",
            "no_match",
            "template:schema_match_pair:no_match",
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
        )
    while rows.count("novelty_pair", "contradiction") < target_per_task_label:
        idx = rows.count("novelty_pair", "contradiction")
        pack = _topic_pack(idx)
        rows.add(
            "novelty_pair",
            f"{pack['relevant']} in preference record {idx}.",
            f"Preference record {idx} says the opposite of this: {pack['irrelevant'].lower()}.",
            "contradiction",
            "template:novelty_pair:contradiction",
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
            "temporal_change": "Second text updates the first with new temporal context.",
            "contradiction": "Second text contradicts the first.",
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
                    while missing > 0 and attempts < max_attempts_per_label:
                        rounds += 1
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
                                if use_multilingual
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
                                ):
                                    accepted += 1
                            if accepted > 0:
                                round_accepted += accepted
                                pbar.update(accepted)

                        llm.record_batch_result(generated=round_generated, accepted=round_accepted)
                        missing = max(0, target_per_task_label - rows.count(task, label))
                        if round_accepted <= 0:
                            attempts += 1
                        else:
                            attempts = 0

                        if round_generated == 0 and round_errors == 0 and label_batch_size > 1:
                            new_batch = max(1, label_batch_size // 2)
                            if new_batch != label_batch_size:
                                _warn(
                                    f"LLM reducing batch size for {task}::{label}: "
                                    f"{label_batch_size} -> {new_batch} (no parseable outputs)."
                                )
                            label_batch_size = new_batch
                        elif round_accepted > 0 and label_batch_size < llm.batch_size:
                            label_batch_size = min(llm.batch_size, label_batch_size * 2)

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
                                f"batch_size={label_batch_size}."
                            )
                        llm.maybe_report(context=f"{family}:{task}::{label}")

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
                    while missing > 0 and attempts < max_attempts_per_label:
                        rounds += 1
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
                                if use_multilingual
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
                                ):
                                    accepted += 1
                            if accepted > 0:
                                round_accepted += accepted
                                pbar.update(accepted)

                        llm.record_batch_result(generated=round_generated, accepted=round_accepted)
                        missing = max(0, target_per_task_label - rows.count(task, label))
                        if round_accepted <= 0:
                            attempts += 1
                        else:
                            attempts = 0

                        if round_generated == 0 and round_errors == 0 and label_batch_size > 1:
                            new_batch = max(1, label_batch_size // 2)
                            if new_batch != label_batch_size:
                                _warn(
                                    f"LLM reducing batch size for {task}::{label}: "
                                    f"{label_batch_size} -> {new_batch} (no parseable outputs)."
                                )
                            label_batch_size = new_batch
                        elif round_accepted > 0 and label_batch_size < llm.batch_size:
                            label_batch_size = min(llm.batch_size, label_batch_size * 2)

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
                                f"batch_size={label_batch_size}."
                            )
                        llm.maybe_report(context=f"pair:{task}::{label}")

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

    _fill_router_tasks_without_llm(
        rows=rows,
        regression_rows=regression_rows,
        task_labels=task_labels,
        regression_tasks=regression_tasks,
        target_per_task_label=target,
        single_pools=single_pools,
        rng=rng,
    )

    if llm is not None:
        _fill_single_task_with_llm(
            rows=rows,
            llm=llm,
            family="router",
            task_labels=task_labels,
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
    _add_fever_schema_match_rows(rows, registry)
    _fill_pair_tasks_without_llm(
        rows,
        task_labels=task_labels,
        target_per_task_label=target,
        rng=rng,
    )
    if llm is not None:
        _fill_pair_task_with_llm(
            rows=rows,
            llm=llm,
            task_labels=task_labels,
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


def _split_by_task_label(
    df: pd.DataFrame, seed: int, ratios: dict[str, float]
) -> dict[str, pd.DataFrame]:
    rng = random.Random(seed)
    train_parts: list[pd.DataFrame] = []
    test_parts: list[pd.DataFrame] = []
    eval_parts: list[pd.DataFrame] = []

    for (_task, _label), group in df.groupby(["task", "label"], sort=False):
        idx = list(group.index)
        rng.shuffle(idx)
        n_train, n_test, n_eval = _split_counts(len(idx), ratios)
        train_parts.append(df.loc[idx[:n_train]])
        test_parts.append(df.loc[idx[n_train : n_train + n_test]])
        eval_parts.append(df.loc[idx[n_train + n_test : n_train + n_test + n_eval]])

    out = {
        "train": pd.concat(train_parts, ignore_index=True) if train_parts else df.iloc[0:0].copy(),
        "test": pd.concat(test_parts, ignore_index=True) if test_parts else df.iloc[0:0].copy(),
        "eval": pd.concat(eval_parts, ignore_index=True) if eval_parts else df.iloc[0:0].copy(),
    }
    for split_name in out:
        out[split_name] = out[split_name].sample(frac=1, random_state=seed).reset_index(drop=True)
    return out


def _write_splits(
    df: pd.DataFrame, *, prefix: str, out_dir: Path, seed: int, ratios: dict[str, float]
) -> dict[str, int]:
    splits = _split_by_task_label(df, seed, ratios)
    counts: dict[str, int] = {}
    for split_name, split_df in splits.items():
        path = out_dir / f"{prefix}_{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        counts[split_name] = len(split_df)
    return counts


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
            synth = len(group[group["source"].str.startswith("llm:", na=False)])
            synthetic_ratio[key] = round(synth / max(1, total), 4)

    return {
        "rows": len(df),
        "task_counts": task_counts,
        "task_label_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in task_label_counts.items()},
        "source_top_20": source_top,
        "source_mix_per_task": source_mix,
        "synthetic_ratio_per_label": synthetic_ratio,
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
        print("Force-full mode: ignoring existing prepared splits.")
        router_existing_df = pd.DataFrame(columns=["text", "task", "label", "source"])
        extractor_existing_df = pd.DataFrame(columns=["text", "task", "label", "source"])
        pair_existing_df = pd.DataFrame(columns=["text_a", "text_b", "task", "label", "source"])
        token_existing_dfs = {
            task_name: pd.DataFrame(columns=["text", "task", "spans", "source", "language"])
            for task_name in TOKEN_TASKS
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

    if not (needs_router or needs_extractor or needs_pair or needs_tokens):
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
                "tasks": sorted(router_task_labels.keys()) + sorted(router_regression_tasks),
                "updated": False,
            },
            "extractor": {
                **_summary(extractor_existing_df),
                "splits": _existing_split_counts(prepared_dir, "extractor"),
                "tasks": list(EXTRACTOR_TASKS),
                "updated": False,
            },
            "pair": {
                **_summary(pair_existing_df),
                "splits": _existing_split_counts(prepared_dir, "pair"),
                "tasks": sorted(pair_task_labels.keys()),
                "updated": False,
            },
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
            router_splits = _write_splits(
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
            extractor_splits = _write_splits(
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
            print("Writing pair splits...")
            pair_splits = _write_splits(
                pair_df,
                prefix="pair",
                out_dir=prepared_dir,
                seed=int(prepare_cfg["seed"]),
                ratios=ratios,
            )
        else:
            pair_df = pair_existing_df
            pair_splits = _existing_split_counts(prepared_dir, "pair")

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
                "tasks": list(EXTRACTOR_TASKS),
                "updated": needs_extractor,
                "missing_before": extractor_missing,
            },
            "pair": {
                **_summary(pair_df),
                "splits": pair_splits,
                "tasks": sorted(pair_task_labels.keys()),
                "updated": needs_pair,
                "missing_before": pair_missing,
            },
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
