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
import random
import re
import sys
import tomllib
from collections import defaultdict
from pathlib import Path

import pandas as pd

try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency
    _tqdm = None

try:
    from datasets import load_dataset as _hf_load_dataset
except Exception:  # pragma: no cover - optional dependency
    _hf_load_dataset = None


MODELS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODELS_ROOT.parent.parent
DEFAULT_CONFIG_PATH = MODELS_ROOT / "model_pipeline.toml"

LOCAL_BOOTSTRAP_SPLITS = ("train", "test", "eval")
ROUTER_TASKS = (
    "memory_type",
    "query_intent",
    "constraint_dimension",
    "context_tag",
    "salience_bin",
    "importance_bin",
    "confidence_bin",
    "decay_profile",
)
EXTRACTOR_TASKS = ("constraint_type", "constraint_scope", "fact_type", "pii_presence")
PAIR_TASKS = ("conflict_detection", "constraint_rerank")

MEMORY_TYPES = {
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
}

CONTEXT_KEYWORDS = {
    "food": ["food", "meal", "restaurant", "cook", "diet", "kitchen", "eat", "drink"],
    "travel": ["travel", "trip", "flight", "hotel", "tour", "airport", "train"],
    "finance": ["bank", "account", "money", "transfer", "payment", "card", "balance", "bill"],
    "health": ["health", "doctor", "medicine", "sleep", "allergy", "exercise", "therapy"],
    "work": ["work", "job", "task", "project", "deadline", "meeting", "office", "deploy"],
    "tech": ["code", "api", "tool", "error", "debug", "server", "model", "data", "python"],
    "social": ["friend", "family", "partner", "wedding", "party", "chat", "conversation"],
}

MEMORY_DECAY_PROFILE = {
    "constraint": "very_slow",
    "semantic_fact": "slow",
    "preference": "slow",
    "procedure": "slow",
    "knowledge": "slow",
    "hypothesis": "medium",
    "observation": "medium",
    "tool_result": "medium",
    "reasoning_step": "medium",
    "plan": "medium",
    "task_state": "fast",
    "conversation": "fast",
    "message": "fast",
    "episodic_event": "fast",
    "scratch": "very_fast",
}

MEMORY_SALIENCE = {
    "constraint": "high",
    "preference": "high",
    "procedure": "high",
    "semantic_fact": "medium",
    "knowledge": "medium",
    "reasoning_step": "medium",
    "plan": "medium",
    "hypothesis": "medium",
    "observation": "medium",
    "tool_result": "medium",
    "episodic_event": "low",
    "conversation": "low",
    "message": "low",
    "task_state": "low",
    "scratch": "low",
}

UNCERTAINTY_RE = re.compile(r"\b(maybe|might|possibly|perhaps|uncertain|guess|i think)\b", re.IGNORECASE)
PII_RE = re.compile(
    r"("
    r"\b\d{3}-\d{2}-\d{4}\b|"
    r"\b(?:\+?\d{1,2}[\s-]?)?(?:\(?\d{3}\)?[\s-]?)\d{3}[\s-]?\d{4}\b|"
    r"[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}|"
    r"\b(?:\d[ -]*?){13,16}\b"
    r")",
    re.IGNORECASE,
)


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
    if "paths" not in cfg or "prepare" not in cfg:
        raise ValueError("Config must contain [paths] and [prepare] sections.")
    return cfg


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
        idx = int(value)
    except Exception:
        return _clean(value, 80)
    if 0 <= idx < len(names):
        return names[idx]
    return ""


def _pick_text(ex: dict, keys: tuple[str, ...], *, limit: int = 1000) -> str:
    for key in keys:
        if key in ex:
            text = _clean(ex.get(key, ""), limit)
            if text:
                return text
    return ""


def _infer_hf_dataset_id(link: str) -> str:
    marker = "huggingface.co/datasets/"
    if marker not in link:
        return ""
    tail = link.split(marker, 1)[1].strip().strip("/")
    if not tail:
        return ""
    return tail.split("?", 1)[0].strip("/")


def _heuristic_query_intent(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(should|must|policy|allowed|safe|risk|can i)\b", t):
        return "constraint_check"
    if re.search(r"\b(error|traceback|exception|stack|api|command|tool|execute|run)\b", t):
        return "tool_query"
    if re.search(r"\b(plan|next step|roadmap|todo|strategy)\b", t):
        return "planning"
    if "?" in t or re.search(r"\b(what|when|where|who|why|how)\b", t):
        return "factual"
    return "conversation"


def _heuristic_constraint_dimension(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(should|must|always|never|allowed|forbidden|policy)\b", t):
        return "policy"
    if re.search(r"\b(goal|objective|plan|trying to|intend|aim)\b", t):
        return "goal"
    if re.search(r"\b(value|principle|important to me|believe)\b", t):
        return "value"
    if re.search(r"\b(because|therefore|so that|as a result)\b", t):
        return "causal"
    if re.search(r"\b(feel|stressed|tired|healthy|state)\b", t):
        return "state"
    return "other"


def _heuristic_context_tag(text: str, default_tag: str = "general") -> str:
    t = text.lower()
    for tag, words in CONTEXT_KEYWORDS.items():
        for word in words:
            if word in t:
                return tag
    return default_tag


def _heuristic_constraint_type(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(must|never|always|required|forbidden|policy|do not|don't)\b", t):
        return "policy"
    if re.search(r"\b(goal|plan to|trying to|intend|objective|aim)\b", t):
        return "goal"
    if re.search(r"\b(value|important to me|believe in|principle)\b", t):
        return "value"
    if re.search(r"\b(because|therefore|so that|as a result)\b", t):
        return "causal"
    if re.search(r"\b(feel|state|status|stressed|tired|healthy)\b", t):
        return "state"
    if re.search(r"\b(prefer|favorite|like|love|dislike|hate)\b", t):
        return "preference"
    return "constraint_other"


def _heuristic_fact_type(text: str) -> str:
    t = text.lower()
    if re.search(r"\b(prefer|favorite|like|love|enjoy|dislike|hate)\b", t):
        return "preference"
    if re.search(r"\b(my name is|i am|i'm)\b", t):
        return "identity"
    if re.search(r"\b(live in|from|located in|city|country)\b", t):
        return "location"
    if re.search(r"\b(work as|job|profession|engineer|teacher|manager|lawyer)\b", t):
        return "occupation"
    return "other_fact"


def _heuristic_confidence_bin(text: str, memory_type: str) -> str:
    if UNCERTAINTY_RE.search(text):
        return "low"
    if memory_type in {"semantic_fact", "knowledge", "procedure"}:
        return "high"
    return "medium"


class _SingleTaskStore:
    def __init__(self, max_per_task_label: int) -> None:
        self.max_per_task_label = max_per_task_label
        self.rows: list[dict] = []
        self._seen: set[tuple[str, str, str]] = set()
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def add(self, task: str, text: object, label: str, source: str) -> bool:
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
        self.rows.append({"text": cleaned, "task": task, "label": label, "source": source})
        return True


class _PairTaskStore:
    def __init__(self, max_per_task_label: int) -> None:
        self.max_per_task_label = max_per_task_label
        self.rows: list[dict] = []
        self._seen: set[tuple[str, str, str, str]] = set()
        self._counts: dict[tuple[str, str], int] = defaultdict(int)

    def add(self, task: str, text_a: object, text_b: object, label: str, source: str) -> bool:
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
        self.rows.append({"text_a": a, "text_b": b, "task": task, "label": label, "source": source})
        return True


class _HFRegistry:
    def __init__(self, *, datasets_cfg: list[dict], cache_dir: Path | None, prepare_cfg: dict) -> None:
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
        max_rows = max(1, int(dcfg.get("max_rows", self.prepare_cfg["max_rows_per_source"])))
        if "[" in split and "]" in split:
            return split
        return f"{split}[:{max_rows}]"

    def limit(self, name: str) -> int:
        dcfg = self.datasets_cfg[name]
        return max(1, int(dcfg.get("max_rows", self.prepare_cfg["max_rows_per_source"])))

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
                self.status[name]["rows"] = int(getattr(ds, "num_rows"))
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


def _iter_dataset_rows(dataset: object, *, limit: int, desc: str):
    total = limit
    if hasattr(dataset, "num_rows"):
        try:
            total = min(limit, int(dataset.num_rows))  # type: ignore[attr-defined]
        except Exception:
            total = limit
    pbar = _progress(total=total, desc=desc, unit="row")
    count = 0
    try:
        for row in dataset:  # type: ignore[operator]
            if count >= limit:
                break
            yield row
            count += 1
            pbar.update(1)
    finally:
        pbar.close()


def _load_local_bootstrap_rows(bootstrap_dir: Path, max_rows_per_source: int, seed: int) -> list[dict]:
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


def _build_router_rows(local_rows: list[dict], registry: _HFRegistry, prepare_cfg: dict) -> list[dict]:
    rows = _SingleTaskStore(int(prepare_cfg["max_per_task_label"]))

    for item in local_rows:
        text = item["text"]
        memory_type = item["label"]
        source = item["source"]
        if memory_type not in MEMORY_TYPES:
            continue
        rows.add("memory_type", text, memory_type, source)
        rows.add("query_intent", text, _heuristic_query_intent(text), source)
        rows.add("constraint_dimension", text, _heuristic_constraint_dimension(text), source)
        rows.add("context_tag", text, _heuristic_context_tag(text), source)
        salience = MEMORY_SALIENCE.get(memory_type, "medium")
        rows.add("salience_bin", text, salience, source)
        rows.add("importance_bin", text, salience, source)
        rows.add("confidence_bin", text, _heuristic_confidence_bin(text, memory_type), source)
        rows.add("decay_profile", text, MEMORY_DECAY_PROFILE.get(memory_type, "medium"), source)

    dataset_sequence = ["banking77", "trec", "massive"]
    pbar = _progress(total=len(dataset_sequence), desc="Router remote datasets", unit="dataset")
    for dataset_name in dataset_sequence:
        pbar.set_description(f"Router remote [{dataset_name}]")
        ds = registry.get(dataset_name)
        if ds is None:
            pbar.update(1)
            continue

        if dataset_name == "banking77":
            label_names = _safe_feature_names(ds, "label")
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Router rows [banking77]"):
                text = _clean(ex.get("text", ""))
                if not text:
                    continue
                label_text = _resolve_class_name(ex.get("label", ""), label_names).replace("_", " ")
                rows.add("query_intent", text, _heuristic_query_intent(text), "hf:banking77")
                rows.add("context_tag", text, _heuristic_context_tag(text, "finance"), "hf:banking77")
                rows.add(
                    "constraint_dimension",
                    text,
                    _heuristic_constraint_dimension(f"{text} {label_text}"),
                    "hf:banking77",
                )

        elif dataset_name == "trec":
            label_names = _safe_feature_names(ds, "label")
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Router rows [trec]"):
                text = _pick_text(ex, ("text", "question", "query", "sentence", "content", "title"))
                if not text:
                    continue
                label_hint = _resolve_class_name(ex.get("label", ""), label_names).replace("_", " ")
                rows.add("query_intent", text, "factual", "hf:trec")
                rows.add("context_tag", text, _heuristic_context_tag(f"{text} {label_hint}".strip(), "general"), "hf:trec")
                rows.add("constraint_dimension", text, "other", "hf:trec")

        elif dataset_name == "massive":
            scenario_names = _safe_feature_names(ds, "scenario")
            intent_names = _safe_feature_names(ds, "intent")
            label_names = _safe_feature_names(ds, "label")
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Router rows [massive]"):
                text = _pick_text(ex, ("utt", "text", "content", "title", "query", "sentence", "question"))
                if not text:
                    continue
                scenario = _resolve_class_name(ex.get("scenario", ""), scenario_names).replace("_", " ")
                intent = _resolve_class_name(ex.get("intent", ""), intent_names).replace("_", " ")
                label_hint = _resolve_class_name(ex.get("label", ""), label_names).replace("_", " ")
                hint = " ".join(x for x in (scenario, intent, label_hint) if x).strip()
                rows.add("query_intent", text, _heuristic_query_intent(text), "hf:massive")
                rows.add(
                    "context_tag",
                    text,
                    _heuristic_context_tag(f"{hint} {text}".strip(), "general"),
                    "hf:massive",
                )
                rows.add(
                    "constraint_dimension",
                    text,
                    _heuristic_constraint_dimension(f"{text} {hint}".strip()),
                    "hf:massive",
                )
        pbar.update(1)
    pbar.close()

    return rows.rows


def _build_extractor_rows(local_rows: list[dict], registry: _HFRegistry, prepare_cfg: dict) -> list[dict]:
    rows = _SingleTaskStore(int(prepare_cfg["max_per_task_label"]))

    for item in local_rows:
        text = item["text"]
        memory_type = item["label"]
        source = item["source"]
        if memory_type == "constraint":
            rows.add("constraint_type", text, _heuristic_constraint_type(text), source)
            rows.add("constraint_scope", text, _heuristic_context_tag(text), source)
        else:
            rows.add("constraint_type", text, "none", source)
            rows.add("constraint_scope", text, "none", source)

        if memory_type in {"semantic_fact", "preference", "knowledge"}:
            rows.add("fact_type", text, _heuristic_fact_type(text), source)
        else:
            rows.add("fact_type", text, "none", source)

        rows.add("pii_presence", text, "pii" if PII_RE.search(text) else "no_pii", source)

    dataset_sequence = ["moral_stories", "pii_masking"]
    pbar = _progress(total=len(dataset_sequence), desc="Extractor remote datasets", unit="dataset")
    for dataset_name in dataset_sequence:
        pbar.set_description(f"Extractor remote [{dataset_name}]")
        ds = registry.get(dataset_name)
        if ds is None:
            pbar.update(1)
            continue

        if dataset_name == "moral_stories":
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Extractor rows [moral_stories]"):
                norm = _clean(ex.get("norm", ""))
                moral_action = _clean(ex.get("moral_action", ""))
                immoral_action = _clean(ex.get("immoral_action", ""))
                situation = _clean(ex.get("situation", ""))
                generic_text = _pick_text(ex, ("text", "content", "title", "sentence", "query", "premise", "hypothesis"))

                if norm:
                    rows.add("constraint_type", norm, "policy", "hf:moral_stories")
                    rows.add("constraint_scope", norm, _heuristic_context_tag(norm), "hf:moral_stories")
                if moral_action:
                    rows.add("constraint_type", moral_action, _heuristic_constraint_type(moral_action), "hf:moral_stories")
                    rows.add("constraint_scope", moral_action, _heuristic_context_tag(moral_action), "hf:moral_stories")
                if immoral_action:
                    rows.add("constraint_type", immoral_action, _heuristic_constraint_type(immoral_action), "hf:moral_stories")
                    rows.add("constraint_scope", immoral_action, _heuristic_context_tag(immoral_action), "hf:moral_stories")
                if situation:
                    rows.add("constraint_type", situation, "none", "hf:moral_stories")

                if not any((norm, moral_action, immoral_action, situation)) and generic_text:
                    rows.add("constraint_type", generic_text, _heuristic_constraint_type(generic_text), "hf:moral_stories")
                    rows.add("constraint_scope", generic_text, _heuristic_context_tag(generic_text), "hf:moral_stories")
                    rows.add("fact_type", generic_text, _heuristic_fact_type(generic_text), "hf:moral_stories")
                    rows.add("pii_presence", generic_text, "pii" if PII_RE.search(generic_text) else "no_pii", "hf:moral_stories")

        elif dataset_name == "pii_masking":
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Extractor rows [pii_masking]"):
                src = _clean(ex.get("source_text", ex.get("source", "")))
                tgt = _clean(ex.get("target_text", ex.get("target", "")))
                if src:
                    rows.add("pii_presence", src, "pii", "hf:pii_masking")
                if tgt:
                    rows.add("pii_presence", tgt, "no_pii", "hf:pii_masking")
        pbar.update(1)
    pbar.close()

    return rows.rows


def _is_conflict_label(raw_label: object, label_names: list[str]) -> bool:
    if label_names:
        name = _resolve_class_name(raw_label, label_names).lower()
        if name:
            return "contradiction" in name or "conflict" in name
    try:
        return int(raw_label) == 2
    except Exception:
        text = _clean(raw_label, 80).lower()
        return "contradiction" in text or "conflict" in text


def _negate_text(text: str) -> str:
    cleaned = _clean(text, 700)
    if not cleaned:
        return ""
    if re.search(r"\bnot\b", cleaned, re.IGNORECASE):
        return re.sub(r"\bnot\b\s*", "", cleaned, count=1, flags=re.IGNORECASE).strip()
    injected = re.sub(
        r"\b(is|are|was|were|do|does|did|can|could|will|would|should|must|have|has|had)\b",
        r"\1 not",
        cleaned,
        count=1,
        flags=re.IGNORECASE,
    )
    if injected != cleaned:
        return injected
    return f"Not true: {cleaned}"


def _add_pair_local_fallback(
    rows: _PairTaskStore,
    local_rows: list[dict],
    seed: int,
    max_rows: int,
    *,
    tasks: set[str] | None = None,
) -> None:
    if not local_rows:
        return
    rng = random.Random(seed)
    pool = [item["text"] for item in local_rows if _clean(item.get("text"))]
    if not pool:
        return

    use_conflict = tasks is None or "conflict_detection" in tasks
    use_rerank = tasks is None or "constraint_rerank" in tasks
    if not use_conflict and not use_rerank:
        return

    n = min(len(local_rows), max_rows)
    pbar = _progress(total=n, desc="Pair rows [local-fallback]", unit="row")
    try:
        for item in local_rows[:n]:
            text = item["text"]
            source = item["source"]

            if use_conflict:
                negated = _negate_text(text)
                if negated:
                    rows.add("conflict_detection", text, negated, "conflict", f"{source}:fallback")
                rows.add("conflict_detection", text, text, "no_conflict", f"{source}:fallback")

            if use_rerank:
                rows.add("constraint_rerank", text, text, "relevant", f"{source}:fallback")
                other = rng.choice(pool)
                if _norm_key(other) == _norm_key(text):
                    other = f"{other} (different context)"
                rows.add("constraint_rerank", text, other, "not_relevant", f"{source}:fallback")
            pbar.update(1)
    finally:
        pbar.close()


def _build_pair_rows(
    local_rows: list[dict],
    registry: _HFRegistry,
    prepare_cfg: dict,
) -> list[dict]:
    rows = _PairTaskStore(int(prepare_cfg["max_per_task_label"]))

    dataset_sequence = ["snli", "multi_nli", "ms_marco", "quora_duplicates"]
    pbar = _progress(total=len(dataset_sequence), desc="Pair remote datasets", unit="dataset")
    for dataset_name in dataset_sequence:
        pbar.set_description(f"Pair remote [{dataset_name}]")
        ds = registry.get(dataset_name)
        if ds is None:
            pbar.update(1)
            continue

        if dataset_name in {"snli", "multi_nli"}:
            label_names = _safe_feature_names(ds, "label")
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc=f"Pair rows [{dataset_name}]"):
                label = "conflict" if _is_conflict_label(ex.get("label", -1), label_names) else "no_conflict"
                rows.add(
                    "conflict_detection",
                    ex.get("premise", ""),
                    ex.get("hypothesis", ""),
                    label,
                    f"hf:{dataset_name}",
                )

        elif dataset_name == "ms_marco":
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Pair rows [ms_marco]"):
                query = _clean(ex.get("query", ""))
                passages = ex.get("passages", {})
                if not query or not isinstance(passages, dict):
                    continue
                selected = passages.get("is_selected", [])
                ptexts = passages.get("passage_text", [])
                if not isinstance(selected, list) or not isinstance(ptexts, list):
                    continue

                positive = ""
                negative = ""
                for flag, ptxt in zip(selected, ptexts):
                    passage = _clean(ptxt, 700)
                    try:
                        f = int(flag)
                    except Exception:
                        f = 0
                    if f == 1 and not positive:
                        positive = passage
                    if f == 0 and not negative:
                        negative = passage
                    if positive and negative:
                        break
                if positive:
                    rows.add("constraint_rerank", query, positive, "relevant", "hf:ms_marco")
                if negative:
                    rows.add("constraint_rerank", query, negative, "not_relevant", "hf:ms_marco")

        elif dataset_name == "quora_duplicates":
            for ex in _iter_dataset_rows(ds, limit=registry.limit(dataset_name), desc="Pair rows [quora_duplicates]"):
                raw = ex.get("label", 0)
                try:
                    rel = int(raw) == 1
                except Exception:
                    rel = _clean(raw, 40).lower() in {"1", "true", "duplicate", "relevant"}
                rows.add(
                    "constraint_rerank",
                    ex.get("sentence1", ""),
                    ex.get("sentence2", ""),
                    "relevant" if rel else "not_relevant",
                    "hf:quora_duplicates",
                )
        pbar.update(1)
    pbar.close()

    allow_fallback = bool(prepare_cfg.get("allow_local_pair_fallback", True))
    if allow_fallback and not rows.rows:
        _warn("No remote pair data loaded; using local weak-supervision fallback.")
        _add_pair_local_fallback(rows, local_rows, int(prepare_cfg["seed"]), int(prepare_cfg["max_rows_per_source"]))
    elif allow_fallback:
        present = {str(r.get("task", "")) for r in rows.rows}
        missing = set(PAIR_TASKS) - present
        if missing:
            _warn(f"Pair tasks missing from remote datasets ({sorted(missing)}); backfilling from local data.")
            _add_pair_local_fallback(
                rows,
                local_rows,
                int(prepare_cfg["seed"]),
                int(prepare_cfg["max_rows_per_source"]),
                tasks=missing,
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


def _split_by_task_label(df: pd.DataFrame, seed: int, ratios: dict[str, float]) -> dict[str, pd.DataFrame]:
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


def _write_splits(df: pd.DataFrame, *, prefix: str, out_dir: Path, seed: int, ratios: dict[str, float]) -> dict[str, int]:
    splits = _split_by_task_label(df, seed, ratios)
    counts: dict[str, int] = {}
    for split_name, split_df in splits.items():
        path = out_dir / f"{prefix}_{split_name}.parquet"
        split_df.to_parquet(path, index=False)
        counts[split_name] = int(len(split_df))
    return counts


def _summary(df: pd.DataFrame) -> dict:
    task_counts = df["task"].value_counts().to_dict() if "task" in df.columns else {}
    source_top = df["source"].value_counts().head(20).to_dict() if "source" in df.columns else {}
    task_label_counts = (
        df.groupby(["task", "label"]).size().sort_values(ascending=False).to_dict()
        if {"task", "label"}.issubset(df.columns)
        else {}
    )
    return {
        "rows": int(len(df)),
        "task_counts": task_counts,
        "task_label_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in task_label_counts.items()},
        "source_top_20": source_top,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for all custom model families.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model_pipeline.toml",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override prepare.seed from config.")
    parser.add_argument("--max-rows-per-source", type=int, default=None, help="Override prepare.max_rows_per_source.")
    parser.add_argument("--max-per-task-label", type=int, default=None, help="Override prepare.max_per_task_label.")
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = _load_config(args.config)

    paths_cfg = dict(config.get("paths", {}))
    prepare_cfg = dict(config.get("prepare", {}))
    datasets_cfg = list(config.get("datasets", []))

    if args.seed is not None:
        prepare_cfg["seed"] = int(args.seed)
    if args.max_rows_per_source is not None:
        prepare_cfg["max_rows_per_source"] = int(args.max_rows_per_source)
    if args.max_per_task_label is not None:
        prepare_cfg["max_per_task_label"] = int(args.max_per_task_label)
    if args.allow_missing_datasets_package:
        prepare_cfg["require_datasets_package"] = False

    prepare_cfg.setdefault("seed", 42)
    prepare_cfg.setdefault("train_ratio", 0.8)
    prepare_cfg.setdefault("test_ratio", 0.1)
    prepare_cfg.setdefault("eval_ratio", 0.1)
    prepare_cfg.setdefault("max_rows_per_source", 60000)
    prepare_cfg.setdefault("max_per_task_label", 50000)
    prepare_cfg.setdefault("auto_download_missing", True)
    prepare_cfg.setdefault("allow_local_pair_fallback", True)
    prepare_cfg.setdefault("require_datasets_package", True)

    if int(prepare_cfg["max_rows_per_source"]) <= 0:
        print("prepare.max_rows_per_source must be > 0.", file=sys.stderr)
        return 1
    if int(prepare_cfg["max_per_task_label"]) <= 0:
        print("prepare.max_per_task_label must be > 0.", file=sys.stderr)
        return 1

    prepared_dir = _resolve_path(str(paths_cfg.get("prepared_dir", "packages/models/prepared_data/modelpack")), base=REPO_ROOT)
    bootstrap_dir = _resolve_path(str(paths_cfg.get("bootstrap_prepared_dir", "packages/models/prepared_data")), base=REPO_ROOT)
    cache_dir_raw = str(paths_cfg.get("datasets_cache_dir", "packages/models/datasets")).strip()
    cache_dir = _resolve_path(cache_dir_raw, base=REPO_ROOT) if cache_dir_raw else None

    if not bootstrap_dir.exists():
        alt_bootstrap = (REPO_ROOT / "packages/models/prepared_data").resolve()
        if alt_bootstrap.exists():
            _warn(f"bootstrap_prepared_dir not found ({bootstrap_dir}); using {alt_bootstrap} instead.")
            bootstrap_dir = alt_bootstrap

    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
    prepared_dir.mkdir(parents=True, exist_ok=True)

    has_enabled_remote = any(bool(d.get("enabled", True)) for d in datasets_cfg)
    if has_enabled_remote and _hf_load_dataset is None and bool(prepare_cfg["require_datasets_package"]):
        print(
            "Missing dependency: python package `datasets` is required by config. "
            "Install with `pip install datasets` or run with --allow-missing-datasets-package.",
            file=sys.stderr,
        )
        return 1

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

    registry = _HFRegistry(datasets_cfg=datasets_cfg, cache_dir=cache_dir, prepare_cfg=prepare_cfg)
    if bool(prepare_cfg["auto_download_missing"]) and not args.disable_download:
        try:
            registry.ensure_required()
        except Exception as exc:
            print(f"Failed to download/load required dataset: {exc}", file=sys.stderr)
            return 1

    print("Loading local bootstrap rows...")
    local_rows = _load_local_bootstrap_rows(
        bootstrap_dir=bootstrap_dir,
        max_rows_per_source=int(prepare_cfg["max_rows_per_source"]),
        seed=int(prepare_cfg["seed"]),
    )
    print(f"Local bootstrap rows: {len(local_rows)}")

    print("Preparing router dataset...")
    router_rows = _build_router_rows(local_rows, registry, prepare_cfg)
    router_df = pd.DataFrame(router_rows)
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

    print("Preparing extractor dataset...")
    extractor_rows = _build_extractor_rows(local_rows, registry, prepare_cfg)
    extractor_df = pd.DataFrame(extractor_rows)
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

    print("Preparing pair dataset...")
    pair_rows = _build_pair_rows(local_rows, registry, prepare_cfg)
    pair_df = pd.DataFrame(pair_rows)
    if pair_df.empty:
        print("Pair dataset is empty; cannot continue.", file=sys.stderr)
        return 1
    pair_splits = _write_splits(
        pair_df,
        prefix="pair",
        out_dir=prepared_dir,
        seed=int(prepare_cfg["seed"]),
        ratios=ratios,
    )

    manifest = {
        "config_path": str(args.config.resolve()),
        "seed": int(prepare_cfg["seed"]),
        "paths": {
            "prepared_dir": str(prepared_dir),
            "bootstrap_prepared_dir": str(bootstrap_dir),
            "datasets_cache_dir": str(cache_dir) if cache_dir else "",
        },
        "prepare_settings": {
            k: prepare_cfg[k]
            for k in [
                "max_rows_per_source",
                "max_per_task_label",
                "train_ratio",
                "test_ratio",
                "eval_ratio",
                "auto_download_missing",
                "allow_local_pair_fallback",
            ]
        },
        "datasets": registry.status,
        "router": {**_summary(router_df), "splits": router_splits, "tasks": list(ROUTER_TASKS)},
        "extractor": {**_summary(extractor_df), "splits": extractor_splits, "tasks": list(EXTRACTOR_TASKS)},
        "pair": {**_summary(pair_df), "splits": pair_splits, "tasks": list(PAIR_TASKS)},
    }

    manifest_path = prepared_dir / "manifest.json"
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    print("Preparation complete.")
    print(f"Prepared dir: {prepared_dir}")
    print(f"Manifest: {manifest_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
