"""
Unified preparation script for all custom model families.

Families prepared:
1) router      (single-text multi-task)
2) extractor   (single-text multi-task)
3) pair        (text-pair multi-task)

Dataset size is controlled by a single config variable: samples_per_task_label
(in model_pipeline.toml [prepare]). target_per_task_label, max_per_task_label,
and max_rows_per_source are derived from it when not set explicitly. Optional
CLI overrides exist for fine-grained control.

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
import time
import tomllib
from collections import defaultdict
from pathlib import Path

import pandas as pd

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


MODELS_ROOT = Path(__file__).resolve().parent.parent
REPO_ROOT = MODELS_ROOT.parent.parent
DEFAULT_CONFIG_PATH = MODELS_ROOT / "model_pipeline.toml"
# Used when deriving max_rows_per_source from samples_per_task_label (router has 55 (task,label) pairs).
_MAX_TASK_LABEL_PAIRS = 60


def _derive_prepare_limits_from_samples_per_task_label(
    samples_per_task_label: int,
) -> dict[str, int]:
    """Derive target_per_task_label, max_per_task_label, max_rows_per_source from the single knob."""
    n = int(samples_per_task_label)
    return {
        "target_per_task_label": n,
        "max_per_task_label": max(n, n * 5),
        "max_rows_per_source": n * _MAX_TASK_LABEL_PAIRS,
    }


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

    def count(self, task: str, label: str) -> int:
        return int(self._counts.get((task, label), 0))

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


def _seed_single_store_from_df(rows: _SingleTaskStore, df: pd.DataFrame) -> None:
    if df.empty:
        return
    for item in df.itertuples(index=False):
        rows.add(
            str(getattr(item, "task", "")),
            getattr(item, "text", ""),
            str(getattr(item, "label", "")),
            str(getattr(item, "source", "prepared:existing")),
        )


def _seed_pair_store_from_df(rows: _PairTaskStore, df: pd.DataFrame) -> None:
    if df.empty:
        return
    for item in df.itertuples(index=False):
        rows.add(
            str(getattr(item, "task", "")),
            getattr(item, "text_a", ""),
            getattr(item, "text_b", ""),
            str(getattr(item, "label", "")),
            str(getattr(item, "source", "prepared:existing")),
        )


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
        self.timeout_seconds = max(10, int(cfg.get("timeout_seconds", 120)))
        self.max_retries = max(1, int(cfg.get("max_retries", 3)))

        if not self.model:
            raise ValueError("LLM model missing. Set LLM_EVAL__MODEL or synthetic_llm.model.")
        if not self.base_url:
            raise ValueError(
                "LLM base URL missing. Set LLM_EVAL__BASE_URL or synthetic_llm.base_url."
            )

        if self.provider and self.provider != "ollama":
            _warn(f"LLM provider is `{self.provider}`. Expected `ollama`.")

        self.url = self.base_url.rstrip("/") + "/chat/completions"
        self.client = httpx.Client(timeout=self.timeout_seconds)

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
                content = data["choices"][0]["message"]["content"]
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("Empty LLM response content.")
                return content
            except Exception as exc:
                last_err = str(exc)
                if attempt >= self.max_retries:
                    break
                time.sleep(min(6, attempt * 1.5))
        raise RuntimeError(f"LLM request failed after {self.max_retries} attempts: {last_err}")

    def generate_single(self, *, task: str, label: str, seed_text: str, n: int) -> list[str]:
        system = "Generate synthetic classification data. Return STRICT JSON only."
        user = (
            f"Task: {task}\nTarget label: {label}\nCount: {n}\n"
            f"Seed example from related dataset (do not copy): {seed_text}\n\n"
            'Return exactly: {"samples":[{"text":"..."}]}\n'
            "Each sample must match target label exactly and be diverse."
        )
        payload = _parse_json_content(self._request(system, user))
        if not payload:
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
        self, *, task: str, label: str, seed_a: str, seed_b: str, n: int
    ) -> list[tuple[str, str]]:
        system = "Generate synthetic text-pair classification data. Return STRICT JSON only."
        user = (
            f"Task: {task}\nTarget label: {label}\nCount: {n}\n"
            f"Seed pair from related dataset (do not copy): A={seed_a} | B={seed_b}\n\n"
            'Return exactly: {"samples":[{"text_a":"...","text_b":"..."}]}\n'
            "Every pair must match target label exactly and be diverse."
        )
        payload = _parse_json_content(self._request(system, user))
        if not payload:
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

    ds_mm = registry.get("ms_marco")
    if ds_mm is not None:
        for ex in _iter_dataset_rows(
            ds_mm, limit=registry.limit("ms_marco"), desc="Existing rows [ms_marco]"
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
            if positive:
                rows.add("constraint_rerank", query, positive, "relevant", "hf:ms_marco")
                rows.add("scope_match", query, positive, "match", "hf:ms_marco")
            if negative:
                rows.add("constraint_rerank", query, negative, "not_relevant", "hf:ms_marco")
                rows.add("scope_match", query, negative, "no_match", "hf:ms_marco")

    ds_quora = registry.get("quora_duplicates")
    if ds_quora is not None:
        for ex in _iter_dataset_rows(
            ds_quora,
            limit=registry.limit("quora_duplicates"),
            desc="Existing rows [quora_duplicates]",
        ):
            if not isinstance(ex, dict):
                continue
            raw = ex.get("label", 0)
            try:
                relevant = int(raw) == 1
            except Exception:
                relevant = _clean(raw, 40).lower() in {"1", "true", "duplicate", "relevant"}
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
) -> None:
    total_missing = 0
    for task, labels in task_labels.items():
        for label in labels:
            total_missing += max(0, target_per_task_label - rows.count(task, label))

    pbar = _progress(total=total_missing, desc=f"LLM synth [{family}]", unit="sample")
    try:
        for task, labels in task_labels.items():
            for label in labels:
                missing = max(0, target_per_task_label - rows.count(task, label))
                if missing <= 0:
                    continue

                attempts = 0
                while missing > 0 and attempts < max_attempts_per_label:
                    seed_text = _pick_seed_text(single_pools, family, rng)
                    guidance = _label_guidance(task, label)
                    batch = min(llm.batch_size, missing)
                    prompt_seed = (
                        f"{seed_text}\n\nLabel guidance: {guidance}" if guidance else seed_text
                    )

                    generated = llm.generate_single(
                        task=task, label=label, seed_text=prompt_seed, n=batch
                    )
                    if not generated:
                        attempts += 1
                        continue

                    accepted = 0
                    for text in generated:
                        if rows.add(task, text, label, f"llm:{task}:{label}"):
                            accepted += 1

                    if accepted == 0:
                        attempts += 1
                        continue

                    missing -= accepted
                    pbar.update(accepted)
                    attempts = 0

                if missing > 0:
                    _warn(
                        f"LLM underfilled {task}::{label}. "
                        f"missing={missing}, current={rows.count(task, label)}."
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
) -> None:
    total_missing = 0
    for task, labels in task_labels.items():
        for label in labels:
            total_missing += max(0, target_per_task_label - rows.count(task, label))

    pbar = _progress(total=total_missing, desc="LLM synth [pair]", unit="sample")
    try:
        for task, labels in task_labels.items():
            for label in labels:
                missing = max(0, target_per_task_label - rows.count(task, label))
                if missing <= 0:
                    continue

                attempts = 0
                while missing > 0 and attempts < max_attempts_per_label:
                    seed_a, seed_b = _pick_seed_pair(pair_pool, single_pools, rng)
                    guidance = _label_guidance(task, label)
                    batch = min(llm.batch_size, missing)
                    if guidance:
                        seed_a = f"{seed_a}\n\nLabel guidance: {guidance}"

                    generated = llm.generate_pair(
                        task=task, label=label, seed_a=seed_a, seed_b=seed_b, n=batch
                    )
                    if not generated:
                        attempts += 1
                        continue

                    accepted = 0
                    for text_a, text_b in generated:
                        if rows.add(task, text_a, text_b, label, f"llm:{task}:{label}"):
                            accepted += 1

                    if accepted == 0:
                        attempts += 1
                        continue

                    missing -= accepted
                    pbar.update(accepted)
                    attempts = 0

                if missing > 0:
                    _warn(
                        f"LLM underfilled {task}::{label}. "
                        f"missing={missing}, current={rows.count(task, label)}."
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
    llm: _LLMGenerator,
    existing_df: pd.DataFrame | None = None,
) -> list[dict]:
    cap = max(int(prepare_cfg["max_per_task_label"]), int(prepare_cfg["target_per_task_label"]))
    rows = _SingleTaskStore(cap)
    rng = random.Random(int(prepare_cfg["seed"]))
    if existing_df is not None and not existing_df.empty:
        _seed_single_store_from_df(rows, existing_df)

    target = int(prepare_cfg["target_per_task_label"])
    missing_total = sum(
        max(0, target - rows.count(task, label))
        for task, labels in ROUTER_TASK_LABELS.items()
        for label in labels
    )
    if missing_total <= 0:
        return rows.rows

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

    _fill_single_task_with_llm(
        rows=rows,
        llm=llm,
        family="router",
        task_labels=ROUTER_TASK_LABELS,
        target_per_task_label=target,
        single_pools=single_pools,
        rng=rng,
        max_attempts_per_label=int(synthetic_cfg.get("max_attempts_per_label", 80)),
    )
    return rows.rows


def _build_extractor_rows(
    *,
    registry: _HFRegistry,
    prepare_cfg: dict,
    synthetic_cfg: dict,
    single_pools: dict[str, list[str]],
    llm: _LLMGenerator,
    existing_df: pd.DataFrame | None = None,
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

    _fill_single_task_with_llm(
        rows=rows,
        llm=llm,
        family="extractor",
        task_labels=EXTRACTOR_TASK_LABELS,
        target_per_task_label=target,
        single_pools=single_pools,
        rng=rng,
        max_attempts_per_label=int(synthetic_cfg.get("max_attempts_per_label", 80)),
    )
    return rows.rows


def _build_pair_rows(
    *,
    registry: _HFRegistry,
    prepare_cfg: dict,
    synthetic_cfg: dict,
    single_pools: dict[str, list[str]],
    pair_pool: list[tuple[str, str]],
    llm: _LLMGenerator,
    existing_df: pd.DataFrame | None = None,
) -> list[dict]:
    cap = max(int(prepare_cfg["max_per_task_label"]), int(prepare_cfg["target_per_task_label"]))
    rows = _PairTaskStore(cap)
    rng = random.Random(int(prepare_cfg["seed"]) + 2)
    if existing_df is not None and not existing_df.empty:
        _seed_pair_store_from_df(rows, existing_df)

    target = int(prepare_cfg["target_per_task_label"])
    missing_total = sum(
        max(0, target - rows.count(task, label))
        for task, labels in PAIR_TASK_LABELS.items()
        for label in labels
    )
    if missing_total <= 0:
        return rows.rows

    _add_existing_pair_rows(rows, registry)
    _fill_pair_task_with_llm(
        rows=rows,
        llm=llm,
        task_labels=PAIR_TASK_LABELS,
        target_per_task_label=target,
        pair_pool=pair_pool,
        single_pools=single_pools,
        rng=rng,
        max_attempts_per_label=int(synthetic_cfg.get("max_attempts_per_label", 80)),
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
    return {
        "rows": len(df),
        "task_counts": task_counts,
        "task_label_counts": {f"{k[0]}::{k[1]}": int(v) for k, v in task_label_counts.items()},
        "source_top_20": source_top,
    }


def _load_existing_family_df(prepared_dir: Path, family: str) -> pd.DataFrame:
    if family in {"router", "extractor"}:
        required = ["text", "task", "label"]
        cols = ["text", "task", "label", "source"]
    elif family == "pair":
        required = ["text_a", "text_b", "task", "label"]
        cols = ["text_a", "text_b", "task", "label", "source"]
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
        parts.append(df[cols].copy())

    if not parts:
        return pd.DataFrame(columns=cols)
    return pd.concat(parts, ignore_index=True)


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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare datasets for all custom model families.")
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="Path to model_pipeline.toml",
    )
    parser.add_argument("--seed", type=int, default=None, help="Override prepare.seed from config.")
    parser.add_argument(
        "--samples-per-task-label",
        type=int,
        default=None,
        help="Override prepare.samples_per_task_label (single knob for dataset size; derives target/max per label and max_rows_per_source).",
    )
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
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    _load_dotenv(REPO_ROOT / ".env")
    config = _load_config(args.config)

    paths_cfg = dict(config.get("paths", {}))
    prepare_cfg = dict(config.get("prepare", {}))
    synthetic_cfg = dict(config.get("synthetic_llm", {}))
    datasets_cfg = list(config.get("datasets", []))

    if args.seed is not None:
        prepare_cfg["seed"] = int(args.seed)
    if args.samples_per_task_label is not None:
        prepare_cfg["samples_per_task_label"] = int(args.samples_per_task_label)
    if args.max_rows_per_source is not None:
        prepare_cfg["max_rows_per_source"] = int(args.max_rows_per_source)
    if args.max_per_task_label is not None:
        prepare_cfg["max_per_task_label"] = int(args.max_per_task_label)
    if args.target_per_task_label is not None:
        prepare_cfg["target_per_task_label"] = int(args.target_per_task_label)
    if args.llm_temperature is not None:
        synthetic_cfg["temperature"] = float(args.llm_temperature)
    if args.allow_missing_datasets_package:
        prepare_cfg["require_datasets_package"] = False

    # Derive target_per_task_label, max_per_task_label, max_rows_per_source from single knob when set.
    if "samples_per_task_label" in prepare_cfg:
        derived = _derive_prepare_limits_from_samples_per_task_label(
            prepare_cfg["samples_per_task_label"]
        )
        for k, v in derived.items():
            prepare_cfg.setdefault(k, v)

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
    synthetic_cfg.setdefault("timeout_seconds", 120)
    synthetic_cfg.setdefault("max_retries", 3)
    synthetic_cfg.setdefault("max_attempts_per_label", 80)
    synthetic_cfg.setdefault("local_raw_scan_rows_per_file", 300)

    if "samples_per_task_label" in prepare_cfg and int(prepare_cfg["samples_per_task_label"]) <= 0:
        print("prepare.samples_per_task_label must be > 0.", file=sys.stderr)
        return 1
    if int(prepare_cfg["max_rows_per_source"]) <= 0:
        print("prepare.max_rows_per_source must be > 0.", file=sys.stderr)
        return 1
    if int(prepare_cfg["target_per_task_label"]) <= 0:
        print("prepare.target_per_task_label must be > 0.", file=sys.stderr)
        return 1
    if int(prepare_cfg["max_per_task_label"]) <= 0:
        print("prepare.max_per_task_label must be > 0.", file=sys.stderr)
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

    if args.force_full:
        print("Force-full mode: ignoring existing prepared splits.")
        router_existing_df = pd.DataFrame(columns=["text", "task", "label", "source"])
        extractor_existing_df = pd.DataFrame(columns=["text", "task", "label", "source"])
        pair_existing_df = pd.DataFrame(columns=["text_a", "text_b", "task", "label", "source"])
    else:
        router_existing_df = _load_existing_family_df(prepared_dir, "router")
        extractor_existing_df = _load_existing_family_df(prepared_dir, "extractor")
        pair_existing_df = _load_existing_family_df(prepared_dir, "pair")

    router_missing, router_missing_total = _missing_task_labels(
        router_existing_df,
        task_labels=ROUTER_TASK_LABELS,
        target_per_task_label=target_per_task_label,
    )
    extractor_missing, extractor_missing_total = _missing_task_labels(
        extractor_existing_df,
        task_labels=EXTRACTOR_TASK_LABELS,
        target_per_task_label=target_per_task_label,
    )
    pair_missing, pair_missing_total = _missing_task_labels(
        pair_existing_df,
        task_labels=PAIR_TASK_LABELS,
        target_per_task_label=target_per_task_label,
    )

    needs_router = args.force_full or router_missing_total > 0
    needs_extractor = args.force_full or extractor_missing_total > 0
    needs_pair = args.force_full or pair_missing_total > 0

    print(
        "Existing coverage missing counts: "
        f"router={router_missing_total}, extractor={extractor_missing_total}, pair={pair_missing_total}"
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

    if not (needs_router or needs_extractor or needs_pair):
        manifest = {
            "config_path": str(args.config.resolve()),
            "seed": int(prepare_cfg["seed"]),
            "incremental": {
                "mode": "missing_only",
                "skipped_all_families": True,
                "force_full": False,
                "missing_counts": {
                    "router": router_missing_total,
                    "extractor": extractor_missing_total,
                    "pair": pair_missing_total,
                },
            },
            "paths": {
                "prepared_dir": str(prepared_dir),
                "bootstrap_prepared_dir": str(bootstrap_dir),
                "datasets_cache_dir": str(cache_dir) if cache_dir else "",
            },
            "datasets": {},
            "router": {
                **_summary(router_existing_df),
                "splits": _existing_split_counts(prepared_dir, "router"),
                "tasks": list(ROUTER_TASKS),
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
                "tasks": list(PAIR_TASKS),
                "updated": False,
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
    datasets_cfg_active = [
        d for d in datasets_cfg if str(d.get("target", "")).strip().lower() in active_targets
    ]

    has_enabled_remote = any(bool(d.get("enabled", True)) for d in datasets_cfg_active)
    if (
        has_enabled_remote
        and _hf_load_dataset is None
        and bool(prepare_cfg["require_datasets_package"])
    ):
        print(
            "Missing dependency: python package `datasets` is required by config. "
            "Install with `pip install datasets` or run with --allow-missing-datasets-package.",
            file=sys.stderr,
        )
        return 1

    registry = _HFRegistry(
        datasets_cfg=datasets_cfg_active, cache_dir=cache_dir, prepare_cfg=prepare_cfg
    )
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

    try:
        llm = _LLMGenerator(synthetic_cfg)
    except Exception as exc:
        print(f"Failed to initialize synthetic LLM generator: {exc}", file=sys.stderr)
        return 1
    print(
        "Using synthetic LLM provider="
        f"{llm.provider or 'unknown'} model={llm.model} base_url={llm.base_url} temperature={llm.temperature}"
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
                existing_df=router_existing_df,
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
                existing_df=pair_existing_df,
            )
            pair_df = pd.DataFrame(_sanitize_row_dicts(pair_rows))
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
        else:
            pair_df = pair_existing_df
            pair_splits = _existing_split_counts(prepared_dir, "pair")

        manifest = {
            "config_path": str(args.config.resolve()),
            "seed": int(prepare_cfg["seed"]),
            "incremental": {
                "mode": "missing_only",
                "force_full": bool(args.force_full),
                "missing_counts": {
                    "router": router_missing_total,
                    "extractor": extractor_missing_total,
                    "pair": pair_missing_total,
                },
                "updated": {
                    "router": needs_router,
                    "extractor": needs_extractor,
                    "pair": needs_pair,
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
                    "samples_per_task_label",
                    "max_rows_per_source",
                    "target_per_task_label",
                    "max_per_task_label",
                    "train_ratio",
                    "test_ratio",
                    "eval_ratio",
                    "auto_download_missing",
                ]
                if k in prepare_cfg
            }
            | {"llm_temperature": synthetic_cfg["temperature"]},
            "synthetic_llm": {
                "provider": llm.provider,
                "model": llm.model,
                "base_url": llm.base_url,
                "temperature": llm.temperature,
                "top_p": llm.top_p,
                "batch_size": llm.batch_size,
                "max_tokens": llm.max_tokens,
                "max_attempts_per_label": int(synthetic_cfg["max_attempts_per_label"]),
                "local_raw_scan_rows_per_file": int(synthetic_cfg["local_raw_scan_rows_per_file"]),
            },
            "datasets": registry.status,
            "router": {
                **_summary(router_df),
                "splits": router_splits,
                "tasks": list(ROUTER_TASKS),
                "updated": needs_router,
                "missing_before": router_missing,
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
                "tasks": list(PAIR_TASKS),
                "updated": needs_pair,
                "missing_before": pair_missing,
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
        try:
            llm.client.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
