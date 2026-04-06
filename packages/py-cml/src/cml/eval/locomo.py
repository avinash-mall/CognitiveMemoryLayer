"""Locomo-Plus evaluation with CML as the RAG backend."""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, cast

from cml.eval.config import ensure_unified_eval_data, find_repo_root, load_repo_dotenv
from cml.eval.types import LocomoEvalConfig

requests: Any | None
HTTPAdapter: type[Any] | None

try:
    import requests as _requests
    from requests.adapters import HTTPAdapter as _HTTPAdapter
except ImportError:
    requests = None
    HTTPAdapter = None
else:
    requests = _requests
    HTTPAdapter = _HTTPAdapter

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

_REPO_ROOT = find_repo_root(Path(__file__).resolve()) or Path.cwd()
load_repo_dotenv(_REPO_ROOT)


class _NoopProgress:
    """Fallback progress helper when tqdm is unavailable."""

    def __init__(self, iterable: object | None = None, total: int | None = None, **kwargs) -> None:
        self._iterable = iterable
        self.total = total

    def __iter__(self):
        if self._iterable is None:
            return iter(())
        return iter(self._iterable)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        return None

    def update(self, n: int = 1) -> None:
        return None

    def write(self, msg: str) -> None:
        print(msg, flush=True)


def _tqdm(iterable: object | None = None, **kwargs):
    if tqdm is None:
        return _NoopProgress(iterable=iterable, total=kwargs.get("total"))
    return tqdm(iterable, **kwargs)


def _progress_write(msg: str) -> None:
    if tqdm is None:
        print(msg, flush=True)
        return
    tqdm.write(msg)


INGESTION_DELAY_SEC = 0.2

# Shared HTTP session for connection reuse across threads
_SESSION: Any = None


def _get_session() -> Any:
    """Return a shared requests.Session with connection pooling."""
    global _SESSION
    if _SESSION is None and requests is not None:
        requests_lib = cast("Any", requests)
        adapter_cls = cast("type[Any]", HTTPAdapter)
        _SESSION = requests_lib.Session()
        adapter = adapter_cls(pool_connections=100, pool_maxsize=200)
        _SESSION.mount("http://", adapter)
        _SESSION.mount("https://", adapter)
    return _SESSION


_CML_WRITE_MAX_429_ATTEMPTS = 15
_CML_WRITE_BACKOFF_CAP_SEC = 65
_CML_READ_MAX_429_ATTEMPTS = 15
_CML_READ_BACKOFF_CAP_SEC = 65
QA_READ_DELAY_SEC = 0.05

# QA prompt for all categories (aligned with Locomo-Plus task instructions)
QA_PROMPT = """Based on the above context, write an answer in the form of a short phrase.
Answer with exact words from the context whenever possible.

Question: {} Short answer:"""

COGNITIVE_PROMPT = """Based on the above context, continue the conversation naturally.
Respond to the following as you would in a real dialogue.

{}

Respond (short):"""


_DATE_FORMATS = [
    "%B %d, %Y",  # "January 15, 2024"
    "%b %d, %Y",  # "Jan 15, 2024"
    "%Y-%m-%d",  # "2024-01-15"
    "%m/%d/%Y",  # "01/15/2024"
    "%d %B %Y",  # "15 January 2024"
    "%B %d %Y",  # "January 15 2024" (no comma)
    "%I:%M %p on %d %B, %Y",  # "02:30 PM on 15 January, 2024" (Locomo session format)
]


def _parse_date_str(date_str: str | None) -> datetime | None:
    """Best-effort parse of a LoCoMo-Plus DATE string into a UTC datetime."""
    if not date_str:
        return None
    cleaned = date_str.strip().rstrip(".")
    for fmt in _DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).replace(tzinfo=UTC)
        except ValueError:
            continue
    return None


def _ensure_eval_dependencies() -> None:
    if requests is None:
        raise ImportError(
            "Evaluation dependency missing: requests. Install with: "
            'pip install "cognitive-memory-layer[eval]"'
        )
    if tqdm is None:
        raise ImportError(
            "Evaluation dependency missing: tqdm. Install with: "
            'pip install "cognitive-memory-layer[eval]"'
        )
    try:
        import_module("openai")
    except ImportError as exc:
        raise ImportError(
            "Evaluation dependency missing: openai. Install with: "
            'pip install "cognitive-memory-layer[eval]"'
        ) from exc


def _ensure_locomo_plus_path(repo_root: Path | None = None) -> bool:
    """Add evaluation/locomo_plus to sys.path if found. Returns True if available."""
    root = repo_root or _REPO_ROOT
    locomo_plus = root / "evaluation" / "locomo_plus"
    if locomo_plus.is_dir():
        if str(locomo_plus) not in sys.path:
            sys.path.insert(0, str(locomo_plus))
        return True
    return False


def _load_unified_samples_builtin(data_file: str) -> list[dict]:
    """Built-in fallback for task_eval.utils.load_unified_samples."""
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Unified input file not found: {data_file}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Unified input JSON must be a list of samples.")
    return data


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Locomo-Plus unified eval with CML; QA uses LLM from .env (LLM_EVAL__* or LLM_INTERNAL__*)"
    )
    p.add_argument(
        "--unified-file", type=str, required=True, help="Path to unified_input_samples_v2.json"
    )
    p.add_argument("--out-dir", type=str, default="evaluation/outputs")
    p.add_argument(
        "--cml-url", type=str, default=os.environ.get("CML_BASE_URL", "http://localhost:8000")
    )
    p.add_argument("--cml-api-key", type=str, default=os.environ.get("CML_API_KEY", "test-key"))
    p.add_argument("--max-results", type=int, default=25)
    p.add_argument("--limit-samples", type=int, default=None)
    p.add_argument("--skip-ingestion", action="store_true")
    p.add_argument(
        "--skip-consolidation",
        action="store_true",
        help="Skip consolidation and reconsolidation between Phase A and Phase B",
    )
    p.add_argument(
        "--score-only", action="store_true", help="Run only Phase C (judge) on existing predictions"
    )
    p.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("LLM_EVAL__MODEL")
        or os.environ.get("LLM_INTERNAL__MODEL", "gpt-4o-mini"),
        help="Model for LLM-as-judge (default: LLM_EVAL__MODEL or LLM_INTERNAL__MODEL from .env or gpt-4o-mini)",
    )
    p.add_argument("--verbose", action="store_true", help="Emit per-sample retrieval diagnostics")
    p.add_argument(
        "--ingestion-workers",
        type=int,
        default=10,
        metavar="N",
        help="Number of concurrent workers for Phase A ingestion (default 10)",
    )
    p.add_argument(
        "--qa-backend",
        type=str,
        default="openai_compatible",
        choices=["openai_compatible", "vllm"],
        help=(
            "Backend for Phase B QA inference. "
            "'openai_compatible' calls LLM_EVAL__BASE_URL via HTTP (default). "
            "'vllm' loads the model in-process on GPU and batches all prompts in one pass."
        ),
    )
    p.add_argument(
        "--judge-backend",
        type=str,
        default="call_llm",
        choices=["call_llm", "call_vllm"],
        help=(
            "Backend for Phase C LLM-as-judge. "
            "'call_llm' uses OpenAI-compatible HTTP API (default). "
            "'call_vllm' runs inference in-process on GPU."
        ),
    )
    return p.parse_args(argv)


def _parse_input_prompt_into_turns(input_prompt: str) -> list[tuple[str, str, str | None]]:
    """
    Parse input_prompt into (session_id, content, timestamp) tuples for CML ingestion.
    Format: DATE: ...\\nCONVERSATION:\\nSpeaker said, "text"\\n  or  Speaker said, "text" and shared X.\\n
    Returns list of (session_id, content, date_str).
    """
    turns: list[tuple[str, str, str | None]] = []
    current_session = "session_1"
    current_date: str | None = None
    session_counter = 1

    for line in input_prompt.split("\n"):
        line = line.strip()
        if not line:
            continue
        if line.startswith("DATE:"):
            current_date = line[5:].strip()
            current_session = f"session_{session_counter}"
            session_counter += 1
            continue
        if line == "CONVERSATION:":
            continue
        if line.startswith("Question:"):
            break
        m = re.match(r'^(.+?) said, "(.+?)"(?:\s+and shared .+)?\.?\s*$', line)
        if m:
            speaker, text = m.group(1).strip(), m.group(2).strip()
            content = f"{speaker}: {text}"
            turns.append((current_session, content, current_date))
    return turns


def _cml_write(
    base_url: str,
    api_key: str,
    tenant_id: str,
    content: str,
    session_id: str,
    metadata: dict,
    turn_id: str,
    timestamp: str | None = None,
) -> None:
    requests_lib = cast("Any", requests)
    url = f"{base_url.rstrip('/')}/api/v1/memory/write"
    payload: dict = {
        "content": content,
        "session_id": session_id,
        "metadata": metadata,
        "turn_id": turn_id,
    }
    if timestamp is not None:
        payload["timestamp"] = timestamp
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id, "X-Eval-Mode": "true"}
    write_timeout = 180  # Allow time for first-request embedding model load
    write_retries = 3  # Retry on connection errors (e.g. server busy, model loading)
    for retry in range(write_retries):
        for attempt in range(_CML_WRITE_MAX_429_ATTEMPTS):
            try:
                resp = _get_session().post(
                    url, json=payload, headers=headers, timeout=write_timeout
                )
                if resp.status_code in [429, 500]:
                    if attempt == _CML_WRITE_MAX_429_ATTEMPTS - 1:
                        resp.raise_for_status()
                    wait = min(_CML_WRITE_BACKOFF_CAP_SEC, 5 * (2**attempt))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return
            except (requests_lib.exceptions.ConnectionError, OSError):
                if retry < write_retries - 1:
                    time.sleep(5 * (retry + 1))
                    break
                raise


def _dashboard_post(
    base_url: str,
    api_key: str,
    path: str,
    body: dict,
    max_attempts: int = 5,
    backoff_cap_sec: int = 30,
) -> dict:
    """POST to a dashboard endpoint; raises on non-2xx. Used for consolidate and reconsolidate."""
    requests_lib = cast("Any", requests)
    url = f"{base_url.rstrip('/')}/api/v1/dashboard{path}"
    headers = {
        "X-API-Key": api_key,
        "Content-Type": "application/json",
        "X-Requested-With": "XMLHttpRequest",
    }
    last_resp = None
    for attempt in range(max_attempts):
        resp = _get_session().post(url, json=body, headers=headers, timeout=120)
        last_resp = resp
        if resp.status_code == 429:
            if attempt == max_attempts - 1:
                resp.raise_for_status()
            wait = min(backoff_cap_sec, 5 * (2**attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    if last_resp is not None:
        raise requests_lib.exceptions.HTTPError("429 Too Many Requests", response=last_resp)
    raise requests_lib.exceptions.HTTPError("No attempts made (max_attempts=0)")


def _cml_read(
    base_url: str,
    api_key: str,
    tenant_id: str,
    query: str,
    max_results: int,
    return_full: bool = False,
) -> str | tuple[str, dict]:
    """Read memories from CML. If *return_full*, also return the raw JSON response."""
    requests_lib = cast("Any", requests)
    url = f"{base_url.rstrip('/')}/api/v1/memory/read"
    payload = {"query": query, "format": "llm_context", "max_results": max_results}
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id}
    last_resp = None
    for attempt in range(_CML_READ_MAX_429_ATTEMPTS):
        resp = _get_session().post(url, json=payload, headers=headers, timeout=60)
        last_resp = resp
        if resp.status_code == 429:
            if attempt == _CML_READ_MAX_429_ATTEMPTS - 1:
                resp.raise_for_status()
            wait = min(_CML_READ_BACKOFF_CAP_SEC, 5 * (2**attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        data = resp.json()
        llm_context = (data.get("llm_context") or "").strip()
        if return_full:
            return llm_context, data
        return llm_context
    if last_resp is not None:
        raise requests_lib.exceptions.HTTPError("429 Too Many Requests", response=last_resp)
    raise requests_lib.exceptions.HTTPError("No attempts made")


# Default OpenAI-compatible base URLs per provider (align with .env.example and src/utils/llm.py)
_LLM_DEFAULT_BASE: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "ollama": "http://localhost:11434/v1",
    "openai_compatible": "http://localhost:8000/v1",
}


def _get_llm_qa_config() -> tuple[str, str, str]:
    """Resolve (base_url, model, api_key) for QA from .env: LLM_EVAL__* with fallback to LLM_INTERNAL__*."""
    provider = (
        (
            os.environ.get("LLM_EVAL__PROVIDER")
            or os.environ.get("LLM_INTERNAL__PROVIDER")
            or "openai"
        )
        .strip()
        .lower()
    )
    model = (
        os.environ.get("LLM_EVAL__MODEL") or os.environ.get("LLM_INTERNAL__MODEL") or "gpt-4o-mini"
    ).strip()
    api_key = (
        os.environ.get("LLM_EVAL__API_KEY")
        or os.environ.get("LLM_INTERNAL__API_KEY")
        or os.environ.get("OPENAI_API_KEY")
        or ""
    ).strip()
    base_url = (
        os.environ.get("LLM_EVAL__BASE_URL") or os.environ.get("LLM_INTERNAL__BASE_URL") or ""
    ).strip()
    if not base_url:
        base_url = _LLM_DEFAULT_BASE.get(provider, "")
    if not base_url and provider in ("gemini", "claude", "anthropic"):
        raise RuntimeError(
            f"LLM_EVAL__BASE_URL or LLM_INTERNAL__BASE_URL is required for provider={provider!r}. "
            "Use an OpenAI-compatible proxy or set LLM_EVAL__PROVIDER=openai_compatible with a proxy URL."
        )
    if not base_url:
        base_url = _LLM_DEFAULT_BASE.get("openai_compatible", "http://localhost:8000/v1")
    if provider != "openai" and not api_key:
        api_key = "dummy"
    base_url = base_url.rstrip("/")
    return base_url, model, api_key


def _llm_chat(user_content: str, max_tokens: int = 256, backend: str = "openai_compatible") -> str:
    """Call LLM for QA. backend='openai_compatible' uses HTTP API; backend='vllm' uses in-process GPU."""
    if backend == "vllm":
        _, model, _ = _get_llm_qa_config()
        try:
            from task_eval.vllm_backend import generate_single
        except ImportError:
            raise ImportError(
                "task_eval.vllm_backend not found. Run from repo root so evaluation/locomo_plus is on sys.path."
            )
        result = generate_single(
            model,
            [{"role": "user", "content": user_content}],
            temperature=0.0,
            max_tokens=max_tokens,
        )
        return result or ""

    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError(
            "openai package is required for evaluation QA. Install with: pip install openai"
        )

    base_url, model, api_key = _get_llm_qa_config()

    def _do_request() -> str:
        client = OpenAI(base_url=base_url, api_key=api_key)
        try:
            resp = client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": user_content}],
                max_tokens=max_tokens,
                temperature=0,
            )
        except Exception:
            return ""
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return ""
        content = getattr(msg, "content", None)
        return (content or "").strip()

    out = _do_request()
    if not out:
        time.sleep(2.0)
        out = _do_request()
    return out or ""


def _ingest_sample(
    cml_url: str,
    cml_api_key: str,
    sample_idx: int,
    sample: dict,
    ingestion_delay_sec: float,
    pbar: tqdm | None = None,
) -> None:
    """Ingest one sample: write all turns sequentially for tenant lp-{sample_idx}."""
    tenant_id = f"lp-{sample_idx}"
    input_prompt = sample.get("input_prompt", "")
    turns = _parse_input_prompt_into_turns(input_prompt)
    turn_timestamps = sample.get("turn_timestamps") or []
    for j, (session_id, content, date_str) in enumerate(turns):
        speaker = content.split(":")[0].strip() if ":" in content else "unknown"
        ts_iso = None
        if j < len(turn_timestamps) and turn_timestamps[j]:
            t = turn_timestamps[j]
            if isinstance(t, str) and t.strip():
                ts_iso = t.strip()
            elif hasattr(t, "isoformat"):
                ts_iso = t.isoformat()
        if ts_iso is None:
            parsed_ts = _parse_date_str(date_str)
            ts_iso = parsed_ts.isoformat() if parsed_ts else None
        metadata = {
            "locomo_plus_idx": sample_idx,
            "turn_idx": j,
            "speaker": speaker,
            "date_str": date_str or "",
            "session_idx": int(session_id.split("_")[-1]) if "_" in session_id else 1,
        }
        _cml_write(
            cml_url,
            cml_api_key,
            tenant_id,
            content,
            session_id,
            metadata,
            f"turn_{j}",
            timestamp=ts_iso,
        )
        if pbar is not None:
            pbar.update(1)
        if ingestion_delay_sec > 0:
            time.sleep(ingestion_delay_sec)


def phase_a_ingestion(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
    ingestion_workers: int = 10,
    checkpoint_file: Path | None = None,
) -> None:
    samples_to_ingest = samples[:limit] if limit else samples

    # Load checkpoint — skip already-completed samples
    completed: set[int] = set()
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            completed = set(data.get("completed_indices", []))
        except (json.JSONDecodeError, OSError):
            pass

    pending = [(i, s) for i, s in enumerate(samples_to_ingest) if i not in completed]

    if not pending:
        print("\n[Phase A] All samples already ingested (checkpoint). Skipping.", flush=True)
        return

    # Apply per-turn delay only when single-threaded (rate limiting)
    ingestion_delay = INGESTION_DELAY_SEC if ingestion_workers == 1 else 0.0
    checkpoint_lock = threading.Lock()

    def _save_checkpoint(idx: int) -> None:
        if checkpoint_file is None:
            return
        with checkpoint_lock:
            completed.add(idx)
            try:
                checkpoint_file.write_text(
                    json.dumps({"completed_indices": sorted(completed)}, ensure_ascii=False),
                    encoding="utf-8",
                )
            except OSError:
                pass

    total_turns = sum(
        len(_parse_input_prompt_into_turns(s.get("input_prompt", ""))) for _, s in pending
    )
    skipped = len(samples_to_ingest) - len(pending)
    skip_msg = f" ({skipped} already ingested)" if skipped else ""
    print(
        f"\n[Phase A] Ingesting {len(pending)} samples ({total_turns} turns){skip_msg}"
        f" into CML ({ingestion_workers} workers)...",
        flush=True,
    )

    with _tqdm(total=total_turns, desc="Ingestion", unit="turn", disable=False) as pbar:
        if ingestion_workers <= 1:
            for i, sample in pending:
                _ingest_sample(cml_url, cml_api_key, i, sample, ingestion_delay, pbar)
                _save_checkpoint(i)
        else:
            with ThreadPoolExecutor(max_workers=ingestion_workers) as executor:
                futures = {
                    executor.submit(
                        _ingest_sample, cml_url, cml_api_key, i, sample, ingestion_delay, pbar
                    ): i
                    for i, sample in pending
                }
                for future in as_completed(futures):
                    future.result()  # Propagate any exception
                    _save_checkpoint(futures[future])


def phase_ab_consolidation(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
    checkpoint_file: Path | None = None,
) -> None:
    """Run consolidation then reconsolidation for each eval tenant (between Phase A and Phase B)."""
    samples_scope = samples[:limit] if limit else samples
    n = len(samples_scope)
    if n == 0:
        return

    # Load checkpoint — skip already-consolidated tenants
    completed_tenants: set[str] = set()
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            completed_tenants = set(data.get("completed_tenants", []))
        except (json.JSONDecodeError, OSError):
            pass

    pending_indices = [i for i in range(n) if f"lp-{i}" not in completed_tenants]
    if not pending_indices:
        print("\n[Phase A-B] All tenants already consolidated (checkpoint). Skipping.", flush=True)
        return

    skipped = n - len(pending_indices)
    skip_msg = f" ({skipped} already done)" if skipped else ""
    print(
        f"\n[Phase A-B] Consolidation and reconsolidation for {len(pending_indices)} tenants{skip_msg}...",
        flush=True,
    )
    for i in _tqdm(pending_indices, desc="Phase A-B", unit="tenant", disable=False):
        tenant_id = f"lp-{i}"
        body = {"tenant_id": tenant_id, "user_id": None}
        requests_lib = cast("Any", requests)
        try:
            _dashboard_post(cml_url, cml_api_key, "/consolidate", body)
            _dashboard_post(cml_url, cml_api_key, "/reconsolidate", body)
        except requests_lib.RequestException as e:
            raise RuntimeError(f"Dashboard step failed for tenant {tenant_id}: {e}") from e
        if checkpoint_file is not None:
            completed_tenants.add(tenant_id)
            try:
                checkpoint_file.write_text(
                    json.dumps(
                        {"completed_tenants": sorted(completed_tenants)}, ensure_ascii=False
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass
    print(f"  Completed for {len(pending_indices)} tenants.", flush=True)


def _count_memory_types(raw_response: dict) -> dict[str, int]:
    """Count memories by type from CML read response for diagnostics."""
    counts: dict[str, int] = {}
    # ReadMemoryResponse has category lists at the top level
    for category_key in ("constraints", "facts", "preferences", "episodes"):
        items = raw_response.get(category_key) or []
        if items:
            counts[category_key] = len(items)
    # Also count from flat memories list for type breakdown
    memories = raw_response.get("memories") or []
    if memories and not counts:
        for mem in memories:
            mtype = mem.get("type", "unknown") if isinstance(mem, dict) else "unknown"
            counts[mtype] = counts.get(mtype, 0) + 1
    return counts


def phase_b_qa(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    max_results: int,
    limit: int | None,
    out_dir: Path,
    verbose: bool = False,
    backend: str = "openai_compatible",
) -> list[dict]:
    _, qa_model, _ = _get_llm_qa_config()
    samples_qa = samples[:limit] if limit else samples
    pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
    records: list[dict] = []
    start_index = 0
    if pred_file.exists():
        try:
            loaded = json.loads(pred_file.read_text(encoding="utf-8"))
            if isinstance(loaded, list) and loaded:
                required = {
                    "question_input",
                    "evidence",
                    "category",
                    "ground_truth",
                    "prediction",
                    "model",
                }
                if all(isinstance(r, dict) and required.issubset(set(r.keys())) for r in loaded):
                    records = loaded
                    start_index = len(records)
                    if start_index >= len(samples_qa):
                        print(
                            f"\n[Phase B] Predictions already complete ({len(records)} records), skipping QA.",
                            flush=True,
                        )
                        return records
                    print(
                        f"\n[Phase B] Resuming QA from sample {start_index} (LLM: {qa_model})...",
                        flush=True,
                    )
        except (json.JSONDecodeError, OSError):
            pass
    backend_label = f"{qa_model}_cml" + ("_vllm" if backend == "vllm" else "")
    if start_index == 0:
        print(
            f"\n[Phase B] Running QA on {len(samples_qa)} samples "
            f"(LLM: {qa_model}, backend: {backend})...",
            flush=True,
        )

    if backend == "vllm":
        # --- Batched GPU path: collect all CML contexts first, then run one batched vLLM call ---
        _ensure_locomo_plus_path()
        from task_eval.vllm_backend import generate_batch

        remaining = samples_qa[start_index:]
        user_contents: list[str] = []
        meta: list[dict] = []

        print(f"  [Phase B/vLLM] Fetching {len(remaining)} CML contexts...", flush=True)
        for i, sample in _tqdm(
            enumerate(remaining, start=start_index), desc="CML read", unit="sample"
        ):
            tenant_id = f"lp-{i}"
            trigger = (sample.get("trigger") or "").strip()
            category = sample.get("category", "")
            read_result = _cml_read(
                cml_url, cml_api_key, tenant_id, trigger, max_results, return_full=verbose
            )
            if verbose:
                assert isinstance(read_result, tuple)
                llm_context, raw_response = read_result
                _progress_write(
                    f"  [{category}] sample={i} types={_count_memory_types(raw_response)} "
                    f"context_len={len(llm_context)}"
                )
            else:
                llm_context = read_result  # type: ignore[assignment]
            if QA_READ_DELAY_SEC > 0:
                time.sleep(QA_READ_DELAY_SEC)
            if category == "Cognitive":
                user_content = (
                    (llm_context or "(No retrieved context.)")
                    + "\n\n"
                    + COGNITIVE_PROMPT.format(trigger)
                )
            else:
                user_content = (
                    (llm_context or "(No retrieved context.)") + "\n\n" + QA_PROMPT.format(trigger)
                )
            user_contents.append(user_content)
            meta.append(
                {
                    "question_input": trigger
                    or ("Context dialogue (cue awareness)" if category == "Cognitive" else trigger),
                    "evidence": sample.get("evidence", ""),
                    "category": category,
                    "ground_truth": sample.get("answer") or "",
                    "time_gap": sample.get("time_gap"),
                }
            )

        print(
            f"  [Phase B/vLLM] Running batched inference on {len(user_contents)} prompts...",
            flush=True,
        )
        conversations = [[{"role": "user", "content": c}] for c in user_contents]
        predictions = generate_batch(qa_model, conversations, temperature=0.0, max_tokens=256)

        for m, prediction in zip(meta, predictions, strict=False):
            record: dict = {
                "question_input": m["question_input"],
                "evidence": m["evidence"],
                "category": m["category"],
                "ground_truth": m["ground_truth"],
                "prediction": prediction,
                "model": backend_label,
            }
            if m.get("time_gap"):
                record["time_gap"] = m["time_gap"]
            records.append(record)
        pred_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")

    else:
        # --- Sequential HTTP path (default) ---
        for i in _tqdm(
            range(start_index, len(samples_qa)), desc="QA", unit="sample", disable=False
        ):
            sample = samples_qa[i]
            tenant_id = f"lp-{i}"
            trigger = (sample.get("trigger") or "").strip()
            category = sample.get("category", "")
            if category == "Cognitive":
                question_input = trigger or "Context dialogue (cue awareness)"
            else:
                question_input = trigger
            read_result = _cml_read(
                cml_url, cml_api_key, tenant_id, trigger, max_results, return_full=verbose
            )
            if verbose:
                assert isinstance(read_result, tuple), "return_full=True yields tuple"
                llm_context, raw_response = read_result
                type_counts = _count_memory_types(raw_response)
                _progress_write(
                    f"  [{category}] sample={i} types={type_counts} context_len={len(llm_context)}"
                )
            else:
                assert isinstance(read_result, str), "return_full=False yields str"
                llm_context = read_result
            if QA_READ_DELAY_SEC > 0:
                time.sleep(QA_READ_DELAY_SEC)

            if category == "Cognitive":
                user_content = (
                    (llm_context or "(No retrieved context.)")
                    + "\n\n"
                    + COGNITIVE_PROMPT.format(trigger)
                )
            else:
                user_content = (
                    (llm_context or "(No retrieved context.)") + "\n\n" + QA_PROMPT.format(trigger)
                )

            prediction = _llm_chat(user_content, backend=backend)
            ground_truth = sample.get("answer")
            if ground_truth is None or (
                isinstance(ground_truth, str) and ground_truth.strip() == ""
            ):
                ground_truth = ""
            record = {
                "question_input": question_input,
                "evidence": sample.get("evidence", ""),
                "category": category,
                "ground_truth": ground_truth,
                "prediction": prediction,
                "model": backend_label,
            }
            if sample.get("time_gap"):
                record["time_gap"] = sample["time_gap"]
            records.append(record)
            pred_file.write_text(
                json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
            )

    empty_count = sum(1 for r in records if not (r.get("prediction") or "").strip())
    if empty_count:
        print(
            f"\n[Phase B] Warning: {empty_count}/{len(records)} predictions are empty. "
            "Check LLM output and CML read context (e.g. --verbose).",
            flush=True,
        )
    return records


def phase_c_judge(
    records: list[dict], out_dir: Path, judge_model: str, judge_backend: str = "call_llm"
) -> None:
    if not _ensure_locomo_plus_path():
        raise ImportError(
            "LLM-as-judge requires the evaluation/locomo_plus directory from the CML repository. "
            "Run from within the repo or set --repo-root to a checkout that contains evaluation/locomo_plus/."
        )
    from task_eval.llm_as_judge import run_judge

    print(
        f"\n[Phase C] LLM-as-judge scoring {len(records)} predictions "
        f"({judge_model}, backend: {judge_backend})...",
        flush=True,
    )

    class Args:
        input_file = ""
        out_file = str(out_dir / "locomo_plus_qa_cml_judged.json")
        model = judge_model
        backend = judge_backend
        temperature = 0.0
        max_tokens = 512
        concurrency = 1
        summary_file = str(out_dir / "locomo_plus_qa_cml_judge_summary.json")

    pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
    pred_file.write_text(json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8")
    args = Args()
    args.input_file = str(pred_file)
    run_judge(args)
    print(f"  Judged output: {args.out_file}", flush=True)
    print(f"  Summary: {args.summary_file}", flush=True)


def run_locomo_plus(config: LocomoEvalConfig) -> list[dict]:
    """Run LoCoMo-Plus evaluation and return prediction records."""
    _ensure_eval_dependencies()
    has_locomo_path = _ensure_locomo_plus_path()
    unified_file = ensure_unified_eval_data(Path(config.unified_file), repo_root=_REPO_ROOT)

    out_dir = Path(config.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if config.score_only:
        pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
        if not pred_file.exists():
            raise FileNotFoundError(f"Score-only requires {pred_file}")
        records = json.loads(pred_file.read_text(encoding="utf-8"))
        phase_c_judge(records, out_dir, config.judge_model, judge_backend=config.judge_backend)
        return records

    if has_locomo_path:
        try:
            from task_eval.utils import load_unified_samples
        except ImportError:
            load_unified_samples = _load_unified_samples_builtin
    else:
        load_unified_samples = _load_unified_samples_builtin

    samples = load_unified_samples(str(unified_file))
    total = len(samples)
    if config.limit_samples:
        samples = samples[: config.limit_samples]
        print(f"Loaded {len(samples)} samples (limit; total available: {total})", flush=True)
    else:
        print(f"Loaded {len(samples)} samples", flush=True)

    if not config.skip_ingestion:
        phase_a_ingestion(
            samples,
            config.cml_url,
            config.cml_api_key,
            config.limit_samples,
            ingestion_workers=config.ingestion_workers,
            checkpoint_file=out_dir / "locomo_ingestion_checkpoint.json",
        )

    if not config.skip_consolidation:
        phase_ab_consolidation(
            samples,
            config.cml_url,
            config.cml_api_key,
            config.limit_samples,
            checkpoint_file=out_dir / "locomo_consolidation_checkpoint.json",
        )

    records = phase_b_qa(
        samples,
        config.cml_url,
        config.cml_api_key,
        config.max_results,
        config.limit_samples,
        out_dir,
        verbose=config.verbose,
        backend=config.qa_backend,
    )

    phase_c_judge(records, out_dir, config.judge_model, judge_backend=config.judge_backend)
    print(f"Predictions: {out_dir / 'locomo_plus_qa_cml_predictions.json'}")
    return records


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = LocomoEvalConfig(
        unified_file=Path(args.unified_file),
        out_dir=Path(args.out_dir),
        cml_url=str(args.cml_url),
        cml_api_key=str(args.cml_api_key),
        max_results=int(args.max_results),
        limit_samples=args.limit_samples,
        skip_ingestion=bool(args.skip_ingestion),
        skip_consolidation=bool(args.skip_consolidation),
        score_only=bool(args.score_only),
        judge_model=str(args.judge_model),
        verbose=bool(args.verbose),
        ingestion_workers=int(args.ingestion_workers),
        qa_backend=str(args.qa_backend),
        judge_backend=str(args.judge_backend),
    )
    try:
        run_locomo_plus(cfg)
    except (FileNotFoundError, ImportError) as exc:
        print(f"Error: {exc}", file=sys.stderr, flush=True)
        return 1
    except Exception as exc:
        print(f"Evaluation failed: {exc}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
