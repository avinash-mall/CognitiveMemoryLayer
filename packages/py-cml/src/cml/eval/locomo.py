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
_SESSION_REQUESTS_ID: int | None = None


def _get_session() -> Any:
    """Return a shared requests.Session or a requests-like fallback client."""
    global _SESSION, _SESSION_REQUESTS_ID
    if requests is None:
        return None

    requests_lib = cast("Any", requests)
    session_factory = getattr(requests_lib, "Session", None)
    if not callable(session_factory):
        return requests_lib

    requests_id = id(requests_lib)
    if _SESSION is None or requests_id != _SESSION_REQUESTS_ID:
        _SESSION = session_factory()
        _SESSION_REQUESTS_ID = requests_id
        adapter_cls = cast("type[Any] | None", HTTPAdapter)
        if adapter_cls is not None and hasattr(_SESSION, "mount"):
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
# Improvement Report Section 8.1: category-aware answering with conciseness rules.
# Balanced: encourage conciseness without over-refusing when context is sparse.
# NOTE: The memory store redacts PII (names → [FIRSTNAME_REDACTED], etc.).
# The prompt instructs the model to match redacted placeholders to question entities.
QA_PROMPT = """Based on the above context from past conversations, answer the question below.

IMPORTANT RULES:
1. Names may appear as [FIRSTNAME_REDACTED] — treat them as the people in the question.
2. If the context contains timestamps or dates, reason about time carefully.
   Calculate durations by counting between dates. "Yesterday" means the day before the date shown.
3. Answer with a short phrase or 1-2 sentences. Use exact words from the context when possible.
4. ALWAYS attempt to answer from the context. Even if the context is incomplete, extract
   whatever relevant information is available and provide your best answer.
5. For common-sense questions, combine the context with general world knowledge to reason
   about the answer. Think about what would logically follow from what was said.
6. For multi-hop questions, connect information across multiple parts of the context.
   Look for indirect references and combine facts to derive the answer.
7. Only say "I don't have information about that from our previous conversations" if the
   context is completely empty OR contains absolutely nothing related to the question.

Question: {}

Short answer:"""

COGNITIVE_PROMPT = """Based on the above context from past conversations, continue the
conversation naturally. Respond to the following as you would in a real dialogue.
If the context contains relevant constraints, preferences, or past decisions, incorporate them.
Names may appear as [FIRSTNAME_REDACTED] — treat them as the people in the conversation.

{}

Respond (short):"""

# Answer markers used by reasoning models (ordered by specificity)
_ANSWER_MARKERS = [
    "Short answer:",
    "Final Answer:",
    "Answer:",
    "Therefore,",
    "In summary:",
    "In conclusion:",
    "The answer is:",
    "Based on the context,",
    "Based on the conversation,",
]


def _extract_answer(raw: str) -> str:
    """Strip chain-of-thought wrapper from a prediction, returning just the answer.

    Critical: never return empty string when the raw input is non-empty.
    The judge needs *something* to evaluate.
    """
    if not raw:
        return raw
    text = raw.strip()
    if not text:
        return text

    # Handle <think>...</think> blocks (may be nested or repeated)
    if "<think>" in text:
        parts = text.split("</think>")
        if len(parts) > 1:
            after_think = parts[-1].strip()
            if after_think:
                # Still check for answer markers in the post-think section
                for marker in _ANSWER_MARKERS:
                    idx = after_think.find(marker)
                    if idx != -1:
                        answer = after_think[idx + len(marker) :].strip()
                        if answer:
                            return answer
                return after_think
            # All content was inside <think> — return last think block content
            # so the judge has something to evaluate
            return parts[-2].replace("<think>", "").strip() or text
        # Unclosed <think> (model truncated before emitting </think>): strip the
        # opening marker so the judge never receives a raw <think> tag as the answer.
        return text.split("<think>", 1)[-1].strip() or text

    # Handle various reasoning prefixes
    _reasoning_prefixes = (
        "Thinking Process:",
        "**Thinking Process",
        "Let me think",
        "Let me analyze",
        "Step 1:",
        "**Step 1",
        "First, let me",
        "I need to",
    )
    is_reasoning = any(text.startswith(p) for p in _reasoning_prefixes)

    if is_reasoning:
        # Try to find an explicit answer marker
        for marker in _ANSWER_MARKERS:
            idx = text.rfind(marker)
            if idx != -1:
                answer = text[idx + len(marker) :].strip()
                # Clean trailing markers or bullet formatting
                if answer:
                    return answer
        # Fallback: return last non-reasoning paragraph
        paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
        if paragraphs:
            # Walk backwards to find first non-bullet, non-reasoning paragraph
            for para in reversed(paragraphs):
                if not para.startswith(("*", "-", "1.", "2.", "3.", "##", "**Step")):
                    return para
            # All paragraphs are bullet lists — return the last one anyway
            return paragraphs[-1]

    # Even for non-reasoning text, check if there's an explicit answer marker
    # (some models wrap short reasoning then give "Answer: X")
    for marker in _ANSWER_MARKERS[:3]:  # Only check the most explicit markers
        idx = text.rfind(marker)
        if idx != -1 and idx > len(text) // 3:  # Marker should be in latter part
            answer = text[idx + len(marker) :].strip()
            if answer and len(answer) < len(text) * 0.8:  # Answer should be shorter than full text
                return answer

    return text


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
        default=20,
        metavar="N",
        help="Number of concurrent workers for Phase A ingestion (default 20)",
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


_WRITE_BATCH_SIZE = 50  # Turns per HTTP request for batch ingestion


def _cml_write_batch(
    base_url: str,
    api_key: str,
    tenant_id: str,
    turns: list[dict],
) -> None:
    """Write multiple turns in a single HTTP call using the batch endpoint."""
    requests_lib = cast("Any", requests)
    url = f"{base_url.rstrip('/')}/api/v1/memory/write_batch"
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id, "X-Eval-Mode": "true"}
    write_timeout = 300  # Batch may take longer
    write_retries = 3
    payload = {"turns": turns}
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
                if resp.status_code == 404:
                    # Server doesn't have batch endpoint — fall back to single writes
                    for turn in turns:
                        _cml_write(
                            base_url,
                            api_key,
                            tenant_id,
                            turn["content"],
                            turn.get("session_id", ""),
                            turn.get("metadata", {}),
                            turn.get("turn_id", ""),
                            timestamp=turn.get("timestamp"),
                        )
                    return
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
    read_retries = 5
    last_exc: Exception | None = None
    for retry in range(read_retries):
        for attempt in range(_CML_READ_MAX_429_ATTEMPTS):
            try:
                resp = _get_session().post(url, json=payload, headers=headers, timeout=120)
                if resp.status_code == 429:
                    if attempt == _CML_READ_MAX_429_ATTEMPTS - 1:
                        resp.raise_for_status()
                    wait = min(_CML_READ_BACKOFF_CAP_SEC, 5 * (2**attempt))
                    time.sleep(wait)
                    continue
                if resp.status_code >= 500:
                    # Server error — retry with backoff
                    last_exc = requests_lib.exceptions.HTTPError(
                        f"{resp.status_code} Server Error", response=resp
                    )
                    break  # go to next retry
                resp.raise_for_status()
                data = resp.json()
                llm_context = (data.get("llm_context") or "").strip()
                if return_full:
                    return llm_context, data
                return llm_context
            except (
                requests_lib.exceptions.ConnectionError,
                requests_lib.exceptions.Timeout,
                OSError,
            ) as exc:
                last_exc = exc
                break  # go to next retry
        else:
            # Inner loop completed without break (all 429 retries exhausted)
            raise requests_lib.exceptions.HTTPError(
                "429 Too Many Requests after all retries",
                response=resp,
            )
        # Backoff before next retry
        wait = min(30, 3 * (2**retry))
        time.sleep(wait)
    # All retries exhausted
    if last_exc is not None:
        raise last_exc
    raise requests_lib.exceptions.HTTPError("No attempts made (read_retries=0)")


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


def _llm_chat(user_content: str, max_tokens: int = 512, backend: str = "openai_compatible") -> str:
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
    # Chain-of-thought control for Qwen/thinking models.
    # Default: DISABLE thinking for eval — Qwen3.5's "Thinking Process:" output
    # fills the max_tokens budget (256) before producing an actual answer, resulting
    # in truncated reasoning with no answer.  The softened prompts already guide the
    # model to reason about dates/timestamps inline.
    # Set LLM_EVAL__ENABLE_THINKING=1 to opt back in (requires larger max_tokens).
    _is_thinking_model = "qwen" in model.lower()
    _enable_thinking = os.environ.get("LLM_EVAL__ENABLE_THINKING", "").strip() in ("1", "true")
    _extra_body = (
        {"chat_template_kwargs": {"enable_thinking": False}}
        if _is_thinking_model and not _enable_thinking
        else None
    )

    import random as _rand

    _max_retries = 5
    _backoff_base = 2.0
    _backoff_cap = 30.0

    def _do_request() -> str:
        client = OpenAI(base_url=base_url, api_key=api_key)
        create_kwargs: dict = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if _extra_body:
            create_kwargs["extra_body"] = _extra_body
        resp = client.chat.completions.create(**create_kwargs)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        if msg is None:
            return ""
        content = getattr(msg, "content", None)
        return (content or "").strip()

    # Retryable exception types
    _retryable: tuple[type[Exception], ...] = (ConnectionError, OSError, TimeoutError)
    try:
        import openai as _openai_mod

        _retryable = (
            *_retryable,
            _openai_mod.APIConnectionError,
            _openai_mod.APITimeoutError,
            _openai_mod.RateLimitError,
            _openai_mod.InternalServerError,
        )
    except (ImportError, AttributeError):
        pass

    for attempt in range(_max_retries):
        try:
            out = _do_request()
            if out:
                return out
            # Empty response — retry once after short delay
            if attempt == 0:
                time.sleep(2.0)
                continue
            return ""
        except _retryable as e:
            if attempt < _max_retries - 1:
                delay = min(_backoff_cap, _backoff_base * (2**attempt)) + _rand.uniform(0, 1)
                time.sleep(delay)
                continue
            print(f"LLM API Error after {_max_retries} retries: {e}", flush=True)
            return ""
        except Exception as e:
            print(f"LLM API Error: {e}", flush=True)
            return ""
    return ""


def _ingest_sample(
    cml_url: str,
    cml_api_key: str,
    sample_idx: int,
    sample: dict,
    ingestion_delay_sec: float,
    pbar: tqdm | None = None,
) -> None:
    """Ingest one sample: batch turns for tenant lp-{sample_idx}."""
    tenant_id = f"lp-{sample_idx}"
    input_prompt = sample.get("input_prompt", "")
    turns = _parse_input_prompt_into_turns(input_prompt)
    turn_timestamps = sample.get("turn_timestamps") or []

    # Build all turn payloads
    turn_payloads: list[dict] = []
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
        payload: dict = {
            "content": content,
            "session_id": session_id,
            "metadata": {
                "locomo_plus_idx": sample_idx,
                "turn_idx": j,
                "speaker": speaker,
                "date_str": date_str or "",
                "session_idx": int(session_id.split("_")[-1]) if "_" in session_id else 1,
            },
            "turn_id": f"turn_{j}",
        }
        if ts_iso is not None:
            payload["timestamp"] = ts_iso
        turn_payloads.append(payload)

    # Send in batches
    for batch_start in range(0, len(turn_payloads), _WRITE_BATCH_SIZE):
        batch = turn_payloads[batch_start : batch_start + _WRITE_BATCH_SIZE]
        _cml_write_batch(cml_url, cml_api_key, tenant_id, batch)
        if pbar is not None:
            pbar.update(len(batch))
        if ingestion_delay_sec > 0:
            time.sleep(ingestion_delay_sec)


def _build_conversation_groups(
    samples: list[dict],
) -> tuple[dict[int, int], dict[int, list[int]]]:
    """Group samples sharing the same conversation (input_prompt minus Question:).

    Returns:
        sample_to_conv: maps sample_idx → canonical_idx (lowest idx in group)
        conv_to_samples: maps canonical_idx → list of all sample indices in group

    LoCoMo-Plus frequently assigns 20-260 different questions to the same
    conversation.  By ingesting only once per conversation (using the canonical
    index as tenant), we avoid massive data duplication.
    """
    import hashlib as _hashlib

    # Extract conversation prefix (everything before "Question:" line)
    def _conv_key(sample: dict) -> str:
        ip = sample.get("input_prompt", "")
        # Strip trailing question — _parse_input_prompt_into_turns stops at "Question:"
        idx = ip.find("\nQuestion:")
        prefix = ip[:idx] if idx != -1 else ip
        return _hashlib.sha256(prefix.encode()).hexdigest()

    hash_to_indices: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        h = _conv_key(s)
        hash_to_indices.setdefault(h, []).append(i)

    sample_to_conv: dict[int, int] = {}
    conv_to_samples: dict[int, list[int]] = {}
    for indices in hash_to_indices.values():
        canonical = min(indices)
        conv_to_samples[canonical] = indices
        for idx in indices:
            sample_to_conv[idx] = canonical

    return sample_to_conv, conv_to_samples


def phase_a_ingestion(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
    ingestion_workers: int = 10,
    checkpoint_file: Path | None = None,
) -> None:
    samples_to_ingest = samples[:limit] if limit else samples

    # Group samples sharing the same conversation to avoid redundant ingestion.
    _sample_to_conv, conv_to_samples = _build_conversation_groups(samples_to_ingest)
    canonical_indices = sorted(conv_to_samples.keys())
    total_samples = len(samples_to_ingest)
    total_convos = len(canonical_indices)
    dedup_savings = total_samples - total_convos
    if dedup_savings > 0:
        print(
            f"\n[Phase A] Conversation dedup: {total_samples} samples → {total_convos} unique conversations "
            f"({dedup_savings} redundant ingestions skipped)",
            flush=True,
        )

    # Load checkpoint — skip already-completed conversations
    completed: set[int] = set()
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            completed = set(data.get("completed_indices", []))
        except (json.JSONDecodeError, OSError):
            pass

    # Only ingest canonical (first) sample per conversation group
    pending = [(i, samples_to_ingest[i]) for i in canonical_indices if i not in completed]

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
            # Mark all samples in this conversation group as completed
            group = conv_to_samples.get(idx, [idx])
            completed.update(group)
            try:
                checkpoint_file.write_text(
                    json.dumps(
                        {
                            "completed_indices": sorted(completed),
                            "conversation_dedup": True,
                        },
                        ensure_ascii=False,
                    ),
                    encoding="utf-8",
                )
            except OSError:
                pass

    total_turns = sum(
        len(_parse_input_prompt_into_turns(s.get("input_prompt", ""))) for _, s in pending
    )
    skipped = len(canonical_indices) - len(pending)
    skip_msg = f" ({skipped} already ingested)" if skipped else ""
    print(
        f"\n[Phase A] Ingesting {len(pending)} conversations ({total_turns} turns){skip_msg}"
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

    # Only consolidate canonical (unique conversation) tenants
    _, conv_to_samples = _build_conversation_groups(samples_scope)
    canonical_indices = sorted(conv_to_samples.keys())

    # Load checkpoint — skip already-consolidated tenants
    completed_tenants: set[str] = set()
    if checkpoint_file is not None and checkpoint_file.exists():
        try:
            data = json.loads(checkpoint_file.read_text(encoding="utf-8"))
            completed_tenants = set(data.get("completed_tenants", []))
        except (json.JSONDecodeError, OSError):
            pass

    pending_indices = [i for i in canonical_indices if f"lp-{i}" not in completed_tenants]
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

    # Map sample index → canonical conversation tenant for deduped ingestion.
    # When ingestion used conversation grouping, multiple samples share one tenant.
    # The ingestion checkpoint records which ingestion mode was used.
    ingestion_checkpoint = out_dir / "locomo_ingestion_checkpoint.json"
    use_conv_tenants = False
    if ingestion_checkpoint.exists():
        try:
            ck = json.loads(ingestion_checkpoint.read_text(encoding="utf-8"))
            use_conv_tenants = bool(ck.get("conversation_dedup"))
        except (json.JSONDecodeError, OSError):
            pass
    if use_conv_tenants:
        sample_to_conv, _ = _build_conversation_groups(samples_qa)
    else:
        sample_to_conv = {i: i for i in range(len(samples_qa))}
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
            tenant_id = f"lp-{sample_to_conv.get(i, i)}"
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
            # Treat context as empty if it's just the markdown header
            effective_context = (
                llm_context
                if len(llm_context or "") > 30
                else "(No relevant memories found for this query.)"
            )
            if category == "Cognitive":
                user_content = effective_context + "\n\n" + COGNITIVE_PROMPT.format(trigger)
            else:
                user_content = effective_context + "\n\n" + QA_PROMPT.format(trigger)
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
        predictions = generate_batch(qa_model, conversations, temperature=0.0, max_tokens=512)

        for m, raw_prediction in zip(meta, predictions, strict=False):
            prediction = _extract_answer(raw_prediction)
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
        sample_error_count = 0
        empty_context_count = 0
        for i in _tqdm(
            range(start_index, len(samples_qa)), desc="QA", unit="sample", disable=False
        ):
            sample = samples_qa[i]
            tenant_id = f"lp-{sample_to_conv.get(i, i)}"
            trigger = (sample.get("trigger") or "").strip()
            category = sample.get("category", "")
            if category == "Cognitive":
                question_input = trigger or "Context dialogue (cue awareness)"
            else:
                question_input = trigger
            ctx_len = 0
            try:
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
                ctx_len = len(llm_context or "")
                if not llm_context:
                    empty_context_count += 1
                if QA_READ_DELAY_SEC > 0:
                    time.sleep(QA_READ_DELAY_SEC)

                # Treat context as empty if it's just the markdown header with no content
                effective_context = (
                    llm_context if ctx_len > 30 else "(No relevant memories found for this query.)"
                )
                if category == "Cognitive":
                    user_content = effective_context + "\n\n" + COGNITIVE_PROMPT.format(trigger)
                else:
                    user_content = effective_context + "\n\n" + QA_PROMPT.format(trigger)

                prediction = _extract_answer(_llm_chat(user_content, backend=backend))
            except Exception as exc:
                sample_error_count += 1
                prediction = f"[Error: {exc}]"
                _progress_write(f"\n[Phase B] Warning: sample {i} ({category}) failed: {exc}")
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
                "context_length": ctx_len,
            }
            if sample.get("time_gap"):
                record["time_gap"] = sample["time_gap"]
            records.append(record)
            # Checkpoint every 50 samples instead of every sample for performance
            if len(records) % 50 == 0 or i == len(samples_qa) - 1:
                pred_file.write_text(
                    json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
                )
        if empty_context_count:
            print(
                f"\n[Phase B] {empty_context_count}/{len(samples_qa)} samples had empty retrieval context.",
                flush=True,
            )
        if sample_error_count:
            print(
                f"\n[Phase B] {sample_error_count}/{len(samples_qa)} samples had errors "
                "(predictions stored as '[Error: ...]').",
                flush=True,
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
