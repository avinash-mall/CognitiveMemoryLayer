#!/usr/bin/env python3
"""
Locomo-Plus evaluation with CML as the RAG backend.

Unified six-category eval: LoCoMo (multi-hop, temporal, common-sense, single-hop, adversarial)
plus Cognitive. Uses LLM-as-judge for scoring (correct=1, partial=0.5, wrong=0).

Phase A: Ingest each unified sample's input_prompt (parsed into turns) into CML.
Phase B: For each sample, CML read with query=trigger, then Ollama generates answer.
Phase C: LLM-as-judge scores predictions; writes judged JSON and summary.

Usage (from project root):
  set PYTHONPATH=evaluation/locomo_plus
  python evaluation/scripts/eval_locomo_plus.py --unified-file evaluation/locomo_plus/data/unified_input_samples_v2.json --out-dir evaluation/outputs
"""

from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

# Load project .env so LLM__* (and CML_*, OLLAMA_*) are available for judge and QA
try:
    from dotenv import load_dotenv

    _repo_root = Path(__file__).resolve().parent.parent.parent
    load_dotenv(_repo_root / ".env")
except ImportError:
    pass
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime

import requests  # type: ignore[import-untyped]
from tqdm import tqdm

INGESTION_DELAY_SEC = 0.2
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


def _ensure_locomo_plus_path() -> None:
    script_dir = Path(__file__).resolve().parent
    locomo_plus = script_dir.parent / "locomo_plus"
    if locomo_plus.is_dir() and str(locomo_plus) not in sys.path:
        sys.path.insert(0, str(locomo_plus))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Locomo-Plus unified eval with CML + Ollama")
    p.add_argument(
        "--unified-file", type=str, required=True, help="Path to unified_input_samples_v2.json"
    )
    p.add_argument("--out-dir", type=str, default="evaluation/outputs")
    p.add_argument(
        "--cml-url", type=str, default=os.environ.get("CML_BASE_URL", "http://localhost:8000")
    )
    p.add_argument("--cml-api-key", type=str, default=os.environ.get("CML_API_KEY", "test-key"))
    p.add_argument(
        "--ollama-url",
        type=str,
        default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"),
    )
    p.add_argument(
        "--ollama-model", type=str, default=os.environ.get("OLLAMA_QA_MODEL", "gpt-oss:20b")
    )
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
        default=os.environ.get("LLM__MODEL", "gpt-4o-mini"),
        help="Model for LLM-as-judge (default: LLM__MODEL from .env or gpt-4o-mini)",
    )
    p.add_argument("--verbose", action="store_true", help="Emit per-sample retrieval diagnostics")
    p.add_argument(
        "--ingestion-workers",
        type=int,
        default=10,
        metavar="N",
        help="Number of concurrent workers for Phase A ingestion (default 10)",
    )
    return p.parse_args()


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
                resp = requests.post(url, json=payload, headers=headers, timeout=write_timeout)
                if resp.status_code in [429, 500]:
                    if attempt == _CML_WRITE_MAX_429_ATTEMPTS - 1:
                        resp.raise_for_status()
                    wait = min(_CML_WRITE_BACKOFF_CAP_SEC, 5 * (2**attempt))
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                return
            except (requests.exceptions.ConnectionError, OSError):
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
    url = f"{base_url.rstrip('/')}/api/v1/dashboard{path}"
    headers = {"X-API-Key": api_key, "Content-Type": "application/json"}
    for attempt in range(max_attempts):
        resp = requests.post(url, json=body, headers=headers, timeout=120)
        if resp.status_code == 429:
            if attempt == max_attempts - 1:
                resp.raise_for_status()
            wait = min(backoff_cap_sec, 5 * (2**attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return resp.json()
    raise requests.exceptions.HTTPError("429 Too Many Requests", response=resp)


def _cml_read(
    base_url: str,
    api_key: str,
    tenant_id: str,
    query: str,
    max_results: int,
    return_full: bool = False,
) -> str | tuple[str, dict]:
    """Read memories from CML. If *return_full*, also return the raw JSON response."""
    url = f"{base_url.rstrip('/')}/api/v1/memory/read"
    payload = {"query": query, "format": "llm_context", "max_results": max_results}
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id}
    for attempt in range(_CML_READ_MAX_429_ATTEMPTS):
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
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
    raise requests.exceptions.HTTPError("429 Too Many Requests", response=resp)


def _ollama_chat(base_url: str, model: str, user_content: str, max_tokens: int = 256) -> str:
    base = base_url.rstrip("/")
    messages = [{"role": "user", "content": user_content}]
    url = f"{base}/api/chat"
    payload = {
        "model": model,
        "messages": messages,
        "stream": False,
        "options": {"num_predict": max_tokens, "temperature": 0},
    }
    resp = requests.post(url, json=payload, timeout=120)
    if resp.status_code == 200:
        msg = (resp.json().get("message") or {}).get("content", "")
        return (msg or "").strip()
    if resp.status_code == 404:
        url_openai = f"{base}/v1/chat/completions"
        resp = requests.post(
            url_openai,
            json={"model": model, "messages": messages, "max_tokens": max_tokens, "temperature": 0},
            timeout=120,
        )
        if resp.status_code == 200:
            choices = resp.json().get("choices", [])
            if choices:
                return (choices[0].get("message") or {}).get("content", "").strip()
    resp.raise_for_status()
    return ""


def _ollama_available(base_url: str, model: str | None = None) -> None:
    base = base_url.rstrip("/")
    try:
        r = requests.get(f"{base}/api/tags", timeout=5)
        if r.status_code == 200 and model:
            models = r.json().get("models") or []
            model_base = model.split(":")[0]
            names = [(m.get("name") or "").split(":")[0] for m in models]
            if names and model_base not in names:
                raise RuntimeError(f"Model {model!r} not in Ollama. Pull with: ollama pull {model}")
        return
    except RuntimeError:
        raise
    except requests.RequestException:
        pass
    raise RuntimeError(f"Cannot reach Ollama at {base_url}. Start with 'ollama serve'.")


def _ingest_sample(
    cml_url: str,
    cml_api_key: str,
    sample_idx: int,
    sample: dict,
    ingestion_delay_sec: float,
) -> None:
    """Ingest one sample: write all turns sequentially for tenant lp-{sample_idx}."""
    tenant_id = f"lp-{sample_idx}"
    input_prompt = sample.get("input_prompt", "")
    turns = _parse_input_prompt_into_turns(input_prompt)
    for j, (session_id, content, date_str) in enumerate(turns):
        speaker = content.split(":")[0].strip() if ":" in content else "unknown"
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
        if ingestion_delay_sec > 0:
            time.sleep(ingestion_delay_sec)


def phase_a_ingestion(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
    ingestion_workers: int = 10,
) -> None:
    samples_to_ingest = samples[:limit] if limit else samples
    # Apply per-turn delay only when single-threaded (rate limiting)
    ingestion_delay = INGESTION_DELAY_SEC if ingestion_workers == 1 else 0.0

    print(
        f"\n[Phase A] Ingesting {len(samples_to_ingest)} samples into CML ({ingestion_workers} workers)...",
        flush=True,
    )

    if ingestion_workers <= 1:
        for i, sample in enumerate(tqdm(samples_to_ingest, desc="Ingestion", unit="sample")):
            _ingest_sample(cml_url, cml_api_key, i, sample, ingestion_delay)
    else:
        with ThreadPoolExecutor(max_workers=ingestion_workers) as executor:
            futures = {
                executor.submit(_ingest_sample, cml_url, cml_api_key, i, sample, ingestion_delay): i
                for i, sample in enumerate(samples_to_ingest)
            }
            for future in tqdm(
                as_completed(futures), total=len(futures), desc="Ingestion", unit="sample"
            ):
                future.result()  # Propagate any exception


def phase_ab_consolidation(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
) -> None:
    """Run consolidation then reconsolidation for each eval tenant (between Phase A and Phase B)."""
    samples_scope = samples[:limit] if limit else samples
    n = len(samples_scope)
    if n == 0:
        return
    print(
        f"\n[Phase A-B] Consolidation and reconsolidation for {n} tenants...",
        flush=True,
    )
    for i in range(n):
        tenant_id = f"lp-{i}"
        body = {"tenant_id": tenant_id, "user_id": None}
        try:
            _dashboard_post(cml_url, cml_api_key, "/consolidate", body)
            _dashboard_post(cml_url, cml_api_key, "/reconsolidate", body)
        except requests.RequestException as e:
            raise RuntimeError(f"Dashboard step failed for tenant {tenant_id}: {e}") from e
    print(f"  Completed for {n} tenants.", flush=True)


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
    ollama_url: str,
    ollama_model: str,
    max_results: int,
    limit: int | None,
    verbose: bool = False,
) -> list[dict]:
    _ollama_available(ollama_url, ollama_model)
    samples_qa = samples[:limit] if limit else samples
    print(f"\n[Phase B] Running QA on {len(samples_qa)} samples ({ollama_model})...", flush=True)
    records: list[dict] = []
    for i, sample in enumerate(tqdm(samples_qa, desc="QA", unit="sample")):
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
            tqdm.write(
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

        prediction = _ollama_chat(ollama_url, ollama_model, user_content)
        ground_truth = sample.get("answer")
        if ground_truth is None or (isinstance(ground_truth, str) and ground_truth.strip() == ""):
            ground_truth = ""
        record = {
            "question_input": question_input,
            "evidence": sample.get("evidence", ""),
            "category": category,
            "ground_truth": ground_truth,
            "prediction": prediction,
            "model": f"{ollama_model}_cml",
        }
        if sample.get("time_gap"):
            record["time_gap"] = sample["time_gap"]
        records.append(record)
    return records


def phase_c_judge(records: list[dict], out_dir: Path, judge_model: str) -> None:
    _ensure_locomo_plus_path()
    from task_eval.llm_as_judge import run_judge

    print(
        f"\n[Phase C] LLM-as-judge scoring {len(records)} predictions ({judge_model})...",
        flush=True,
    )

    class Args:
        input_file = ""
        out_file = str(out_dir / "locomo_plus_qa_cml_judged.json")
        model = judge_model
        backend = "call_llm"
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


def main() -> None:
    args = parse_args()
    _ensure_locomo_plus_path()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.score_only:
        pred_file = out_dir / "locomo_plus_qa_cml_predictions.json"
        if not pred_file.exists():
            sys.exit(f"Score-only requires {pred_file}")
        records = json.loads(pred_file.read_text(encoding="utf-8"))
        phase_c_judge(records, out_dir, args.judge_model)
        return

    from task_eval.utils import load_unified_samples

    samples = load_unified_samples(args.unified_file)
    total = len(samples)
    if args.limit_samples:
        samples = samples[: args.limit_samples]
        print(f"Loaded {len(samples)} samples (limit; total available: {total})", flush=True)
    else:
        print(f"Loaded {len(samples)} samples", flush=True)

    if not args.skip_ingestion:
        phase_a_ingestion(
            samples,
            args.cml_url,
            args.cml_api_key,
            args.limit_samples,
            ingestion_workers=args.ingestion_workers,
        )

    if not args.skip_consolidation:
        phase_ab_consolidation(samples, args.cml_url, args.cml_api_key, args.limit_samples)

    records = phase_b_qa(
        samples,
        args.cml_url,
        args.cml_api_key,
        args.ollama_url,
        args.ollama_model,
        args.max_results,
        args.limit_samples,
        verbose=args.verbose,
    )

    phase_c_judge(records, out_dir, args.judge_model)
    print(f"Predictions: {out_dir / 'locomo_plus_qa_cml_predictions.json'}")


if __name__ == "__main__":
    main()
