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

import requests
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

COGNITIVE_PROMPT = """Your task: This is a memory-aware dialogue setting.
You are continuing or reflecting on a prior conversation.
Show that you are aware of the relevant memory or context from the evidence when you respond.
Your answer should naturally connect to or acknowledge that context.

Trigger/query: {}

Respond (short):"""


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
        "--score-only", action="store_true", help="Run only Phase C (judge) on existing predictions"
    )
    p.add_argument("--judge-model", type=str, default="gpt-4o-mini", help="Model for LLM-as-judge")
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
) -> None:
    url = f"{base_url.rstrip('/')}/api/v1/memory/write"
    payload = {
        "content": content,
        "session_id": session_id,
        "metadata": metadata,
        "turn_id": turn_id,
    }
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id, "X-Eval-Mode": "true"}
    for attempt in range(_CML_WRITE_MAX_429_ATTEMPTS):
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 429:
            if attempt == _CML_WRITE_MAX_429_ATTEMPTS - 1:
                resp.raise_for_status()
            wait = min(_CML_WRITE_BACKOFF_CAP_SEC, 5 * (2**attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        return
    raise requests.exceptions.HTTPError("429 Too Many Requests", response=resp)


def _cml_read(base_url: str, api_key: str, tenant_id: str, query: str, max_results: int) -> str:
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
        return (resp.json().get("llm_context") or "").strip()
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


def phase_a_ingestion(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
) -> None:
    for i, sample in enumerate(tqdm(samples[:limit] if limit else samples, desc="Ingestion")):
        tenant_id = f"lp-{i}"
        input_prompt = sample.get("input_prompt", "")
        turns = _parse_input_prompt_into_turns(input_prompt)
        for j, (session_id, content, _date) in enumerate(turns):
            _cml_write(
                cml_url,
                cml_api_key,
                tenant_id,
                content,
                session_id,
                {"locomo_plus_idx": i, "turn_idx": j},
                f"turn_{j}",
            )
            time.sleep(INGESTION_DELAY_SEC)


def phase_b_qa(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    ollama_url: str,
    ollama_model: str,
    max_results: int,
    limit: int | None,
) -> list[dict]:
    _ollama_available(ollama_url, ollama_model)
    records: list[dict] = []
    for i, sample in enumerate(tqdm(samples[:limit] if limit else samples, desc="QA")):
        tenant_id = f"lp-{i}"
        trigger = (sample.get("trigger") or "").strip()
        category = sample.get("category", "")
        if category == "Cognitive":
            question_input = trigger or "Context dialogue (cue awareness)"
        else:
            question_input = trigger
        llm_context = _cml_read(cml_url, cml_api_key, tenant_id, trigger, max_results)
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
    print(f"Judged output: {args.out_file}")
    print(f"Summary: {args.summary_file}")


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
    if args.limit_samples:
        samples = samples[: args.limit_samples]

    if not args.skip_ingestion:
        phase_a_ingestion(samples, args.cml_url, args.cml_api_key, args.limit_samples)

    records = phase_b_qa(
        samples,
        args.cml_url,
        args.cml_api_key,
        args.ollama_url,
        args.ollama_model,
        args.max_results,
        args.limit_samples,
    )

    phase_c_judge(records, out_dir, args.judge_model)
    print(f"Predictions: {out_dir / 'locomo_plus_qa_cml_predictions.json'}")


if __name__ == "__main__":
    main()
