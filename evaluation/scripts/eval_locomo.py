#!/usr/bin/env python3
"""
LoCoMo evaluation with CML as the RAG backend.

Phase A: Ingest each sample's conversation into CML (one tenant per sample_id).
Phase B: For each QA item, read from CML (llm_context), call Ollama to generate answer.
Phase C: Score with LoCoMo's eval_question_answering and analyze_aggr_acc.

Usage:
  Set PYTHONPATH to include the LoCoMo repo root (e.g. evaluation/locomo).
  From project root:
    set PYTHONPATH=evaluation/locomo
    python evaluation/scripts/eval_locomo.py --data-file evaluation/locomo/data/locomo10.json --out-dir evaluation/outputs

  Env: CML_BASE_URL, CML_API_KEY, OLLAMA_BASE_URL, OLLAMA_QA_MODEL (optional).
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import requests
from tqdm import tqdm

# Throttle writes (set AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600 for bulk eval)
INGESTION_DELAY_SEC = 0.2
# 429 retry: max attempts and backoff (seconds); cap so we don't wait forever
_CML_WRITE_MAX_429_ATTEMPTS = 15
_CML_WRITE_BACKOFF_CAP_SEC = 65

# QA prompts aligned with LoCoMo gpt_utils
QA_PROMPT = """
Based on the above context, write an answer in the form of a short phrase for the following question. Answer with exact words from the context whenever possible.

Question: {} Short answer:
"""

QA_PROMPT_CAT_5 = """
Based on the above context, answer the following question.

Question: {} Short answer:
"""


def _ensure_locomo_path() -> None:
    """Ensure LoCoMo repo is on sys.path for task_eval imports."""
    script_dir = Path(__file__).resolve().parent
    # evaluation/scripts -> evaluation -> evaluation/locomo
    locomo_root = script_dir.parent / "locomo"
    if locomo_root.is_dir() and str(locomo_root) not in sys.path:
        sys.path.insert(0, str(locomo_root))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="LoCoMo QA evaluation with CML + Ollama")
    p.add_argument("--data-file", type=str, required=True, help="Path to locomo10.json")
    p.add_argument("--out-dir", type=str, default="evaluation/outputs", help="Output directory for QA and stats JSON")
    p.add_argument("--cml-url", type=str, default=os.environ.get("CML_BASE_URL", "http://localhost:8000"), help="CML API base URL")
    p.add_argument("--cml-api-key", type=str, default=os.environ.get("CML_API_KEY", "test-key"), help="CML API key")
    p.add_argument("--ollama-url", type=str, default=os.environ.get("OLLAMA_BASE_URL", "http://localhost:11434"), help="Ollama base URL (no /v1)")
    p.add_argument("--ollama-model", type=str, default=os.environ.get("OLLAMA_QA_MODEL", "gpt-oss-20b"), help="Ollama model for QA")
    p.add_argument("--max-results", type=int, default=25, help="CML read max_results (top-k)")
    p.add_argument("--limit-samples", type=int, default=None, help="Limit to first N samples (for testing)")
    p.add_argument("--skip-ingestion", action="store_true", help="Skip Phase A (use existing CML state)")
    p.add_argument("--overwrite", action="store_true", help="Overwrite existing predictions in output file")
    p.add_argument("--no-eval-mode", dest="eval_mode", action="store_false", default=True, help="Disable X-Eval-Mode on writes and gating stats")
    p.add_argument("--log-timing", action="store_true", help="Log latency and token usage; write timing summary JSON")
    return p.parse_args()


def _parse_timestamp(s: str | None) -> str | None:
    """Return ISO timestamp string if parseable, else None (API expects ISO datetime)."""
    if not s or not s.strip():
        return None
    # Accept ISO-like format
    try:
        from datetime import datetime
        if "T" in s and "-" in s:
            dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
            return dt.isoformat()
    except Exception:
        pass
    # Try common Locomo-style "1:56 pm on 8 May, 2023" (optional: python-dateutil)
    try:
        from dateutil import parser as dateutil_parser
        dt = dateutil_parser.parse(s)
        return dt.isoformat()
    except Exception:
        pass
    return None  # omit timestamp so API does not reject


def _cml_write(
    base_url: str,
    api_key: str,
    tenant_id: str,
    content: str,
    session_id: str,
    timestamp: str | None,
    metadata: dict,
    turn_id: str | None,
    eval_mode: bool = True,
) -> dict | None:
    """Write one turn to CML. When eval_mode=True, returns response JSON (for eval_outcome/eval_reason)."""
    url = f"{base_url.rstrip('/')}/api/v1/memory/write"
    payload = {
        "content": content,
        "session_id": session_id,
        "metadata": metadata,
        "turn_id": turn_id,
    }
    iso_ts = _parse_timestamp(timestamp)
    if iso_ts:
        payload["timestamp"] = iso_ts
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id}
    if eval_mode:
        headers["X-Eval-Mode"] = "true"
    last_429_response = None
    for attempt in range(_CML_WRITE_MAX_429_ATTEMPTS):
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        if resp.status_code == 429:
            last_429_response = resp
            if attempt == _CML_WRITE_MAX_429_ATTEMPTS - 1:
                break
            retry_after = resp.headers.get("Retry-After")
            if retry_after is not None:
                try:
                    wait = min(int(retry_after), _CML_WRITE_BACKOFF_CAP_SEC)
                except ValueError:
                    wait = min(_CML_WRITE_BACKOFF_CAP_SEC, 5 * (2**attempt))
            else:
                wait = min(_CML_WRITE_BACKOFF_CAP_SEC, 5 * (2**attempt))
            time.sleep(wait)
            continue
        resp.raise_for_status()
        if eval_mode:
            return resp.json()
        return None
    raise requests.exceptions.HTTPError(
        "429 Too Many Requests (rate limit). Set AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600 "
        "in project root .env and restart the API, then re-run. See ProjectPlan/LocomoEval/RunEvaluation.md.",
        response=last_429_response,
    )


def _cml_read(
    base_url: str,
    api_key: str,
    tenant_id: str,
    query: str,
    max_results: int,
) -> tuple[str, list[str]]:
    """Return (llm_context, list of dia_ids from retrieved memories for recall)."""
    url = f"{base_url.rstrip('/')}/api/v1/memory/read"
    payload = {
        "query": query,
        "format": "llm_context",
        "max_results": max_results,
    }
    resp = requests.post(
        url,
        json=payload,
        headers={"X-API-Key": api_key, "X-Tenant-ID": tenant_id},
        timeout=60,
    )
    resp.raise_for_status()
    data = resp.json()
    llm_context = data.get("llm_context") or ""
    # Collect dia_ids from retrieved memories for recall
    dia_ids: list[str] = []
    for mem in (
        data.get("memories", [])
        + data.get("episodes", [])
        + data.get("facts", [])
        + data.get("preferences", [])
    ):
        meta = mem.get("metadata") or {}
        if "dia_id" in meta:
            dia_ids.append(meta["dia_id"])
    return llm_context, dia_ids


def _ollama_chat(
    base_url: str,
    model: str,
    user_content: str,
    max_tokens: int = 64,
    return_usage: bool = False,
) -> str | tuple[str, dict]:
    """Return content string, or (content, usage_dict) when return_usage=True."""
    url = f"{base_url.rstrip('/')}/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": user_content}],
        "max_tokens": max_tokens,
        "temperature": 0,
    }
    resp = requests.post(url, json=payload, timeout=120)
    resp.raise_for_status()
    data = resp.json()
    choices = data.get("choices", [])
    content = ""
    if choices:
        content = (choices[0].get("message") or {}).get("content", "").strip()
    usage = (data.get("usage") or {}) if return_usage else {}
    if return_usage:
        return content, usage
    return content


def _get_session_nums(conversation: dict) -> list[int]:
    keys = [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")]
    return sorted(int(k.split("_")[-1]) for k in keys)


def phase_a_ingestion(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
    eval_mode: bool = True,
    gating_results: list[dict] | None = None,
) -> None:
    """Ingest each sample's conversation into CML (one tenant per sample_id)."""
    if gating_results is None and eval_mode:
        gating_results = []
    for sample in tqdm(
        (samples[:limit] if limit is not None else samples), desc="Ingestion"
    ):
        tenant_id = sample["sample_id"]
        conv = sample["conversation"]
        session_nums = _get_session_nums(conv)
        for k in session_nums:
            session_id = f"session_{k}"
            date_time_str = conv.get(f"session_{k}_date_time", "")
            session_timestamp = date_time_str if date_time_str else None
            for dialog in conv.get(f"session_{k}", []):
                content = dialog["speaker"] + ": " + dialog["text"]
                if dialog.get("blip_caption"):
                    content += " [shared: " + dialog["blip_caption"] + "]"
                out = _cml_write(
                    base_url=cml_url,
                    api_key=cml_api_key,
                    tenant_id=tenant_id,
                    content=content,
                    session_id=session_id,
                    timestamp=session_timestamp,
                    metadata={
                        "locomo_session_date_time": date_time_str,
                        "speaker": dialog["speaker"],
                        "dia_id": dialog.get("dia_id"),
                    },
                    turn_id=dialog.get("dia_id"),
                    eval_mode=eval_mode,
                )
                if eval_mode and out and gating_results is not None:
                    gating_results.append({
                        "eval_outcome": out.get("eval_outcome"),
                        "eval_reason": out.get("eval_reason"),
                    })
                time.sleep(INGESTION_DELAY_SEC)


def _get_cat5_answer(model_prediction: str, answer_key: dict) -> str:
    """Map model output to (a) or (b) for adversarial QA (category 5). Returns the answer text."""
    raw = model_prediction.strip().lower()
    if len(raw) == 1:
        return answer_key["a"] if "a" in raw else answer_key["b"]
    if "(a)" in model_prediction:
        return answer_key["a"]
    if "(b)" in model_prediction:
        return answer_key["b"]
    return model_prediction


def phase_b_qa(
    samples: list[dict],
    out_data_by_id: dict[str, dict],
    cml_url: str,
    cml_api_key: str,
    ollama_url: str,
    ollama_model: str,
    max_results: int,
    prediction_key: str,
    limit: int | None,
    overwrite: bool,
    log_timing: bool = False,
    timing_entries: list[dict] | None = None,
) -> None:
    """For each sample and each QA, read from CML, call Ollama, store prediction and context for recall."""
    if log_timing and timing_entries is None:
        timing_entries = []
    for sample in tqdm(
        (samples[:limit] if limit is not None else samples), desc="QA"
    ):
        tenant_id = sample["sample_id"]
        out_data = out_data_by_id[sample["sample_id"]]
        for i, qa_item in enumerate(out_data["qa"]):
            if prediction_key in qa_item and not overwrite:
                continue
            question = qa_item["question"]
            category = qa_item.get("category", 3)
            t0 = time.perf_counter()
            llm_context, dia_ids = _cml_read(cml_url, cml_api_key, tenant_id, question, max_results)
            cml_read_sec = time.perf_counter() - t0
            if category == 2:
                question_for_prompt = question + " Use DATE of CONVERSATION to answer with an approximate date."
            elif category == 5:
                # Adversarial: present two options (a) and (b)
                if random.random() < 0.5:
                    question_for_prompt = question + " Select the correct answer: (a) {} (b) {}. Short answer:".format(
                        "Not mentioned in the conversation", qa_item["answer"]
                    )
                    answer_key = {"a": "Not mentioned in the conversation", "b": qa_item["answer"]}
                else:
                    question_for_prompt = question + " Select the correct answer: (a) {} (b) {}. Short answer:".format(
                        qa_item["answer"], "Not mentioned in the conversation"
                    )
                    answer_key = {"a": qa_item["answer"], "b": "Not mentioned in the conversation"}
            else:
                question_for_prompt = question
                answer_key = {}
            prompt = (QA_PROMPT_CAT_5 if category == 5 else QA_PROMPT).format(question_for_prompt)
            user_content = (llm_context or "(No retrieved context.)") + "\n\n" + prompt
            t1 = time.perf_counter()
            if log_timing and timing_entries is not None:
                result = _ollama_chat(
                    ollama_url, ollama_model, user_content, return_usage=True
                )
                answer, usage = (result[0], result[1]) if isinstance(result, tuple) else (result, {})
                ollama_sec = time.perf_counter() - t1
                timing_entries.append({
                    "sample_id": tenant_id,
                    "qa_index": i,
                    "cml_read_sec": round(cml_read_sec, 3),
                    "ollama_sec": round(ollama_sec, 3),
                    "prompt_tokens": usage.get("prompt_tokens"),
                    "completion_tokens": usage.get("completion_tokens"),
                    "total_tokens": usage.get("total_tokens"),
                })
            else:
                answer = _ollama_chat(ollama_url, ollama_model, user_content)
                if isinstance(answer, tuple):
                    answer = answer[0]
            if category == 5:
                ans_str = answer[0] if isinstance(answer, tuple) else answer
                answer = _get_cat5_answer(ans_str, answer_key)
            out_data["qa"][i][prediction_key] = answer.strip() if isinstance(answer, str) else str(answer).strip()
            if dia_ids:
                out_data["qa"][i][prediction_key + "_context"] = dia_ids


def main() -> None:
    args = parse_args()
    _ensure_locomo_path()

    with open(args.data_file, encoding="utf-8") as f:
        samples = json.load(f)
    if not isinstance(samples, list):
        samples = [samples]

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_file = out_dir / "locomo10_qa_cml.json"
    stats_file = out_dir / "locomo10_qa_cml_stats.json"

    # Build per-sample output (copy qa from data, preserve existing keys if loading)
    if out_file.exists():
        existing = {d["sample_id"]: d for d in json.loads(out_file.read_text(encoding="utf-8"))}
    else:
        existing = {}
    out_data_by_id = {}
    for s in samples:
        sid = s["sample_id"]
        qa_list = existing[sid]["qa"] if sid in existing else s["qa"]
        out_data_by_id[sid] = {
            "sample_id": sid,
            "qa": [dict(q) for q in qa_list],
        }

    prediction_key = f"cml_top_{args.max_results}_prediction"
    model_key = f"{args.ollama_model}_cml_top_{args.max_results}"

    gating_results: list[dict] = []
    if not args.skip_ingestion:
        phase_a_ingestion(
            samples,
            args.cml_url,
            args.cml_api_key,
            args.limit_samples,
            eval_mode=args.eval_mode,
            gating_results=gating_results,
        )
        if args.eval_mode and gating_results:
            stored = sum(1 for g in gating_results if g.get("eval_outcome") == "stored")
            skipped = sum(1 for g in gating_results if g.get("eval_outcome") == "skipped")
            reasons: dict[str, int] = {}
            for g in gating_results:
                r = (g.get("eval_reason") or "unknown").strip()
                if len(r) > 120:
                    r = r[:117] + "..."
                reasons[r] = reasons.get(r, 0) + 1
            gating_stats = {
                "total_writes": len(gating_results),
                "stored_count": stored,
                "skipped_count": skipped,
                "skip_reason_counts": reasons,
            }
            gating_file = out_dir / "locomo10_gating_stats.json"
            gating_file.write_text(json.dumps(gating_stats, indent=2), encoding="utf-8")
            print(f"Gating: {gating_file}")

    timing_entries: list[dict] = []
    phase_b_qa(
        samples,
        out_data_by_id,
        args.cml_url,
        args.cml_api_key,
        args.ollama_url,
        args.ollama_model,
        args.max_results,
        prediction_key,
        args.limit_samples,
        args.overwrite,
        log_timing=args.log_timing,
        timing_entries=timing_entries,
    )

    if args.log_timing and timing_entries:
        cml_times = [e["cml_read_sec"] for e in timing_entries]
        ollama_times = [e["ollama_sec"] for e in timing_entries]
        pt: list[int] = [
            int(v) for e in timing_entries
            if (v := e.get("prompt_tokens")) is not None and isinstance(v, (int, float))
        ]
        ct: list[int] = [
            int(v) for e in timing_entries
            if (v := e.get("completion_tokens")) is not None and isinstance(v, (int, float))
        ]
        tt: list[int] = [
            int(v) for e in timing_entries
            if (v := e.get("total_tokens")) is not None and isinstance(v, (int, float))
        ]
        timing_summary = {
            "per_question": timing_entries,
            "aggregate": {
                "count": len(timing_entries),
                "mean_cml_read_sec": round(sum(cml_times) / len(cml_times), 3) if cml_times else None,
                "mean_ollama_sec": round(sum(ollama_times) / len(ollama_times), 3) if ollama_times else None,
                "p95_cml_read_sec": round(sorted(cml_times)[min(len(cml_times) - 1, max(0, int(len(cml_times) * 0.95)))], 3) if cml_times else None,
                "p95_ollama_sec": round(sorted(ollama_times)[min(len(ollama_times) - 1, max(0, int(len(ollama_times) * 0.95)))], 3) if ollama_times else None,
                "total_prompt_tokens": sum(pt) if pt else None,
                "total_completion_tokens": sum(ct) if ct else None,
                "total_tokens": sum(tt) if tt else None,
            },
        }
        timing_file = out_dir / "locomo10_qa_cml_timing.json"
        timing_file.write_text(json.dumps(timing_summary, indent=2), encoding="utf-8")
        print(f"Timing: {timing_file}")

    # Write raw output (list of per-sample dicts) for analyze_aggr_acc
    out_list = list(out_data_by_id.values())
    out_file.write_text(json.dumps(out_list, indent=2), encoding="utf-8")

    # Phase C: LoCoMo scoring (requires bert_score, nltk, regex, numpy)
    has_predictions = any(
        prediction_key in q for d in out_list for q in d["qa"]
    )
    if has_predictions:
        from task_eval.evaluation import eval_question_answering  # type: ignore[import-untyped]
        from task_eval.evaluation_stats import analyze_aggr_acc  # type: ignore[import-untyped]

        for out_data in out_list:
            if not out_data["qa"]:
                continue
            exact_matches, _lengths, recall = eval_question_answering(
                out_data["qa"], prediction_key
            )
            for j, f1_val in enumerate(exact_matches):
                out_data["qa"][j][model_key + "_f1"] = round(f1_val, 3)
            if recall:
                for j, r in enumerate(recall):
                    out_data["qa"][j][model_key + "_recall"] = round(r, 3)

        out_file.write_text(json.dumps(out_list, indent=2), encoding="utf-8")

        analyze_aggr_acc(
            args.data_file,
            str(out_file),
            str(stats_file),
            model_key,
            model_key + "_f1",
            rag=True,
        )
        print(f"Stats:  {stats_file}")
    print(f"Output: {out_file}")


if __name__ == "__main__":
    main()
