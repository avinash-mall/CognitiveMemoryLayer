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
) -> None:
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
    for attempt in range(6):
        resp = requests.post(
            url,
            json=payload,
            headers={"X-API-Key": api_key, "X-Tenant-ID": tenant_id},
            timeout=60,
        )
        if resp.status_code == 429 and attempt < 5:
            time.sleep(3.0 * (attempt + 1))
            continue
        resp.raise_for_status()
        break


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


def _ollama_chat(base_url: str, model: str, user_content: str, max_tokens: int = 64) -> str:
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
    if not choices:
        return ""
    return (choices[0].get("message") or {}).get("content", "").strip()


def _get_session_nums(conversation: dict) -> list[int]:
    keys = [k for k in conversation if k.startswith("session_") and not k.endswith("_date_time")]
    return sorted(int(k.split("_")[-1]) for k in keys)


def phase_a_ingestion(
    samples: list[dict],
    cml_url: str,
    cml_api_key: str,
    limit: int | None,
) -> None:
    """Ingest each sample's conversation into CML (one tenant per sample_id)."""
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
                _cml_write(
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
                )
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
) -> None:
    """For each sample and each QA, read from CML, call Ollama, store prediction and context for recall."""
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
            llm_context, dia_ids = _cml_read(cml_url, cml_api_key, tenant_id, question, max_results)
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
            answer = _ollama_chat(ollama_url, ollama_model, user_content)
            if category == 5:
                answer = _get_cat5_answer(answer, answer_key)
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

    if not args.skip_ingestion:
        phase_a_ingestion(samples, args.cml_url, args.cml_api_key, args.limit_samples)

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
    )

    # Write raw output (list of per-sample dicts) for analyze_aggr_acc
    out_list = list(out_data_by_id.values())
    out_file.write_text(json.dumps(out_list, indent=2), encoding="utf-8")

    # Phase C: LoCoMo scoring (requires bert_score, nltk, regex, numpy)
    has_predictions = any(
        prediction_key in q for d in out_list for q in d["qa"]
    )
    if has_predictions:
        from task_eval.evaluation import eval_question_answering
        from task_eval.evaluation_stats import analyze_aggr_acc

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
