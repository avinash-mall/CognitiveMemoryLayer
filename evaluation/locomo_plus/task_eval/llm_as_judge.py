"""
LLM-as-a-judge: reads prediction JSON, scores via prompt templates,
writes JSON with judge_label, judge_reason, judge_score and prints summary.
Scoring: correct=1, partial=0.5, wrong=0.
"""

import argparse
import json
import re
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

try:
    from tqdm import tqdm
except ImportError:

    def tqdm(iterable, **kwargs):
        return iterable


import sys

_root = Path(__file__).resolve().parent.parent
if str(_root) not in sys.path:
    sys.path.insert(0, str(_root))

from task_eval.prompt import PROMPT_TEMPLATES
from task_eval.utils import call_model

LABEL_TO_SCORE = {"correct": 1.0, "partial": 0.5, "wrong": 0.0}


def get_judge_prompt(category: str, evidence: str, pred: str, gold: str = "") -> str:
    template = PROMPT_TEMPLATES.get(category) or PROMPT_TEMPLATES["default"]
    return template.format(gold=gold or "", pred=pred or "", evidence=evidence or "")


def label_to_score(label: str) -> float | None:
    """Map label to score. Returns None for unknown/failed labels (pipeline error)."""
    normalized = (label or "").strip().lower()
    if normalized in LABEL_TO_SCORE:
        return LABEL_TO_SCORE[normalized]
    return None


_VALID_LABELS = frozenset(LABEL_TO_SCORE.keys())


def _parse_judge_response(raw: str) -> tuple:
    label, reason = "", ""
    raw = (raw or "").strip()
    if not raw or raw.startswith("[API Error:") or raw.startswith("[vLLM Error:"):
        return "_parse_failed", raw[:200] if raw else ""

    # --- Strategy 1: regex for {"label": "...", "reason": "..."} anywhere ---
    try:
        m = re.search(
            r'\{[^{}]*"label"\s*:\s*["\']([^"\']+)["\'][^{}]*"reason"\s*:\s*["\']([^"\']*)["\']',
            raw,
            re.DOTALL,
        )
        if m:
            label, reason = m.group(1).strip().lower(), (m.group(2) or "").strip()
            if label in _VALID_LABELS:
                return label, reason
    except Exception:
        pass

    # --- Strategy 2: try json.loads on the full response ---
    try:
        obj = json.loads(raw)
        label = (obj.get("label") or "").strip().lower()
        reason = (obj.get("reason") or "").strip()
        if label in _VALID_LABELS:
            return label, reason
    except Exception:
        pass

    # --- Strategy 3: find last JSON-like object (handles CoT + JSON) ---
    try:
        candidates = re.findall(r'\{[^{}]*\}', raw)
        for candidate in reversed(candidates):
            try:
                obj = json.loads(candidate)
                lbl = (obj.get("label") or "").strip().lower()
                rsn = (obj.get("reason") or "").strip()
                if lbl in _VALID_LABELS:
                    return lbl, rsn
            except (json.JSONDecodeError, AttributeError):
                continue
    except Exception:
        pass

    # --- Strategy 4: substring fallback with negative checks ---
    raw_lower = raw.lower()
    if re.search(r'(?<!\bnot )(?<!\bin)correct', raw_lower) and "incorrect" not in raw_lower:
        label = "correct"
    elif "wrong" in raw_lower:
        label = "wrong"
    elif "partial" in raw_lower:
        label = "partial"

    if label and label in _VALID_LABELS:
        return label, raw[:200]

    return "_parse_failed", raw[:200] if raw else ""


_JUDGE_MAX_RETRIES = 3
_JUDGE_BACKOFF_BASE = 2.0


def _judge_one_record(record: dict, args) -> dict:
    r = dict(record)
    cat = r.get("category") or "default"
    evidence = r.get("evidence", "")
    pred = r.get("prediction", "")
    gold = r.get("ground_truth") or r.get("answer", "") or ""
    prompt = get_judge_prompt(cat, evidence, pred, gold)

    import random
    import time

    extra_body = getattr(args, "extra_body", None)
    label, reason = "_parse_failed", ""
    for attempt in range(_JUDGE_MAX_RETRIES):
        kwargs: dict = {
            "temperature": args.temperature,
            "max_tokens": args.max_tokens,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        raw = call_model(
            prompt,
            model=args.model,
            backend=args.backend,
            **kwargs,
        )
        label, reason = _parse_judge_response(raw)
        if label != "_parse_failed":
            break
        if attempt < _JUDGE_MAX_RETRIES - 1:
            delay = min(30, _JUDGE_BACKOFF_BASE * (2 ** attempt)) + random.uniform(0, 0.5)
            time.sleep(delay)

    score = label_to_score(label)
    r["judge_label"] = label
    r["judge_reason"] = reason
    r["judge_score"] = score
    return r


def _compute_summary(results: list) -> dict:
    total_score = 0.0
    valid_count = 0
    error_count = 0
    by_cat: defaultdict[str, dict[str, float | int]] = defaultdict(
        lambda: {"score": 0.0, "count": 0, "valid": 0, "errors": 0}
    )
    for r in results:
        cat = r.get("category") or "default"
        score = r.get("judge_score")
        by_cat[cat]["count"] += 1
        if score is None:
            error_count += 1
            by_cat[cat]["errors"] += 1
        else:
            s = float(score)
            total_score += s
            valid_count += 1
            by_cat[cat]["score"] += s
            by_cat[cat]["valid"] += 1
    by_category: dict[str, dict[str, float | int]] = {}
    for cat, v in sorted(by_cat.items()):
        n_valid = v["valid"]
        by_category[cat] = {
            "score": round(v["score"], 2),
            "count": v["count"],
            "valid": n_valid,
            "errors": v["errors"],
            "avg": round(v["score"] / n_valid, 4) if n_valid else 0.0,
        }
    return {
        "total_score": round(total_score, 2),
        "total_samples": len(results),
        "valid_samples": valid_count,
        "error_count": error_count,
        "max_possible": valid_count,
        "overall_avg": round(total_score / valid_count, 4) if valid_count else 0.0,
        "by_category": by_category,
    }


def _print_summary(summary: dict) -> None:
    print("\n" + "=" * 60)
    print("Judge score summary")
    print("=" * 60)
    print(f" Total samples:  {summary['total_samples']}")
    print(f" Valid samples:  {summary.get('valid_samples', summary['total_samples'])}")
    err = summary.get("error_count", 0)
    if err:
        print(f" Errors (judge): {err} (excluded from averages)")
    print(f" Total score: {summary['total_score']} / {summary['max_possible']}")
    print(f" Average: {summary['overall_avg']} (correct=1, partial=0.5, wrong=0)")
    print("-" * 60)
    print(" By category:")
    for cat, v in summary["by_category"].items():
        errs = v.get("errors", 0)
        err_str = f", {errs} errors" if errs else ""
        print(f"  {cat}: score {v['score']} / {v.get('valid', v['count'])} valid samples, avg {v['avg']}{err_str}")
    print("=" * 60 + "\n")


_JUDGE_CHECKPOINT_INTERVAL = 100


def _load_partial_results(out_path: Path, total: int) -> tuple[list[dict | None], int]:
    """Load partially-judged results from a prior run. Returns (results, start_index)."""
    if not out_path.exists():
        return [None] * total, 0
    try:
        with open(out_path, encoding="utf-8") as f:
            prior = json.load(f)
        if not isinstance(prior, list) or len(prior) != total:
            return [None] * total, 0
        # Find first un-judged or failed record
        start = 0
        for idx, r in enumerate(prior):
            if r is None or not isinstance(r, dict):
                start = idx
                break
            lbl = (r.get("judge_label") or "").strip()
            if not lbl or lbl == "_parse_failed":
                start = idx
                break
        else:
            start = len(prior)  # All done
        resumed = sum(1 for r in prior[:start] if r is not None and r.get("judge_label"))
        if resumed:
            print(f"[Judge] Resuming from record {start} ({resumed} already judged)")
        return prior, start
    except (json.JSONDecodeError, OSError):
        return [None] * total, 0


def run_judge(args):
    with open(args.input_file, encoding="utf-8") as f:
        records = json.load(f)
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    concurrency = max(1, int(args.concurrency))

    # Auto-detect thinking models and suppress CoT for judge calls
    if not hasattr(args, "extra_body") or args.extra_body is None:
        model_lower = (args.model or "").lower()
        if "qwen" in model_lower:
            args.extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    results, start_index = _load_partial_results(out_path, len(records))

    if start_index >= len(records):
        print(f"[Judge] All {len(records)} records already judged, skipping.")
    elif concurrency <= 1:
        for idx in tqdm(range(start_index, len(records)), desc="Judge", disable=False):
            results[idx] = _judge_one_record(records[idx], args)
            if (idx - start_index + 1) % _JUDGE_CHECKPOINT_INTERVAL == 0:
                with open(out_path, "w", encoding="utf-8") as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
    else:
        # For concurrent, process only remaining records
        pending_indices = list(range(start_index, len(records)))
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(_judge_one_record, records[i], args): i for i in pending_indices
            }
            for done_count, future in enumerate(
                tqdm(as_completed(future_to_idx), total=len(pending_indices), desc="Judge", disable=False),
                start=1,
            ):
                idx = future_to_idx[future]
                results[idx] = future.result()
                if done_count % _JUDGE_CHECKPOINT_INTERVAL == 0:
                    with open(out_path, "w", encoding="utf-8") as f:
                        json.dump(results, f, ensure_ascii=False, indent=2)

    # Final write
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} judged records to {out_path}")

    summary = _compute_summary(results)
    _print_summary(summary)
    if args.summary_file:
        summary_path = Path(args.summary_file)
        summary_path.parent.mkdir(parents=True, exist_ok=True)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        print(f"Summary written to {summary_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LLM-as-judge on prediction JSON.")
    parser.add_argument("--input-file", required=True, help="Prediction JSON from evaluate_qa.py")
    parser.add_argument(
        "--out-file", required=True, help="Output JSON with judge_label, judge_reason"
    )
    parser.add_argument("--model", type=str, default="gpt-4o-mini")
    parser.add_argument(
        "--backend",
        type=str,
        default="call_llm",
        choices=["call_test", "call_llm", "call_vllm"],
    )
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512, dest="max_tokens")
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument(
        "--summary-file", type=str, default="", help="Optional: write score summary JSON"
    )
    args = parser.parse_args()
    run_judge(args)
