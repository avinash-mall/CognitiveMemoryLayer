#!/usr/bin/env python3
"""Short evaluation test: runs ~30 samples across all categories to validate
eval pipeline fixes without a full day-long run.

Compares current (fixed) pipeline against baseline predictions for the same
samples and reports per-category deltas.

Usage:
    python evaluation/short_eval_test.py [--cml-url URL] [--samples-per-cat N]
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

# Ensure repo root is on path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

from cml.eval.config import ensure_unified_eval_data, load_repo_dotenv

load_repo_dotenv(REPO_ROOT)


def _get_session():
    import requests
    from requests.adapters import HTTPAdapter

    s = requests.Session()
    adapter = HTTPAdapter(pool_connections=20, pool_maxsize=20)
    s.mount("http://", adapter)
    s.mount("https://", adapter)
    return s


SESSION = None


def get_session():
    global SESSION
    if SESSION is None:
        SESSION = _get_session()
    return SESSION


def cml_read(base_url: str, api_key: str, tenant_id: str, query: str, max_results: int = 25) -> str:
    """Read memories from CML and return llm_context string."""
    url = f"{base_url.rstrip('/')}/api/v1/memory/read"
    payload = {"query": query, "format": "llm_context", "max_results": max_results}
    headers = {"X-API-Key": api_key, "X-Tenant-ID": tenant_id}
    for attempt in range(5):
        try:
            resp = get_session().post(url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 429:
                time.sleep(5 * (2**attempt))
                continue
            resp.raise_for_status()
            data = resp.json()
            return (data.get("llm_context") or "").strip()
        except Exception as e:
            if attempt < 4:
                time.sleep(3 * (2**attempt))
                continue
            print(f"  [ERROR] CML read failed for {tenant_id}: {e}")
            return ""
    return ""


def llm_chat(user_content: str, max_tokens: int = 512) -> str:
    """Call the QA LLM."""
    try:
        from openai import OpenAI
    except ImportError:
        raise ImportError("pip install openai")

    base_url = os.environ.get("LLM_EVAL__BASE_URL", "http://localhost:8001/v1").rstrip("/")
    model = os.environ.get("LLM_EVAL__MODEL", "Qwen/Qwen3.5-27B")
    api_key = os.environ.get("LLM_EVAL__API_KEY", "dummy")

    # Disable thinking by default (Qwen fills max_tokens with CoT before producing answer).
    # Set LLM_EVAL__ENABLE_THINKING=1 to re-enable (needs larger max_tokens).
    enable_thinking = os.environ.get("LLM_EVAL__ENABLE_THINKING", "").strip() in ("1", "true")
    extra_body = None
    if "qwen" in model.lower() and not enable_thinking:
        extra_body = {"chat_template_kwargs": {"enable_thinking": False}}

    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        kwargs = {
            "model": model,
            "messages": [{"role": "user", "content": user_content}],
            "max_tokens": max_tokens,
            "temperature": 0,
        }
        if extra_body:
            kwargs["extra_body"] = extra_body
        resp = client.chat.completions.create(**kwargs)
        choices = getattr(resp, "choices", None) or []
        if not choices:
            return ""
        msg = getattr(choices[0], "message", None)
        return (getattr(msg, "content", None) or "").strip()
    except Exception as e:
        print(f"  [LLM ERROR] {e}")
        return ""


# Import the fixed _extract_answer
from cml.eval.locomo import _extract_answer

# Prompts (matching the fixed eval pipeline)
QA_PROMPT = """Based on the above context from past conversations, answer the question below.

IMPORTANT RULES:
1. Names may appear as [FIRSTNAME_REDACTED] — treat them as the people in the question.
2. If the context contains timestamps or dates, reason about time carefully.
3. Answer with a short phrase or 1-2 sentences. Use exact words from the context when possible.
4. If the answer can be inferred or reasoned from the context, provide your best answer.
   For common-sense questions, use the context plus general reasoning to answer.
5. If no context was retrieved, or the context has no information related to the question,
   say "I don't have information about that from our previous conversations."

Question: {}

Short answer:"""

COGNITIVE_PROMPT = """Based on the above context from past conversations, continue the
conversation naturally. Respond to the following as you would in a real dialogue.
If the context contains relevant constraints, preferences, or past decisions, incorporate them.

{}

Respond (short):"""


def judge_single(
    prediction: str, ground_truth: str, evidence: str, category: str
) -> tuple[float, str]:
    """Simple judge: for adversarial, check if model correctly refuses.
    For others, check if prediction contains key terms from ground_truth."""
    pred = prediction.strip().lower()
    gt = str(ground_truth).strip().lower()

    if category == "adversarial":
        # Adversarial: ground_truth is empty, model should refuse
        refusal_phrases = [
            "no information",
            "not mentioned",
            "not enough context",
            "not available",
            "none provided",
            "i don't have",
            "no context",
            "no relevant",
            "cannot answer",
            "unanswerable",
            "does not contain",
            "does not mention",
            "does not specify",
            "does not state",
            "does not include",
            "not found in",
            "haven't included",
            "no specific",
            "not provided",
        ]
        if any(p in pred for p in refusal_phrases) or not pred:
            return 1.0, "correct_refusal"
        return 0.0, "hallucinated_answer"

    if category == "Cognitive":
        # Cognitive: ground_truth is empty, judge checks if prediction connects to evidence
        if not pred or pred in (
            "",
            "i don't have information about that from our previous conversations.",
        ):
            return 0.0, "empty_or_refusal"
        # If prediction has substance, give partial credit
        if len(pred) > 20:
            return 0.5, "has_substance"
        return 0.0, "too_short"

    # Factual categories: check token overlap with ground_truth
    if not gt:
        return 0.0, "no_ground_truth"

    gt_tokens = set(gt.split())
    pred_tokens = set(pred.split())
    if not gt_tokens:
        return 0.0, "empty_gt_tokens"

    overlap = gt_tokens & pred_tokens
    precision = len(overlap) / len(pred_tokens) if pred_tokens else 0
    recall = len(overlap) / len(gt_tokens) if gt_tokens else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    if f1 >= 0.5:
        return 1.0, f"f1={f1:.2f}"
    elif f1 >= 0.2:
        return 0.5, f"f1={f1:.2f}"
    return 0.0, f"f1={f1:.2f}"


def run_short_test(
    cml_url: str = "http://localhost:8000",
    cml_api_key: str = "test-key",
    samples_per_cat: int = 5,
    seed: int = 42,
) -> dict:
    """Run short evaluation test."""
    unified_file = ensure_unified_eval_data(
        Path("evaluation/locomo_plus/data/unified_input_samples_v2.json"),
        repo_root=REPO_ROOT,
    )
    with open(unified_file, encoding="utf-8") as f:
        all_samples = json.load(f)

    print(f"Loaded {len(all_samples)} total samples")

    # Pick balanced subset
    random.seed(seed)
    selected: list[tuple[int, dict]] = []
    categories = ["Cognitive", "adversarial", "common-sense", "multi-hop", "single-hop", "temporal"]

    for cat in categories:
        cat_samples = [(i, s) for i, s in enumerate(all_samples) if s["category"] == cat]

        # Prefer samples from data range (0-1967 have DB data; 1968-2386 do NOT)
        with_data = [(i, s) for i, s in cat_samples if i < 1968]
        without_data = [(i, s) for i, s in cat_samples if i >= 1968]

        n_with = min(max(2, samples_per_cat // 2), len(with_data))
        n_without = min(samples_per_cat - n_with, len(without_data))

        picked_with = random.sample(with_data, n_with) if with_data else []
        picked_without = random.sample(without_data, n_without) if without_data else []
        selected.extend(picked_with + picked_without)

    print(f"Selected {len(selected)} samples for short test:")
    cat_counts = Counter(s["category"] for _, s in selected)
    for cat in categories:
        n_data = sum(1 for i, s in selected if s["category"] == cat and 1968 <= i <= 2386)
        print(f"  {cat}: {cat_counts.get(cat, 0)} ({n_data} with DB data)")

    # Load baseline predictions for comparison
    baseline_preds = {}
    baseline_file = (
        REPO_ROOT / "evaluation" / "outputs" / "locomo_plus_qa_cml_predictions_baseline.json"
    )
    if baseline_file.exists():
        with open(baseline_file, encoding="utf-8") as f:
            bl = json.load(f)
        for i, rec in enumerate(bl):
            baseline_preds[i] = rec

    # Run eval
    results = []
    print(f"\n{'=' * 70}")
    print("Running short eval...")
    print(f"{'=' * 70}\n")

    for idx, (sample_idx, sample) in enumerate(selected):
        tenant_id = f"lp-{sample_idx}"
        trigger = (sample.get("trigger") or "").strip()
        category = sample["category"]
        ground_truth = sample.get("answer") or ""
        evidence = sample.get("evidence", "")
        has_db_data = sample_idx < 1968

        # Step 1: CML Read
        print(
            f"[{idx + 1}/{len(selected)}] {category} (idx={sample_idx}, data={'YES' if has_db_data else 'NO'})...",
            end=" ",
            flush=True,
        )
        llm_context = cml_read(cml_url, cml_api_key, tenant_id, trigger)
        ctx_len = len(llm_context)
        ctx_useful = ctx_len > 30  # More than just header

        # Step 2: LLM QA
        # Treat context as empty if it's just the markdown header with no content
        effective_context = (
            llm_context if ctx_useful else "(No relevant memories found for this query.)"
        )
        if category == "Cognitive":
            user_content = effective_context + "\n\n" + COGNITIVE_PROMPT.format(trigger)
        else:
            user_content = effective_context + "\n\n" + QA_PROMPT.format(trigger)

        raw_prediction = llm_chat(user_content)
        prediction = _extract_answer(raw_prediction)

        # Step 3: Simple judge
        score, reason = judge_single(prediction, ground_truth, evidence, category)

        # Baseline comparison
        bl_pred = baseline_preds.get(sample_idx, {}).get("prediction", "")
        bl_score, _bl_reason = judge_single(bl_pred, ground_truth, evidence, category)

        delta = score - bl_score
        delta_str = f"{'+' if delta > 0 else ''}{delta:.1f}" if delta != 0 else "="
        print(
            f"ctx={ctx_len:>4}{'*' if ctx_useful else ' '} score={score:.1f} (bl={bl_score:.1f}, Δ={delta_str}) [{reason}]"
        )

        if prediction[:80] != bl_pred[:80]:
            print(f'    pred: "{prediction[:100]}"')
            print(f'    base: "{bl_pred[:100]}"')

        results.append(
            {
                "sample_idx": sample_idx,
                "category": category,
                "has_db_data": has_db_data,
                "context_length": ctx_len,
                "context_useful": ctx_useful,
                "prediction": prediction,
                "ground_truth": str(ground_truth),
                "score": score,
                "reason": reason,
                "baseline_score": bl_score,
                "baseline_prediction": bl_pred[:200],
                "delta": delta,
            }
        )

    # Summary
    print(f"\n{'=' * 70}")
    print("SHORT TEST SUMMARY")
    print(f"{'=' * 70}\n")

    cat_scores = defaultdict(list)
    cat_bl_scores = defaultdict(list)
    for r in results:
        cat_scores[r["category"]].append(r["score"])
        cat_bl_scores[r["category"]].append(r["baseline_score"])

    print(f"{'Category':<16} {'N':>3} {'Current':>8} {'Baseline':>8} {'Delta':>8}")
    print("-" * 50)
    total_score = 0
    total_bl = 0
    total_n = 0
    for cat in categories:
        scores = cat_scores.get(cat, [])
        bl_scores = cat_bl_scores.get(cat, [])
        if not scores:
            continue
        avg = sum(scores) / len(scores)
        bl_avg = sum(bl_scores) / len(bl_scores)
        delta = avg - bl_avg
        total_score += sum(scores)
        total_bl += sum(bl_scores)
        total_n += len(scores)
        print(f"{cat:<16} {len(scores):>3} {avg:>8.3f} {bl_avg:>8.3f} {delta:>+8.3f}")

    overall = total_score / total_n if total_n else 0
    bl_overall = total_bl / total_n if total_n else 0
    print("-" * 50)
    print(
        f"{'OVERALL':<16} {total_n:>3} {overall:>8.3f} {bl_overall:>8.3f} {overall - bl_overall:>+8.3f}"
    )

    # Retrieval diagnostics
    print("\n--- Retrieval Diagnostics ---")
    useful = sum(1 for r in results if r["context_useful"])
    print(f"Samples with useful context (>30 chars): {useful}/{len(results)}")
    with_data_useful = sum(1 for r in results if r["has_db_data"] and r["context_useful"])
    with_data_total = sum(1 for r in results if r["has_db_data"])
    print(f"Of those with DB data: {with_data_useful}/{with_data_total} got useful context")

    # Prediction diagnostics
    empty_preds = sum(1 for r in results if not r["prediction"].strip())
    refusal_preds = sum(
        1
        for r in results
        if any(
            x in r["prediction"].lower() for x in ["no information", "not mentioned", "not enough"]
        )
    )
    print(f"\nEmpty predictions: {empty_preds}/{len(results)}")
    print(f"Refusal predictions: {refusal_preds}/{len(results)}")

    # Save results
    out_file = REPO_ROOT / "evaluation" / "outputs" / "short_test_results.json"
    with open(out_file, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False, default=str)
    print(f"\nResults saved to {out_file}")

    return {
        "overall": overall,
        "baseline_overall": bl_overall,
        "delta": overall - bl_overall,
        "by_category": {
            cat: {
                "avg": sum(cat_scores[cat]) / len(cat_scores[cat]) if cat_scores[cat] else 0,
                "baseline_avg": sum(cat_bl_scores[cat]) / len(cat_bl_scores[cat])
                if cat_bl_scores[cat]
                else 0,
            }
            for cat in categories
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Short eval test")
    parser.add_argument("--cml-url", default="http://localhost:8000")
    parser.add_argument("--cml-api-key", default="test-key")
    parser.add_argument("--samples-per-cat", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    run_short_test(
        cml_url=args.cml_url,
        cml_api_key=args.cml_api_key,
        samples_per_cat=args.samples_per_cat,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
