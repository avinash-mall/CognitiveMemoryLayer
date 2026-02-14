"""
Unified evaluation over six categories, outputs JSON for judge.

Flow:
1. Load unified input JSON (each sample: input_prompt, evidence, category, etc.).
2. For each sample: extract question, call call_test/call_llm/call_vllm for prediction.
3. Build one record per sample: question_input, evidence, category, ground_truth, prediction, model.
4. Write all records to --out-file for downstream judge.
"""

import argparse
import json
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

from task_eval.utils import (
    build_output_record,
    call_model,
    extract_question_from_input_prompt,
    load_unified_samples,
)


def _process_one_sample(sample, args):
    """Process one sample and return one record."""
    input_prompt = sample.get("input_prompt", "")
    category = sample.get("category", "")
    if category == "Cognitive":
        question_input = (sample.get("trigger") or "").strip() or "Context dialogue (cue awareness)"
    else:
        question_input = extract_question_from_input_prompt(input_prompt)

    if not input_prompt:
        return build_output_record(
            sample,
            prediction="(no input_prompt)",
            model=args.model,
            question_input=question_input,
        )

    prediction = call_model(
        input_prompt,
        model=args.model,
        backend=args.backend,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        category=category,
    )
    return build_output_record(sample, prediction, args.model, question_input=question_input)


def evaluate_dataset(args):
    out_path = Path(args.out_file)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    samples = load_unified_samples(args.data_file)
    concurrency = max(1, int(args.concurrency))

    if concurrency <= 1:
        results = []
        for sample in tqdm(samples, desc=f"Evaluating {args.model}"):
            results.append(_process_one_sample(sample, args))
    else:
        results = [None] * len(samples)
        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            future_to_idx = {
                executor.submit(_process_one_sample, s, args): i for i, s in enumerate(samples)
            }
            for future in tqdm(as_completed(future_to_idx), total=len(samples), desc=f"Evaluating {args.model}"):
                idx = future_to_idx[future]
                results[idx] = future.result()

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(results)} records to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run model on unified input, write predictions JSON.")
    parser.add_argument("--data-file", required=True, type=str, help="Path to unified input JSON")
    parser.add_argument("--out-file", required=True, type=str, help="Output JSON path")
    parser.add_argument("--model", type=str, default="mock", help="Model name")
    parser.add_argument(
        "--backend",
        type=str,
        default="call_test",
        choices=["call_test", "call_llm", "call_vllm"],
        help="Backend: call_test, call_llm, call_vllm",
    )
    parser.add_argument("--temperature", type=float, default=0.3)
    parser.add_argument("--max-tokens", type=int, default=1024, dest="max_tokens")
    parser.add_argument("--concurrency", type=int, default=1)
    args = parser.parse_args()
    evaluate_dataset(args)
