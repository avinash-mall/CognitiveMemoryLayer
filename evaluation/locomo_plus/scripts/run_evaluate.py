#!/usr/bin/env python3
"""Windows-friendly driver: run evaluate_qa.py with default paths."""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCOMO_PLUS = SCRIPT_DIR.parent
DATA_FILE = LOCOMO_PLUS / "data" / "unified_input_samples_v2.json"
OUT_DIR = LOCOMO_PLUS.parent / "outputs"
OUT_FILE = OUT_DIR / "locomo_plus_predictions.json"

sys.path.insert(0, str(LOCOMO_PLUS))

from task_eval.evaluate_qa import evaluate_dataset

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--model", default="mock")
    p.add_argument("--backend", default="call_test", choices=["call_test", "call_llm", "call_vllm"])
    p.add_argument("--temperature", type=float, default=0.3)
    p.add_argument("--concurrency", type=int, default=1)
    args = p.parse_args()
    args.data_file = str(DATA_FILE)
    args.out_file = str(OUT_FILE)
    args.max_tokens = 1024
    evaluate_dataset(args)
