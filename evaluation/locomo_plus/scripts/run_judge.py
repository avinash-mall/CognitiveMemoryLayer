#!/usr/bin/env python3
"""Windows-friendly driver: run llm_as_judge.py with default paths."""
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
LOCOMO_PLUS = SCRIPT_DIR.parent
OUT_DIR = LOCOMO_PLUS.parent / "outputs"
INPUT_FILE = OUT_DIR / "locomo_plus_predictions.json"
OUT_FILE = OUT_DIR / "locomo_plus_judged.json"

sys.path.insert(0, str(LOCOMO_PLUS))

from task_eval.llm_as_judge import run_judge

if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--input-file", default=str(INPUT_FILE))
    p.add_argument("--out-file", default=str(OUT_FILE))
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--concurrency", type=int, default=1)
    p.add_argument("--summary-file", default="")
    args = p.parse_args()
    args.backend = "call_llm"
    args.temperature = 0.0
    args.max_tokens = 512
    run_judge(args)
