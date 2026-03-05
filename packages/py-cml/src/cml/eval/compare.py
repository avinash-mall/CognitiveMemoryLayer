"""Compare CML scores with LoCoMo-Plus paper baselines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

COL_NAMES = ["single-hop", "multi-hop", "temporal", "commonsense", "adversarial"]
FACTUAL_CATEGORIES = ["single-hop", "multi-hop", "temporal", "common-sense", "adversarial"]
COGNITIVE_CATEGORY = "Cognitive"

PAPER_BASELINES: list[tuple[str, list[float], float]] = [
    ("Qwen2.5-3B-Instruct", [68.25, 38.65, 18.38, 48.44, 11.69], 10.82),
    ("Qwen2.5-7B-Instruct", [70.72, 39.54, 21.81, 37.50, 20.22], 9.57),
    ("Qwen2.5-14B-Instruct", [76.33, 48.23, 38.94, 57.29, 68.09], 19.24),
    ("Qwen3-14B", [65.96, 46.45, 53.89, 59.38, 60.45], 19.09),
    ("gpt-4o (full context)", [78.13, 52.30, 45.79, 69.79, 48.99], 21.05),
    ("gemini-2.5-flash", [77.71, 54.26, 66.04, 66.67, 65.84], 24.67),
    ("gemini-2.5-pro", [77.83, 52.48, 73.83, 63.54, 73.03], 26.06),
    ("RAG (text-embedding-002)", [40.00, 16.73, 37.81, 15.73, 49.44], 13.91),
    ("RAG (text-embedding-large)", [49.76, 22.78, 40.00, 21.35, 59.73], 15.55),
    ("Mem0 (GPT-4o)", [80.20, 48.10, 39.40, 66.20, 30.50], 15.80),
    ("SeCom (GPT-4o)", [77.60, 50.90, 42.30, 71.40, 31.80], 14.90),
    ("A-Mem (GPT-4o)", [76.90, 55.60, 49.30, 68.10, 35.20], 17.20),
]


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def summary_to_row(summary: dict, method: str) -> tuple[str, list[float], float, float]:
    by_cat = summary.get("by_category") or {}
    factual_avgs: list[float] = []
    for cat in FACTUAL_CATEGORIES:
        factual_avgs.append(float((by_cat.get(cat) or {}).get("avg", 0.0)) * 100.0)

    avg_factual = sum(factual_avgs) / len(factual_avgs) if factual_avgs else 0.0
    cognitive_avg = float((by_cat.get(COGNITIVE_CATEGORY) or {}).get("avg", 0.0)) * 100.0
    return method, factual_avgs, cognitive_avg, avg_factual - cognitive_avg


def fmt_row(method: str, factual: list[float], locomoplus: float, gap: float) -> list[str]:
    avg = sum(factual) / len(factual) if factual else 0.0
    return [method, *[f"{v:.2f}" for v in factual], f"{avg:.2f}", f"{locomoplus:.2f}", f"{gap:.2f}"]


def compare_locomo_scores(summary_path: Path, method: str) -> str:
    summary = load_summary(summary_path)
    cml_method, cml_factual, cml_lp, cml_gap = summary_to_row(summary, method)

    cols = ["Method", *COL_NAMES, "average", "LoCoMo-Plus", "Gap"]
    widths = [max(len(c), 8) for c in cols]
    rows: list[list[str]] = []

    current = fmt_row(cml_method, cml_factual, cml_lp, cml_gap)
    rows.append(current)

    for baseline_method, factual, lp in PAPER_BASELINES:
        avg = sum(factual) / len(factual)
        rows.append(fmt_row(baseline_method, factual, lp, avg - lp))

    for row in rows:
        for i, value in enumerate(row):
            widths[i] = max(widths[i], len(value))

    def sep() -> str:
        return " | ".join("-" * w for w in widths)

    lines = [
        "Comparison: This project vs Locomo-Plus paper baselines (Table 1, arXiv:2602.10715)",
        "All numbers are LLM-as-judge scores in %. Same protocol: constraint consistency, no task disclosure.",
        "",
        " | ".join(c.rjust(widths[i]) for i, c in enumerate(cols)),
        sep(),
    ]
    lines.extend(" | ".join(v.rjust(widths[i]) for i, v in enumerate(row)) for row in rows)
    return "\n".join(lines)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CML scores with Locomo-Plus paper baselines")
    parser.add_argument("--summary", type=Path, default=Path("evaluation") / "outputs" / "locomo_plus_qa_cml_judge_summary.json")
    parser.add_argument("--method", type=str, default="CML+gpt-oss:20b")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.summary.exists():
        print(f"Summary not found: {args.summary}")
        return 1
    print()
    print(compare_locomo_scores(args.summary, args.method))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
