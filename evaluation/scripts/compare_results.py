#!/usr/bin/env python3
"""
Print a comparison table of LoCoMo QA results from one or more stats JSON files.

Each stats file has structure: { "<model_name>": { "category_counts", "cum_accuracy_by_category", "recall_by_category" (if RAG) } }.
Usage (from project root):
  python evaluation/scripts/compare_results.py evaluation/outputs/locomo10_qa_cml_stats.json
  python evaluation/scripts/compare_results.py evaluation/outputs/locomo10_qa_cml_stats.json evaluation/locomo/outputs/locomo10_qa_stats.json
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def overall_metrics(data: dict) -> tuple[float | None, float | None]:
    """Return (overall_accuracy, overall_recall). Either can be None if missing."""
    counts = data.get("category_counts") or {}
    acc_cum = data.get("cum_accuracy_by_category") or {}
    total_n = sum(counts.values())
    if not total_n:
        return None, None
    total_acc = sum(acc_cum.get(k, 0) for k in counts)
    overall_acc = round(total_acc / total_n, 3)
    recall_by_cat = data.get("recall_by_category") or {}
    if not recall_by_cat:
        return overall_acc, None
    weighted_recall = sum(recall_by_cat.get(k, 0) * c for k, c in counts.items())
    overall_recall = round(weighted_recall / total_n, 3)
    return overall_acc, overall_recall


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: compare_results.py <stats1.json> [stats2.json ...]", file=sys.stderr)
        sys.exit(1)

    rows = []
    for path in sys.argv[1:]:
        p = Path(path)
        if not p.exists():
            print(f"Skip (not found): {p}", file=sys.stderr)
            continue
        with open(p, encoding="utf-8") as f:
            stats = json.load(f)
        if not isinstance(stats, dict):
            continue
        for model_name, data in stats.items():
            if not isinstance(data, dict) or "category_counts" not in data:
                continue
            acc, rec = overall_metrics(data)
            rows.append((model_name, acc, rec))

    if not rows:
        print("No model results found.", file=sys.stderr)
        sys.exit(1)

    # Table header
    col_name = "Model"
    col_acc = "Overall accuracy"
    col_rec = "Overall recall"
    w_name = max(len(col_name), max(len(r[0]) for r in rows), 20)
    w_acc = max(len(col_acc), 6)
    w_rec = max(len(col_rec), 6)
    fmt = f"  {{:{w_name}}}  {{:{w_acc}}}  {{:{w_rec}}}"
    print(fmt.format(col_name, col_acc, col_rec))
    print("  " + "-" * (w_name + w_acc + w_rec + 4))
    for model_name, acc, rec in rows:
        acc_s = f"{acc:.3f}" if acc is not None else "—"
        rec_s = f"{rec:.3f}" if rec is not None else "—"
        print(fmt.format(model_name, acc_s, rec_s))


if __name__ == "__main__":
    main()
