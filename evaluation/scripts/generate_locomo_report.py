#!/usr/bin/env python3
"""
Generate a performance table for LoCoMo (factual memory) and Locomo-Plus (cognitive memory).

Matches the paper format:
  Table: Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap

Reads the judge summary from eval_locomo_plus. Values are percentages (0-100).
Gap = LoCoMo average - LoCoMo-Plus (performance drop from factual to cognitive).

Usage (from project root):
  python evaluation/scripts/generate_locomo_report.py [--summary summary.json] [--method "CML+gpt-oss-20b"]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

# Category order matching paper table (lookup keys in judge summary)
FACTUAL_CATEGORIES = ["single-hop", "multi-hop", "temporal", "common-sense", "adversarial"]
COGNITIVE_CATEGORY = "Cognitive"

# Display column names (commonsense no hyphen for table)
COL_NAMES = ["single-hop", "multi-hop", "temporal", "commonsense", "adversarial"]


def _pct(x: float) -> str:
    """Format 0-1 score as percentage string."""
    return f"{x * 100:.2f}"


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_row(summary: dict, method: str) -> tuple[str, list[str], str, str]:
    """
    Extract one row from judge summary.
    Returns (method, list of 5 factual percentages, locomoplus_pct, gap_pct).
    """
    by_cat = summary.get("by_category") or {}
    factual_avgs: list[float] = []
    for cat in FACTUAL_CATEGORIES:
        v = by_cat.get(cat) or {}
        avg = v.get("avg")
        if avg is not None:
            factual_avgs.append(float(avg))
        else:
            factual_avgs.append(0.0)
    avg_factual = sum(factual_avgs) / len(factual_avgs) if factual_avgs else 0.0

    cog = by_cat.get(COGNITIVE_CATEGORY) or {}
    cog_avg = float(cog.get("avg", 0.0))
    gap = avg_factual - cog_avg

    factual_pcts = [_pct(a) for a in factual_avgs]
    locomoplus_pct = _pct(cog_avg)
    gap_pct = f"{gap * 100:.2f}"

    return method, factual_pcts, locomoplus_pct, gap_pct


def format_table(
    rows: list[tuple[str, list[str], str, str]],
    title: str | None = None,
) -> str:
    """Format rows as a table matching the paper layout."""
    cols = ["Method", *COL_NAMES, "average", "LoCoMo-Plus", "Gap"]
    widths = [max(len(c), 10) for c in cols]
    for method, factual, lp, gap in rows:
        avg_val = f"{(sum(float(p) for p in factual) / len(factual)):.2f}" if factual else "0.00"
        row_vals = [method, *factual, avg_val, lp, gap]
        for i, v in enumerate(row_vals):
            if i < len(widths):
                widths[i] = max(widths[i], len(str(v)))

    def fmt_row(values: list[str]) -> str:
        return " | ".join(str(v).rjust(w) for v, w in zip(values, widths, strict=False))

    lines: list[str] = []
    if title:
        lines.extend(
            [title, "The Gap column indicates the performance drop from LoCoMo to LoCoMo-Plus.", ""]
        )
    lines.extend(
        [
            fmt_row(cols),
            " | ".join("-" * w for w in widths),
        ]
    )
    for method, factual, lp, gap in rows:
        avg_val = f"{(sum(float(p) for p in factual) / len(factual)):.2f}" if factual else "0.00"
        row_vals = [method, *factual, avg_val, lp, gap]
        lines.append(fmt_row(row_vals))

    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description="Generate LoCoMo / Locomo-Plus performance table")
    p.add_argument(
        "--summary",
        type=str,
        default=None,
        help="Path to locomo_plus_qa_cml_judge_summary.json",
    )
    p.add_argument(
        "--method",
        type=str,
        default=None,
        help='Method name for the table row (default "CML")',
    )
    p.add_argument(
        "--no-title",
        action="store_true",
        help="Omit table title (compact output)",
    )
    args = p.parse_args()

    root = Path(__file__).resolve().parent.parent
    summary_path = (
        Path(args.summary)
        if args.summary
        else root / "outputs" / "locomo_plus_qa_cml_judge_summary.json"
    )

    if not summary_path.exists():
        print(f"Summary file not found: {summary_path}", file=sys.stderr)
        print("Run eval_locomo_plus.py first to produce the judge summary.", file=sys.stderr)
        sys.exit(1)

    summary = load_summary(summary_path)
    method = args.method or "CML"
    row = extract_row(summary, method)
    title = (
        None
        if args.no_title
        else "Table 1: Overall performance on LoCoMo (factual memory) and LoCoMo-Plus (cognitive memory). Gap = LoCoMo avg - LoCoMo-Plus."
    )
    table = format_table([row], title=title)
    print(flush=True)
    print(table)
    print(flush=True)


if __name__ == "__main__":
    main()
