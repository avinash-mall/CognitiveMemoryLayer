"""Generate LoCoMo/Locomo-Plus report tables."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

FACTUAL_CATEGORIES = ["single-hop", "multi-hop", "temporal", "common-sense", "adversarial"]
COGNITIVE_CATEGORY = "Cognitive"
COL_NAMES = ["single-hop", "multi-hop", "temporal", "commonsense", "adversarial"]


def _pct(value: float) -> str:
    return f"{value * 100:.2f}"


def load_summary(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def extract_row(summary: dict, method: str) -> tuple[str, list[str], str, str]:
    by_cat = summary.get("by_category") or {}
    factual_avgs: list[float] = []
    for cat in FACTUAL_CATEGORIES:
        value = by_cat.get(cat) or {}
        factual_avgs.append(float(value.get("avg", 0.0)))

    avg_factual = sum(factual_avgs) / len(factual_avgs) if factual_avgs else 0.0
    cognitive_avg = float((by_cat.get(COGNITIVE_CATEGORY) or {}).get("avg", 0.0))
    gap = avg_factual - cognitive_avg

    factual_pcts = [_pct(v) for v in factual_avgs]
    return method, factual_pcts, _pct(cognitive_avg), f"{gap * 100:.2f}"


def format_table(rows: list[tuple[str, list[str], str, str]], title: str | None) -> str:
    cols = ["Method", *COL_NAMES, "average", "LoCoMo-Plus", "Gap"]
    widths = [max(len(c), 10) for c in cols]
    for method, factual, locomoplus, gap in rows:
        avg = f"{(sum(float(p) for p in factual) / len(factual)):.2f}" if factual else "0.00"
        values = [method, *factual, avg, locomoplus, gap]
        for i, value in enumerate(values):
            widths[i] = max(widths[i], len(str(value)))

    def fmt(values: list[str]) -> str:
        return " | ".join(str(v).rjust(w) for v, w in zip(values, widths, strict=False))

    lines: list[str] = []
    if title:
        lines.append(title)
        lines.append("The Gap column indicates the performance drop from LoCoMo to LoCoMo-Plus.")
        lines.append("")

    lines.append(fmt(cols))
    lines.append(" | ".join("-" * w for w in widths))
    for method, factual, locomoplus, gap in rows:
        avg = f"{(sum(float(p) for p in factual) / len(factual)):.2f}" if factual else "0.00"
        lines.append(fmt([method, *factual, avg, locomoplus, gap]))

    return "\n".join(lines)


def generate_locomo_report(summary_path: Path, method: str, no_title: bool = False) -> str:
    summary = load_summary(summary_path)
    row = extract_row(summary, method)
    title = (
        None
        if no_title
        else "Table 1: Overall performance on LoCoMo (factual memory) and LoCoMo-Plus (cognitive memory). Gap = LoCoMo avg - LoCoMo-Plus."
    )
    return format_table([row], title=title)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate LoCoMo / Locomo-Plus performance table")
    parser.add_argument("--summary", type=Path, default=Path("evaluation") / "outputs" / "locomo_plus_qa_cml_judge_summary.json")
    parser.add_argument("--method", type=str, default="CML")
    parser.add_argument("--no-title", action="store_true", help="Omit table title")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.summary.exists():
        print(f"Summary file not found: {args.summary}")
        return 1
    print()
    print(generate_locomo_report(args.summary, args.method, no_title=bool(args.no_title)))
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
