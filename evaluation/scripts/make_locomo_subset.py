#!/usr/bin/env python3
"""Build a stratified LoCoMo-Plus subset for quick A/B runs.

Why this exists rather than ``--limit-samples``: that flag is a prefix slice
(``samples[:N]``) and the dataset is ordered by category. Cognitive occupies indices
1986-2386, so any N below ~2000 yields *zero* Cognitive samples — the category most
worth measuring. The first 600 contain none either.

Selection is by whole conversation, not by sample, because ingestion groups on the
conversation prefix (``cml.eval.locomo._build_conversation_groups``). Picking scattered
samples would drag in most of the 411 conversations and save no ingestion time at all.

Greedy: repeatedly take the conversation giving the most progress toward the per-category
quotas *per turn ingested*, so the subset stays cheap as well as representative.

    python evaluation/scripts/make_locomo_subset.py           # write subset + index file
    python evaluation/scripts/make_locomo_subset.py --baseline # also score a judged artifact
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
FULL = REPO / "evaluation/locomo_plus/data/unified_input_samples_v2.json"
SUBSET = REPO / "evaluation/locomo_plus/data/unified_input_subset_v2.json"
INDICES = REPO / "evaluation/locomo_plus/data/unified_input_subset_v2.indices.json"

# Tuned against the committed full-run judged artifact so the subset reproduces its
# per-category averages. At these quotas five of six categories land within ±0.03 of the
# full run and overall within +0.036, for ~7% of the ingestion.
#
# Both 40s are floors that cost real time to raise, and raising them is mostly not worth
# it: common-sense 20 -> 40 halves its calibration error (+0.22 -> +0.11) for only +768
# turns, but Cognitive 40 -> 80 *doubles* ingestion (16.7k -> 33.6k turns) for 40 more
# samples. Cognitive and common-sense therefore stay the low-power categories — only
# large movements in them mean anything.
QUOTAS = {
    "Cognitive": 40,
    "adversarial": 40,
    "common-sense": 40,
    "multi-hop": 40,
    "single-hop": 40,
    "temporal": 40,
}


def conversation_key(sample: dict) -> str:
    """The conversation prefix — everything before the trailing 'Question:' line.

    Mirrors ``cml.eval.locomo._build_conversation_groups`` so this script selects the
    same units the harness ingests.
    """
    prompt = sample["input_prompt"]
    cut = prompt.find("Question:")
    return prompt[:cut] if cut >= 0 else prompt


def select(samples: list[dict]) -> list[int]:
    """Return the sample indices of the chosen conversations, in original order."""
    groups: dict[str, list[int]] = {}
    for idx, sample in enumerate(samples):
        groups.setdefault(conversation_key(sample), []).append(idx)

    keys = list(groups)
    cats = [collections.Counter(samples[i]["category"] for i in groups[k]) for k in keys]
    # Turn count proxy: the conversation prefix is one line per turn.
    turns = [max(k.count("\n") + 1, 1) for k in keys]

    def shortfall(have: collections.Counter) -> int:
        return sum(max(0, need - have[cat]) for cat, need in QUOTAS.items())

    have: collections.Counter = collections.Counter()
    chosen: list[int] = []
    remaining = set(range(len(keys)))
    while shortfall(have) > 0 and remaining:
        best = max(
            remaining, key=lambda k: (shortfall(have) - shortfall(have + cats[k])) / turns[k]
        )
        if shortfall(have) - shortfall(have + cats[best]) <= 0:
            break  # nothing left contributes to an unmet quota
        remaining.discard(best)
        chosen.append(best)
        have += cats[best]

    picked = sorted(i for k in chosen for i in groups[keys[k]])
    total_turns = sum(turns[k] for k in chosen)
    print(f"conversations : {len(chosen)} of {len(keys)}")
    print(f"turns         : {total_turns:,} ({total_turns / 242658:.1%} of the full run)")
    print(f"samples       : {len(picked)}")
    print(f"coverage      : {dict(sorted(have.items()))}")
    return picked


def category_averages(scored: list[dict]) -> dict[str, float]:
    by: dict[str, list[float]] = collections.defaultdict(list)
    for rec in scored:
        by[rec["category"]].append(float(rec["judge_score"]))
    return {cat: round(sum(v) / len(v), 4) for cat, v in sorted(by.items())}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--baseline",
        type=Path,
        nargs="?",
        const=REPO / "evaluation/results/locomo_plus_2026-07-31_judged.json",
        help="Judged artifact to restrict to the subset, giving the 'before' arm for free.",
    )
    args = ap.parse_args()

    samples = json.loads(FULL.read_text())
    picked = select(samples)

    SUBSET.write_text(json.dumps([samples[i] for i in picked], indent=1))
    INDICES.write_text(json.dumps(picked))
    print(f"\nwrote {SUBSET.relative_to(REPO)}")

    if not args.baseline:
        return

    judged = json.loads(args.baseline.read_text())
    if len(judged) != len(samples):
        raise SystemExit(
            f"{args.baseline.name} has {len(judged)} records but the dataset has "
            f"{len(samples)} — cannot align by index."
        )
    # The judged artifact is emitted in dataset order; confirm rather than assume.
    mismatched = [
        i
        for i, (j, s) in enumerate(zip(judged, samples, strict=True))
        if j["category"] != s["category"]
    ]
    if mismatched:
        raise SystemExit(f"category mismatch at {len(mismatched)} indices, first {mismatched[:5]}")

    restricted = [judged[i] for i in picked]
    full_avg = category_averages(judged)
    sub_avg = category_averages(restricted)
    print("\nArm 0 — committed full run restricted to the subset (no re-run needed)")
    print(f"{'category':14}{'full':>9}{'subset':>9}{'delta':>9}")
    for cat in sorted(full_avg):
        d = sub_avg.get(cat, float("nan")) - full_avg[cat]
        print(f"{cat:14}{full_avg[cat]:>9.4f}{sub_avg.get(cat, 0):>9.4f}{d:>+9.4f}")
    fo = sum(float(r["judge_score"]) for r in judged) / len(judged)
    so = sum(float(r["judge_score"]) for r in restricted) / len(restricted)
    print(f"{'OVERALL':14}{fo:>9.4f}{so:>9.4f}{so - fo:>+9.4f}")


if __name__ == "__main__":
    main()
