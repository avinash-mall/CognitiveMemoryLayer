"""Replace structured rows for decay_profile and confidence_bin with updated profiles.

Root cause: the old profiles produced feature tokens that are indistinguishable at key
ordinal boundaries:
  - decay_profile::slow vs ::medium: both produced support_count=medium, access=medium, age=medium
  - decay_profile::slow vs ::very_slow: slow had support_count=medium OR high (50/50)
  - confidence_bin::low vs ::medium: low had support_count=low OR medium (50/50)
  - confidence_bin::medium vs ::high: high had support_count=medium OR high (50/50)

Fix: restructure support_count and access_count bases so each ordinal boundary has a
CLEAN bucket-level discriminator:

  decay_profile ordinal boundaries:
    B0 (≥fast vs very_fast): support none/low → very_fast; medium/high → fast+
    B1 (≥medium vs fast):    age high → fast; age medium/low → medium+
    B2 (≥slow vs medium):    support high → slow/very_slow; support medium → medium
    B3 (very_slow vs slow):  access high → very_slow; access medium → slow

  confidence_bin ordinal boundaries:
    B0 (≥medium vs low):     support none/low → low; support medium/high → medium+
    B1 (high vs medium):     support high → high; support medium → medium

Changed profile values:
  decay_profile:
    very_fast  support_count: 1 → 0  (range 0-1 → none/low)
    fast       support_count: 1 → 2  (range 2-3 → medium)
    slow       support_count: 3 → 4  (range 4-5 → high)
    very_slow  access_count:  4 → 6  (range 6-8 → high)

  confidence_bin:
    low        support_count: 1 → 0  (range 0-1 → none/low)
    high       support_count: 3 → 4  (range 4-5 → high)

Run from repo root:
    python packages/models/scripts/recalibrate_decay_confidence_profiles.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import pandas as pd  # noqa: E402

from cml.modeling.prepare import (  # noqa: E402
    _cycle_choice,
    _other_topic_pack,
    _structured_row_extras,
    _topic_pack,
)

PREPARED_DIR = REPO_ROOT / "packages" / "models" / "prepared_data" / "modelpack"

# ── Updated profiles ──────────────────────────────────────────────────────────
# These must stay in sync with _fill_structured_ordinal_router_rows in prepare.py.

DECAY_PROFILE_PROFILES: dict[str, dict] = {
    "very_fast": {
        "memory_types": ["scratch", "episodic_event", "observation"],
        "importance": 0.24,
        "confidence": 0.38,
        "access_count": 0,
        "age_days": 170,
        "dependency_count": 0,
        "support_count": 0,   # range 0-1 → none/low (B0 discriminator)
        "mixed_topic": True,
    },
    "fast": {
        "memory_types": ["episodic_event", "observation", "semantic_fact"],
        "importance": 0.34,
        "confidence": 0.48,
        "access_count": 1,
        "age_days": 95,
        "dependency_count": 0,
        "support_count": 2,   # range 2-3 → always medium (was 1)
        "mixed_topic": True,
    },
    "medium": {
        "memory_types": ["observation", "semantic_fact", "preference"],
        "importance": 0.46,
        "confidence": 0.58,
        "access_count": 2,
        "age_days": 56,
        "dependency_count": 1,
        "support_count": 2,   # range 2-3 → always medium
        "mixed_topic": False,
    },
    "slow": {
        "memory_types": ["semantic_fact", "preference", "constraint"],
        "importance": 0.58,
        "confidence": 0.68,
        "access_count": 3,
        "age_days": 30,
        "dependency_count": 2,
        "support_count": 4,   # range 4-5 → always high (was 3, B2 discriminator)
        "mixed_topic": False,
    },
    "very_slow": {
        "memory_types": ["constraint", "semantic_fact", "preference"],
        "importance": 0.69,
        "confidence": 0.76,
        # access_count=7: range 7-9 → always high (was 4, B3 discriminator).
        # Base=7 not 6 because boundary rows apply access−1; 7−1=6 still "high" (≥6).
        "access_count": 7,
        "age_days": 14,
        "dependency_count": 3,
        "support_count": 4,
        "mixed_topic": False,
    },
}

CONFIDENCE_BIN_PROFILES: dict[str, dict] = {
    "low": {
        "memory_types": ["observation", "episodic_event", "semantic_fact"],
        "importance": 0.42,
        "confidence": 0.26,
        "access_count": 1,
        "age_days": 38,
        "dependency_count": 0,
        "support_count": 0,   # range 0-1 → none/low (B0 discriminator, was 1)
        "mixed_topic": True,
    },
    "medium": {
        "memory_types": ["semantic_fact", "observation", "preference"],
        "importance": 0.48,
        "confidence": 0.55,
        "access_count": 2,
        "age_days": 28,
        "dependency_count": 1,
        "support_count": 2,   # range 2-3 → always medium
        "mixed_topic": False,
    },
    "high": {
        "memory_types": ["knowledge", "constraint", "semantic_fact"],
        "importance": 0.56,
        "confidence": 0.78,
        "access_count": 3,
        "age_days": 18,
        "dependency_count": 2,
        "support_count": 4,   # range 4-5 → always high (B1 discriminator, was 3)
        "mixed_topic": False,
    },
}

TASK_PROFILES = {
    "decay_profile": DECAY_PROFILE_PROFILES,
    "confidence_bin": CONFIDENCE_BIN_PROFILES,
}

# Text templates (must mirror _fill_structured_ordinal_router_rows)
_REVIEW_CLAUSES = (
    "still comes up when the same thread returns",
    "shares the workspace with one nearby follow-up",
    "mostly stays in the background until the topic repeats",
    "keeps a small footprint in current review passes",
)
_SUPPORT_CLAUSES = (
    "reviewers still keep one supporting line around",
    "a nearby summary keeps a reference in circulation",
    "one follow-up note still echoes the detail",
    "the working set keeps a compact reminder attached",
)


def _build_structured_rows(task: str, label: str, count: int) -> list[dict]:
    profile = TASK_PROFILES[task][label]
    rows: list[dict] = []
    for idx in range(count):
        pack = _topic_pack(idx)
        other = _other_topic_pack(idx)
        boundary = idx % 6 == 0

        importance = float(profile["importance"]) + (0.035 * (idx % 3))
        confidence = float(profile["confidence"]) + (0.04 * ((idx + 1) % 3))
        access_count = int(profile["access_count"]) + (idx % 3)
        age_days = int(profile["age_days"]) + (idx % 30)
        dependency_count = int(profile["dependency_count"]) + (idx % 2)
        support_count = int(profile["support_count"]) + (idx % 2)
        mixed_topic = bool(profile["mixed_topic"])
        memory_type = _cycle_choice(
            list(profile["memory_types"]),
            idx,
            offset=1 if boundary else 0,
        )

        if boundary:
            label_val = label
            if label_val in {"low", "very_fast"}:
                importance += 0.08
                confidence += 0.05
                access_count += 1
                age_days = max(6, age_days - 20)
            elif label_val in {"high", "very_slow"}:
                importance -= 0.07
                confidence -= 0.05
                access_count = max(1, access_count - 1)
                age_days += 12
            else:
                importance += 0.02
                confidence += 0.01
                mixed_topic = True
            mixed_topic = True if label_val not in {"high", "very_slow"} else mixed_topic

        context_tags = [pack["topic"]]
        if mixed_topic:
            context_tags.append(other["topic"])

        review_clause = _cycle_choice(_REVIEW_CLAUSES, idx, offset=(1 if mixed_topic else 0))
        support_clause = _cycle_choice(_SUPPORT_CLAUSES, idx, offset=(1 if boundary else 0))

        # Seed fragment fallback (no seed pool available in this script)
        seed = f"{pack['topic'].title()} preference reference {idx}"

        text_variants = [
            f"Memory note {idx}: {pack['fact']}. Review notes say it {review_clause}.",
            f"Memory note {idx}: {pack['relevant']}. The current thread still mentions {seed.lower()} while {support_clause}.",
            f"Memory note {idx}: {pack['gist']}. A nearby {other['topic']} thread stays visible and {support_clause}.",
        ]
        if boundary:
            text_variants.append(
                f"Memory note {idx}: {pack['gist']}. Reviewers note a small overlap with {other['topic']} while {seed.lower()} remains attached."
            )
        text = text_variants[idx % len(text_variants)]

        extras = _structured_row_extras(
            topic=pack["topic"],
            memory_type=memory_type,
            importance=importance,
            confidence=confidence,
            access_count=access_count,
            age_days=age_days,
            dependency_count=dependency_count,
            support_count=support_count,
            mixed_topic=mixed_topic,
            context_tags=context_tags,
            group_id=f"structured:{task}:{idx}",
        )
        row = {
            "text": text,
            "task": task,
            "label": label,
            "source": f"structured:{task}:{label}",
            "language": "en",
            **extras,
        }
        rows.append(row)
    return rows


def patch_split(split: str) -> None:
    path = PREPARED_DIR / f"router_{split}.parquet"
    df = pd.read_parquet(path)
    orig_len = len(df)

    tasks = list(TASK_PROFILES.keys())
    mask = df["task"].isin(tasks) & df["source"].str.startswith("structured:", na=False)
    removed = mask.sum()
    df = df[~mask].copy()
    print(f"  [{split}] removed {removed} old structured rows")

    new_rows_list: list[dict] = []
    for task, profiles in TASK_PROFILES.items():
        label_counts = (
            pd.read_parquet(path)[pd.read_parquet(path)["task"] == task]
            .groupby("label")
            .size()
            .to_dict()
        )
        for label, count in label_counts.items():
            if label not in profiles:
                print(f"  WARNING: unknown label '{label}' for task '{task}', skipping")
                continue
            rows = _build_structured_rows(task, label, count)
            new_rows_list.extend(rows)
            print(f"  [{split}] generated {len(rows)} rows for {task}::{label}")

    new_df = pd.DataFrame(new_rows_list)
    # Align columns to match existing parquet schema
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[df.columns]

    result = pd.concat([df, new_df], ignore_index=True)
    result.to_parquet(path, index=False)
    print(f"  [{split}] {orig_len} → {len(result)} rows (delta: {len(result) - orig_len})")


def main() -> None:
    for split in ["train", "test", "eval"]:
        print(f"\nPatching router_{split}.parquet …")
        patch_split(split)
    print("\nDone. Re-train with:")
    print("  python3 -m cml.modeling.train --tasks decay_profile confidence_bin")


if __name__ == "__main__":
    main()
