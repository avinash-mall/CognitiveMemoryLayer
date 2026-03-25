"""Fix schema_match_pair SNLI label mapping in pair_train.parquet.

The previous patch (patch_schema_match_pair.py) used:
  entailment(0) → match, neutral(1)+contradiction(2) → no_match

The correct mapping (consistent with derived:hf:snli scope_match derivation) is:
  entailment(0)+contradiction(2) → match, neutral(1) → no_match

Contradiction pairs describe the *same scenario* from opposing views, so they share
the same schema and should be match, not no_match.

This script:
1. Removes the incorrectly-labeled hf:snli rows from pair_train.parquet
2. Re-adds them with the correct label mapping (contradiction → match)

Run from repo root:
    python packages/models/scripts/fix_schema_match_pair_labels.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import pyarrow as pa
import pyarrow.ipc as ipc
import pandas as pd
import numpy as np

MODELS_DIR = REPO_ROOT / "packages" / "models"
PREPARED_DIR = MODELS_DIR / "prepared_data" / "modelpack"
TASK = "schema_match_pair"
RANDOM_SEED = 43  # different seed from original patch


def _read_snli_arrow() -> pd.DataFrame:
    """Load SNLI train split with correct schema_match_pair label mapping.

    Correct mapping:
      0=entailment → match  (same scene, implication)
      2=contradiction → match  (same scene, opposing claim — same schema)
      1=neutral → no_match  (unrelated scene — different schema)
    """
    snli_dir = MODELS_DIR / "datasets" / "stanfordnlp___snli"
    arrow_candidates = list(snli_dir.glob("**/snli-train.arrow"))
    if not arrow_candidates:
        raise FileNotFoundError(f"SNLI train arrow not found under {snli_dir}")
    arrow_file = arrow_candidates[0]
    print(f"[fix] Loading SNLI from {arrow_file}")
    with pa.memory_map(str(arrow_file), "r") as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    df = table.to_pandas()
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)
    # entailment(0) + contradiction(2) → match; neutral(1) → no_match
    df["schema_label"] = df["label"].map({0: "match", 2: "match", 1: "no_match"})
    return df


def _build_snli_rows(snli_df: pd.DataFrame, n_match: int, n_no_match: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    match_pool = snli_df[snli_df["schema_label"] == "match"].reset_index(drop=True)
    no_match_pool = snli_df[snli_df["schema_label"] == "no_match"].reset_index(drop=True)

    match_idx = rng.choice(len(match_pool), size=min(n_match, len(match_pool)), replace=False)
    no_match_idx = rng.choice(
        len(no_match_pool), size=min(n_no_match, len(no_match_pool)), replace=False
    )

    rows = []
    for idx in match_idx:
        row = match_pool.iloc[int(idx)]
        rows.append(
            {
                "task": TASK,
                "text_a": str(row["premise"]),
                "text_b": str(row["hypothesis"]),
                "label": "match",
                "source": "hf:snli",
                "language": "en",
                "group_id": f"hf:snli:match:{int(idx)}",
            }
        )
    for idx in no_match_idx:
        row = no_match_pool.iloc[int(idx)]
        rows.append(
            {
                "task": TASK,
                "text_a": str(row["premise"]),
                "text_b": str(row["hypothesis"]),
                "label": "no_match",
                "source": "hf:snli",
                "language": "en",
                "group_id": f"hf:snli:no_match:{int(idx)}",
            }
        )
    return pd.DataFrame(rows)


def main() -> None:
    train_path = PREPARED_DIR / "pair_train.parquet"
    df_train = pd.read_parquet(train_path)

    smp_all = df_train[df_train["task"] == TASK]
    print(f"[fix] Existing {TASK} rows: {len(smp_all)}")
    print(f"[fix] Sources: {smp_all['source'].value_counts().to_dict()}")
    print(f"[fix] Labels: {smp_all['label'].value_counts().to_dict()}")

    # Remove the incorrectly-labeled hf:snli rows
    snli_mask = (df_train["task"] == TASK) & (df_train["source"] == "hf:snli")
    n_removed = int(snli_mask.sum())
    df_clean = df_train[~snli_mask].reset_index(drop=True)
    print(f"[fix] Removed {n_removed} hf:snli rows (wrong contradiction mapping)")

    # Check what remains
    smp_remaining = df_clean[df_clean["task"] == TASK]
    n_match_have = int((smp_remaining["label"] == "match").sum())
    n_no_match_have = int((smp_remaining["label"] == "no_match").sum())
    print(f"[fix] Remaining: match={n_match_have}, no_match={n_no_match_have}")

    # Target ~4600 per label (restore prior total)
    target_per_label = 4600
    n_match_need = max(0, target_per_label - n_match_have)
    n_no_match_need = max(0, target_per_label - n_no_match_have)
    print(f"[fix] Need to add: match={n_match_need}, no_match={n_no_match_need}")

    snli_df = _read_snli_arrow()
    print(f"[fix] SNLI pool: match={int((snli_df['schema_label']=='match').sum())}, no_match={int((snli_df['schema_label']=='no_match').sum())}")

    new_rows = _build_snli_rows(snli_df, n_match_need, n_no_match_need)

    # Deduplicate against existing rows by (text_a, text_b)
    existing_keys = set(
        zip(df_clean["text_a"].astype(str), df_clean["text_b"].astype(str))
    )
    new_rows = new_rows[
        ~new_rows.apply(lambda r: (r["text_a"], r["text_b"]) in existing_keys, axis=1)
    ].reset_index(drop=True)
    print(f"[fix] After dedup: adding {len(new_rows)} new SNLI rows")
    print(
        f"[fix] New rows: match={int((new_rows['label']=='match').sum())}, no_match={int((new_rows['label']=='no_match').sum())}"
    )

    df_patched = pd.concat([df_clean, new_rows], ignore_index=True)

    smp_final = df_patched[df_patched["task"] == TASK]
    print(f"[fix] Final {TASK} rows: {len(smp_final)}")
    print(f"[fix] Final sources: {smp_final['source'].value_counts().to_dict()}")
    print(f"[fix] Final labels: {smp_final['label'].value_counts().to_dict()}")

    df_patched.to_parquet(train_path, index=False)
    print(f"[fix] Saved updated pair_train.parquet ({len(df_patched)} total rows)")


if __name__ == "__main__":
    main()
