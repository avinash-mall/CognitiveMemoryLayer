"""Patch schema_match_pair training data: remove FEVER rows, add real SNLI pairs.

FEVER evidence text is stored as "Title sentence N" references (not actual sentences),
so FEVER rows in schema_match_pair carry no semantic signal for text_b. Removing them
and replacing with SNLI entailment pairs gives the model real NLI-style training signal.

Run from repo root:
    python packages/models/scripts/patch_schema_match_pair.py
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
RANDOM_SEED = 42


def _read_snli_arrow() -> pd.DataFrame:
    """Load SNLI train split from local Arrow file.

    Label mapping: 0=entailment, 1=neutral, 2=contradiction, -1=invalid.
    schema_match_pair: entailment→match, neutral/contradiction→no_match.
    """
    snli_dir = MODELS_DIR / "datasets" / "stanfordnlp___snli"
    arrow_candidates = list(snli_dir.glob("**/snli-train.arrow"))
    if not arrow_candidates:
        raise FileNotFoundError(f"SNLI train arrow not found under {snli_dir}")
    arrow_file = arrow_candidates[0]
    print(f"[patch] Loading SNLI from {arrow_file}")
    with pa.memory_map(str(arrow_file), "r") as source:
        reader = ipc.open_stream(source)
        table = reader.read_all()
    df = table.to_pandas()
    # Keep only valid labels
    df = df[df["label"].isin([0, 1, 2])].reset_index(drop=True)
    df["schema_label"] = df["label"].map({0: "match", 1: "no_match", 2: "no_match"})
    return df


def _build_snli_rows(snli_df: pd.DataFrame, n_match: int, n_no_match: int) -> pd.DataFrame:
    rng = np.random.default_rng(RANDOM_SEED)

    match_pool = snli_df[snli_df["schema_label"] == "match"].reset_index(drop=True)
    no_match_pool = snli_df[snli_df["schema_label"] == "no_match"].reset_index(drop=True)

    match_idx = rng.choice(len(match_pool), size=min(n_match, len(match_pool)), replace=False)
    no_match_idx = rng.choice(len(no_match_pool), size=min(n_no_match, len(no_match_pool)), replace=False)

    rows = []
    for idx in match_idx:
        row = match_pool.iloc[int(idx)]
        rows.append({
            "task": TASK,
            "text_a": str(row["premise"]),
            "text_b": str(row["hypothesis"]),
            "label": "match",
            "source": "hf:snli",
            "language": "en",
            "group_id": f"hf:snli:match:{int(idx)}",
        })
    for idx in no_match_idx:
        row = no_match_pool.iloc[int(idx)]
        rows.append({
            "task": TASK,
            "text_a": str(row["premise"]),
            "text_b": str(row["hypothesis"]),
            "label": "no_match",
            "source": "hf:snli",
            "language": "en",
            "group_id": f"hf:snli:no_match:{int(idx)}",
        })
    return pd.DataFrame(rows)


def main() -> None:
    train_path = PREPARED_DIR / "pair_train.parquet"
    df_train = pd.read_parquet(train_path)

    # Count existing schema_match_pair rows by source
    smp_all = df_train[df_train["task"] == TASK]
    print(f"[patch] Existing {TASK} rows: {len(smp_all)}")
    print(f"[patch] Sources: {smp_all['source'].value_counts().to_dict()}")
    print(f"[patch] Labels: {smp_all['label'].value_counts().to_dict()}")

    # Remove FEVER rows
    fever_mask = (df_train["task"] == TASK) & (df_train["source"] == "hf:fever")
    n_removed = fever_mask.sum()
    df_train_clean = df_train[~fever_mask].reset_index(drop=True)
    print(f"[patch] Removed {n_removed} FEVER rows")

    # How many of each label do we still have?
    smp_remaining = df_train_clean[df_train_clean["task"] == TASK]
    n_match_have = int((smp_remaining["label"] == "match").sum())
    n_no_match_have = int((smp_remaining["label"] == "no_match").sum())
    print(f"[patch] Remaining after removal: match={n_match_have}, no_match={n_no_match_have}")

    # Target: same total as before (4600 per label)
    target_per_label = max(n_match_have + n_removed // 2, 4600)
    n_match_need = max(0, target_per_label - n_match_have)
    n_no_match_need = max(0, target_per_label - n_no_match_have)
    print(f"[patch] Need to add: match={n_match_need}, no_match={n_no_match_need}")

    if n_match_need > 0 or n_no_match_need > 0:
        snli_df = _read_snli_arrow()
        new_rows = _build_snli_rows(snli_df, n_match_need, n_no_match_need)

        # Deduplicate against existing rows by (text_a, text_b)
        existing_keys = set(
            zip(df_train_clean["text_a"].astype(str), df_train_clean["text_b"].astype(str))
        )
        new_rows = new_rows[
            ~new_rows.apply(lambda r: (r["text_a"], r["text_b"]) in existing_keys, axis=1)
        ].reset_index(drop=True)
        print(f"[patch] After dedup: adding {len(new_rows)} new SNLI rows")

        df_train_patched = pd.concat([df_train_clean, new_rows], ignore_index=True)
    else:
        df_train_patched = df_train_clean

    smp_final = df_train_patched[df_train_patched["task"] == TASK]
    print(f"[patch] Final {TASK} rows: {len(smp_final)}")
    print(f"[patch] Final sources: {smp_final['source'].value_counts().to_dict()}")
    print(f"[patch] Final labels: {smp_final['label'].value_counts().to_dict()}")

    df_train_patched.to_parquet(train_path, index=False)
    print(f"[patch] Saved updated pair_train.parquet ({len(df_train_patched)} total rows)")


if __name__ == "__main__":
    main()
