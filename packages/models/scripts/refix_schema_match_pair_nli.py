"""Re-fix schema_match_pair NLI labels in pair_train/test/eval parquets.

After a full prepare run, the pair parquets have schema_match_pair rows with the old
wrong label mapping: only entailment→match, contradiction→no_match.

Correct mapping (as documented in dataset_shortlist.md):
  entailment   → match     (supporting claim, same scenario)
  contradiction → match     (opposing claim, SAME SCENARIO — shares the schema)
  neutral      → no_match   (different topic/domain)

This script:
  1. Removes existing schema_match_pair rows from all three splits
  2. Re-loads snli, multi_nli, anli from the HuggingFace cache
  3. Re-generates schema_match_pair rows with the correct mapping
  4. Appends the fixed rows back and saves

Also fixes in prepare.py: line that generates these rows. Run this script if you
need to apply the fix to already-prepared parquets without a full re-prepare.

Run from repo root:
    python packages/models/scripts/refix_schema_match_pair_nli.py
"""
from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import pandas as pd  # noqa: E402

PREPARED_DIR = REPO_ROOT / "packages" / "models" / "prepared_data" / "modelpack"
CACHE_DIR = REPO_ROOT / "packages" / "models" / "datasets"

DATASET_CONFIGS = [
    ("snli", "stanfordnlp/snli", "", "train"),
    ("multi_nli", "nyu-mll/multi_nli", "", "train"),
    ("anli", "facebook/anli", "", "train_r3"),
]


def _nli_label_name(label_int: int, label_names: list[str]) -> str:
    if 0 <= label_int < len(label_names):
        return label_names[label_int]
    return "neutral"


def _load_nli_dataset(dataset_id: str, config: str, split: str):
    """Load an NLI dataset from HuggingFace cache (offline mode)."""
    try:
        from datasets import load_dataset  # type: ignore
        ds = load_dataset(
            dataset_id,
            config if config else None,
            split=split,
            cache_dir=str(CACHE_DIR),
        )
        return ds
    except Exception as e:
        print(f"  WARNING: could not load {dataset_id}: {e}")
        return None


def _safe_feature_names(ds) -> list[str]:
    """Extract label class names from dataset features."""
    try:
        feat = ds.features.get("label")
        if feat is not None and hasattr(feat, "names"):
            return list(feat.names)
    except Exception:
        pass
    return ["entailment", "neutral", "contradiction"]


def build_nli_schema_rows(max_rows_per_source: int = 200_000) -> pd.DataFrame:
    """Build schema_match_pair rows from all NLI datasets with correct mapping."""
    all_rows: list[dict] = []

    for name, dataset_id, config, split in DATASET_CONFIGS:
        print(f"  Loading {name} ({dataset_id}) …")
        ds = _load_nli_dataset(dataset_id, config, split)
        if ds is None:
            print(f"  SKIP {name} (not available)")
            continue

        label_names = _safe_feature_names(ds)
        limit = min(max_rows_per_source, len(ds))

        added = 0
        for i, ex in enumerate(ds):
            if i >= limit:
                break
            if not isinstance(ex, dict):
                continue
            premise = str(ex.get("premise", ex.get("hypothesis1", ""))).strip()
            hypothesis = str(ex.get("hypothesis", ex.get("hypothesis2", ""))).strip()
            if not premise or not hypothesis:
                continue
            label_int = ex.get("label", -1)
            label_name = _nli_label_name(int(label_int), label_names)
            if label_name not in {"entailment", "contradiction", "neutral"}:
                continue

            # Correct mapping: entailment + contradiction → match (same schema)
            schema_label = "match" if label_name in {"entailment", "contradiction"} else "no_match"
            all_rows.append({
                "text_a": premise,
                "text_b": hypothesis,
                "task": "schema_match_pair",
                "label": schema_label,
                "source": f"hf:{name}",
                "language": "en",
                "group_id": None,
                "score": None,
            })
            added += 1

        print(f"  {name}: {added} rows")

    df = pd.DataFrame(all_rows)
    print(f"  Total NLI rows for schema_match_pair: {len(df)}")
    return df


def stratified_sample(df: pd.DataFrame, n_per_label: int, seed: int = 42) -> pd.DataFrame:
    """Return at most n_per_label rows per label via stratified sampling."""
    groups = []
    for label, grp in df.groupby("label"):
        if len(grp) > n_per_label:
            grp = grp.sample(n_per_label, random_state=seed)
        groups.append(grp)
    return pd.concat(groups, ignore_index=True)


def patch_split(split: str, nli_pool: pd.DataFrame, seed: int = 42) -> None:
    path = PREPARED_DIR / f"pair_{split}.parquet"
    df = pd.read_parquet(path)
    orig_len = len(df)

    # Remove existing schema_match_pair rows
    mask_smp = df["task"] == "schema_match_pair"
    removed = mask_smp.sum()
    n_per_label_orig = df[mask_smp].groupby("label").size().to_dict()
    df = df[~mask_smp].copy()
    print(f"  [{split}] removed {removed} old schema_match_pair rows "
          f"({n_per_label_orig})")

    # Sample from the pool to match original per-label counts (or max available)
    n_per_label = max(n_per_label_orig.values()) if n_per_label_orig else 5000
    sampled = stratified_sample(nli_pool, n_per_label, seed=seed + hash(split) % 997)

    # Align columns to match existing parquet schema
    new_df = sampled.copy()
    for col in df.columns:
        if col not in new_df.columns:
            new_df[col] = None
    new_df = new_df[df.columns]

    result = pd.concat([df, new_df], ignore_index=True)
    result.to_parquet(path, index=False)
    new_counts = result[result["task"] == "schema_match_pair"]["label"].value_counts().to_dict()
    print(f"  [{split}] {orig_len} → {len(result)} rows; "
          f"schema_match_pair: {new_counts}")


def main() -> None:
    print("Loading NLI data from cache …")
    nli_pool = build_nli_schema_rows()

    label_dist = nli_pool["label"].value_counts().to_dict()
    print(f"Pool distribution: {label_dist}")
    if not label_dist:
        print("ERROR: no NLI rows loaded. Ensure datasets are cached.")
        sys.exit(1)

    for split in ["train", "test", "eval"]:
        print(f"\nPatching pair_{split}.parquet …")
        patch_split(split, nli_pool, seed=42)

    print("\nDone. Retrain with:")
    print("  python3 -m cml.modeling.train --tasks schema_match_pair")


if __name__ == "__main__":
    main()
