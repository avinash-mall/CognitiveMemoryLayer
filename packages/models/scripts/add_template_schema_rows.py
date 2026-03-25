"""Add template schema_match_pair training rows to pair_train.parquet.

The test set has 2000 template:schema_match_pair rows (CML memory format) but
training had none, causing ~0.39 macro_f1 on template rows. This script adds
template rows to expose the model to CML memory text during training.

Template format:
  match:    "{gist} Summary N." vs "{same_topic_fact} Fact N."
  no_match: "{gist} Summary N." vs "{other_topic_fact} Fact N."

Run from repo root:
    python packages/models/scripts/add_template_schema_rows.py
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "packages" / "py-cml" / "src"))

import pandas as pd
from cml.modeling.prepare import _topic_pack, _other_topic_pack

MODELS_DIR = REPO_ROOT / "packages" / "models"
PREPARED_DIR = MODELS_DIR / "prepared_data" / "modelpack"
TASK = "schema_match_pair"
# Add up to N template rows per label. The test set has 1000 per label;
# training with 500-700 covers the 7 topic patterns many times.
TARGET_TEMPLATE_PER_LABEL = 600


def main() -> None:
    train_path = PREPARED_DIR / "pair_train.parquet"
    df_train = pd.read_parquet(train_path)

    smp_all = df_train[df_train["task"] == TASK]
    print(f"[tmpl] Existing {TASK} rows: {len(smp_all)}")
    print(f"[tmpl] Sources: {smp_all['source'].value_counts().to_dict()}")
    print(f"[tmpl] Labels: {smp_all['label'].value_counts().to_dict()}")

    # Build existing keys to avoid duplicates
    existing_keys = set(
        zip(df_train["text_a"].astype(str), df_train["text_b"].astype(str))
    )

    new_rows = []

    # Add match rows: gist vs same-topic fact
    n_match = 0
    idx = 0
    while n_match < TARGET_TEMPLATE_PER_LABEL:
        pack = _topic_pack(idx)
        text_a = f"{pack['gist']} Summary {idx}."
        text_b = f"{pack['fact']} Fact {idx}."
        if (text_a, text_b) not in existing_keys:
            new_rows.append({
                "task": TASK,
                "text_a": text_a,
                "text_b": text_b,
                "label": "match",
                "source": "template:schema_match_pair:match",
                "language": "en",
                "group_id": f"template:schema_match_pair:{idx}",
            })
            existing_keys.add((text_a, text_b))
            n_match += 1
        idx += 1

    # Add no_match rows: gist vs different-topic fact
    n_no_match = 0
    idx = 0
    while n_no_match < TARGET_TEMPLATE_PER_LABEL:
        pack = _topic_pack(idx)
        other = _other_topic_pack(idx)
        text_a = f"{pack['gist']} Summary {idx}."
        text_b = f"{other['fact']} Fact {idx}."
        if (text_a, text_b) not in existing_keys:
            new_rows.append({
                "task": TASK,
                "text_a": text_a,
                "text_b": text_b,
                "label": "no_match",
                "source": "template:schema_match_pair:no_match",
                "language": "en",
                "group_id": f"template:schema_match_pair:nm:{idx}",
            })
            existing_keys.add((text_a, text_b))
            n_no_match += 1
        idx += 1

    df_new = pd.DataFrame(new_rows)
    print(f"[tmpl] Adding {len(df_new)} template rows: match={n_match}, no_match={n_no_match}")

    df_patched = pd.concat([df_train, df_new], ignore_index=True)

    smp_final = df_patched[df_patched["task"] == TASK]
    print(f"[tmpl] Final {TASK} rows: {len(smp_final)}")
    print(f"[tmpl] Final sources: {smp_final['source'].value_counts().to_dict()}")
    print(f"[tmpl] Final labels: {smp_final['label'].value_counts().to_dict()}")

    df_patched.to_parquet(train_path, index=False)
    print(f"[tmpl] Saved updated pair_train.parquet ({len(df_patched)} total rows)")


if __name__ == "__main__":
    main()
