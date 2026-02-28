"""Unit tests for incremental/missing-only behavior in model prepare script."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

pd = pytest.importorskip("pandas")

from packages.models.scripts import prepare as p


class _DummyLLM:
    def __init__(self) -> None:
        self.single_calls = 0
        self.pair_calls = 0

    def generate_single(self, *, task: str, label: str, seed_text: str, n: int):
        self.single_calls += 1
        return [f"{task}:{label}:{i}" for i in range(n)]

    def generate_pair(self, *, task: str, label: str, seed_a: str, seed_b: str, n: int):
        self.pair_calls += 1
        return [(f"{task}:{label}:a{i}", f"{task}:{label}:b{i}") for i in range(n)]


def _full_single_df(task_labels: dict[str, list[str]], *, target: int = 1) -> Any:
    rows = []
    for task, labels in task_labels.items():
        for label in labels:
            for i in range(target):
                rows.append(
                    {
                        "text": f"{task}:{label}:{i}",
                        "task": task,
                        "label": label,
                        "source": "prepared:test",
                    }
                )
    return pd.DataFrame(rows)


def _full_pair_df(task_labels: dict[str, list[str]], *, target: int = 1) -> Any:
    rows = []
    for task, labels in task_labels.items():
        for label in labels:
            for i in range(target):
                rows.append(
                    {
                        "text_a": f"{task}:{label}:a{i}",
                        "text_b": f"{task}:{label}:b{i}",
                        "task": task,
                        "label": label,
                        "source": "prepared:test",
                    }
                )
    return pd.DataFrame(rows)


def test_missing_task_labels_counts():
    df = pd.DataFrame(
        [
            {"text": "a", "task": "query_domain", "label": "food", "source": "x"},
            {"text": "b", "task": "query_domain", "label": "food", "source": "x"},
            {"text": "c", "task": "query_domain", "label": "travel", "source": "x"},
        ]
    )
    missing, total = p._missing_task_labels(
        df,
        task_labels={"query_domain": ["food", "travel", "finance"]},
        target_per_task_label=2,
    )
    assert missing["query_domain::food"] == 0
    assert missing["query_domain::travel"] == 1
    assert missing["query_domain::finance"] == 2
    assert total == 3


def test_load_existing_family_df_reads_splits(tmp_path: Path):
    out = tmp_path / "prepared"
    out.mkdir(parents=True, exist_ok=True)

    train = pd.DataFrame(
        [{"text": "a", "task": "query_domain", "label": "food", "source": "train"}]
    )
    test = pd.DataFrame(
        [{"text": "b", "task": "query_domain", "label": "travel", "source": "test"}]
    )
    train.to_parquet(out / "router_train.parquet", index=False)
    test.to_parquet(out / "router_test.parquet", index=False)

    df = p._load_existing_family_df(out, "router")
    assert len(df) == 2
    assert set(df["label"]) == {"food", "travel"}


def test_build_router_rows_skips_llm_if_existing_complete():
    existing = _full_single_df(p.ROUTER_TASK_LABELS, target=1)
    llm = _DummyLLM()
    rows = p._build_router_rows(
        local_rows=[],
        registry=object(),  # unused when fully satisfied
        prepare_cfg={"seed": 7, "target_per_task_label": 1, "max_per_task_label": 1},
        synthetic_cfg={"max_attempts_per_label": 1},
        single_pools={"router": ["seed"], "extractor": ["seed"], "pair": ["seed"]},
        llm=llm,
        existing_df=existing,
    )
    assert len(rows) == len(existing)
    assert llm.single_calls == 0


def test_build_extractor_rows_skips_llm_if_existing_complete():
    existing = _full_single_df(p.EXTRACTOR_TASK_LABELS, target=1)
    llm = _DummyLLM()
    rows = p._build_extractor_rows(
        registry=object(),  # unused when fully satisfied
        prepare_cfg={"seed": 11, "target_per_task_label": 1, "max_per_task_label": 1},
        synthetic_cfg={"max_attempts_per_label": 1},
        single_pools={"router": ["seed"], "extractor": ["seed"], "pair": ["seed"]},
        llm=llm,
        existing_df=existing,
    )
    assert len(rows) == len(existing)
    assert llm.single_calls == 0


def test_build_pair_rows_skips_llm_if_existing_complete():
    existing = _full_pair_df(p.PAIR_TASK_LABELS, target=1)
    llm = _DummyLLM()
    rows = p._build_pair_rows(
        registry=object(),  # unused when fully satisfied
        prepare_cfg={"seed": 13, "target_per_task_label": 1, "max_per_task_label": 1},
        synthetic_cfg={"max_attempts_per_label": 1},
        single_pools={"router": ["seed"], "extractor": ["seed"], "pair": ["seed"]},
        pair_pool=[("a", "b")],
        llm=llm,
        existing_df=existing,
    )
    assert len(rows) == len(existing)
    assert llm.pair_calls == 0
