"""Unit tests for modeling prepare API wrappers."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pytest

pytest.importorskip("pandas")
import pandas as pd

import cml.modeling.prepare as prepare_module
from cml.modeling.memory_type_features import (
    MEMORY_TYPE_FEATURE_COLUMNS,
    derive_memory_type_feature_columns,
    derive_memory_type_feature_tokens_from_row,
    derive_memory_type_feature_tokens_from_text,
)
from cml.modeling.pair_features import build_pair_lexical_features
from cml.modeling.types import PrepareConfig


def test_prepare_data_builds_expected_argv(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(prepare_module, "main", _fake_main)
    cfg = PrepareConfig(
        config_path=tmp_path / "model_pipeline.toml",
        seed=42,
        target_per_task_label=100,
        force_full=True,
        no_multilingual=True,
    )
    rc = prepare_module.prepare_data(cfg)
    assert rc == 0
    assert "--config" in captured["argv"]
    assert "--seed" in captured["argv"]
    assert "--force-full" in captured["argv"]
    assert "--no-multilingual" in captured["argv"]


def test_novelty_pair_labels_are_normalized_to_changed() -> None:
    assert prepare_module._normalize_pair_task_label("novelty_pair", "contradiction") == "changed"
    assert prepare_module._normalize_pair_task_label("novelty_pair", "temporal_change") == "changed"
    assert prepare_module._normalize_pair_task_label("novelty_pair", "duplicate") == "duplicate"


def test_enabled_embedding_pair_tasks_extracts_shared_model_name() -> None:
    tasks, model_name = prepare_module._enabled_embedding_pair_tasks(
        [
            {
                "task_name": "memory_rerank_pair",
                "family": "pair",
                "enabled": True,
                "trainer": "embedding_pair",
                "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            },
            {
                "task_name": "retrieval_constraint_relevance_pair",
                "family": "pair",
                "enabled": True,
                "feature_backend": "embedding_pair",
                "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
            },
        ]
    )

    assert tasks == {"memory_rerank_pair", "retrieval_constraint_relevance_pair"}
    assert model_name == "sentence-transformers/all-MiniLM-L6-v2"


def test_force_full_inputs_do_not_load_existing_prepared_rows(monkeypatch, tmp_path: Path) -> None:
    def _fail_family(*args, **kwargs):
        raise AssertionError("force_full should not load existing family frames")

    def _fail_token(*args, **kwargs):
        raise AssertionError("force_full should not load existing token frames")

    monkeypatch.setattr(prepare_module, "_load_existing_family_df", _fail_family)
    monkeypatch.setattr(prepare_module, "_load_existing_token_df", _fail_token)

    router_df, extractor_df, pair_df, token_dfs = prepare_module._load_existing_prepared_inputs(
        tmp_path,
        force_full=True,
    )

    assert router_df.empty
    assert extractor_df.empty
    assert pair_df.empty
    assert set(token_dfs) == set(prepare_module.TOKEN_TASKS)
    assert all(df.empty for df in token_dfs.values())


def test_prepare_main_manifest_omits_adversarial_fields(monkeypatch, tmp_path: Path) -> None:
    config_path = tmp_path / "model_pipeline.toml"
    config_path.write_text("", encoding="utf-8")
    prepared_dir = tmp_path / "prepared"
    bootstrap_dir = tmp_path / "bootstrap"
    cache_dir = tmp_path / "cache"

    monkeypatch.setattr(prepare_module, "_load_dotenv", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        prepare_module,
        "_load_config",
        lambda _path: {
            "paths": {
                "prepared_dir": str(prepared_dir),
                "bootstrap_prepared_dir": str(bootstrap_dir),
                "datasets_cache_dir": str(cache_dir),
            },
            "prepare": {},
            "synthetic_llm": {},
            "multilingual": {},
            "datasets": [],
            "tasks": [],
        },
    )
    monkeypatch.setattr(
        prepare_module,
        "_load_existing_prepared_inputs",
        lambda *_args, **_kwargs: (pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), {}),
    )
    monkeypatch.setattr(prepare_module, "_missing_task_labels", lambda *_args, **_kwargs: ({}, 0))
    monkeypatch.setattr(
        prepare_module, "_missing_regression_tasks", lambda *_args, **_kwargs: ({}, 0)
    )
    monkeypatch.setattr(prepare_module, "_missing_token_tasks", lambda *_args, **_kwargs: ({}, 0))
    monkeypatch.setattr(
        prepare_module, "_pair_embedding_cache_summary", lambda *_args, **_kwargs: {}
    )
    monkeypatch.setattr(prepare_module, "_existing_split_counts", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prepare_module, "_existing_split_hashes", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prepare_module, "_existing_split_integrity", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(prepare_module, "_existing_label_coverage", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(
        prepare_module,
        "_existing_train_source_diagnostics",
        lambda *_args, **_kwargs: {},
    )
    monkeypatch.setattr(
        prepare_module, "_existing_token_split_counts", lambda *_args, **_kwargs: {}
    )

    rc = prepare_module.main(["--config", str(config_path)])

    assert rc == 0
    manifest = json.loads((prepared_dir / "manifest.json").read_text(encoding="utf-8"))
    assert "adversarial_fixtures" not in manifest
    assert "locked_adversarial_eval_tasks" not in manifest


def test_llm_generator_prefers_env_over_config(monkeypatch) -> None:
    class _FakeClient:
        def __init__(self, *args, **kwargs) -> None:
            pass

        def close(self) -> None:
            pass

    class _FakeHttpx:
        Limits = staticmethod(lambda **kwargs: kwargs)
        Client = _FakeClient

    monkeypatch.setattr(prepare_module, "httpx", _FakeHttpx)
    monkeypatch.setattr(
        prepare_module._LLMGenerator,
        "_fetch_model_metadata",
        lambda self: None,
    )
    monkeypatch.setattr(
        prepare_module,
        "_thinking_disable_policy",
        lambda **_kwargs: {},
    )
    monkeypatch.setenv("LLM_EVAL__PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_EVAL__MODEL", "Qwen/Qwen3.5-27B")
    monkeypatch.setenv("LLM_EVAL__BASE_URL", "http://localhost:8001/v1")

    llm = prepare_module._LLMGenerator(
        {
            "provider": "vllm",
            "model": "Qwen/Qwen3.5-9B",
            "base_url": "http://localhost:8000/v1",
        }
    )

    assert llm.provider == "openai_compatible"
    assert llm.model == "Qwen/Qwen3.5-27B"
    assert llm.base_url == "http://localhost:8001/v1"


def test_inject_hardened_router_rows_generates_reserved_rows() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=12)

    prepare_module._inject_hardened_router_rows(
        rows=rows,
        task_labels={
            "consolidation_gist_quality": ["accept", "reject"],
            "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
        },
        target_per_task_label=12,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
    )

    assert rows.count("consolidation_gist_quality", "accept") == 2
    assert rows.count("consolidation_gist_quality", "reject") == 2
    for label in ("keep", "decay", "silence", "compress", "delete"):
        assert rows.count("forgetting_action_policy", label) == 2

    sources = {str(row["source"]) for row in rows.rows}
    assert "template_hardened:consolidation_gist_quality:accept" in sources
    assert "template_hardened:consolidation_gist_quality:reject" in sources
    assert "template_hardened:forgetting_action_policy:compress" in sources

    gist_prefixes = {
        " ".join(str(row["text"]).split()[:2])
        for row in rows.rows
        if row["task"] == "consolidation_gist_quality"
    }
    assert gist_prefixes & {"Cluster review", "Session draft", "Memory brief"}

    forgetting_prefixes = {
        " ".join(str(row["text"]).split()[:2])
        for row in rows.rows
        if row["task"] == "forgetting_action_policy"
    }
    assert forgetting_prefixes == {"Retention review"}


def test_group_aware_split_keeps_group_ids_in_single_split() -> None:
    rows = []
    for idx in range(9):
        group_id = f"group-{idx}"
        rows.append(
            {
                "task": "schema_match_pair",
                "label": "match" if idx % 2 == 0 else "no_match",
                "text_a": f"query {idx}",
                "text_b": f"memory {idx}",
                "source": "hf:test",
                "group_id": group_id,
            }
        )
        rows.append(
            {
                "task": "reconsolidation_candidate_pair",
                "label": "relevant" if idx % 2 == 0 else "not_relevant",
                "text_a": f"query {idx}",
                "text_b": f"candidate {idx}",
                "source": "hf:test",
                "group_id": group_id,
            }
        )
    df = pd.DataFrame(rows)

    splits = prepare_module._split_by_task_label(
        df,
        seed=42,
        ratios={"train": 0.8, "test": 0.1, "eval": 0.1},
    )
    integrity = prepare_module._split_integrity_summary(splits)

    assert integrity["ok"] is True
    assert sum(integrity["overlap_counts"].values()) == 0
    assert sum(len(frame) for frame in splits.values()) == len(df)


def test_group_aware_split_reserves_required_source_prefix_in_train() -> None:
    df = pd.DataFrame(
        [
            {
                "task": "schema_match_pair",
                "label": "match",
                "text_a": "claim fever",
                "text_b": "evidence fever",
                "source": "hf:fever",
                "group_id": "g-fever",
            },
            {
                "task": "schema_match_pair",
                "label": "match",
                "text_a": "claim derived 1",
                "text_b": "evidence derived 1",
                "source": "derived:hf:snli",
                "group_id": "g-derived-1",
            },
            {
                "task": "schema_match_pair",
                "label": "no_match",
                "text_a": "claim derived 2",
                "text_b": "evidence derived 2",
                "source": "derived:hf:snli",
                "group_id": "g-derived-2",
            },
            {
                "task": "schema_match_pair",
                "label": "no_match",
                "text_a": "claim derived 3",
                "text_b": "evidence derived 3",
                "source": "derived:hf:multi_nli",
                "group_id": "g-derived-3",
            },
        ]
    )

    splits = prepare_module._split_by_task_label(
        df,
        seed=11,
        ratios={"train": 0.25, "test": 0.5, "eval": 0.25},
    )

    train_schema = splits["train"][splits["train"]["task"].astype(str) == "schema_match_pair"]
    assert train_schema["source"].astype(str).str.startswith("hf:fever").any()


def test_router_structured_fill_keeps_group_ids_and_non_template_majority() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=12)
    prepare_module._inject_hardened_router_rows(
        rows=rows,
        task_labels={
            "consolidation_gist_quality": ["accept", "reject"],
            "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
        },
        target_per_task_label=12,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
    )
    prepare_module._fill_router_tasks_without_llm(
        rows=rows,
        regression_rows=prepare_module._RegressionTaskStore(max_per_task=12),
        task_labels={
            "consolidation_gist_quality": ["accept", "reject"],
            "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
        },
        regression_tasks=set(),
        target_per_task_label=12,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
    )
    df = pd.DataFrame(rows.rows)
    diagnostics = prepare_module._validate_source_coverage(df, split_name="router:train")

    assert df["group_id"].astype(str).str.len().gt(0).all()
    assert diagnostics["consolidation_gist_quality"]["template_ratio"] <= 0.5
    assert diagnostics["forgetting_action_policy"]["template_ratio"] <= 0.5
    assert any(
        str(source).startswith("structured:consolidation_gist_quality") for source in df["source"]
    )
    assert any(
        str(source).startswith("structured:forgetting_action_policy") for source in df["source"]
    )


def test_router_quality_structured_fill_respects_source_caps_when_llm_available() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=20)
    prepare_module._inject_hardened_router_rows(
        rows=rows,
        task_labels={
            "consolidation_gist_quality": ["accept", "reject"],
            "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
        },
        target_per_task_label=20,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
    )

    prepare_module._fill_router_tasks_without_llm(
        rows=rows,
        regression_rows=prepare_module._RegressionTaskStore(max_per_task=20),
        task_labels={
            "consolidation_gist_quality": ["accept", "reject"],
            "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
        },
        regression_tasks=set(),
        target_per_task_label=20,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
        llm_available=True,
    )

    def _count(task: str, label: str, prefix: str) -> int:
        return sum(
            1
            for row in rows.rows
            if str(row["task"]) == task
            and str(row["label"]) == label
            and str(row["source"]).startswith(prefix)
        )

    assert _count("consolidation_gist_quality", "accept", "structured:") == 7
    assert _count("consolidation_gist_quality", "reject", "structured:") == 7
    assert _count("forgetting_action_policy", "keep", "structured:") == 7
    assert rows.count("consolidation_gist_quality", "accept") == 11
    assert rows.count("forgetting_action_policy", "keep") == 11


def test_router_quality_source_validation_passes_after_llm_top_up() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=20)
    task_labels = {
        "consolidation_gist_quality": ["accept", "reject"],
        "forgetting_action_policy": ["keep", "decay", "silence", "compress", "delete"],
    }
    prepare_module._inject_hardened_router_rows(
        rows=rows,
        task_labels=task_labels,
        target_per_task_label=20,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
    )
    prepare_module._fill_router_tasks_without_llm(
        rows=rows,
        regression_rows=prepare_module._RegressionTaskStore(max_per_task=20),
        task_labels=task_labels,
        regression_tasks=set(),
        target_per_task_label=20,
        single_pools={"router": ["Remember the weekly planning update for Alice."]},
        rng=random.Random(7),
        llm_available=True,
    )

    for task, labels in task_labels.items():
        for label in labels:
            while rows.count(task, label) < 20:
                idx = rows.count(task, label)
                rows.add(
                    task,
                    f"LLM {task} {label} {idx}",
                    label,
                    f"llm:{task}:{label}",
                    extras={"group_id": f"llm:{task}:{label}:{idx}"},
                )

    diagnostics = prepare_module._validate_source_coverage(
        pd.DataFrame(rows.rows),
        split_name="router:train",
    )

    assert diagnostics["consolidation_gist_quality"]["template_ratio"] <= 0.35
    assert diagnostics["forgetting_action_policy"]["template_ratio"] <= 0.35


def test_fill_structured_memory_type_rows_backfills_missing_labels() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=3)

    prepare_module._fill_structured_memory_type_rows(
        rows=rows,
        task_labels={
            "memory_type": ["plan", "knowledge", "reasoning_step", "message", "observation"]
        },
        target_per_task_label=3,
        single_pools={"router": ["Review the release checklist tomorrow."]},
        rng=random.Random(5),
    )

    for label in ("plan", "knowledge", "reasoning_step", "message", "observation"):
        assert rows.count("memory_type", label) == 3
    assert all(str(row["source"]).startswith("structured:memory_type:") for row in rows.rows)


def test_fill_structured_memory_type_rows_scales_past_topic_cycle() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=12)

    prepare_module._fill_structured_memory_type_rows(
        rows=rows,
        task_labels={"memory_type": ["episodic_event"]},
        target_per_task_label=12,
        single_pools={"router": ["Review the release checklist tomorrow."]},
        rng=random.Random(5),
    )

    assert rows.count("memory_type", "episodic_event") == 12
    assert len({str(row["text"]) for row in rows.rows}) == 12


def test_fill_structured_extractor_rows_scales_constraint_type_past_topic_cycle() -> None:
    rows = prepare_module._SingleTaskStore(max_per_task_label=12)

    prepare_module._fill_structured_extractor_rows(
        rows=rows,
        target_per_task_label=12,
        single_pools={"router": ["Review the release checklist tomorrow."]},
        rng=random.Random(5),
    )

    constraint_rows = [
        row
        for row in rows.rows
        if str(row["task"]) == "constraint_type" and str(row["label"]) == "policy"
    ]

    assert len(constraint_rows) == 12
    assert len({str(row["text"]) for row in constraint_rows}) == 12


def test_regression_backfill_advances_past_seeded_duplicate() -> None:
    regression_rows = prepare_module._RegressionTaskStore(max_per_task=4)
    pack = prepare_module._topic_pack(1)
    added = regression_rows.add(
        "write_importance_regression",
        f"Stable preference 1: {pack['relevant']} for future recommendations.",
        0.76,
        "prepared:existing",
        extras=prepare_module._structured_row_extras(
            topic=pack["topic"],
            memory_type="preference",
            importance=0.78,
            confidence=0.92,
            access_count=5,
            age_days=7,
            dependency_count=1,
        ),
    )
    assert added is True

    prepare_module._fill_router_tasks_without_llm(
        rows=prepare_module._SingleTaskStore(max_per_task_label=4),
        regression_rows=regression_rows,
        task_labels={},
        regression_tasks={"write_importance_regression"},
        target_per_task_label=2,
        single_pools={"router": ["Review the release checklist tomorrow."]},
        rng=random.Random(5),
    )

    assert regression_rows.count("write_importance_regression") == 2
    texts = {str(row["text"]) for row in regression_rows.rows}
    assert f"Stable preference 1: {pack['relevant']} for future recommendations." in texts
    assert any(text.startswith("Useful fact 2:") for text in texts)


def test_validate_split_label_coverage_rejects_single_label_slice() -> None:
    split_frames = {
        "train": pd.DataFrame(
            [
                {"task": "confidence_bin", "label": "low", "text": "a"},
                {"task": "confidence_bin", "label": "low", "text": "b"},
            ]
        ),
        "test": pd.DataFrame([{"task": "confidence_bin", "label": "low", "text": "c"}]),
        "eval": pd.DataFrame([{"task": "confidence_bin", "label": "low", "text": "d"}]),
    }

    with pytest.raises(ValueError, match="expected at least 2 labels"):
        prepare_module._validate_split_label_coverage(split_frames)


def test_schema_template_fill_is_capped_and_assigns_group_ids() -> None:
    rows = prepare_module._PairTaskStore(max_per_task_label=20)

    prepare_module._fill_template_schema_rows(rows, target_per_task_label=20)

    assert rows.count("schema_match_pair", "match") == 2
    assert rows.count("schema_match_pair", "no_match") == 2
    assert all(str(row["group_id"]).startswith("template:schema_match_pair:") for row in rows.rows)


def test_load_fever_rows_from_jsonl_extracts_evidence_sentence() -> None:
    path = Path("packages/py-cml/tests/unit/_tmp_fever_rows.jsonl")
    try:
        path.write_text(
            "\n".join(
                [
                    json.dumps(
                        {
                            "id": 1,
                            "label": "SUPPORTS",
                            "claim": "The Eiffel Tower is in Paris.",
                            "evidence": [[[12, 0, "Eiffel_Tower", 0]]],
                        }
                    ),
                    json.dumps(
                        {
                            "id": 2,
                            "label": "NOT ENOUGH INFO",
                            "claim": "An unsupported claim.",
                            "evidence": [],
                        }
                    ),
                ]
            ),
            encoding="utf-8",
        )

        rows = prepare_module._load_fever_rows_from_jsonl(path, limit=10)

        assert rows == [
            {
                "claim": "The Eiffel Tower is in Paris.",
                "label": "SUPPORTS",
                "evidence_sentence": "Eiffel Tower sentence 0",
            }
        ]
    finally:
        path.unlink(missing_ok=True)


def test_validate_source_coverage_rejects_template_dominated_schema_split() -> None:
    df = pd.DataFrame(
        [
            {
                "task": "schema_match_pair",
                "label": "match",
                "text_a": f"claim {idx}",
                "text_b": f"evidence {idx}",
                "source": "template:schema_match_pair:match",
                "group_id": f"template:{idx}",
            }
            for idx in range(8)
        ]
        + [
            {
                "task": "schema_match_pair",
                "label": "no_match",
                "text_a": f"claim other {idx}",
                "text_b": f"evidence other {idx}",
                "source": "template:schema_match_pair:no_match",
                "group_id": f"template-other:{idx}",
            }
            for idx in range(8)
        ]
    )

    with pytest.raises(ValueError, match="template ratio too high"):
        prepare_module._validate_source_coverage(df, split_name="pair:train")


def test_augment_pair_hard_negatives_uses_cached_embeddings(tmp_path: Path) -> None:
    cache_path = tmp_path / "pair_text_embeddings.parquet"
    pd.DataFrame(
        [
            {"text": "query alpha", "embedding": [1.0, 0.0]},
            {"text": "query beta", "embedding": [0.0, 1.0]},
            {"text": "memory near alpha", "embedding": [0.95, 0.05]},
            {"text": "memory near beta", "embedding": [0.05, 0.95]},
            {"text": "memory distractor one", "embedding": [0.2, 0.8]},
            {"text": "memory distractor two", "embedding": [0.8, 0.2]},
        ]
    ).to_parquet(cache_path, index=False)
    df = pd.DataFrame(
        [
            {
                "task": "memory_rerank_pair",
                "label": "relevant",
                "text_a": "query alpha",
                "text_b": "memory near alpha",
                "source": "hf:test",
                "group_id": "g-alpha",
            },
            {
                "task": "memory_rerank_pair",
                "label": "relevant",
                "text_a": "query beta",
                "text_b": "memory near beta",
                "source": "hf:test",
                "group_id": "g-beta",
            },
            {
                "task": "memory_rerank_pair",
                "label": "not_relevant",
                "text_a": "query alpha",
                "text_b": "memory distractor one",
                "source": "hf:test",
                "group_id": "g-alpha",
            },
            {
                "task": "memory_rerank_pair",
                "label": "not_relevant",
                "text_a": "query beta",
                "text_b": "memory distractor two",
                "source": "hf:test",
                "group_id": "g-beta",
            },
        ]
    )

    augmented, _summary = prepare_module._augment_pair_hard_negatives(
        df,
        cache_path=cache_path,
        task_names={"memory_rerank_pair"},
    )

    mined = augmented[augmented["source"].astype(str) == "derived_hard_negative:memory_rerank_pair"]
    task_rows = augmented[augmented["task"].astype(str) == "memory_rerank_pair"]
    ratio = len(mined) / max(1, len(task_rows))
    assert ratio <= 0.6
    assert len(mined) >= 2
    assert set(mined["group_id"].astype(str)) == {"g-alpha", "g-beta"}


def test_apply_router_feature_enrichment_adds_memory_type_columns() -> None:
    df = pd.DataFrame(
        [
            {
                "task": "memory_type",
                "label": "plan",
                "source": "test",
                "text": '{"step": "Review the release plan with Alice tomorrow?"}',
            }
        ]
    )

    enriched = prepare_module._apply_router_feature_enrichment(df)

    for column in MEMORY_TYPE_FEATURE_COLUMNS:
        assert column in enriched.columns

    row = enriched.iloc[0].to_dict()
    assert row["question_mark_count"] == 1
    assert row["has_json_like_shape"] is True
    assert row["temporal_marker_count"] >= 1
    assert row["named_entity_like_count"] >= 1


def test_apply_router_feature_enrichment_defaults_non_memory_type_rows() -> None:
    df = pd.DataFrame(
        [
            {
                "task": "query_domain",
                "label": "work",
                "source": "test",
                "text": "Please review the release schedule tomorrow.",
            }
        ]
    )

    enriched = prepare_module._apply_router_feature_enrichment(df)

    for column in MEMORY_TYPE_FEATURE_COLUMNS:
        assert column in enriched.columns

    row = enriched.iloc[0].to_dict()
    assert row["text_length_chars"] == 0
    assert row["question_mark_count"] == 0
    assert row["has_plan_structure"] is False


def test_should_precompute_router_features_skips_large_memory_type_sets() -> None:
    df = pd.DataFrame(
        [
            {"task": "memory_type", "text": f"Plan row {idx}", "label": "plan", "source": "test"}
            for idx in range(prepare_module._MAX_PRECOMPUTED_ROUTER_FEATURE_ROWS + 1)
        ]
    )

    assert prepare_module._should_precompute_router_features(df) is False


def test_memory_type_feature_tokens_match_text_and_row_derivation() -> None:
    text = '{"step": "Please review this plan with Alice tomorrow?"}'
    row = {"text": text, **derive_memory_type_feature_columns(text)}

    assert derive_memory_type_feature_tokens_from_row(
        row
    ) == derive_memory_type_feature_tokens_from_text(text)


def test_llm_round_strategy_reduces_batch_for_truncation() -> None:
    batch_size, use_multilingual, abort, reason = prepare_module._llm_round_strategy(
        label_batch_size=8,
        max_batch_size=8,
        round_generated=6,
        round_accepted=0,
        round_errors=0,
        attempts_without_progress=2,
        parse_fail_delta=0,
        finish_length_delta=5,
        finish_stop_delta=0,
        use_multilingual=True,
    )

    assert batch_size == 4
    assert use_multilingual is True
    assert abort is False
    assert reason is not None and "truncated outputs" in reason


def test_llm_round_strategy_disables_multilingual_and_aborts_stall() -> None:
    batch_size, use_multilingual, abort, reason = prepare_module._llm_round_strategy(
        label_batch_size=1,
        max_batch_size=4,
        round_generated=0,
        round_accepted=0,
        round_errors=0,
        attempts_without_progress=8,
        parse_fail_delta=3,
        finish_length_delta=0,
        finish_stop_delta=0,
        use_multilingual=True,
    )

    assert batch_size == 1
    assert use_multilingual is False
    assert abort is True
    assert reason is not None and "disabled multilingual" in reason


def test_match_model_metadata_prefers_exact_id() -> None:
    payload = {
        "object": "list",
        "data": [
            {"id": "Other/Model", "owned_by": "vllm"},
            {"id": "Qwen/Qwen3.5-9B", "owned_by": "vllm"},
        ],
    }

    matched = prepare_module._match_model_metadata("Qwen/Qwen3.5-9B", payload)

    assert matched is not None
    assert matched["id"] == "Qwen/Qwen3.5-9B"


def test_thinking_disable_policy_uses_vllm_kwargs_for_qwen() -> None:
    policy = prepare_module._thinking_disable_policy(
        provider="openai_compatible",
        base_url="http://localhost:8001/v1",
        model_name="Qwen/Qwen3.5-9B",
        model_metadata={"id": "Qwen/Qwen3.5-9B", "owned_by": "vllm"},
    )

    assert policy["chat_template_kwargs"] == {
        "enable_thinking": False,
        "thinking": False,
    }
    assert policy["assistant_prefill"] is None


def test_thinking_disable_policy_adds_assistant_prefill_for_gpt_oss() -> None:
    policy = prepare_module._thinking_disable_policy(
        provider="openai_compatible",
        base_url="http://localhost:8001/v1",
        model_name="gpt-oss-20b",
        model_metadata={"id": "gpt-oss-20b", "owned_by": "vllm"},
    )

    assert policy["assistant_prefill"] == "<think></think>\n"


def test_thinking_disable_policy_sets_reasoning_effort_for_openai_gpt_5_1() -> None:
    policy = prepare_module._thinking_disable_policy(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-5.1-mini",
        model_metadata={"id": "gpt-5.1-mini", "owned_by": "openai"},
    )

    assert policy["reasoning_effort"] == "none"
    assert policy["omit_sampling_controls"] is False


def test_thinking_disable_policy_sets_minimal_for_openai_gpt_5() -> None:
    policy = prepare_module._thinking_disable_policy(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-5-mini",
        model_metadata={"id": "gpt-5-mini", "owned_by": "openai"},
    )

    assert policy["reasoning_effort"] == "minimal"
    assert policy["omit_sampling_controls"] is True


def test_thinking_disable_policy_sets_none_for_gemini_flash() -> None:
    policy = prepare_module._thinking_disable_policy(
        provider="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model_name="gemini-2.5-flash",
        model_metadata={"id": "gemini-2.5-flash", "owned_by": "google"},
    )

    assert policy["reasoning_effort"] == "none"


def test_thinking_disable_policy_warns_for_gemini_pro_and_deepseek_reasoner() -> None:
    gemini_policy = prepare_module._thinking_disable_policy(
        provider="openai_compatible",
        base_url="https://generativelanguage.googleapis.com/v1beta/openai",
        model_name="gemini-2.5-pro",
        model_metadata={"id": "gemini-2.5-pro", "owned_by": "google"},
    )
    deepseek_policy = prepare_module._thinking_disable_policy(
        provider="openai_compatible",
        base_url="https://api.deepseek.com/v1",
        model_name="deepseek-reasoner",
        model_metadata={"id": "deepseek-reasoner", "owned_by": "deepseek"},
    )

    assert "does not support fully disabling thinking" in gemini_policy["warning"]
    assert "Use deepseek-chat" in deepseek_policy["warning"]


def test_thinking_disable_policy_warns_for_openai_pro_and_codex_variants() -> None:
    pro_policy = prepare_module._thinking_disable_policy(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-5.2-pro",
        model_metadata={"id": "gpt-5.2-pro", "owned_by": "openai"},
    )
    codex_policy = prepare_module._thinking_disable_policy(
        provider="openai",
        base_url="https://api.openai.com/v1",
        model_name="gpt-5.2-codex",
        model_metadata={"id": "gpt-5.2-codex", "owned_by": "openai"},
    )

    assert "cannot be switched to no-thinking" in pro_policy["warning"]
    assert "Codex-only reasoning variant" in codex_policy["warning"]


def test_looks_like_thinking_output_detects_reasoning_preamble() -> None:
    assert prepare_module._looks_like_thinking_output(
        "Okay, let's tackle this problem. The user wants me to generate JSON."
    )
    assert not prepare_module._looks_like_thinking_output('{"samples":[{"text":"done"}]}')


# ---------- Multilingual feature tests ----------


def test_memory_type_features_detect_chinese_temporal_markers() -> None:
    columns = derive_memory_type_feature_columns("请明天上午检查发布清单")
    assert columns["temporal_marker_count"] >= 1
    assert columns["has_imperative_hint"] is True


def test_memory_type_features_detect_japanese_plan_structure() -> None:
    text = "まず設計を確認し、次にコードを書き、最後にテストする予定"
    columns = derive_memory_type_feature_columns(text)
    assert columns["has_plan_structure"] is True


def test_memory_type_features_detect_spanish_first_person() -> None:
    columns = derive_memory_type_feature_columns("yo prefiero usar Python para automatización")
    assert columns["has_first_person_pronoun"] is True


def test_memory_type_features_detect_russian_imperative() -> None:
    columns = derive_memory_type_feature_columns("запомни этот номер телефона")
    assert columns["has_imperative_hint"] is True


def test_memory_type_features_detect_german_temporal() -> None:
    columns = derive_memory_type_feature_columns("gestern habe ich den Code aktualisiert")
    assert columns["temporal_marker_count"] >= 1


def test_memory_type_features_detect_arabic_plan_structure() -> None:
    text = "سوف نبدأ أولاً بالتصميم ثم ننتقل إلى التنفيذ"
    columns = derive_memory_type_feature_columns(text)
    assert columns["has_plan_structure"] is True


def test_memory_type_features_detect_korean_first_person() -> None:
    columns = derive_memory_type_feature_columns("나는 파이썬 도구를 선호합니다")
    assert columns["has_first_person_pronoun"] is True


def test_memory_type_features_detect_hindi_imperative() -> None:
    columns = derive_memory_type_feature_columns("कृपया इस डेटा को अपडेट करें")
    assert columns["has_imperative_hint"] is True


def test_semantic_hint_tokens_detect_multilingual_keywords() -> None:
    tokens_zh = derive_memory_type_feature_tokens_from_text("我认为也许可以用这个方案")
    assert "hint=analytical" in tokens_zh
    assert "hint=first_person" in tokens_zh

    tokens_fr = derive_memory_type_feature_tokens_from_text(
        "je pense qu'il faut planifier la semaine"
    )
    assert "hint=analytical" in tokens_fr
    assert "hint=planning" in tokens_fr
    assert "hint=time_anchored" in tokens_fr


def test_pair_lexical_features_tokenize_chinese() -> None:
    feats = build_pair_lexical_features("航班报销截止日期", "航班报销截止日期")
    assert feats[0] > 0.5  # token overlap should be high for identical text


def test_pair_lexical_features_tokenize_arabic() -> None:
    feats = build_pair_lexical_features(
        "الموعد النهائي لاسترداد تذكرة الطيران",
        "طلب ترقية المقعد",
    )
    assert feats[0] < 0.5  # different texts, low overlap


def test_pair_lexical_features_tokenize_korean() -> None:
    feats = build_pair_lexical_features("항공권 환불 기한", "항공권 환불 기한")
    assert feats[0] > 0.5  # identical text, high overlap


def test_pair_lexical_features_negation_detection_multilingual() -> None:
    feats_zh = build_pair_lexical_features("没有问题", "有问题")
    assert feats_zh[4] == 1.0  # negation mismatch (没 in one, not the other)

    feats_ru = build_pair_lexical_features("не готово", "готово")
    assert feats_ru[4] == 1.0  # не is negation
