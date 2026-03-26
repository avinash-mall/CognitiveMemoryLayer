"""Tests for release gate enforcement, per-epoch eval, and dataset usage_role handling."""

from __future__ import annotations

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")

import cml.modeling.prepare as prepare_module
import cml.modeling.train as train_module

# ── Release gate enforcement ──────────────────────────────────────────────────


def test_release_gate_passes_when_above_min_threshold() -> None:
    result = train_module._release_gate_results(
        "forgetting_action_policy",
        {
            "test": {
                "macro_f1": 0.94,
                "classification_report": {
                    "forgetting_action_policy::decay": {"recall": 0.91},
                    "forgetting_action_policy::delete": {"recall": 0.95},
                },
            }
        },
    )
    assert result["passed"] is True
    assert all(c["passed"] for c in result["checks"])


def test_release_gate_fails_when_below_min_threshold() -> None:
    result = train_module._release_gate_results(
        "forgetting_action_policy",
        {
            "test": {
                "macro_f1": 0.94,
                "classification_report": {
                    "forgetting_action_policy::decay": {"recall": 0.73},  # below 0.90
                    "forgetting_action_policy::delete": {"recall": 0.95},
                },
            }
        },
    )
    assert result["passed"] is False
    failed = [c for c in result["checks"] if not c["passed"]]
    assert any(c["metric"] == "decay_recall" for c in failed)


def test_release_gate_passes_when_calibration_error_below_max() -> None:
    result = train_module._release_gate_results(
        "constraint_dimension",
        {"test": {"macro_f1": 0.89, "calibration_error": 0.055}},
    )
    assert result["passed"] is True
    cal_check = next(c for c in result["checks"] if c["metric"] == "calibration_error")
    assert cal_check["passed"] is True


def test_release_gate_fails_when_calibration_error_above_max() -> None:
    result = train_module._release_gate_results(
        "constraint_dimension",
        {"test": {"macro_f1": 0.89, "calibration_error": 0.083}},
    )
    assert result["passed"] is False
    cal_check = next(c for c in result["checks"] if c["metric"] == "calibration_error")
    assert cal_check["passed"] is False
    assert cal_check["actual"] == pytest.approx(0.083)


def test_release_gate_fails_when_metric_not_found() -> None:
    result = train_module._release_gate_results(
        "schema_match_pair",
        {"test": {}},  # macro_f1 and calibration_error missing
    )
    assert result["passed"] is False
    assert all(c["actual"] is None for c in result["checks"])


def test_release_gate_returns_passed_for_unknown_task() -> None:
    result = train_module._release_gate_results("unknown_task", {"test": {"accuracy": 0.5}})
    assert result["passed"] is True
    assert result["checks"] == []


def test_release_gate_write_importance_regression_max_gate() -> None:
    result = train_module._release_gate_results(
        "write_importance_regression",
        {"test": {"test_mae": 0.05}},
    )
    assert result["passed"] is True

    result_fail = train_module._release_gate_results(
        "write_importance_regression",
        {"test": {"test_mae": 0.12}},  # above max 0.10
    )
    assert result_fail["passed"] is False


def test_release_gate_memory_type_macro_f1() -> None:
    result = train_module._release_gate_results(
        "memory_type",
        {"test": {"macro_f1": 0.87}},
    )
    # plan_f1 is missing → gate fails
    assert result["passed"] is False
    assert any(c["metric"] == "plan_f1" and c["actual"] is None for c in result["checks"])


def test_release_gate_memory_type_plan_f1_from_report() -> None:
    result = train_module._release_gate_results(
        "memory_type",
        {
            "test": {
                "macro_f1": 0.87,
                "classification_report": {
                    "memory_type::plan": {"f1-score": 0.80},
                },
            }
        },
    )
    plan_check = next(c for c in result["checks"] if c["metric"] == "plan_f1")
    assert plan_check["passed"] is True
    assert plan_check["actual"] == pytest.approx(0.80)


# ── Per-epoch eval tracking ───────────────────────────────────────────────────


def test_train_classifier_with_monitoring_records_eval_macro_f1_per_epoch() -> None:
    """Epoch stats must contain eval_macro_f1 when validation data is provided."""
    _model, epoch_stats, _summary = train_module._train_classifier_with_monitoring(
        train_x=["apple fruit", "banana fruit", "cat animal", "dog animal"],
        train_y=["router::fruit", "router::fruit", "router::animal", "router::animal"],
        valid_x=["kiwi fruit", "mouse animal"],
        valid_y=["router::fruit", "router::animal"],
        train_cfg={
            "alpha": 1e-3,
            "seed": 0,
            "max_iter": 3,
            "max_features": 64,
            "min_df": 1,
        },
        run_name="test_epoch_tracking",
    )

    assert len(epoch_stats) == 3
    for stat in epoch_stats:
        assert "epoch" in stat
        assert "train_loss" in stat
        assert "valid_macro_f1" in stat, (
            f"epoch {stat['epoch']} missing valid_macro_f1; got keys: {list(stat.keys())}"
        )


# ── Dataset usage_role / task_targets parsing ─────────────────────────────────


def _make_registry(datasets: list[dict]) -> prepare_module._HFRegistry:
    return prepare_module._HFRegistry(
        datasets_cfg=datasets,
        cache_dir=None,
        prepare_cfg={"max_rows_per_source": 1000, "require_datasets_package": False},
    )


def test_hf_registry_parses_usage_role_and_task_targets() -> None:
    reg = _make_registry(
        [
            {
                "name": "ds_supervision",
                "enabled": True,
                "usage_role": "supervision",
                "task_targets": ["memory_type", "context_tag"],
                "target": "",
            },
            {
                "name": "ds_eval_only",
                "enabled": True,
                "usage_role": "eval_only",
                "task_targets": ["schema_match_pair"],
                "target": "",
            },
        ]
    )
    sup = reg.status["ds_supervision"]
    assert sup["usage_role"] == "supervision"
    assert sup["task_targets"] == ["memory_type", "context_tag"]

    eo = reg.status["ds_eval_only"]
    assert eo["usage_role"] == "eval_only"
    assert eo["task_targets"] == ["schema_match_pair"]


def test_hf_registry_eval_only_excluded_from_seed_pool() -> None:
    """Datasets with usage_role='eval_only' must not appear in _SEED_POOL_USAGE_ROLES."""
    assert "eval_only" not in prepare_module._SEED_POOL_USAGE_ROLES


def test_hf_registry_supervision_and_llm_seed_included_in_seed_pool() -> None:
    assert "supervision" in prepare_module._SEED_POOL_USAGE_ROLES
    assert "llm_seed" in prepare_module._SEED_POOL_USAGE_ROLES


def test_hf_registry_empty_usage_role_treated_as_default_included() -> None:
    """Datasets with no usage_role (empty string) default to inclusion in seed pool."""
    assert "" in prepare_module._SEED_POOL_USAGE_ROLES


def test_hf_registry_disabled_datasets_not_in_status() -> None:
    reg = _make_registry(
        [
            {"name": "active_ds", "enabled": True, "usage_role": "supervision", "target": ""},
            {"name": "disabled_ds", "enabled": False, "usage_role": "supervision", "target": ""},
        ]
    )
    assert "active_ds" in reg.status
    assert "disabled_ds" not in reg.status


def test_hf_registry_task_targets_strips_whitespace() -> None:
    reg = _make_registry(
        [
            {
                "name": "padded_ds",
                "enabled": True,
                "usage_role": "supervision",
                "task_targets": ["  memory_type  ", " context_tag"],
                "target": "",
            }
        ]
    )
    assert reg.status["padded_ds"]["task_targets"] == ["memory_type", "context_tag"]


def test_hf_registry_task_targets_skips_empty_strings() -> None:
    reg = _make_registry(
        [
            {
                "name": "sparse_ds",
                "enabled": True,
                "usage_role": "supervision",
                "task_targets": ["memory_type", "", "  "],
                "target": "",
            }
        ]
    )
    assert reg.status["sparse_ds"]["task_targets"] == ["memory_type"]


# ── Compound metadata token tests ─────────────────────────────────────────────


def _fap_record(access_count, age_days):
    """Minimal record stub with FAP-relevant fields."""

    class _Row:
        pass

    r = _Row()
    r.access_count = access_count
    r.age_days = age_days
    r.support_count = None
    r.dependency_count = None
    r.namespace = None
    r.mixed_topic = None
    return r


_FAP_COLS = {"access_count", "age_days", "support_count", "dependency_count"}


def test_fap_keep_signal_fires_for_high_access_low_age() -> None:
    rec = _fap_record(access_count=7, age_days=5)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_keep_signal=yes" in tokens


def test_fap_keep_signal_does_not_fire_for_medium_access() -> None:
    rec = _fap_record(access_count=3, age_days=5)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_keep_signal=yes" not in tokens


def test_fap_decay_signal_fires_for_medium_access_medium_age() -> None:
    rec = _fap_record(access_count=3, age_days=48)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_decay_signal=yes" in tokens


def test_fap_decay_signal_does_not_fire_for_compress_profile() -> None:
    """compress: access=medium, age=high (≥90) — should NOT trigger decay signal."""
    rec = _fap_record(access_count=2, age_days=104)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_decay_signal=yes" not in tokens


def test_fap_decay_signal_does_not_fire_for_keep_profile() -> None:
    """keep: access=high (≥6) — should NOT trigger decay signal."""
    rec = _fap_record(access_count=7, age_days=5)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_decay_signal=yes" not in tokens


def test_fap_decay_signal_does_not_fire_for_silence_profile() -> None:
    """silence: access=low (1, <2) — should NOT trigger decay signal."""
    rec = _fap_record(access_count=1, age_days=132)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_decay_signal=yes" not in tokens


def test_fap_decay_signal_does_not_fire_for_delete_profile() -> None:
    """delete: access=0 (none/low) — should NOT trigger decay signal."""
    rec = _fap_record(access_count=0, age_days=366)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_decay_signal=yes" not in tokens


def test_fap_decay_signal_does_not_fire_for_none_access() -> None:
    """LLM rows have access_count=None — should NOT trigger decay signal."""
    rec = _fap_record(access_count=None, age_days=60)
    tokens = train_module._single_metadata_tokens(
        rec, _FAP_COLS, task_name="forgetting_action_policy"
    )
    assert "fap_decay_signal=yes" not in tokens


def test_fap_decay_signal_does_not_fire_for_other_task() -> None:
    """Non-FAP tasks must never emit FAP compound tokens."""
    rec = _fap_record(access_count=3, age_days=48)
    tokens = train_module._single_metadata_tokens(rec, _FAP_COLS, task_name="memory_type")
    assert "fap_decay_signal=yes" not in tokens
    assert "fap_keep_signal=yes" not in tokens


def test_fap_hardened_decay_profile_support_count_never_high() -> None:
    """Hardened decay training rows must never have support_count=high (≥4).
    The previous profile used support_count=3 + idx%2, giving 4 (high) for odd-idx rows.
    This caused odd-idx decay test rows to be confused with keep (which also has support=high).
    New profile uses support_count=2 + idx%2 = 2 or 3 (both medium, threshold=4).
    """
    from cml.modeling.train import _build_forgetting_policy_hardened_rows, _count_bucket

    hardened = _build_forgetting_policy_hardened_rows()
    decay_rows = hardened[hardened["label"] == "decay"]
    for _, row in decay_rows.iterrows():
        bucket = _count_bucket(int(row["support_count"]), high=4, medium=2)
        assert bucket != "high", (
            f"Decay hardened row has support_count=high ({row['support_count']}). "
            "This leaks 'keep'-like features into decay training rows."
        )
