from __future__ import annotations

from cml.modeling import train as train_mod


def test_release_gate_results_supports_max_calibration_and_label_recall() -> None:
    summary = {
        "test": {
            "macro_f1": 0.95,
            "calibration_error": 0.04,
            "classification_report": {
                "forgetting_action_policy::decay": {"recall": 0.93},
                "forgetting_action_policy::delete": {"recall": 0.91},
            },
        }
    }

    result = train_mod._release_gate_results("forgetting_action_policy", summary)

    assert result["passed"] is True


def test_optimize_binary_threshold_respects_precision_floor() -> None:
    result = train_mod._optimize_binary_threshold(
        [0.95, 0.82, 0.61, 0.24, 0.18],
        [
            "schema_match_pair::match",
            "schema_match_pair::match",
            "schema_match_pair::no_match",
            "schema_match_pair::no_match",
            "schema_match_pair::no_match",
        ],
        positive_label="schema_match_pair::match",
        precision_floor=0.85,
    )

    assert 0.2 <= float(result["default_threshold"]) <= 0.95
    assert result["positive_label"] == "schema_match_pair::match"
