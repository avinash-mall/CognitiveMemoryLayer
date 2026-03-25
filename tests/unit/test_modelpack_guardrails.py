from __future__ import annotations

import numpy as np

from src.utils.modelpack import ModelPackRuntime


class _FixedModel:
    def __init__(self, classes: list[str], probs: list[float], pred: str) -> None:
        self.classes_ = classes
        self._probs = np.asarray([probs], dtype=float)
        self._pred = pred

    def predict(self, features):
        return np.asarray([self._pred for _ in features], dtype=object)

    def predict_proba(self, features):
        return np.repeat(self._probs, repeats=len(features), axis=0)


def _runtime_with_tasks(task_models: dict[str, object]) -> ModelPackRuntime:
    runtime = ModelPackRuntime(models_dir=None)
    runtime._loaded = True
    runtime._task_models = dict(task_models)
    runtime._models = {}
    return runtime


def _runtime_with_family(family: str, model: object) -> ModelPackRuntime:
    runtime = ModelPackRuntime(models_dir=None)
    runtime._loaded = True
    runtime._task_models = {}
    runtime._models = {family: model}
    return runtime


def test_gist_guardrail_rejects_low_confidence_accept() -> None:
    runtime = _runtime_with_tasks(
        {
            "consolidation_gist_quality": _FixedModel(
                classes=[
                    "consolidation_gist_quality::reject",
                    "consolidation_gist_quality::accept",
                ],
                probs=[0.26, 0.74],
                pred="consolidation_gist_quality::accept",
            )
        }
    )

    pred = runtime.predict_single("consolidation_gist_quality", "Digest this cluster.")

    assert pred is not None
    assert pred.label == "reject"


def test_forgetting_guardrail_blocks_low_confidence_compress() -> None:
    runtime = _runtime_with_tasks(
        {
            "forgetting_action_policy": _FixedModel(
                classes=[
                    "forgetting_action_policy::keep",
                    "forgetting_action_policy::compress",
                ],
                probs=[0.30, 0.70],
                pred="forgetting_action_policy::compress",
            )
        }
    )

    pred = runtime.predict_single(
        "forgetting_action_policy",
        "Old memory text.",
        metadata={"importance": 0.2, "access_count": 0, "age_days": 120, "dependency_count": 0},
    )

    assert pred is not None
    assert pred.label == "keep"


def test_forgetting_guardrail_requires_gist_accept_for_compress() -> None:
    runtime = _runtime_with_tasks(
        {
            "forgetting_action_policy": _FixedModel(
                classes=[
                    "forgetting_action_policy::keep",
                    "forgetting_action_policy::compress",
                ],
                probs=[0.08, 0.92],
                pred="forgetting_action_policy::compress",
            ),
            "consolidation_gist_quality": _FixedModel(
                classes=[
                    "consolidation_gist_quality::reject",
                    "consolidation_gist_quality::accept",
                ],
                probs=[0.55, 0.45],
                pred="consolidation_gist_quality::reject",
            ),
        }
    )

    pred = runtime.predict_single(
        "forgetting_action_policy",
        "Compress candidate text.",
        metadata={"importance": 0.1, "access_count": 0, "age_days": 240, "dependency_count": 0},
    )

    assert pred is not None
    assert pred.label == "keep"


def test_dedicated_only_task_does_not_fallback_to_family_model() -> None:
    runtime = _runtime_with_family(
        "router",
        _FixedModel(
            classes=["memory_type::semantic_fact", "memory_type::preference"],
            probs=[0.9, 0.1],
            pred="memory_type::semantic_fact",
        ),
    )

    pred = runtime.predict_single("memory_type", "User likes tea.")

    assert pred is None
