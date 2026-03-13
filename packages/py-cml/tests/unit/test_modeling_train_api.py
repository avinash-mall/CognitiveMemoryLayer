"""Unit tests for modeling train API wrappers."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

pytest.importorskip("pandas")
pytest.importorskip("sklearn")
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import cml.modeling.train as train_module
from cml.modeling.memory_type_features import derive_memory_type_feature_columns
from cml.modeling.pair_features import build_pair_dense_features
from cml.modeling.runtime_models import (
    CumulativeOrdinalClassifier,
    EmbeddingPairClassifier,
    HierarchicalTextClassifier,
    TaskConditionalCalibratedClassifier,
)
from cml.modeling.train import TaskSpec
from cml.modeling.types import TrainConfig


def test_train_models_builds_expected_argv(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(train_module, "main", _fake_main)
    cfg = TrainConfig(
        config_path=tmp_path / "model_pipeline.toml",
        families="router,pair",
        max_iter=20,
        tasks="novelty_pair",
        objective_types="classification",
        export_thresholds=True,
    )
    rc = train_module.train_models(cfg)
    assert rc == 0
    assert "--families" in captured["argv"]
    assert "router,pair" in captured["argv"]
    assert "--export-thresholds" in captured["argv"]
    assert "--strict" in captured["argv"]


def test_train_models_allow_skips(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(train_module, "main", _fake_main)
    cfg = TrainConfig(
        config_path=tmp_path / "model_pipeline.toml",
        strict=False,
    )
    rc = train_module.train_models(cfg)
    assert rc == 0
    assert "--allow-skips" in captured["argv"]


def test_train_models_forwards_early_stopping_and_calibration(monkeypatch, tmp_path: Path) -> None:
    captured: dict[str, list[str]] = {}

    def _fake_main(argv=None):
        captured["argv"] = list(argv or [])
        return 0

    monkeypatch.setattr(train_module, "main", _fake_main)
    cfg = TrainConfig(
        config_path=tmp_path / "model_pipeline.toml",
        early_stopping=True,
        early_stopping_patience=4,
        early_stopping_metric="macro_f1",
        early_stopping_min_delta=0.01,
        calibration_method="sigmoid",
        calibration_split="eval",
    )

    rc = train_module.train_models(cfg)

    assert rc == 0
    assert "--early-stopping" in captured["argv"]
    assert "true" in captured["argv"]
    assert "--early-stopping-patience" in captured["argv"]
    assert "4" in captured["argv"]
    assert "--early-stopping-metric" in captured["argv"]
    assert "--early-stopping-min-delta" in captured["argv"]
    assert "--calibration-method" in captured["argv"]
    assert "--calibration-split" in captured["argv"]


def test_train_task_dispatches_on_trainer(monkeypatch, tmp_path: Path) -> None:
    called: dict[str, str] = {}

    def _fake_embedding_pair(spec, *, prepared_dir, output_dir, train_cfg):
        called["trainer"] = spec.trainer
        return {"task": spec.task_name, "model_path": str(output_dir / "dummy.joblib")}

    monkeypatch.setattr(train_module, "_train_embedding_pair", _fake_embedding_pair)

    summary = train_module._train_task(
        TaskSpec(
            task_name="memory_rerank_pair",
            family="pair",
            input_type="pair",
            objective="pair_ranking",
            labels=[],
            artifact_name="memory_rerank_pair",
            metrics=["mrr@10"],
            trainer="embedding_pair",
        ),
        prepared_dir=tmp_path / "prepared",
        output_dir=tmp_path / "trained",
        train_cfg={"seed": 42},
        strict=True,
    )

    assert called["trainer"] == "embedding_pair"
    assert summary["task"] == "memory_rerank_pair"


def test_maybe_calibrate_matrix_classifier_returns_summary() -> None:
    np = pytest.importorskip("numpy")

    x = np.asarray(
        [
            [0.0, 0.1],
            [0.1, 0.0],
            [0.2, 0.2],
            [1.0, 1.1],
            [1.1, 1.0],
            [1.2, 1.2],
        ],
        dtype=float,
    )
    y = ["demo::low", "demo::low", "demo::low", "demo::high", "demo::high", "demo::high"]
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=200))
    model.fit(x, y)

    calibrated, summary = train_module._maybe_calibrate_matrix_classifier(
        model,
        calibration_x=x,
        calibration_y=y,
        train_cfg={"calibration_method": "sigmoid", "calibration_split": "eval"},
    )

    assert summary is not None
    assert summary["method"] == "sigmoid"
    assert summary["split"] == "eval"
    assert summary["rows"] == len(y)
    assert calibrated is not model


def test_train_classifier_with_monitoring_uses_explicit_balanced_weights() -> None:
    model, epoch_stats, summary = train_module._train_classifier_with_monitoring(
        train_x=[
            "alpha shared context",
            "alpha repeated context",
            "alpha dominant class",
            "omega minority class",
        ],
        train_y=[
            "demo::major",
            "demo::major",
            "demo::major",
            "demo::minor",
        ],
        valid_x=[],
        valid_y=[],
        train_cfg={
            "alpha": 1e-4,
            "seed": 7,
            "max_iter": 2,
            "max_features": 256,
            "min_df": 1,
        },
        run_name="demo",
    )

    classifier = model.named_steps["classifier"]

    assert len(epoch_stats) == 2
    assert summary["actual_epochs"] == 2
    assert classifier.class_weight == pytest.approx(
        {
            "demo::major": 4.0 / 6.0,
            "demo::minor": 2.0,
        }
    )


def test_write_task_metrics_includes_top_level_calibration_block(tmp_path: Path) -> None:
    calibration = {
        "method": "sigmoid",
        "split": "eval",
        "rows": 12,
        "pre_ece": 0.21,
        "post_ece": 0.08,
        "pre_accuracy": 0.7,
        "post_accuracy": 0.75,
        "accuracy_delta": 0.05,
    }

    train_module._write_task_metrics(
        tmp_path,
        "demo_task",
        metrics_test={"task": "demo_task", "overall": {"accuracy": 0.8}},
        metrics_eval={"task": "demo_task", "overall": {"accuracy": 0.78}},
        calibration=calibration,
    )

    payload = json.loads((tmp_path / "demo_task_metrics_test.json").read_text(encoding="utf-8"))
    assert payload["calibration"] == calibration


def test_train_single_regression_writes_eval_metrics_artifact(tmp_path: Path) -> None:
    prepared_dir = tmp_path / "prepared"
    trained_dir = tmp_path / "trained"
    prepared_dir.mkdir()
    trained_dir.mkdir()

    train_rows = pd.DataFrame(
        [
            {"task": "write_importance_regression", "text": f"train row {idx}", "score": float(idx) / 10.0}
            for idx in range(8)
        ]
    )
    test_rows = pd.DataFrame(
        [
            {"task": "write_importance_regression", "text": "test row 1", "score": 0.15},
            {"task": "write_importance_regression", "text": "test row 2", "score": 0.75},
        ]
    )
    eval_rows = pd.DataFrame(
        [
            {"task": "write_importance_regression", "text": "eval row 1", "score": 0.25},
            {"task": "write_importance_regression", "text": "eval row 2", "score": 0.85},
        ]
    )

    train_rows.to_parquet(prepared_dir / "router_train.parquet", index=False)
    test_rows.to_parquet(prepared_dir / "router_test.parquet", index=False)
    eval_rows.to_parquet(prepared_dir / "router_eval.parquet", index=False)

    summary = train_module._train_single_regression(
        TaskSpec(
            task_name="write_importance_regression",
            family="router",
            input_type="single",
            objective="single_regression",
            labels=[],
            artifact_name="write_importance_regression",
            metrics=["mae", "rmse"],
        ),
        prepared_dir=prepared_dir,
        output_dir=trained_dir,
        train_cfg={
            "alpha": 1e-4,
            "seed": 7,
            "max_iter": 2,
            "max_features": 128,
            "min_df": 1,
        },
    )

    assert (trained_dir / "write_importance_regression_epoch_stats.json").exists()
    assert (trained_dir / "write_importance_regression_metrics_test.json").exists()
    assert (trained_dir / "write_importance_regression_metrics_eval.json").exists()
    assert "test" in summary
    assert "eval" in summary

    payload = json.loads(
        (trained_dir / "write_importance_regression_metrics_eval.json").read_text(encoding="utf-8")
    )
    assert payload["metrics"]["eval_mae"] >= 0.0
    assert payload["metrics"]["eval_rmse"] >= 0.0


def test_encode_memory_type_features_matches_prepared_columns_and_fallback() -> None:
    text = '{"step": "Please review this plan with Alice tomorrow?"}'
    prepared_df = pd.DataFrame(
        [
            {
                "task": "memory_type",
                "label": "plan",
                "text": text,
                **derive_memory_type_feature_columns(text),
            }
        ]
    )
    fallback_df = pd.DataFrame([{"task": "memory_type", "label": "plan", "text": text}])

    assert train_module._encode_memory_type_features(prepared_df, "router") == train_module._encode_memory_type_features(
        fallback_df,
        "router",
    )


class _MacroRuntimeModel:
    classes_ = ["personal"]

    def __init__(self) -> None:
        self.last_features: list[str] = []

    def predict_proba(self, features):
        self.last_features = [str(feature) for feature in features]
        return [[1.0] for _ in features]

    def predict(self, features):
        self.last_features = [str(feature) for feature in features]
        return ["personal" for _ in features]


class _FineRuntimeModel:
    classes_ = ["preference", "episodic_event"]

    def __init__(self) -> None:
        self.last_features: list[str] = []

    def predict_proba(self, features):
        self.last_features = [str(feature) for feature in features]
        return [[0.9, 0.1] for _ in features]

    def predict(self, features):
        self.last_features = [str(feature) for feature in features]
        return ["preference" for _ in features]


def test_hierarchical_runtime_enriches_memory_type_features() -> None:
    stage1 = _MacroRuntimeModel()
    stage2 = _FineRuntimeModel()
    model = HierarchicalTextClassifier(
        task_name="memory_type",
        stage1_model=stage1,
        stage2_models={"personal": stage2},
        macro_to_labels={"personal": ["preference", "episodic_event"]},
        classes_=["memory_type::preference", "memory_type::episodic_event"],
    )

    pred = model.predict(['task=memory_type [text] {"step": "Please review this plan with Alice tomorrow?"}'])

    assert pred.tolist() == ["memory_type::preference"]
    assert "mt_json_like=true" in stage1.last_features[0]
    assert "hint=time_anchored" in stage1.last_features[0]
    assert "mt_json_like=true" in stage2.last_features[0]


class _FakeEncoder:
    def __init__(self, vectors):
        self.vectors = vectors

    def encode(self, texts, **_kwargs):
        np = pytest.importorskip("numpy")
        return np.asarray([self.vectors[text] for text in texts], dtype=np.float32)


class _CaptureDenseClassifier:
    classes_ = ["memory_rerank_pair::not_relevant", "memory_rerank_pair::relevant"]

    def __init__(self) -> None:
        self.last_dense = None

    def predict_proba(self, dense):
        np = pytest.importorskip("numpy")
        self.last_dense = np.asarray(dense, dtype=float)
        return np.asarray([[0.25, 0.75] for _ in range(len(dense))], dtype=float)


def test_embedding_pair_runtime_dense_features_match_train_time(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    vectors = {
        "query alpha 42": np.asarray([1.0, 0.0], dtype=np.float32),
        "memory beta not 42": np.asarray([0.6, 0.8], dtype=np.float32),
    }
    classifier = _CaptureDenseClassifier()
    model = EmbeddingPairClassifier(
        task_name="memory_rerank_pair",
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        classifier=classifier,
        classes_=classifier.classes_,
    )
    monkeypatch.setattr(model, "_ensure_encoder", lambda: _FakeEncoder(vectors))

    probs = model.predict_proba(["task=memory_rerank_pair [a] query alpha 42 [b] memory beta not 42"])
    expected = build_pair_dense_features(
        vectors["query alpha 42"],
        vectors["memory beta not 42"],
        text_a="query alpha 42",
        text_b="memory beta not 42",
    )

    assert probs.shape == (1, 2)
    assert classifier.last_dense is not None
    assert np.allclose(classifier.last_dense[0], expected)


class _FixedDenseClassifier:
    classes_ = ["memory_rerank_pair::not_relevant", "memory_rerank_pair::relevant"]

    def __init__(self, probs):
        self._probs = probs

    def predict_proba(self, _dense):
        return self._probs


def test_embedding_pair_runtime_predict_uses_group_top1_for_ranking_batches(monkeypatch) -> None:
    np = pytest.importorskip("numpy")
    vectors = {
        "query alpha": np.asarray([1.0, 0.0], dtype=np.float32),
        "memory best": np.asarray([0.9, 0.1], dtype=np.float32),
        "memory other": np.asarray([0.4, 0.6], dtype=np.float32),
        "query beta": np.asarray([0.0, 1.0], dtype=np.float32),
        "memory solo": np.asarray([0.0, 1.0], dtype=np.float32),
    }
    classifier = _FixedDenseClassifier(
        np.asarray(
            [
                [0.45, 0.55],
                [0.6, 0.4],
                [0.3, 0.7],
            ],
            dtype=float,
        )
    )
    model = EmbeddingPairClassifier(
        task_name="memory_rerank_pair",
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        classifier=classifier,
        classes_=classifier.classes_,
    )
    monkeypatch.setattr(model, "_ensure_encoder", lambda: _FakeEncoder(vectors))

    preds = model.predict(
        [
            "task=memory_rerank_pair [a] query alpha [b] memory best",
            "task=memory_rerank_pair [a] query alpha [b] memory other",
            "task=memory_rerank_pair [a] query beta [b] memory solo",
        ]
    )

    assert preds.tolist() == [
        "memory_rerank_pair::relevant",
        "memory_rerank_pair::not_relevant",
        "memory_rerank_pair::relevant",
    ]


def test_encode_features_omits_direct_target_metadata_for_ordinal_tasks() -> None:
    df = pd.DataFrame(
        [
            {
                "task": "importance_bin",
                "label": "high",
                "text": "Remember the policy update.",
                "memory_type": "constraint",
                "importance": 0.92,
                "confidence": 0.81,
                "access_count": 5,
                "age_days": 6,
                "dependency_count": 2,
                "support_count": 4,
                "mixed_topic": False,
                "context_tags": ["work"],
            },
            {
                "task": "confidence_bin",
                "label": "low",
                "text": "Remember the uncertain observation.",
                "memory_type": "observation",
                "importance": 0.41,
                "confidence": 0.18,
                "access_count": 1,
                "age_days": 44,
                "dependency_count": 0,
                "support_count": 1,
                "mixed_topic": True,
                "context_tags": ["travel"],
            },
            {
                "task": "salience_bin",
                "label": "medium",
                "text": "Remember the moderately active thread.",
                "memory_type": "semantic_fact",
                "importance": 0.63,
                "confidence": 0.72,
                "access_count": 3,
                "age_days": 18,
                "dependency_count": 2,
                "support_count": 3,
                "mixed_topic": False,
                "context_tags": ["health"],
            },
        ]
    )

    features = train_module._encode_features(df, "router")

    assert "importance_bin=" not in features[0]
    assert "confidence_bin=" not in features[0]
    assert "confidence_bin=" not in features[1]
    assert "importance_bin=" not in features[1]
    assert "access_count=" in features[0]
    assert "memory_type=" not in features[2]
    assert "importance_bin=" not in features[2]
    assert "confidence_bin=" not in features[2]
    assert "access_count=" in features[2]


class _IdentityVectorizer:
    def transform(self, features):
        return list(features)


class _BoundaryModel:
    def __init__(self, probs):
        self._probs = probs

    def predict_proba(self, _matrix):
        return self._probs


def test_cumulative_ordinal_classifier_enforces_monotonic_boundaries() -> None:
    np = pytest.importorskip("numpy")
    model = CumulativeOrdinalClassifier(
        task_name="importance_bin",
        vectorizer=_IdentityVectorizer(),
        boundary_models=[
            _BoundaryModel(np.asarray([[0.2, 0.8], [0.3, 0.7]], dtype=float)),
            _BoundaryModel(np.asarray([[0.1, 0.9], [0.7, 0.3]], dtype=float)),
        ],
        label_order=["low", "medium", "high"],
        classes_=["importance_bin::low", "importance_bin::medium", "importance_bin::high"],
    )

    boundary_probs = model._boundary_probs(["row-1", "row-2"])
    probs = model.predict_proba(["row-1", "row-2"])

    assert np.all(boundary_probs[:, 1] <= boundary_probs[:, 0])
    assert np.allclose(probs.sum(axis=1), 1.0)
    assert model.predict(["row-1", "row-2"]).tolist() == [
        "importance_bin::high",
        "importance_bin::medium",
    ]


class _FamilyBaseModel:
    classes_ = [
        "alpha::low",
        "alpha::high",
        "beta::left",
        "beta::right",
    ]

    def predict_proba(self, _features):
        np = pytest.importorskip("numpy")
        return np.asarray(
            [
                [0.45, 0.45, 0.05, 0.05],
                [0.05, 0.05, 0.45, 0.45],
            ],
            dtype=float,
        )


class _PassThroughCalibrator:
    classes_ = [0, 1]

    def predict_proba(self, features):
        np = pytest.importorskip("numpy")
        base = np.exp(np.asarray(features, dtype=float))
        return base / base.sum(axis=1, keepdims=True)


def test_task_conditional_calibration_keeps_probability_mass_within_task() -> None:
    np = pytest.importorskip("numpy")
    wrapped = TaskConditionalCalibratedClassifier(
        base_model=_FamilyBaseModel(),
        classes_=_FamilyBaseModel.classes_,
        task_label_indices={"alpha": [0, 1], "beta": [2, 3]},
        calibrators={"alpha": _PassThroughCalibrator(), "beta": _PassThroughCalibrator()},
        calibrator_classes={"alpha": [0, 1], "beta": [0, 1]},
    )

    probs = wrapped.predict_proba(
        [
            "task=alpha [text] one",
            "task=beta [text] two",
        ]
    )

    assert np.allclose(probs.sum(axis=1), 1.0)
    assert probs[0, 2] == pytest.approx(0.0)
    assert probs[1, 0] == pytest.approx(0.0)
