"""Tests for runtime device selection in local HF-backed model wrappers."""

from __future__ import annotations

import sys
from pathlib import Path
from types import ModuleType

from cml.modeling.runtime_models import EmbeddingPairClassifier, TransformerTextClassifier
from cml.runtime_device import (
    DEFAULT_MODEL_RUNTIME_DEVICE,
    get_model_runtime_device_preference,
    resolve_runtime_device_name,
)
from cml.token_runtime import HFTokenSpanPredictor


class _FakeCudaNamespace:
    def __init__(self, available: bool) -> None:
        self._available = available

    def is_available(self) -> bool:
        return self._available


class _FakeTorchModule(ModuleType):
    def __init__(self, available: bool) -> None:
        super().__init__("torch")
        self.cuda = _FakeCudaNamespace(available)

    @staticmethod
    def device(name: str) -> str:
        return name


def test_model_runtime_device_default_is_cpu(monkeypatch) -> None:
    monkeypatch.delenv("CML_MODELS_DEVICE", raising=False)
    assert get_model_runtime_device_preference() == DEFAULT_MODEL_RUNTIME_DEVICE


def test_model_runtime_device_auto_uses_cuda_when_available(monkeypatch) -> None:
    monkeypatch.setenv("CML_MODELS_DEVICE", "auto")
    assert resolve_runtime_device_name(_FakeTorchModule(available=True)) == "cuda"


def test_embedding_pair_classifier_passes_resolved_device(monkeypatch) -> None:
    captured: dict[str, str] = {}
    monkeypatch.setenv("CML_MODELS_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule(available=True))

    sentence_transformers = ModuleType("sentence_transformers")

    class _FakeSentenceTransformer:
        def __init__(self, _model_name: str, *, device: str):
            captured["device"] = device

    sentence_transformers.SentenceTransformer = _FakeSentenceTransformer
    monkeypatch.setitem(sys.modules, "sentence_transformers", sentence_transformers)

    model = EmbeddingPairClassifier(
        task_name="memory_rerank_pair",
        model_name_or_path="sentence-transformers/all-MiniLM-L6-v2",
        classifier=object(),
        classes_=["memory_rerank_pair::not_relevant", "memory_rerank_pair::relevant"],
    )

    model._ensure_encoder()

    assert captured["device"] == "cpu"


def test_transformer_text_classifier_respects_cpu_default(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.delenv("CML_MODELS_DEVICE", raising=False)
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule(available=True))

    captured: dict[str, str] = {}
    transformers = ModuleType("transformers")

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(_path: str, use_fast: bool = True):
            return {"use_fast": use_fast}

    class _FakeModel:
        def to(self, device: str):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = "true"
            return self

        @staticmethod
        def from_pretrained(_path: str):
            return _FakeModel()

    transformers.AutoTokenizer = _FakeTokenizer
    transformers.AutoModelForSequenceClassification = _FakeModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model_dir = tmp_path / "constraint_dimension_hf"
    model_dir.mkdir()
    model = TransformerTextClassifier(
        task_name="constraint_dimension",
        model_dir=str(model_dir),
        classes_=["constraint_dimension::a", "constraint_dimension::b"],
    )

    model._ensure_loaded()

    assert model._device == "cpu"
    assert captured["device"] == "cpu"


def test_token_span_predictor_respects_cpu_override(monkeypatch, tmp_path: Path) -> None:
    monkeypatch.setenv("CML_MODELS_DEVICE", "cpu")
    monkeypatch.setitem(sys.modules, "torch", _FakeTorchModule(available=True))

    captured: dict[str, str] = {}
    transformers = ModuleType("transformers")

    class _FakeTokenizer:
        @staticmethod
        def from_pretrained(_path: str, use_fast: bool = True):
            return {"use_fast": use_fast}

    class _FakeTokenModel:
        def to(self, device: str):
            captured["device"] = device
            return self

        def eval(self):
            captured["eval"] = "true"
            return self

        @staticmethod
        def from_pretrained(_path: str):
            return _FakeTokenModel()

    transformers.AutoTokenizer = _FakeTokenizer
    transformers.AutoModelForTokenClassification = _FakeTokenModel
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    model_dir = tmp_path / "fact_extraction_structured_hf"
    model_dir.mkdir()
    predictor = HFTokenSpanPredictor(
        task_name="fact_extraction_structured",
        model_dir=str(model_dir),
        id_to_label={0: "O", 1: "B-FACT"},
    )

    predictor._ensure_loaded()

    assert predictor._device == "cpu"
    assert captured["device"] == "cpu"
