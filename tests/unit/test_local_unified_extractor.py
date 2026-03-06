"""Unit tests for LocalUnifiedWriteExtractor."""

from __future__ import annotations

import asyncio
from unittest.mock import MagicMock

import pytest

from src.extraction.local_unified_extractor import LocalUnifiedWriteExtractor
from src.utils.modelpack import ModelPrediction, ScorePrediction, SpanPrediction


def _make_modelpack(**overrides):
    """Create a mock modelpack with configurable task model availability."""
    mp = MagicMock()
    available_tasks = set(overrides.get("available_tasks", []))
    mp.has_task_model = MagicMock(side_effect=lambda t: t in available_tasks)
    mp.available = overrides.get("available", True)

    if overrides.get("spans_result") is not None:
        mp.predict_spans.return_value = overrides["spans_result"]
    else:
        mp.predict_spans.return_value = None

    if overrides.get("score_result") is not None:
        mp.predict_score_single.return_value = overrides["score_result"]
    else:
        mp.predict_score_single.return_value = None

    if overrides.get("single_result") is not None:
        mp.predict_single.return_value = overrides["single_result"]
    else:
        mp.predict_single.return_value = None

    return mp


class TestLocalUnifiedExtractorAvailability:
    def test_available_when_fact_model_loaded(self):
        mp = _make_modelpack(available_tasks=["fact_extraction_structured"])
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        assert ext.available is True

    def test_available_when_importance_model_loaded(self):
        mp = _make_modelpack(available_tasks=["write_importance_regression"])
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        assert ext.available is True

    def test_available_when_pii_model_loaded(self):
        mp = _make_modelpack(available_tasks=["pii_span_detection"])
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        assert ext.available is True

    def test_not_available_when_no_models(self):
        mp = _make_modelpack(available_tasks=[], available=False)
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        assert ext.available is False

    def test_available_when_router_loaded(self):
        mp = _make_modelpack(available_tasks=[], available=True)
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        assert ext.available is True


class TestLocalUnifiedExtractorExtract:
    def test_returns_default_result_when_no_models(self):
        mp = _make_modelpack(available_tasks=[])
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("test text"))
        assert result["source"] == "local_unified"
        assert result["importance"] == 0.5
        assert result["facts"] == []
        assert result["pii_spans"] == []
        assert result["memory_type"] is None

    def test_extracts_facts_from_model(self):
        spans = SpanPrediction(
            task="fact_extraction_structured",
            spans=((0, 4, "preference"), (5, 10, "identity")),
        )
        mp = _make_modelpack(
            available_tasks=["fact_extraction_structured"],
            spans_result=spans,
        )
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("test text here"))
        assert len(result["facts"]) == 2
        assert result["facts"][0]["label"] == "preference"

    def test_extracts_importance_from_model(self):
        score = ScorePrediction(task="write_importance_regression", score=0.85)
        mp = _make_modelpack(
            available_tasks=["write_importance_regression"],
            score_result=score,
        )
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("important text"))
        assert result["importance"] == pytest.approx(0.85)

    def test_extracts_pii_spans_from_model(self):
        pii = SpanPrediction(
            task="pii_span_detection",
            spans=((0, 5, "EMAIL"),),
        )
        mp = _make_modelpack(
            available_tasks=["pii_span_detection"],
            spans_result=pii,
        )
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("a@b.c text"))
        assert len(result["pii_spans"]) == 1
        assert result["pii_spans"][0]["label"] == "EMAIL"

    def test_extracts_memory_type_from_router(self):
        pred = ModelPrediction(task="memory_type", label="preference", confidence=0.9)
        mp = _make_modelpack(available_tasks=[], single_result=pred)
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("I love pizza"))
        assert result["memory_type"] == "preference"

    def test_fallback_on_model_exception(self):
        mp = _make_modelpack(available_tasks=["fact_extraction_structured"])
        mp.predict_spans.side_effect = RuntimeError("model crashed")
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("text"))
        assert result["facts"] == []
        assert result["source"] == "local_unified"

    def test_importance_clamped_to_unit_interval(self):
        score = ScorePrediction(task="write_importance_regression", score=1.5)
        mp = _make_modelpack(
            available_tasks=["write_importance_regression"],
            score_result=score,
        )
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("text"))
        assert result["importance"] == 1.0

    def test_empty_text_returns_defaults(self):
        mp = _make_modelpack(available_tasks=["write_importance_regression"])
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract(""))
        assert result["importance"] == 0.5

    def test_extracts_context_tag_from_router(self):
        pred = ModelPrediction(task="context_tag", label="food", confidence=0.85)
        mp = _make_modelpack(available=True, single_result=pred)
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(
            ext.extract("I prefer vegetarian cuisine")
        )
        assert result["context_tags"] == ["food"]

    def test_extracts_confidence_bin_maps_to_float(self):
        for label, expected in [("low", 0.35), ("medium", 0.65), ("high", 0.9)]:
            pred = ModelPrediction(task="confidence_bin", label=label, confidence=0.8)
            mp = _make_modelpack(available=True, single_result=pred)
            ext = LocalUnifiedWriteExtractor(modelpack=mp)
            result = asyncio.get_event_loop().run_until_complete(ext.extract("some text"))
            assert result["confidence"] == pytest.approx(expected)

    def test_extracts_decay_profile_maps_to_float(self):
        for label, expected in [
            ("very_fast", 0.35),
            ("fast", 0.2),
            ("medium", 0.1),
            ("slow", 0.05),
            ("very_slow", 0.02),
        ]:
            pred = ModelPrediction(task="decay_profile", label=label, confidence=0.8)
            mp = _make_modelpack(available=True, single_result=pred)
            ext = LocalUnifiedWriteExtractor(modelpack=mp)
            result = asyncio.get_event_loop().run_until_complete(ext.extract("ephemeral note"))
            assert result["decay_rate"] == pytest.approx(expected)

    def test_context_tag_confidence_decay_fallback_when_predict_fails(self):
        mp = _make_modelpack(available=True, single_result=None)
        ext = LocalUnifiedWriteExtractor(modelpack=mp)
        result = asyncio.get_event_loop().run_until_complete(ext.extract("text"))
        assert result["context_tags"] == []
        assert result["confidence"] == 0.5
        assert result["decay_rate"] is None
