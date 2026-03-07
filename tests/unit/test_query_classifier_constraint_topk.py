"""Unit tests for constraint_dimensions and suggested_top_k from Query Classifier LLM."""

import pytest

from src.retrieval.classifier import QueryClassifier
from src.retrieval.query_types import QueryIntent
from src.utils.llm import LLMClient


class _NoModelPack:
    available = False

    @staticmethod
    def predict_single(task: str, text: str):
        _ = task
        _ = text
        return None


@pytest.mark.asyncio
async def test_llm_classify_returns_constraint_dimensions_and_suggested_top_k(monkeypatch):
    """When LLM returns constraint_dimensions and suggested_top_k, QueryAnalysis contains them."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.complete_json = AsyncMock(
        return_value={
            "intent": "constraint_check",
            "entities": ["dietary"],
            "time_reference": None,
            "confidence": 0.9,
            "constraint_dimensions": ["dietary", "goal"],
            "suggested_top_k": 12,
        }
    )
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {"features": type("F", (), {"use_llm_enabled": True})()},
        )(),
    )
    classifier = QueryClassifier(llm_client=mock_llm, modelpack=_NoModelPack())
    result = await classifier.classify("Should I eat the seafood?")
    mock_llm.complete_json.assert_called()
    assert result.constraint_dimensions is not None
    assert "dietary" in result.constraint_dimensions
    assert "goal" in result.constraint_dimensions
    assert result.constraint_dimensions_from_llm is True
    assert result.suggested_top_k == 12


@pytest.mark.asyncio
async def test_llm_classify_valid_suggested_top_k_used(monkeypatch):
    """Valid suggested_top_k (5-20) from LLM is used."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.complete_json = AsyncMock(
        return_value={
            "intent": "preference_lookup",
            "entities": [],
            "time_reference": None,
            "confidence": 0.9,
            "suggested_top_k": 5,
        }
    )
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {"features": type("F", (), {"use_llm_enabled": True})()},
        )(),
    )
    classifier = QueryClassifier(llm_client=mock_llm, modelpack=_NoModelPack())
    result = await classifier.classify("What do I like?")
    assert result.intent == QueryIntent.PREFERENCE_LOOKUP
    assert result.suggested_top_k == 5


@pytest.mark.asyncio
async def test_llm_classify_suggested_top_k_out_of_range_falls_back(monkeypatch):
    """suggested_top_k outside 5-20 falls back to intent default."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.complete_json = AsyncMock(
        return_value={
            "intent": "episodic_recall",
            "entities": [],
            "time_reference": None,
            "confidence": 0.9,
            "suggested_top_k": 50,
        }
    )
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {"features": type("F", (), {"use_llm_enabled": True})()},
        )(),
    )
    classifier = QueryClassifier(llm_client=mock_llm, modelpack=_NoModelPack())
    result = await classifier.classify("What did we discuss?")
    assert result.intent == QueryIntent.EPISODIC_RECALL
    assert result.suggested_top_k == 10
