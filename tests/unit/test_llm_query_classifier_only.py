"""Unit tests for query classifier modelpack/LLM priority."""

import pytest

from src.retrieval.classifier import QueryClassifier
from src.retrieval.query_types import QueryIntent
from src.utils.llm import LLMClient


@pytest.mark.asyncio
async def test_classifier_prefers_modelpack_over_llm_when_available(monkeypatch):
    """Classifier should use modelpack first and avoid LLM call when modelpack predicts."""
    from types import SimpleNamespace
    from unittest.mock import AsyncMock, MagicMock

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.complete_json = AsyncMock(
        return_value={
            "intent": "preference_lookup",
            "entities": ["food"],
            "time_reference": None,
            "confidence": 0.9,
        }
    )
    mock_modelpack = MagicMock()
    mock_modelpack.available = True
    mock_modelpack.predict_single = MagicMock(
        side_effect=lambda task, text: (
            SimpleNamespace(label="preference_lookup", confidence=0.91)
            if task == "query_intent"
            else None
        )
    )
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: type(
            "S",
            (),
            {"features": type("F", (), {"use_llm_enabled": True})()},
        )(),
    )
    classifier = QueryClassifier(llm_client=mock_llm, modelpack=mock_modelpack)
    result = await classifier.classify("What do I like to eat?")
    mock_llm.complete_json.assert_not_called()
    assert result.intent == QueryIntent.PREFERENCE_LOOKUP
