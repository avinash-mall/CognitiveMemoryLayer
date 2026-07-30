"""Unit tests for query classifier LLM path."""

import pytest

from src.retrieval.classifier import QueryClassifier
from src.retrieval.query_types import QueryIntent
from src.utils.llm import LLMClient


@pytest.mark.asyncio
async def test_classifier_uses_llm_when_enabled(monkeypatch):
    """Classifier should call the LLM when a client is set and use_llm_enabled is on."""
    from unittest.mock import AsyncMock

    mock_llm = AsyncMock(spec=LLMClient)
    mock_llm.complete_json = AsyncMock(
        return_value={
            "intent": "preference_lookup",
            "entities": ["food"],
            "time_reference": None,
            "confidence": 0.9,
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
    classifier = QueryClassifier(llm_client=mock_llm)
    result = await classifier.classify("What do I like to eat?")
    mock_llm.complete_json.assert_called()
    assert result.intent == QueryIntent.PREFERENCE_LOOKUP
