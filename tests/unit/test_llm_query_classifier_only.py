"""Unit tests for use_llm_query_classifier_only feature flag."""

import pytest

from src.retrieval.classifier import QueryClassifier
from src.retrieval.query_types import QueryIntent
from src.utils.llm import LLMClient


@pytest.mark.asyncio
async def test_use_llm_query_classifier_only_skips_fast_path(monkeypatch):
    """When use_llm_query_classifier_only=true, classifier skips fast path and calls LLM."""
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
            {
                "features": type(
                    "F", (), {"use_llm_enabled": True, "use_llm_query_classifier_only": True}
                )()
            },
        )(),
    )
    classifier = QueryClassifier(llm_client=mock_llm)
    result = await classifier.classify("What do I like to eat?")
    # Should have used LLM (mock would be called)
    mock_llm.complete_json.assert_called()
    assert (
        result.intent == QueryIntent.PREFERENCE_LOOKUP or "preference" in str(result.intent).lower()
    )
