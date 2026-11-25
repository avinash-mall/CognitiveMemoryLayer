"""Unit tests for LLMFactExtractor with mocked LLM."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.extraction.fact_extractor import ExtractedFact, LLMFactExtractor


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns configurable response."""
    m = MagicMock()
    m.complete = AsyncMock()
    return m


class TestLLMFactExtractor:
    """LLMFactExtractor with mocked LLM."""

    @pytest.mark.asyncio
    async def test_extract_returns_extracted_facts(self, mock_llm):
        mock_llm.complete.return_value = """[
            {"text": "User prefers dark mode", "type": "preference"},
            {"text": "User is a developer", "type": "identity"}
        ]"""
        extractor = LLMFactExtractor(llm_client=mock_llm)
        result = await extractor.extract("I prefer dark mode. I'm a developer.")
        assert len(result) == 2
        assert all(isinstance(f, ExtractedFact) for f in result)
        assert result[0].text == "User prefers dark mode"
        assert result[0].type == "preference"
        assert result[1].type == "identity"

    @pytest.mark.asyncio
    async def test_extract_returns_empty_for_empty_text(self, mock_llm):
        extractor = LLMFactExtractor(llm_client=mock_llm)
        result = await extractor.extract("")
        assert result == []
        result = await extractor.extract("   ")
        assert result == []

    @pytest.mark.asyncio
    async def test_llm_fact_extractor_returns_empty_on_invalid_json(self, mock_llm):
        mock_llm.complete.return_value = "not json"
        extractor = LLMFactExtractor(llm_client=mock_llm)
        result = await extractor.extract("Some conversation.")
        assert result == []
