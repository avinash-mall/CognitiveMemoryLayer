"""Unit tests for extraction (entity, relation, fact) with mocked LLM."""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.core.schemas import EntityMention, Relation
from src.extraction.entity_extractor import EntityExtractor
from src.extraction.fact_extractor import ExtractedFact, FactExtractor, LLMFactExtractor
from src.extraction.relation_extractor import RelationExtractor


@pytest.fixture
def mock_llm():
    """Mock LLM client that returns configurable response."""
    m = MagicMock()
    m.complete = AsyncMock()
    return m


class TestEntityExtractor:
    """EntityExtractor with mocked LLM."""

    @pytest.mark.asyncio
    async def test_extract_returns_entity_mentions(self, mock_llm):
        mock_llm.complete.return_value = """[
            {"text": "Paris", "normalized": "Paris, France", "type": "LOCATION"},
            {"text": "Alice", "normalized": "Alice", "type": "PERSON"}
        ]"""
        extractor = EntityExtractor(llm_client=mock_llm)
        result = await extractor.extract("Alice went to Paris.")
        assert len(result) == 2
        assert all(isinstance(e, EntityMention) for e in result)
        texts = [e.text for e in result]
        assert "Paris" in texts
        assert "Alice" in texts
        types = [e.entity_type for e in result]
        assert "LOCATION" in types
        assert "PERSON" in types

    @pytest.mark.asyncio
    async def test_extract_strips_markdown_fences(self, mock_llm):
        mock_llm.complete.return_value = """```json
[{"text": "Berlin", "normalized": "Berlin", "type": "LOCATION"}]
```"""
        extractor = EntityExtractor(llm_client=mock_llm)
        result = await extractor.extract("Berlin is nice.")
        assert len(result) == 1
        assert result[0].text == "Berlin"
        assert result[0].entity_type == "LOCATION"

    @pytest.mark.asyncio
    async def test_entity_extractor_returns_empty_on_invalid_json(self, mock_llm):
        mock_llm.complete.return_value = "not json at all"
        extractor = EntityExtractor(llm_client=mock_llm)
        result = await extractor.extract("Some text.")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_includes_context_in_prompt_when_provided(self, mock_llm):
        mock_llm.complete.return_value = "[]"
        extractor = EntityExtractor(llm_client=mock_llm)
        await extractor.extract("Hello.", context="User greeting.")
        call_args = mock_llm.complete.call_args
        assert "Context: User greeting." in call_args[0][0]


class TestRelationExtractor:
    """RelationExtractor with mocked LLM."""

    @pytest.mark.asyncio
    async def test_extract_returns_relations(self, mock_llm):
        mock_llm.complete.return_value = """[
            {"subject": "John", "predicate": "lives_in", "object": "Paris", "confidence": 0.9}
        ]"""
        extractor = RelationExtractor(llm_client=mock_llm)
        result = await extractor.extract("John lives in Paris.")
        assert len(result) == 1
        assert isinstance(result[0], Relation)
        assert result[0].subject == "John"
        assert result[0].predicate == "lives_in"
        assert result[0].object == "Paris"
        assert result[0].confidence == 0.9

    @pytest.mark.asyncio
    async def test_extract_normalizes_predicate_to_snake_case(self, mock_llm):
        mock_llm.complete.return_value = """[
            {"subject": "user", "predicate": "prefers something", "object": "tea", "confidence": 0.8}
        ]"""
        extractor = RelationExtractor(llm_client=mock_llm)
        result = await extractor.extract("I prefer tea.")
        assert len(result) == 1
        assert result[0].predicate == "prefers_something"

    @pytest.mark.asyncio
    async def test_relation_extractor_returns_empty_on_invalid_json(self, mock_llm):
        mock_llm.complete.return_value = "not json"
        extractor = RelationExtractor(llm_client=mock_llm)
        result = await extractor.extract("Some text.")
        assert result == []

    @pytest.mark.asyncio
    async def test_extract_batch_returns_list_of_relation_lists(self, mock_llm):
        mock_llm.complete.return_value = """[
            {"subject": "A", "predicate": "knows", "object": "B", "confidence": 0.9}
        ]"""
        extractor = RelationExtractor(llm_client=mock_llm)
        items = [("Alice knows Bob.", []), ("Carol works in Paris.", [])]
        result = await extractor.extract_batch(items)
        assert len(result) == 2
        assert len(result[0]) == 1
        assert result[0][0].subject == "A"
        assert result[0][0].predicate == "knows"
        assert result[0][0].object == "B"
        assert len(result[1]) == 1
        mock_llm.complete.assert_called()

    @pytest.mark.asyncio
    async def test_extract_batch_empty_list(self, mock_llm):
        extractor = RelationExtractor(llm_client=mock_llm)
        result = await extractor.extract_batch([])
        assert result == []


class TestFactExtractor:
    """FactExtractor base (no-op) and LLMFactExtractor."""

    @pytest.mark.asyncio
    async def test_base_extract_returns_empty(self):
        extractor = FactExtractor()
        result = await extractor.extract("Any text.")
        assert result == []


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
