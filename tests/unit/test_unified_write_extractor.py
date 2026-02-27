"""Unit tests for UnifiedWritePathExtractor: entities, entity types, relations."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock
from uuid import uuid4

import pytest

from src.core.schemas import EntityMention, Relation
from src.extraction.unified_write_extractor import (
    UnifiedExtractionResult,
    UnifiedWritePathExtractor,
)
from src.memory.working.models import ChunkType, SemanticChunk


def _chunk(text: str, chunk_type: ChunkType = ChunkType.STATEMENT) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=chunk_type,
        salience=0.7,
        timestamp=datetime.now(UTC),
    )


_ALLOWED_TYPES = {
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
    "TIME",
    "MONEY",
    "PRODUCT",
    "EVENT",
    "CONCEPT",
    "PREFERENCE",
    "ATTRIBUTE",
}


@pytest.fixture
def mock_llm():
    m = AsyncMock()
    m.complete_json = AsyncMock()
    return m


class TestUnifiedExtractorEntities:
    """Entity extraction: typed objects, allowed types, exclusion of system prompts."""

    @pytest.mark.asyncio
    async def test_entities_have_text_normalized_type_from_allowed_set(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": [
                {"text": "Berlin", "normalized": "Berlin", "type": "LOCATION"},
                {"text": "Maria", "normalized": "Maria", "type": "PERSON"},
                {"text": "SAP", "normalized": "SAP", "type": "ORGANIZATION"},
            ],
            "relations": [],
            "salience": 0.7,
            "importance": 0.6,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("I live in Berlin and my sister Maria works at SAP.")
        result = await extractor.extract(chunk)
        assert len(result.entities) == 3
        for e in result.entities:
            assert isinstance(e, EntityMention)
            assert e.text
            assert e.normalized
            assert e.entity_type in _ALLOWED_TYPES
        texts = {e.text for e in result.entities}
        assert "Berlin" in texts
        assert "Maria" in texts
        assert "SAP" in texts
        types = {e.entity_type for e in result.entities}
        assert "LOCATION" in types
        assert "PERSON" in types
        assert "ORGANIZATION" in types

    @pytest.mark.asyncio
    async def test_system_prompt_chunk_yields_empty_entities(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": [],
            "relations": [],
            "salience": 0.3,
            "importance": 0.2,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("You are a helpful assistant. Always be concise.")
        result = await extractor.extract(chunk)
        assert result.entities == []
        assert result.relations == []

    @pytest.mark.asyncio
    async def test_invalid_entity_type_maps_to_concept(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": [
                {"text": "foo", "normalized": "foo", "type": "INVALID_TYPE"},
            ],
            "relations": [],
            "salience": 0.5,
            "importance": 0.5,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("foo bar")
        result = await extractor.extract(chunk)
        assert len(result.entities) == 1
        assert result.entities[0].entity_type == "CONCEPT"

    @pytest.mark.asyncio
    async def test_backward_compat_string_entities(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": ["Paris", "Google"],
            "relations": [],
            "salience": 0.5,
            "importance": 0.5,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("I visited Paris and Google.")
        result = await extractor.extract(chunk)
        assert len(result.entities) == 2
        for e in result.entities:
            assert e.text == e.normalized
            assert e.entity_type == "CONCEPT"


class TestUnifiedExtractorRelations:
    """Relation extraction: subject, predicate, object; snake_case predicate."""

    @pytest.mark.asyncio
    async def test_relations_have_subject_predicate_object_snake_case(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": [
                {"text": "John", "normalized": "John", "type": "PERSON"},
                {"text": "Sarah", "normalized": "Sarah", "type": "PERSON"},
                {"text": "Microsoft", "normalized": "Microsoft", "type": "ORGANIZATION"},
            ],
            "relations": [
                {"subject": "John", "predicate": "knows", "object": "Sarah", "confidence": 0.9},
                {
                    "subject": "Sarah",
                    "predicate": "works_at",
                    "object": "Microsoft",
                    "confidence": 0.85,
                },
            ],
            "salience": 0.7,
            "importance": 0.6,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("John knows Sarah. Sarah works at Microsoft.")
        result = await extractor.extract(chunk)
        assert len(result.relations) == 2
        for r in result.relations:
            assert isinstance(r, Relation)
            assert r.subject
            assert r.predicate
            assert r.object
            assert "_" in r.predicate or r.predicate.islower()
        rel0 = next(r for r in result.relations if r.subject == "John")
        assert rel0.predicate == "knows"
        assert rel0.object == "Sarah"
        rel1 = next(r for r in result.relations if r.object == "Microsoft")
        assert rel1.predicate == "works_at"

    @pytest.mark.asyncio
    async def test_relations_backward_compat_source_target_type(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": [],
            "relations": [
                {"source": "user", "target": "Paris", "type": "lives_in"},
            ],
            "salience": 0.5,
            "importance": 0.5,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("I live in Paris.")
        result = await extractor.extract(chunk)
        assert len(result.relations) == 1
        r = result.relations[0]
        assert r.subject == "user"
        assert r.object == "Paris"
        assert r.predicate == "lives_in"


class TestUnifiedExtractorParametrized:
    """Parametrized tests for fixture chunks (parsing correctness)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "chunk_text",
        [
            "I live in Berlin and my sister Maria works at SAP.",
            "You are a helpful assistant. Always be concise.",
            "Remember: user prefers vegan.",
            "John knows Sarah. Sarah works at Microsoft.",
        ],
    )
    async def test_fixture_chunks_parse_correctly(self, mock_llm, chunk_text):
        mock_llm.complete_json.return_value = {
            "entities": [
                {"text": "Berlin", "normalized": "Berlin", "type": "LOCATION"},
                {"text": "Maria", "normalized": "Maria", "type": "PERSON"},
                {"text": "SAP", "normalized": "SAP", "type": "ORGANIZATION"},
            ],
            "relations": [
                {"subject": "user", "predicate": "lives_in", "object": "Berlin", "confidence": 0.9},
                {"subject": "Maria", "predicate": "works_at", "object": "SAP", "confidence": 0.9},
            ],
            "salience": 0.6,
            "importance": 0.5,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk(chunk_text)
        result = await extractor.extract(chunk)
        assert isinstance(result, UnifiedExtractionResult)
        for e in result.entities:
            assert e.entity_type in _ALLOWED_TYPES
        for r in result.relations:
            assert r.subject and r.predicate and r.object


class TestUnifiedExtractorEmptyAndBatch:
    """Edge cases and batch extraction."""

    @pytest.mark.asyncio
    async def test_empty_text_returns_defaults(self, mock_llm):
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("")
        result = await extractor.extract(chunk)
        assert result.entities == []
        assert result.relations == []
        assert 0.5 <= result.salience <= 1.0
        mock_llm.complete_json.assert_not_called()

    @pytest.mark.asyncio
    async def test_extract_batch_uses_unified_schema(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "0": {
                "entities": [{"text": "Berlin", "normalized": "Berlin", "type": "LOCATION"}],
                "relations": [
                    {
                        "subject": "user",
                        "predicate": "lives_in",
                        "object": "Berlin",
                        "confidence": 0.9,
                    }
                ],
                "salience": 0.7,
                "importance": 0.6,
            },
            "1": {
                "entities": [],
                "relations": [],
                "salience": 0.3,
                "importance": 0.3,
            },
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunks = [_chunk("I live in Berlin."), _chunk("You are a helpful assistant.")]
        results = await extractor.extract_batch(chunks)
        assert len(results) == 2
        assert len(results[0].entities) == 1
        assert results[0].entities[0].entity_type == "LOCATION"
        assert len(results[0].relations) == 1
        assert results[0].relations[0].predicate == "lives_in"
        assert results[1].entities == []
        assert results[1].relations == []

    @pytest.mark.asyncio
    async def test_extract_falls_back_when_llm_call_fails(self, mock_llm):
        mock_llm.complete_json.side_effect = RuntimeError("llm backend unavailable")
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("I live in Berlin.")
        result = await extractor.extract(chunk)
        assert isinstance(result, UnifiedExtractionResult)
        assert result.entities == []
        assert result.relations == []
        assert result.salience == chunk.salience

    @pytest.mark.asyncio
    async def test_extract_batch_falls_back_when_llm_call_fails(self, mock_llm):
        mock_llm.complete_json.side_effect = RuntimeError("llm backend unavailable")
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunks = [_chunk("I live in Berlin."), _chunk("I prefer dark mode.")]
        results = await extractor.extract_batch(chunks)
        assert len(results) == 2
        assert all(isinstance(r, UnifiedExtractionResult) for r in results)
        assert all(r.entities == [] for r in results)


class TestUnifiedExtractorConfidenceContextTagsDecayRate:
    """Unified extractor parses confidence, context_tags, decay_rate from LLM output."""

    @pytest.mark.asyncio
    async def test_extract_returns_confidence_context_tags_decay_rate(self, mock_llm):
        mock_llm.complete_json.return_value = {
            "entities": [],
            "relations": [],
            "constraints": [],
            "facts": [],
            "salience": 0.6,
            "importance": 0.7,
            "confidence": 0.9,
            "context_tags": ["personal", "dietary", "work"],
            "decay_rate": 0.1,
        }
        extractor = UnifiedWritePathExtractor(mock_llm)
        chunk = _chunk("I prefer vegan food")
        result = await extractor.extract(chunk)
        assert result.confidence == 0.9
        assert result.context_tags == ["personal", "dietary", "work"]
        assert result.decay_rate == 0.1

    def test_parse_result_clamps_confidence(self):
        extractor = UnifiedWritePathExtractor(None)
        chunk = _chunk("Test")
        result = extractor._parse_result(
            {
                "entities": [],
                "relations": [],
                "constraints": [],
                "facts": [],
                "salience": 0.5,
                "importance": 0.5,
                "confidence": 1.5,
            },
            chunk,
        )
        assert result.confidence == 1.0
        result2 = extractor._parse_result(
            {
                "entities": [],
                "relations": [],
                "constraints": [],
                "facts": [],
                "salience": 0.5,
                "importance": 0.5,
                "confidence": -0.1,
            },
            chunk,
        )
        assert result2.confidence == 0.0

    def test_parse_result_default_confidence(self):
        extractor = UnifiedWritePathExtractor(None)
        chunk = _chunk("Test")
        result = extractor._parse_result(
            {
                "entities": [],
                "relations": [],
                "constraints": [],
                "facts": [],
                "salience": 0.5,
                "importance": 0.5,
            },
            chunk,
        )
        assert result.confidence == 0.5

    def test_parse_result_validates_decay_rate_range(self):
        extractor = UnifiedWritePathExtractor(None)
        chunk = _chunk("Test")
        result = extractor._parse_result(
            {
                "entities": [],
                "relations": [],
                "constraints": [],
                "facts": [],
                "salience": 0.5,
                "importance": 0.5,
                "decay_rate": 2.0,
            },
            chunk,
        )
        assert result.decay_rate is None
        result2 = extractor._parse_result(
            {
                "entities": [],
                "relations": [],
                "constraints": [],
                "facts": [],
                "salience": 0.5,
                "importance": 0.5,
                "decay_rate": 0.05,
            },
            chunk,
        )
        assert result2.decay_rate == 0.05

    def test_parse_result_context_tags_filters_non_strings(self):
        extractor = UnifiedWritePathExtractor(None)
        chunk = _chunk("Test")
        result = extractor._parse_result(
            {
                "entities": [],
                "relations": [],
                "constraints": [],
                "facts": [],
                "salience": 0.5,
                "importance": 0.5,
                "context_tags": ["a", 42, "b", "", "  ", "c"],
            },
            chunk,
        )
        assert result.context_tags == ["a", "b", "c"]
