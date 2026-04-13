"""Unit tests for HybridRetriever (_retrieve_constraints, _fact_to_record)."""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.enums import MemoryType
from src.memory.neocortical.schemas import FactCategory, SemanticFact
from src.retrieval.planner import RetrievalSource, RetrievalStep
from src.retrieval.retriever import HybridRetriever


@pytest.fixture
def mock_hippocampal():
    hippocampal = MagicMock()
    hippocampal.search = AsyncMock(return_value=[])
    return hippocampal


@pytest.fixture
def mock_neocortical():
    neocortical = MagicMock()
    neocortical.facts = MagicMock()
    neocortical.facts.get_facts_by_category = AsyncMock(return_value=[])
    neocortical.facts.get_facts_by_categories = AsyncMock(return_value=[])
    return neocortical


def test_fact_to_record_constraint_routing(mock_hippocampal, mock_neocortical):
    """_fact_to_record maps items with type=CONSTRAINT or source=constraints to MemoryType.CONSTRAINT."""
    retriever = HybridRetriever(mock_hippocampal, mock_neocortical)
    fact = SemanticFact(
        id=str(uuid4()),
        tenant_id="t",
        category=FactCategory.POLICY,
        key="user:policy:diet",
        subject="user",
        predicate="diet",
        value="no shellfish",
        value_type="str",
        confidence=0.9,
        updated_at=datetime.now(UTC),
    )
    item = {
        "type": MemoryType.CONSTRAINT.value,
        "source": "constraints",
        "text": "[Policy] no shellfish",
        "relevance": 0.8,
    }

    record = retriever._fact_to_record(fact, item)

    assert record.type == MemoryType.CONSTRAINT


def test_fact_to_record_fact_routing(mock_hippocampal, mock_neocortical):
    """_fact_to_record maps items with type=SEMANTIC_FACT to MemoryType.SEMANTIC_FACT."""
    retriever = HybridRetriever(mock_hippocampal, mock_neocortical)
    fact = SemanticFact(
        id=str(uuid4()),
        tenant_id="t",
        category=FactCategory.PREFERENCE,
        key="user:preference:cuisine",
        subject="user",
        predicate="cuisine",
        value="Italian",
        value_type="str",
        confidence=0.8,
        updated_at=datetime.now(UTC),
    )
    item = {
        "type": MemoryType.SEMANTIC_FACT.value,
        "source": "facts",
        "text": "cuisine: Italian",
        "relevance": 0.9,
    }

    record = retriever._fact_to_record(fact, item)

    assert record.type == MemoryType.SEMANTIC_FACT


def test_fact_to_record_prefers_valid_from_for_timestamp(mock_hippocampal, mock_neocortical):
    """_fact_to_record preserves the fact's event time when valid_from is present."""
    retriever = HybridRetriever(mock_hippocampal, mock_neocortical)
    historical_ts = datetime(2023, 1, 15, 12, 0, tzinfo=UTC)
    fact = SemanticFact(
        id=str(uuid4()),
        tenant_id="t",
        category=FactCategory.LOCATION,
        key="user:location:current_city",
        subject="user",
        predicate="current_city",
        value="New York",
        value_type="str",
        confidence=0.9,
        valid_from=historical_ts,
        updated_at=datetime.now(UTC),
    )
    item = {
        "type": MemoryType.SEMANTIC_FACT.value,
        "source": "facts",
        "text": "current_city: New York",
        "relevance": 0.95,
    }

    record = retriever._fact_to_record(fact, item)

    assert record.timestamp == historical_ts


def test_fact_to_record_source_constraints_routes_to_constraint(mock_hippocampal, mock_neocortical):
    """_fact_to_record routes to CONSTRAINT when source=constraints even without type."""
    retriever = HybridRetriever(mock_hippocampal, mock_neocortical)
    fact = SemanticFact(
        id=str(uuid4()),
        tenant_id="t",
        category=FactCategory.VALUE,
        key="user:value:honesty",
        subject="user",
        predicate="honesty",
        value="I value honesty",
        value_type="str",
        confidence=0.85,
        updated_at=datetime.now(UTC),
    )
    item = {
        "source": "constraints",
        "text": "[Value] I value honesty",
        "relevance": 0.75,
    }

    record = retriever._fact_to_record(fact, item)

    assert record.type == MemoryType.CONSTRAINT


@pytest.mark.asyncio
async def test_retrieve_constraints_uses_batch_category_query(mock_hippocampal, mock_neocortical):
    """_retrieve_constraints prefers single IN-query batch fact fetch when available."""
    retriever = HybridRetriever(mock_hippocampal, mock_neocortical)
    mock_hippocampal.search = AsyncMock(return_value=[])
    mock_neocortical.facts.get_facts_by_categories = AsyncMock(
        return_value=[
            SemanticFact(
                id=str(uuid4()),
                tenant_id="t",
                category=FactCategory.POLICY,
                key="user:policy:diet",
                subject="user",
                predicate="diet",
                value="Avoid shellfish at all costs",
                value_type="str",
                confidence=0.9,
                updated_at=datetime.now(UTC),
            )
        ]
    )

    step = RetrievalStep(
        source=RetrievalSource.CONSTRAINTS,
        query="Should I order shellfish?",
        top_k=5,
        constraint_categories=["policy"],
    )

    rows = await retriever._retrieve_constraints("t", step)

    assert rows
    mock_neocortical.facts.get_facts_by_categories.assert_awaited_once()
