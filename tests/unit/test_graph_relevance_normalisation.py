"""Graph hits must arrive on the same scale as every other retrieval prong.

Neo4j co-occurrence scores are unbounded — 315, 265 and 744 were all observed on one
query — while the vector and fact prongs emit cosine-like 0..1. Ranking happens on the
raw value before the reranker's clamp, so unnormalised graph hits took every top-k slot
and pushed real episodes below the packet's ``episode_relevance_threshold``, which drops
them from the Recent Events section entirely. The model then had facts and preferences
but no conversation, and refused temporal and single-hop questions.

This was invisible for as long as the graph stayed empty, which it was until eval mode
stopped skipping graph sync.
"""

from datetime import UTC, datetime
from uuid import NAMESPACE_URL, uuid5

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.retrieval.retriever import (
    GRAPH_RELEVANCE_CEILING,
    GRAPH_RELEVANCE_FLOOR,
    HybridRetriever,
)


class _Neocortical:
    def __init__(self, results):
        self._results = results

    async def multi_hop_query(self, tenant_id, seed_entities=None):
        return self._results


class _Store:
    """Resolves an evidence id to a record whose text is the entity that cited it, so
    normalisation can still be asserted per entity now that the prong returns text."""

    async def get_by_ids_batch(self, record_ids):
        return [
            MemoryRecord(
                id=rid,
                tenant_id="t1",
                context_tags=[],
                type=MemoryType.EPISODIC_EVENT,
                text=_names[rid],
                provenance=Provenance(source=MemorySource.USER_EXPLICIT),
                timestamp=datetime.now(UTC),
            )
            for rid in record_ids
        ]


_names: dict = {}


def _eid(name: str):
    rid = uuid5(NAMESPACE_URL, f"graph-norm/{name}")
    _names[rid] = name
    return rid


def _entity(name: str, score: float) -> dict:
    return {
        "entity": name,
        "relevance_score": score,
        "relations": [
            {
                "predicate": "WORKS_AT",
                "related_entity": "a big software company",
                "relation_properties": {"evidence_ids": [str(_eid(name))]},
            }
        ],
        "facts": [],
    }


def _retriever(results):
    r = HybridRetriever.__new__(HybridRetriever)
    r.neocortical = _Neocortical(results)
    r.hippocampal = type("_H", (), {"store": _Store()})()
    return r


class _Step:
    seeds = ["user"]
    top_k = 10


class TestGraphNormalisation:
    @pytest.mark.asyncio
    async def test_unbounded_scores_land_inside_the_band(self):
        r = _retriever([_entity("user", 315.67), _entity("Tim", 265.33), _entity("John", 744.5)])
        items = await r._retrieve_graph("t1", _Step())

        assert len(items) == 3
        for it in items:
            assert GRAPH_RELEVANCE_FLOOR <= it["relevance"] <= GRAPH_RELEVANCE_CEILING

    @pytest.mark.asyncio
    async def test_ordering_among_graph_hits_is_preserved(self):
        """A flat clamp to 1.0 would make every graph hit indistinguishable."""
        r = _retriever([_entity("user", 315.67), _entity("Tim", 265.33), _entity("John", 744.5)])
        items = await r._retrieve_graph("t1", _Step())

        by_name = {it["text"]: it["relevance"] for it in items}
        assert by_name["John"] > by_name["user"] > by_name["Tim"]

    @pytest.mark.asyncio
    async def test_a_graph_hit_cannot_outrank_a_strong_episode(self):
        """0.9-relevance episodes are direct answers; an entity profile is a summary."""
        r = _retriever([_entity("user", 999999.0)])
        items = await r._retrieve_graph("t1", _Step())

        assert items[0]["relevance"] < 0.9

    @pytest.mark.asyncio
    async def test_the_weakest_graph_hit_still_clears_the_episode_threshold(self):
        """Normalising must not push graph results below the packet's default 0.5
        cutoff — that would trade one silent exclusion for another."""
        r = _retriever([_entity("a", 1.0), _entity("b", 500.0)])
        items = await r._retrieve_graph("t1", _Step())

        assert min(it["relevance"] for it in items) > 0.5

    @pytest.mark.asyncio
    async def test_identical_scores_do_not_divide_by_zero(self):
        r = _retriever([_entity("a", 7.0), _entity("b", 7.0)])
        items = await r._retrieve_graph("t1", _Step())

        assert [it["relevance"] for it in items] == [GRAPH_RELEVANCE_CEILING] * 2

    @pytest.mark.asyncio
    async def test_no_graph_results_is_not_an_error(self):
        r = _retriever([])
        assert await r._retrieve_graph("t1", _Step()) == []
