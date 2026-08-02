"""The graph prong is an index, not content.

``multi_hop_query`` runs Personalized PageRank — the mechanism HippoRAG reports up to 20%
multi-hop gains from — but it used to return ``{entity, relations, facts}`` rendered as
"Entity: user\\n  - LOCATION: Seattle". That is the Entity-Only failure mode: EcphoryRAG's
ablation finds entity records alone perform far worse than entities plus their source
chunks, because the model needs grounded text to reason over. We measured the same thing
from the other side — every populated-graph arm scored below the empty-graph baseline on
multi-hop (0.33 -> 0.23/0.27/0.26).

The join key was already in the payload: relation edges are written with
``evidence_ids=[record.id]`` and the Cypher returns ``properties(r)``. On the full2 corpus
464,511 of 583,346 edges (79.6%) carry one. It reached the retriever and was discarded one
function short of being useful.
"""

from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.retrieval.planner import RetrievalSource, RetrievalStep
from src.retrieval.retriever import (
    GRAPH_RELEVANCE_CEILING,
    GRAPH_RELEVANCE_FLOOR,
    HybridRetriever,
)

TENANT = "t"


def _record(text: str, rid=None, tenant_id: str = TENANT) -> MemoryRecord:
    return MemoryRecord(
        id=rid or uuid4(),
        tenant_id=tenant_id,
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=datetime.now(UTC),
    )


def _entity(name: str, score: float, evidence: list[str]) -> dict:
    """A PPR result whose single relation cites `evidence`."""
    return {
        "entity": name,
        "relevance_score": score,
        "facts": [],
        "relations": [
            {
                "predicate": "MENTIONS",
                "related_entity": "something",
                "relation_properties": {"evidence_ids": evidence},
            }
        ],
    }


def _retriever(ppr_results: list[dict], records: list[MemoryRecord]) -> HybridRetriever:
    by_id = {r.id: r for r in records}

    async def _fetch(record_ids):
        """Return only what was asked for, so a missing cap shows up as a real failure
        rather than being masked by a fixture that hands back everything."""
        return [by_id[i] for i in record_ids if i in by_id]

    hippocampal = MagicMock()
    hippocampal.store.get_by_ids_batch = AsyncMock(side_effect=_fetch)
    neocortical = MagicMock()
    neocortical.multi_hop_query = AsyncMock(return_value=ppr_results)
    return HybridRetriever(hippocampal, neocortical)


def _step(seeds: list[str]) -> RetrievalStep:
    return RetrievalStep(source=RetrievalSource.GRAPH, seeds=seeds)


class TestResolvesToEpisodicText:
    @pytest.mark.asyncio
    async def test_returns_source_text_not_entity_profiles(self):
        rid = str(uuid4())
        r = _retriever([_entity("user", 10.0, [rid])], [_record("I moved to Seattle in May.", rid)])

        items = await r._retrieve_graph(TENANT, _step(["user"]))

        assert [i["text"] for i in items] == ["I moved to Seattle in May."]
        assert not any(i["text"].startswith("Entity:") for i in items)

    @pytest.mark.asyncio
    async def test_the_resolved_record_is_carried_through(self):
        """`_to_retrieved_memories` passes a real MemoryRecord straight through, so the
        packet gets the episode's own type, timestamp and metadata — the event_date
        rendering and the reranker's recency term both depend on that."""
        rid = str(uuid4())
        record = _record("I moved to Seattle in May.", rid)
        r = _retriever([_entity("user", 10.0, [rid])], [record])

        items = await r._retrieve_graph(TENANT, _step(["user"]))

        assert items[0]["record"] is record
        assert items[0]["type"] == MemoryType.EPISODIC_EVENT.value
        assert items[0]["source"] == "graph"

    @pytest.mark.asyncio
    async def test_ppr_order_survives_as_the_relevance_band(self):
        """PPR rank still decides ordering — the entity is the index, so its score is
        what ranks the text it points at."""
        top, bottom = str(uuid4()), str(uuid4())
        r = _retriever(
            [_entity("user", 100.0, [top]), _entity("Tim", 1.0, [bottom])],
            [_record("top hit", top), _record("bottom hit", bottom)],
        )

        by_text = {i["text"]: i["relevance"] for i in await r._retrieve_graph(TENANT, _step(["u"]))}

        assert by_text["top hit"] == pytest.approx(GRAPH_RELEVANCE_CEILING)
        assert by_text["bottom hit"] == pytest.approx(GRAPH_RELEVANCE_FLOOR)

    @pytest.mark.asyncio
    async def test_an_episode_cited_by_several_entities_keeps_its_best_score(self):
        """Reachability from more than one high-PPR entity is evidence for, not against."""
        shared = str(uuid4())
        r = _retriever(
            [_entity("user", 100.0, [shared]), _entity("Tim", 1.0, [shared])],
            [_record("shared", shared)],
        )

        items = await r._retrieve_graph(TENANT, _step(["u"]))

        assert len(items) == 1
        assert items[0]["relevance"] == pytest.approx(GRAPH_RELEVANCE_CEILING)


class TestCandidateCap:
    @pytest.mark.asyncio
    async def test_resolved_records_are_capped_at_step_top_k(self):
        """Measured: 10 entities resolved to 309 distinct records on a real tenant,
        because get_entity_facts_batch has no LIMIT and a hub entity drags in hundreds
        of edges. Uncapped, the graph prong alone swamps every other prong's candidates.
        """
        ids = [str(uuid4()) for _ in range(50)]
        r = _retriever(
            [_entity("user", 10.0, ids)],
            [_record(f"r{i}", rid) for i, rid in enumerate(ids)],
        )
        step = _step(["user"])
        step.top_k = 7

        items = await r._retrieve_graph(TENANT, step)

        assert len(items) == 7
        # The cap must be applied before the fetch, not after — the point is to not
        # pay for 309 rows we then discard.
        assert len(r.hippocampal.store.get_by_ids_batch.await_args.args[0]) == 7

    @pytest.mark.asyncio
    async def test_the_cap_keeps_the_best_scoring_evidence(self):
        weak, strong = str(uuid4()), str(uuid4())
        r = _retriever(
            [_entity("user", 100.0, [strong]), _entity("Tim", 1.0, [weak])],
            [_record("strong", strong)],
        )
        step = _step(["user"])
        step.top_k = 1

        items = await r._retrieve_graph(TENANT, step)

        assert [i["text"] for i in items] == ["strong"]


class TestDegradesQuietly:
    @pytest.mark.asyncio
    async def test_edges_without_evidence_yield_nothing(self):
        """The fact-sync path writes edges with no properties at all — 20.4% of the
        full2 corpus. Those entities simply do not contribute; returning the profile as
        a consolation prize is the failure mode this whole change removes."""
        r = _retriever([_entity("user", 10.0, [])], [])
        assert await r._retrieve_graph(TENANT, _step(["user"])) == []

    @pytest.mark.asyncio
    async def test_no_ppr_results(self):
        r = _retriever([], [])
        assert await r._retrieve_graph(TENANT, _step(["user"])) == []

    @pytest.mark.asyncio
    async def test_no_seeds_skips_the_graph_entirely(self):
        r = _retriever([_entity("user", 10.0, [str(uuid4())])], [])
        assert await r._retrieve_graph(TENANT, _step([])) == []
        r.neocortical.multi_hop_query.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_malformed_evidence_id_does_not_take_the_prong_down(self):
        """evidence_ids is free-form graph property data, not a validated column."""
        good = str(uuid4())
        r = _retriever(
            [_entity("user", 10.0, ["not-a-uuid", good])],
            [_record("survived", good)],
        )
        assert [i["text"] for i in await r._retrieve_graph(TENANT, _step(["u"]))] == ["survived"]


class TestTenantIsolation:
    @pytest.mark.asyncio
    async def test_a_foreign_record_is_dropped(self):
        """`get_by_ids_batch` is a bare id lookup with no tenant predicate. Edge evidence
        comes from this tenant's own graph so this should never fire — but cross-tenant
        leakage is not a property worth trusting transitively."""
        mine, theirs = str(uuid4()), str(uuid4())
        r = _retriever(
            [_entity("user", 10.0, [mine, theirs])],
            [_record("mine", mine), _record("theirs", theirs, tenant_id="other")],
        )

        assert [i["text"] for i in await r._retrieve_graph(TENANT, _step(["u"]))] == ["mine"]
