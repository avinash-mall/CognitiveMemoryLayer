"""Edge ``evidence_ids`` must union across assertions, not be overwritten by the last one.

Relation edges carry ``evidence_ids`` — the episodic record IDs that asserted them — and
the graph prong resolves entity hits back to grounded text through exactly that field
(``_retrieve_graph``). Both writers set edge properties with ``r += $properties``, which
*replaces* the list on every re-assertion, so an edge stated across ten sessions pointed
at one episode: the most recent. The join key existed, was written, and silently lost
nine tenths of its value.

These tests pin the Cypher *shape* rather than a return value, because that is what
broke — the queries are strings and nothing else in the suite reads them. Verified
against live Neo4j 5.26 before being written down.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest

from src.storage.neo4j import _EVIDENCE_IDS_CAP, Neo4jGraphStore


class _Session:
    def __init__(self):
        self.queries: list[str] = []
        self.params: list[dict] = []

    async def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        result = MagicMock()
        result.data = AsyncMock(return_value=[{"edge_id": "e1"}])
        result.single = AsyncMock(return_value={"edge_id": "e1"})
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _store(session):
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return Neo4jGraphStore(driver)


async def _merge_one(session, properties):
    await _store(session).merge_edge("t", "t", "user", "lives_in", "Paris", properties)


async def _merge_batch(session, properties):
    await _store(session).merge_edges_batch(
        "t",
        "t",
        [{"subject": "user", "predicate": "lives_in", "object": "Paris", "properties": properties}],
    )


class TestEvidenceIdsAreNotInThePropertyMap:
    """``r += $properties`` is a whole-key overwrite. Keeping evidence_ids out of that
    map is what makes the union possible at all."""

    @pytest.mark.asyncio
    async def test_merge_edge_pulls_it_out(self):
        session = _Session()
        await _merge_one(session, {"evidence_ids": ["ep1"], "confidence": 0.8})

        assert "evidence_ids" not in session.params[0]["properties"]
        assert session.params[0]["evidence_ids"] == ["ep1"]

    @pytest.mark.asyncio
    async def test_merge_edges_batch_pulls_it_out(self):
        session = _Session()
        await _merge_batch(session, {"evidence_ids": ["ep1"], "confidence": 0.8})

        edge = session.params[0]["batch"][0]
        assert "evidence_ids" not in edge["properties"]
        assert edge["evidence_ids"] == ["ep1"]

    @pytest.mark.asyncio
    async def test_the_callers_dict_is_not_mutated(self):
        """merge_edge used to bind ``properties`` by reference and mutate it for the
        namespace key; popping evidence_ids out of a caller's dict would delete the
        field from the orchestrator's own edge payload on the way past."""
        session = _Session()
        props = {"evidence_ids": ["ep1"], "confidence": 0.8}
        await _merge_one(session, props)

        assert props["evidence_ids"] == ["ep1"]


class TestTheCypherUnions:
    @pytest.mark.asyncio
    @pytest.mark.parametrize("merge", [_merge_one, _merge_batch])
    async def test_prior_ids_are_carried_forward(self, merge):
        session = _Session()
        await merge(session, {"evidence_ids": ["ep1"]})
        query = session.queries[0]

        assert "coalesce(r.evidence_ids, [])" in query
        assert "SET r.evidence_ids" in query

    @pytest.mark.asyncio
    @pytest.mark.parametrize("merge", [_merge_one, _merge_batch])
    async def test_incoming_ids_are_deduplicated_against_the_existing_list(self, merge):
        """Without the filter, re-asserting the same edge from the same episode appends
        a duplicate every write."""
        session = _Session()
        await merge(session, {"evidence_ids": ["ep1"]})

        assert "WHERE NOT x IN" in session.queries[0]

    @pytest.mark.asyncio
    @pytest.mark.parametrize("merge", [_merge_one, _merge_batch])
    async def test_the_list_is_capped(self, merge):
        """ponytail: a hot edge re-asserted every session would grow without bound, and
        the reader truncates to step.top_k anyway."""
        session = _Session()
        await merge(session, {"evidence_ids": ["ep1"]})

        assert f"[-{_EVIDENCE_IDS_CAP}..]" in session.queries[0]


class TestEdgesThatNeverCarryEvidence:
    """The fact-sync path (``neocortical/store.py``) passes no properties at all. 118,835
    edges on the full2 corpus are NULL for this field and must stay NULL — writing ``[]``
    would make "never had evidence" indistinguishable from "had it, lost it"."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("merge", [_merge_one, _merge_batch])
    async def test_an_empty_write_is_a_no_op_branch(self, merge):
        session = _Session()
        await merge(session, {"confidence": 0.8})
        query = session.queries[0]

        assert "ELSE r.evidence_ids" in query
        assert "WHEN size(" in query

    @pytest.mark.asyncio
    async def test_merge_edge_passes_an_empty_list_not_none(self):
        """``size(null)`` is a Cypher type error, so the guard must see a list."""
        session = _Session()
        await _merge_one(session, {"confidence": 0.8})

        assert session.params[0]["evidence_ids"] == []

    @pytest.mark.asyncio
    async def test_merge_edges_batch_passes_an_empty_list_not_none(self):
        session = _Session()
        await _merge_batch(session, {"confidence": 0.8})

        assert session.params[0]["batch"][0]["evidence_ids"] == []
