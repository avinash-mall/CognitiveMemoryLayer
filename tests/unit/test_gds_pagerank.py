"""The GDS path must project a named graph, run, and drop it.

This code shipped for a long time calling ``gds.pageRank.stream({nodeQuery: ...,
relationshipQuery: ...})`` — GDS 1.x anonymous projection, removed in GDS 2.0. Against a
real GDS 2.13 server that fails with "Type mismatch: expected String but was Map", and
the except-branch quietly swapped in the path-count fallback. So the system reported
running Personalized PageRank while never once running it, on any deployment, whether or
not the plugin was installed.

The failure was invisible because the fallback returns plausible results. These tests
pin the call *shape* rather than the output, because shape is what broke.
"""

from unittest.mock import AsyncMock, MagicMock

import pytest
from neo4j.exceptions import ClientError as Neo4jClientError

from src.storage.neo4j import Neo4jGraphStore


class _Session:
    def __init__(self, *, fail_on_project: bool = False):
        self.queries: list[str] = []
        self.params: list[dict] = []
        self.fail_on_project = fail_on_project

    async def run(self, query, **params):
        self.queries.append(query)
        self.params.append(params)
        if self.fail_on_project and "gds.graph.project" in query:
            raise Neo4jClientError("no procedure gds.graph.project.cypher")
        result = MagicMock()
        result.data = AsyncMock(return_value=[{"entity": "Gina", "score": 0.15}])
        return result

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        return False


def _store(session):
    driver = MagicMock()
    driver.session = MagicMock(return_value=session)
    return Neo4jGraphStore(driver)


class TestGdsPath:
    @pytest.mark.asyncio
    async def test_it_projects_then_streams_then_drops(self):
        session = _Session()
        await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        kinds = [
            "project" if "gds.graph.project" in q else "stream" if "pageRank" in q else "drop"
            for q in session.queries
        ]
        assert kinds == ["project", "stream", "drop"]

    @pytest.mark.asyncio
    async def test_the_stream_call_passes_a_graph_name_not_a_projection_map(self):
        """The exact regression: a map here is a GDS 1.x call that 2.x rejects."""
        session = _Session()
        await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        stream = next(q for q in session.queries if "pageRank" in q)
        assert "gds.pageRank.stream($graph_name" in stream
        assert "nodeQuery" not in stream
        assert "relationshipQuery" not in stream

    @pytest.mark.asyncio
    async def test_the_projection_is_dropped_even_when_the_stream_fails(self):
        """A leaked projection pins its nodes in heap for the life of the database, and
        one per query would be a slow-motion outage."""
        session = _Session()
        original = session.run

        async def run(query, **params):
            if "pageRank" in query:
                raise RuntimeError("algo blew up")
            return await original(query, **params)

        session.run = run
        with pytest.raises(RuntimeError):
            await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        assert any("gds.graph.drop" in q for q in session.queries)

    @pytest.mark.asyncio
    async def test_each_call_uses_a_distinct_projection_name(self):
        """Concurrent reads would otherwise collide on one shared graph name — the
        second projection fails, or the first is dropped out from under the second."""
        names = []
        for _ in range(3):
            session = _Session()
            await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])
            names.append(session.params[0]["graph_name"])

        assert len(set(names)) == 3

    @pytest.mark.asyncio
    async def test_the_same_name_is_projected_streamed_and_dropped(self):
        """Dropping a different name than was projected leaks one graph per query."""
        session = _Session()
        await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        used = {p["graph_name"] for p in session.params if "graph_name" in p}
        assert len(used) == 1

    @pytest.mark.asyncio
    async def test_tenant_scoping_is_parameterised_not_interpolated(self):
        """The projection queries are strings handed to Cypher; a tenant id spliced into
        them would be an injection point on a multi-tenant boundary."""
        session = _Session()
        await _store(session).personalized_pagerank("t' OR 1=1 --", "t", seed_entities=["Gina"])

        project = next(q for q in session.queries if "gds.graph.project" in q)
        assert "OR 1=1" not in project
        assert session.params[0]["tenant_id"] == "t' OR 1=1 --"


class TestFallback:
    @pytest.mark.asyncio
    async def test_a_missing_plugin_falls_back_to_the_path_count(self):
        """Deployments without GDS must keep working — the fallback is a proximity
        score, not PageRank, which is why its output is unbounded."""
        session = _Session(fail_on_project=True)
        out = await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        assert out == [{"entity": "Gina", "score": 0.15}]
        assert any("min_distance" in q or "path_count" in q for q in session.queries)

    @pytest.mark.asyncio
    async def test_the_fallback_is_bounded_to_two_hops(self):
        """Depth 3 reached 504 entities against depth 2's 502 on a real tenant while
        counting ~63x as many paths, which blew the prong's 2s step budget."""
        session = _Session(fail_on_project=True)
        await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        fallback = next(q for q in session.queries if "min_distance" in q)
        assert "[*1..2]" in fallback
