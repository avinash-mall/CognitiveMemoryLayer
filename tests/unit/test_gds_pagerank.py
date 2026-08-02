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


@pytest.fixture(autouse=True)
def _gds_on(monkeypatch):
    """These tests describe the GDS path, which is off by default — see
    TestPageRankIsOffByDefault for why."""
    from src.core.config import get_settings

    monkeypatch.setenv("FEATURES__GRAPH_PAGERANK_ENABLED", "true")
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


class TestPageRankIsOffByDefault:
    """Measured, not assumed, and not a fear of the plugin.

    Full 2,387-sample frozen corpus with GDS 2.13.4 live: PPR scored **0.5031** against
    the traversal fallback's **0.5046** — 0.0015, under 4 samples, noise — while read
    latency went from ~174ms to ~700ms and the first call per worker took 2554ms, blowing
    the 2s step budget. Four to five times the cost for nothing measurable.

    Which algorithm runs used to be decided implicitly by whether a plugin happened to be
    installed. That is the same silent-behaviour class that let a GDS 1.x call survive
    here undetected, so it is now an explicit flag.
    """

    def test_the_flag_defaults_off(self):
        from src.core.config import FeatureFlags

        assert FeatureFlags().graph_pagerank_enabled is False

    @pytest.mark.asyncio
    async def test_disabled_goes_straight_to_the_traversal(self, monkeypatch):
        from src.core.config import get_settings

        monkeypatch.setenv("FEATURES__GRAPH_PAGERANK_ENABLED", "false")
        get_settings.cache_clear()
        session = _Session()

        out = await _store(session).personalized_pagerank("t", "t", seed_entities=["Gina"])

        assert out == [{"entity": "Gina", "score": 0.15}]
        assert not any("gds." in q for q in session.queries)
        assert any("min_distance" in q for q in session.queries)
        get_settings.cache_clear()


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


def looks_like_pagerank(scores: list[float]) -> bool:
    """Discriminate real PPR output from the path-count fallback, from scores alone.

    The fallback computes ``1.0 / (min_distance + 1) * path_count`` with an integer
    ``path_count`` and ``min_distance`` in {1, 2} — so every fallback score is a multiple
    of 1/2 or 1/3, and multiplying by 6 always yields an integer. PageRank mass is
    normalised and lands on arbitrary floats below 1.

    Sharper than "bounded vs unbounded": a small graph can produce fallback scores under
    1.0 too (path_count=1, min_distance=2 gives 0.333), so a magnitude check alone would
    pass the fallback off as PageRank on exactly the small tenants this system has.
    """
    if not scores:
        return False
    if max(scores) >= 1.0:
        return False
    return not all(abs(s * 6 - round(s * 6)) < 1e-9 for s in scores)


class TestScoreShapeDiscriminates:
    """The assertion whose absence let a broken GDS call survive here indefinitely.

    Both paths return a ranked entity list that looks entirely reasonable, so nothing in
    the output *shape* said which one ran. These are the observed values from each.
    """

    def test_real_ppr_from_the_live_graph_is_recognised(self):
        # Measured through the real code path against GDS 2.13.4 on the full2-199 tenant.
        assert looks_like_pagerank([0.16675, 0.16045, 0.01125, 0.00862, 0.00582])

    def test_fallback_scores_are_recognised_even_when_small(self):
        """A sparse tenant makes the fallback emit values under 1.0, so a magnitude
        check alone would wave it through. `min_distance` is in {1, 2} because the
        traversal is `[*1..2]`, so the only possible denominators are 2 and 3:
        1/(1+1)*1 = 0.5, 1/(2+1)*1 = 0.333…, 1/(2+1)*2 = 0.666…
        """
        assert not looks_like_pagerank([0.5, 1 / 3, 2 / 3])

    def test_a_large_fallback_score_is_rejected_on_magnitude(self):
        """315 and 744 were both observed on one real query."""
        assert not looks_like_pagerank([744.0, 315.0, 265.0])

    def test_no_scores_is_not_mistaken_for_pagerank(self):
        assert not looks_like_pagerank([])
