"""Retrieving one turn should cue the turns encoded around it.

Human recall shows a temporal contiguity effect — retrieving an item preferentially cues
items encoded at nearby positions, and reinstating the encoding context recovers items
otherwise scored as forgotten. In a conversation log "adjacent in encoding" is exact, so
this costs an ordered timestamp lookup rather than new structure.

The failure it targets: a query matches the turn that *asks* something while the answer
sits in the next turn, or matches a reply whose referent is the turn before. Similarity
alone retrieves the match and drops the answer.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from src.core.config import get_settings
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance
from src.retrieval.retriever import _CONTIGUITY_RELEVANCE_FACTOR, HybridRetriever

TENANT = "t"
T0 = datetime(2026, 5, 1, 12, 0, tzinfo=UTC)


@pytest.fixture(autouse=True)
def _clear_settings_cache():
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()


def _record(
    text: str,
    *,
    offset_min: int = 0,
    session: str | None = "s1",
    turn_idx: int | None = None,
) -> MemoryRecord:
    return MemoryRecord(
        id=uuid4(),
        tenant_id=TENANT,
        context_tags=[],
        source_session_id=session,
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
        timestamp=T0 + timedelta(minutes=offset_min),
        metadata={} if turn_idx is None else {"turn_idx": turn_idx},
    )


def _hit(record: MemoryRecord, relevance: float = 0.8, source: str = "vector") -> dict:
    return {
        "type": MemoryType.EPISODIC_EVENT.value,
        "source": source,
        "text": record.text,
        "relevance": relevance,
        "record": record,
    }


def _retriever(neighbours: list[MemoryRecord]) -> HybridRetriever:
    r = HybridRetriever.__new__(HybridRetriever)
    store = MagicMock()
    store.scan = AsyncMock(return_value=neighbours)
    r.hippocampal = MagicMock()
    r.hippocampal.store = store
    return r


class TestExpansion:
    @pytest.mark.asyncio
    async def test_neighbouring_turns_join_the_candidate_set(self):
        seed = _record("Did you ever finish the painting?", offset_min=0)
        answer = _record("Yes, I finished it last Tuesday.", offset_min=1)
        r = _retriever([seed, answer])
        results = [_hit(seed)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert [x["text"] for x in results] == [seed.text, answer.text]
        assert results[1]["source"] == "contiguity"

    @pytest.mark.asyncio
    async def test_a_neighbour_never_outranks_its_seed(self):
        """Context, not a competitor. A neighbour that outranked the episode which
        actually matched the query would be the graph-profile mistake in a new costume."""
        seed = _record("seed", offset_min=0)
        neighbour = _record("neighbour", offset_min=1)
        r = _retriever([seed, neighbour])
        results = [_hit(seed, relevance=0.8)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert results[1]["relevance"] == pytest.approx(0.8 * _CONTIGUITY_RELEVANCE_FACTOR)
        assert results[1]["relevance"] < results[0]["relevance"]

    @pytest.mark.asyncio
    async def test_the_seed_itself_is_not_duplicated(self):
        seed = _record("seed", offset_min=0)
        r = _retriever([seed])
        results = [_hit(seed)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert len(results) == 1

    @pytest.mark.asyncio
    async def test_the_scan_is_bounded_to_the_seeds_session_and_window(self):
        seed = _record("seed", offset_min=0)
        r = _retriever([seed])

        await r._expand_temporal_contiguity(TENANT, [_hit(seed)])

        filters = r.hippocampal.store.scan.await_args.kwargs["filters"]
        assert filters["source_session_id"] == "s1"
        assert filters["status"] == "active"
        assert filters["since"] < seed.timestamp < filters["until"]


class TestEncodingOrder:
    """The corpus this targets does not order by timestamp.

    Every turn of a LoCoMo session carries the *identical* timestamp — 28 turns, span
    0.000000s — so ordering by it is arbitrary and "neighbours" would be a random sample
    of the session. `written_at` is no better: concurrent ingestion workers scramble it
    (turn order 0, 4, 3, 6, 7, 5 observed on a real session). The explicit `turn_idx` the
    writer supplies is the only faithful encoding order.
    """

    @pytest.mark.asyncio
    async def test_neighbours_are_by_turn_index_when_timestamps_are_identical(self):
        turns = [_record(f"turn{i}", offset_min=0, turn_idx=i) for i in range(9)]
        seed = turns[4]
        # Handed back in scrambled order, exactly as the DB returns them.
        r = _retriever([turns[i] for i in (7, 0, 4, 2, 8, 1, 6, 3, 5)])
        results = [_hit(seed)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert sorted(x["text"] for x in results[1:]) == ["turn2", "turn3", "turn5", "turn6"]

    @pytest.mark.asyncio
    async def test_the_seed_is_excluded_from_its_own_neighbourhood(self):
        turns = [_record(f"turn{i}", offset_min=0, turn_idx=i) for i in range(5)]
        r = _retriever(turns)
        results = [_hit(turns[2])]

        await r._expand_temporal_contiguity(TENANT, results)

        assert "turn2" not in [x["text"] for x in results[1:]]

    @pytest.mark.asyncio
    async def test_a_seed_at_the_edge_of_a_session_is_not_an_error(self):
        turns = [_record(f"turn{i}", offset_min=0, turn_idx=i) for i in range(5)]
        r = _retriever(turns)
        results = [_hit(turns[0])]

        await r._expand_temporal_contiguity(TENANT, results)

        assert sorted(x["text"] for x in results[1:]) == ["turn1", "turn2"]

    @pytest.mark.asyncio
    async def test_timestamps_are_the_fallback_when_no_turn_index_exists(self):
        turns = [_record(f"t{i}", offset_min=i) for i in range(5)]
        r = _retriever([turns[3], turns[0], turns[2], turns[4], turns[1]])
        results = [_hit(turns[2])]

        await r._expand_temporal_contiguity(TENANT, results)

        assert sorted(x["text"] for x in results[1:]) == ["t0", "t1", "t3", "t4"]

    @pytest.mark.asyncio
    async def test_a_partial_turn_index_population_falls_back_rather_than_interleaving(self):
        """Two incomparable ordering keys in one session is worse than the weaker key."""
        a = _record("a", offset_min=0, turn_idx=5)
        b = _record("b", offset_min=1)
        c = _record("c", offset_min=2, turn_idx=0)
        r = _retriever([a, b, c])
        results = [_hit(b)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert sorted(x["text"] for x in results[1:]) == ["a", "c"]


class TestBounds:
    @pytest.mark.asyncio
    async def test_only_the_strongest_seeds_are_expanded(self):
        """One scan per seed, so seed count is what bounds the work."""
        records = [_record(f"r{i}", offset_min=i) for i in range(10)]
        r = _retriever([])
        results = [_hit(rec, relevance=0.5 + i / 100) for i, rec in enumerate(records)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert r.hippocampal.store.scan.await_count == 3

    @pytest.mark.asyncio
    async def test_non_vector_hits_are_not_seeds(self):
        """Graph hits already resolve to episodes and facts have no turn neighbourhood;
        expanding them would multiply prongs against each other."""
        r = _retriever([])
        await r._expand_temporal_contiguity(TENANT, [_hit(_record("g"), source="graph")])
        r.hippocampal.store.scan.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_record_without_a_session_is_skipped(self):
        r = _retriever([])
        await r._expand_temporal_contiguity(TENANT, [_hit(_record("x", session=None))])
        r.hippocampal.store.scan.assert_not_awaited()

    @pytest.mark.asyncio
    async def test_a_scan_failure_does_not_take_retrieval_down(self):
        seed = _record("seed")
        r = _retriever([])
        r.hippocampal.store.scan = AsyncMock(side_effect=RuntimeError("db gone"))
        results = [_hit(seed)]

        await r._expand_temporal_contiguity(TENANT, results)

        assert [x["text"] for x in results] == ["seed"]

    @pytest.mark.asyncio
    async def test_disabled_is_a_no_op(self, monkeypatch):
        monkeypatch.setenv("FEATURES__TEMPORAL_CONTIGUITY_ENABLED", "false")
        get_settings.cache_clear()
        r = _retriever([_record("neighbour", offset_min=1)])
        results = [_hit(_record("seed"))]

        await r._expand_temporal_contiguity(TENANT, results)

        assert len(results) == 1
        r.hippocampal.store.scan.assert_not_awaited()
