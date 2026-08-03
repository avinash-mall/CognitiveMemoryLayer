"""Event time and fact validity must be queryable, not just stored.

Two mirror-image write-only bugs, both closed here.

*Facts* have carried ``valid_from``, ``valid_to`` and ``is_current`` columns all along,
and reads already filter on them (``valid_to IS NULL OR valid_to >= now``). What no path
ever did was set an interval *from content*: ``valid_from`` was stamped with the write
time on every create and ``valid_to`` only on supersession, so "I was vegetarian until
June 2024" stored a fact valid from the moment we heard it, with an open end. The
storage layer was ready; the extractor was not.

*Episodes* had the opposite problem. Temporal resolution has extracted ``event_date`` at
scale since ``bb3a4a5``, but it lived only in the metadata JSON — never a column, never
indexed, read in exactly one place for rendering. The planner's time filter and
``vector_search`` both hit ``timestamp``, which is turn time. We extracted event time
correctly and then could not query by it.
"""

from datetime import UTC, datetime, timedelta

import pytest

from src.storage.postgres import _parse_event_date, _time_column


class TestEventDateParsing:
    """The column is populated from the metadata the extractor already writes, so the
    parser has to accept every shape observed in the store."""

    def test_a_date_only_value_parses(self):
        assert _parse_event_date({"event_date": "2023-10-24"}) == datetime(2023, 10, 24)

    def test_a_full_iso_value_parses(self):
        # The commonest shape in the store: 23,463 rows, all ISO-prefixed.
        assert _parse_event_date({"event_date": "2023-06-15T21:38:00"}) == datetime(
            2023, 6, 15, 21, 38
        )

    def test_a_datetime_value_is_normalised_not_rejected(self):
        got = _parse_event_date({"event_date": datetime(2023, 6, 15, tzinfo=UTC)})
        assert got == datetime(2023, 6, 15)
        assert got.tzinfo is None  # column is naive UTC

    @pytest.mark.parametrize(
        "meta",
        [None, {}, {"event_date": None}, {"event_date": ""}, {"event_date": "  "}],
    )
    def test_absent_is_none(self, meta):
        assert _parse_event_date(meta) is None

    def test_an_unparseable_value_does_not_raise(self):
        """A bad event_date must not fail the write — the turn text is the thing worth
        keeping, and the metadata copy survives either way."""
        assert _parse_event_date({"event_date": "last Tuesday"}) is None
        assert _parse_event_date({"event_date": "2023-13-45"}) is None


class TestWhichColumnTimeFiltersUse:
    def test_the_default_is_turn_time(self):
        """Every caller that predates event_date meant turn time, and one of them is
        load-bearing: temporal contiguity anchors a window on a seed's own
        record.timestamp to pull neighbouring turns. Resolving that window against event
        time would drop exactly the neighbours whose turn recalls a distant event."""
        assert "coalesce" not in str(_time_column(None)).lower()
        assert "coalesce" not in str(_time_column({"since": datetime.now(UTC)})).lower()

    def test_the_planner_basis_coalesces_to_event_time(self):
        sql = str(_time_column({"time_basis": "event"})).lower()
        assert "coalesce" in sql
        assert "event_date" in sql
        assert "timestamp" in sql

    def test_an_unknown_basis_falls_back_to_turn_time(self):
        assert "coalesce" not in str(_time_column({"time_basis": "wall"})).lower()

    def test_it_coalesces_rather_than_comparing_the_bare_column(self):
        """Only 4.3% of stored records (23,463 of 549,582) carry an event_date. A plain
        `event_date >= x` would hide the other 95.7% and read as a scoring collapse
        rather than a filter bug."""
        sql = str(_time_column({"time_basis": "event"})).lower()
        assert sql.index("event_date") < sql.index("memory_records.timestamp")


class TestThePlannerAsksForEventTime:
    def _filter(self, reference):
        from types import SimpleNamespace

        from src.retrieval.planner import RetrievalPlanner

        analysis = SimpleNamespace(time_reference=reference, user_timezone=None)
        return RetrievalPlanner()._build_time_filter(analysis)

    @pytest.mark.parametrize("ref", ["today", "yesterday", "last week", "last month", "recent"])
    def test_every_resolved_reference_is_tagged_event_time(self, ref):
        """A question's time reference is about when the event happened, not when the
        turn describing it was typed."""
        got = self._filter(ref)
        assert got is not None
        assert got["time_basis"] == "event"
        assert "since" in got

    def test_no_time_reference_still_yields_no_filter(self):
        assert self._filter(None) is None


class TestFactValidityFromContent:
    """``upsert_fact`` had no way to accept an end date; these pin the plumbing."""

    def test_extracted_facts_carry_an_interval(self):
        from src.extraction.write_time_facts import ExtractedFact
        from src.memory.neocortical.schemas import FactCategory

        fact = ExtractedFact(
            key="user:preference:diet",
            category=FactCategory.PREFERENCE,
            predicate="diet",
            value="vegetarian",
            confidence=0.9,
        )
        assert fact.valid_from is None
        assert fact.valid_to is None

    def test_the_extractor_parses_iso_bounds(self):
        from src.extraction.unified_write_extractor import _parse_iso_date

        assert _parse_iso_date("2024-06-30") == datetime(2024, 6, 30)
        assert _parse_iso_date("") is None
        assert _parse_iso_date(None) is None
        assert _parse_iso_date("June 2024") is None

    def test_the_prompt_asks_for_validity_bounds(self):
        """The columns and the reader existed; only the extraction did not, so the
        prompt is the actual fix and a silent revert of it is invisible."""
        from src.extraction.unified_write_extractor import _UNIFIED_PROMPT

        assert "valid_to" in _UNIFIED_PROMPT
        assert "valid_from" in _UNIFIED_PROMPT

    async def _create(self, valid_to):
        from unittest.mock import AsyncMock, MagicMock

        from src.memory.neocortical.fact_store import SemanticFactStore
        from src.memory.neocortical.schemas import FactCategory

        session = MagicMock()
        session.add = MagicMock()
        session.commit = AsyncMock()

        return await SemanticFactStore(session_factory=MagicMock())._create_fact(
            session,
            tenant_id="t",
            key="user:preference:diet",
            category=FactCategory.PREFERENCE,
            predicate="diet",
            value="vegetarian",
            confidence=0.9,
            evidence_ids=["ep1"],
            valid_from=datetime(2020, 1, 1),
            context_tags=[],
            valid_to=valid_to,
        )

    @pytest.mark.asyncio
    async def test_an_open_ended_fact_is_current(self):
        fact = await self._create(None)
        assert fact.valid_to is None
        assert fact.is_current is True

    @pytest.mark.asyncio
    async def test_a_fact_that_ends_in_the_future_is_still_current(self):
        end = datetime.now(UTC).replace(tzinfo=None) + timedelta(days=365)
        fact = await self._create(end)
        assert fact.valid_to == end
        assert fact.is_current is True

    @pytest.mark.asyncio
    async def test_a_fact_already_over_is_not_current_when_created(self):
        """ "I was vegetarian until June 2024" is history the moment it is written. The
        reader's valid_to filter would hide it, but is_current is what supersession and
        the fact prong key off, so both have to agree — a fact that is filtered out of
        reads while still flagged current would block its own successor."""
        end = datetime.now(UTC).replace(tzinfo=None) - timedelta(days=365)
        fact = await self._create(end)
        assert fact.valid_to == end
        assert fact.is_current is False
