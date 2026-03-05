"""Unit tests for semantic fact lineage APIs."""
from __future__ import annotations

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, MagicMock


class _FakeFactRow:
    """Minimal stand-in for SemanticFactModel rows."""

    def __init__(self, *, id, key, value, supersedes_id=None, tenant_id="t1",
                 is_current=True, version=1, created_at=None):
        self.id = id
        self.key = key
        self.value = value
        self.supersedes_id = supersedes_id
        self.tenant_id = tenant_id
        self.is_current = is_current
        self.version = version
        self.created_at = created_at or datetime.now(UTC)


def _mock_session_factory(session):
    """Wrap a mock session so factory() returns an async context manager."""
    ctx = AsyncMock()
    ctx.__aenter__.return_value = session
    ctx.__aexit__.return_value = False
    return MagicMock(return_value=ctx)


class TestGetFactLineage:
    def test_returns_empty_for_no_key_or_id(self):
        """Should return empty list if neither key nor fact_id provided."""
        from src.memory.neocortical.fact_store import SemanticFactStore

        store = MagicMock(spec=SemanticFactStore)
        store.get_fact_lineage = SemanticFactStore.get_fact_lineage.__get__(store)
        result = asyncio.get_event_loop().run_until_complete(
            store.get_fact_lineage("tenant1")
        )
        assert result == []

    def test_single_fact_returns_single_entry(self):
        """A fact with no supersedes_id should return just itself."""
        fact = _FakeFactRow(id="f1", key="user:pref:food", value="pizza", supersedes_id=None)

        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = fact

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        store = MagicMock()
        store.session_factory = _mock_session_factory(mock_session)

        from src.memory.neocortical.fact_store import SemanticFactStore

        store.get_fact_lineage = SemanticFactStore.get_fact_lineage.__get__(store)

        result = asyncio.get_event_loop().run_until_complete(
            store.get_fact_lineage("t1", key="user:pref:food")
        )
        assert len(result) == 1
        assert result[0]["fact_id"] == "f1"
        assert result[0]["key"] == "user:pref:food"

    def test_multi_step_chain_returns_oldest_first(self):
        """A chain of A->B->C should return [A, B, C] (oldest first)."""
        fact_a = _FakeFactRow(id="a", key="k", value="v1", supersedes_id=None, version=1)
        fact_b = _FakeFactRow(id="b", key="k", value="v2", supersedes_id="a", version=2)
        fact_c = _FakeFactRow(id="c", key="k", value="v3", supersedes_id="b", version=3, is_current=True)

        call_count = [0]

        def make_result(fact):
            r = MagicMock()
            r.scalar_one_or_none.return_value = fact
            return r

        async def mock_execute(query):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_result(fact_c)
            elif call_count[0] == 2:
                return make_result(fact_b)
            elif call_count[0] == 3:
                return make_result(fact_a)
            return make_result(None)

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        store = MagicMock()
        store.session_factory = _mock_session_factory(mock_session)

        from src.memory.neocortical.fact_store import SemanticFactStore

        store.get_fact_lineage = SemanticFactStore.get_fact_lineage.__get__(store)

        result = asyncio.get_event_loop().run_until_complete(
            store.get_fact_lineage("t1", key="k")
        )
        assert len(result) == 3
        assert result[0]["fact_id"] == "a"
        assert result[1]["fact_id"] == "b"
        assert result[2]["fact_id"] == "c"


class TestGetSupersededChain:
    def test_returns_empty_when_no_successor(self):
        """A fact with no successor should return empty chain."""
        mock_result = MagicMock()
        mock_result.scalar_one_or_none.return_value = None

        mock_session = AsyncMock()
        mock_session.execute = AsyncMock(return_value=mock_result)

        store = MagicMock()
        store.session_factory = _mock_session_factory(mock_session)

        from src.memory.neocortical.fact_store import SemanticFactStore

        store.get_superseded_chain = SemanticFactStore.get_superseded_chain.__get__(store)

        result = asyncio.get_event_loop().run_until_complete(
            store.get_superseded_chain("t1", "f1")
        )
        assert result == []

    def test_forward_chain_returns_successors(self):
        """Walking forward from A should find B then C."""
        fact_b = _FakeFactRow(id="b", key="k", value="v2", supersedes_id="a")
        fact_c = _FakeFactRow(id="c", key="k", value="v3", supersedes_id="b")

        call_count = [0]

        def make_result(fact):
            r = MagicMock()
            r.scalar_one_or_none.return_value = fact
            return r

        async def mock_execute(query):
            call_count[0] += 1
            if call_count[0] == 1:
                return make_result(fact_b)
            elif call_count[0] == 2:
                return make_result(fact_c)
            return make_result(None)

        mock_session = AsyncMock()
        mock_session.execute = mock_execute

        store = MagicMock()
        store.session_factory = _mock_session_factory(mock_session)

        from src.memory.neocortical.fact_store import SemanticFactStore

        store.get_superseded_chain = SemanticFactStore.get_superseded_chain.__get__(store)

        result = asyncio.get_event_loop().run_until_complete(
            store.get_superseded_chain("t1", "a")
        )
        assert len(result) == 2
        assert result[0]["fact_id"] == "b"
        assert result[1]["fact_id"] == "c"
