"""Consolidation leaves its source episodes active; they must not crowd out the gist."""

from datetime import UTC, datetime, timedelta

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.reranker import MemoryReranker


def _memory(
    text: str,
    *,
    source: MemorySource = MemorySource.USER_EXPLICIT,
    mem_type: MemoryType = MemoryType.EPISODIC_EVENT,
    retrieval_source: str = "vector",
    key: str | None = None,
    metadata: dict | None = None,
    relevance: float = 0.8,
) -> RetrievedMemory:
    return RetrievedMemory(
        record=MemoryRecord(
            tenant_id="t",
            context_tags=[],
            type=mem_type,
            text=text,
            key=key,
            metadata=metadata or {},
            provenance=Provenance(source=source),
            timestamp=datetime.now(UTC) - timedelta(days=1),
            confidence=0.9,
        ),
        relevance_score=relevance,
        retrieval_source=retrieval_source,
    )


class TestGistDemotion:
    """Consolidation leaves source episodes active; they must not crowd out the gist."""

    @pytest.mark.asyncio
    async def test_episode_is_demoted_when_its_gist_is_present(self):
        gist = _memory(
            "prefers vegetarian food",
            mem_type=MemoryType.SEMANTIC_FACT,
            retrieval_source="facts",
            key="user:preference:diet",
            relevance=0.7,
        )
        episode = _memory(
            "I ordered the vegetarian platter",
            metadata={"consolidated": True, "consolidated_into_fact_key": "user:preference:diet"},
            relevance=0.9,
        )

        reranker = MemoryReranker()
        _, breakdown = await reranker.rerank_with_breakdown([episode, gist], "diet")

        by_text = {b["text"]: b for b in breakdown}
        assert "demoted_superseded_by_gist" in by_text["I ordered the vegetarian platter"]["notes"]
        # Higher raw relevance, yet the gist now outranks it.
        assert breakdown[0]["text"] == "prefers vegetarian food"

    @pytest.mark.asyncio
    async def test_no_demotion_when_the_gist_is_absent(self):
        """Detail surviving only in the episode must stay reachable."""
        episode = _memory(
            "I ordered the vegetarian platter",
            metadata={"consolidated": True, "consolidated_into_fact_key": "user:preference:diet"},
        )
        other = _memory("unrelated chatter", relevance=0.1)

        reranker = MemoryReranker()
        _, breakdown = await reranker.rerank_with_breakdown([episode, other], "diet")

        assert breakdown[0]["text"] == "I ordered the vegetarian platter"
        assert all("demoted_superseded_by_gist" not in b["notes"] for b in breakdown)

    @pytest.mark.asyncio
    async def test_unrelated_gist_does_not_demote(self):
        """The demotion is keyed on the specific fact the episode was folded into,
        not on the mere presence of any fact in the result set."""
        gist = _memory(
            "works as an engineer",
            mem_type=MemoryType.SEMANTIC_FACT,
            retrieval_source="facts",
            key="user:identity:occupation",
        )
        episode = _memory(
            "I ordered the vegetarian platter",
            metadata={"consolidated_into_fact_key": "user:preference:diet"},
        )

        reranker = MemoryReranker()
        _, breakdown = await reranker.rerank_with_breakdown([episode, gist], "q")
        assert all("demoted_superseded_by_gist" not in b["notes"] for b in breakdown)
