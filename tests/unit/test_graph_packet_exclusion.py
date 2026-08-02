"""Graph results can be kept out of the packet, and the slots must go to episodes.

``multi_hop_query`` has no hop loop — it returns entity profiles that summarise a
neighbourhood rather than answering anything. On the LoCoMo-Plus subset every arm with a
populated graph scored below the empty-graph baseline on multi-hop (0.33 -> 0.23/0.27/
0.26), so being able to drop the prong from the packet is the cheap alternative to
building iterative retrieval.

The exclusion runs before the rerank cap. Filtering afterwards would leave the slots a
graph hit consumed already spent, which is the whole thing this avoids.
"""

from datetime import UTC, datetime

from src.core.config import FeatureFlags
from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.memory_retriever import drop_graph_results


def _mem(text: str, source: str) -> RetrievedMemory:
    return RetrievedMemory(
        record=MemoryRecord(
            tenant_id="t",
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text=text,
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
            timestamp=datetime.now(UTC),
        ),
        relevance_score=0.5,
        retrieval_source=source,
    )


def _candidates() -> list[RetrievedMemory]:
    return [
        _mem("Entity: user\n  - LOCATION: Seattle", "graph"),
        _mem("Entity: Tim\n  - WORKS_AT: a company", "graph"),
        _mem("Jon: I lost my job yesterday.", "vector"),
        _mem("user:preference:diet: vegetarian", "facts"),
        _mem("I never eat shellfish.", "constraints"),
    ]


class TestDropGraphResults:
    def test_disabled_is_a_passthrough(self):
        candidates = _candidates()
        assert drop_graph_results(candidates, enabled=False) == candidates

    def test_enabled_removes_only_the_graph_prong(self):
        kept = drop_graph_results(_candidates(), enabled=True)
        assert {m.retrieval_source for m in kept} == {"vector", "facts", "constraints"}
        assert not any(m.record.text.startswith("Entity:") for m in kept)

    def test_relative_order_of_survivors_is_preserved(self):
        """The reranker sorts afterwards, but a filter that shuffles would make any
        upstream ordering decision untestable."""
        kept = drop_graph_results(_candidates(), enabled=True)
        assert [m.record.text for m in kept] == [
            "Jon: I lost my job yesterday.",
            "user:preference:diet: vegetarian",
            "I never eat shellfish.",
        ]

    def test_all_graph_input_yields_an_empty_candidate_set(self):
        """Retrieval returning nothing is a valid outcome, not an error — the packet
        then carries facts and constraints from other prongs, or nothing."""
        graph_only = [_mem("Entity: a", "graph"), _mem("Entity: b", "graph")]
        assert drop_graph_results(graph_only, enabled=True) == []

    def test_empty_input(self):
        assert drop_graph_results([], enabled=True) == []


class TestFlagDefault:
    def test_graph_results_are_included_by_default(self):
        """On by measurement, and it took three arms to earn it.

        1. Prong emitting entity neighbourhood profiles — excluding it was worth
           0.480 -> 0.513 on a frozen corpus, so it shipped off.
        2. Prong resolving entities to episodic text, ranked by the traversal score:
           **0.4292 against a 0.4860 baseline**, every factual category down and
           adversarial *up* +0.074. Resolution was right; ranking was the bug. The
           traversal score normalises into a constant 0.55-0.85 band sitting above the
           median vector cosine (~0.62), so graph candidates displaced better-matching
           episodes regardless of the question.
        3. Same candidates ranked by cosine against the query: **0.4979, +0.012 over
           baseline**, measured in isolation with contiguity held off. Cognitive +0.020,
           single-hop +0.016, multi-hop +0.016, and adversarial +0.011 — no refusal
           trade at all, because the added content is relevant rather than merely
           well-ranked.

        The lesson worth keeping: the graph is a recall mechanism. Letting it also set
        relevance is what made it harmful both times.
        """
        assert FeatureFlags().graph_results_in_packet is True
