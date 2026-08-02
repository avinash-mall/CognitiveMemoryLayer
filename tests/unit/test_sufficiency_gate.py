"""A memory layer should be able to say it found nothing.

Relevance ranking answers "which of these is best", which is defined even when every
candidate is junk. Sufficiency answers "is there anything here at all", which is what a
caller needs in order to decline. Nothing in the system computed the second one: every
threshold silently shrank the packet instead of signalling.

Delegating the judgement to the answering model does not work — abstention does not
improve with model scale, and reasoning fine-tuning degrades it by ~24% — so the memory
layer scoring it from its own retrieval distribution is the more trustworthy signal.
"""

from datetime import UTC, datetime

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryPacket, MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.memory_retriever import (
    SUFFICIENCY_STRENGTH_FLOOR,
    assess_sufficiency,
)
from src.retrieval.packet_builder import MemoryPacketBuilder

WEAK_NOTE = "No strongly matching memory was found"


def _mem(score: float) -> RetrievedMemory:
    return RetrievedMemory(
        record=MemoryRecord(
            tenant_id="t",
            context_tags=[],
            type=MemoryType.EPISODIC_EVENT,
            text=f"memory at {score}",
            provenance=Provenance(source=MemorySource.USER_EXPLICIT),
            timestamp=datetime.now(UTC),
        ),
        relevance_score=score,
        retrieval_source="vector",
    )


class TestAssessSufficiency:
    def test_a_strong_match_is_sufficient(self):
        assert assess_sufficiency([_mem(0.9), _mem(0.3)])["sufficient"] is True

    def test_a_flat_field_of_weak_matches_is_not(self):
        """The ranking is perfectly well-defined here and completely uninformative —
        this is the case that produces a confident answer to an unanswerable question."""
        result = assess_sufficiency([_mem(0.2), _mem(0.19), _mem(0.18)])

        assert result["sufficient"] is False
        assert result["retrieved"] == 3
        assert result["supporting_memories"] == 0

    def test_retrieving_nothing_is_not_an_error(self):
        result = assess_sufficiency([])

        assert result["sufficient"] is False
        assert result["retrieved"] == 0
        assert result["evidence_strength"] == 0.0

    def test_strength_is_the_best_single_piece_of_evidence(self):
        """One direct hit is enough to answer from; averaging would let a long tail of
        weak neighbours veto it, and contiguity expansion adds exactly such a tail."""
        assert assess_sufficiency([_mem(0.9)] + [_mem(0.1)] * 20)["sufficient"] is True

    def test_the_boundary_is_inclusive(self):
        assert assess_sufficiency([_mem(SUFFICIENCY_STRENGTH_FLOOR)])["sufficient"] is True


class TestItReachesTheReader:
    def _context(self, packet: MemoryPacket) -> str:
        return MemoryPacketBuilder().to_llm_context(packet, max_tokens=3000)

    def test_a_weak_packet_is_labelled_in_the_rendered_context(self):
        packet = MemoryPacket(query="q", recent_episodes=[_mem(0.2)])
        packet.sufficiency = assess_sufficiency([_mem(0.2)])

        assert WEAK_NOTE in self._context(packet)

    def test_a_strong_packet_is_not_labelled(self):
        """Nudging toward refusal on every query would trade one failure for another."""
        packet = MemoryPacket(query="q", recent_episodes=[_mem(0.9)])
        packet.sufficiency = assess_sufficiency([_mem(0.9)])

        assert WEAK_NOTE not in self._context(packet)

    def test_a_packet_without_the_signal_renders_unchanged(self):
        """`sufficiency` is optional on the schema, so every other producer of a packet
        must keep working untouched."""
        packet = MemoryPacket(query="q", recent_episodes=[_mem(0.9)])

        assert WEAK_NOTE not in self._context(packet)

    @pytest.mark.parametrize("field", ["open_questions", "warnings", "sufficiency"])
    def test_the_response_contract_carries_the_signal(self, field):
        """All three were computed and reached no caller before this."""
        from cml_contracts.models import ReadMemoryResponse

        assert field in ReadMemoryResponse.model_fields
