"""Source monitoring: the packet must distinguish testimony from system output.

The write path generates content (prospective implications, consolidation gists,
revised beliefs) that the read path used to render identically to something the
user actually said.
"""

from datetime import UTC, datetime, timedelta

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.packet_builder import MemoryPacketBuilder


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


class TestSourceMonitoring:
    """A model reading the packet must be able to tell testimony from inference."""

    def test_inferred_episode_is_marked_and_user_statement_is_not(self):
        packet = MemoryPacketBuilder().build(
            [
                _memory("I have never been to Japan"),
                _memory(
                    "User may be planning a trip to Japan",
                    source=MemorySource.AGENT_INFERRED,
                ),
            ],
            query="japan",
        )
        rendered = MemoryPacketBuilder().to_llm_context(packet)

        assert "User may be planning a trip to Japan [inferred]" in rendered
        # The real statement stays unmarked — testimony is the default.
        assert "I have never been to Japan [inferred]" not in rendered
        assert "I have never been to Japan" in rendered

    def test_consolidation_and_revision_get_their_own_markers(self):
        builder = MemoryPacketBuilder()
        packet = builder.build(
            [
                _memory("Gist of many meals", source=MemorySource.CONSOLIDATION),
                _memory("Belief updated later", source=MemorySource.RECONSOLIDATION),
            ],
            query="food",
        )
        rendered = builder.to_llm_context(packet)
        assert "[consolidated]" in rendered
        assert "[revised]" in rendered

    def test_json_format_carries_the_same_signal(self):
        builder = MemoryPacketBuilder()
        packet = builder.build(
            [_memory("speculation", source=MemorySource.AGENT_INFERRED)],
            query="q",
        )
        rendered = builder.to_llm_context(packet, format="json")
        assert '"source": "inferred"' in rendered

    def test_fact_prong_is_not_labelled(self):
        """semantic_facts has no provenance column, so _fact_to_record stamps every
        row AGENT_INFERRED as a placeholder. Rendering that would label every fact
        identically — no signal, and it would weaken constraint wording on the
        strength of a value nobody recorded."""
        builder = MemoryPacketBuilder()
        packet = builder.build(
            [
                _memory(
                    "user:preference:cuisine: italian",
                    source=MemorySource.AGENT_INFERRED,
                    mem_type=MemoryType.SEMANTIC_FACT,
                    retrieval_source="facts",
                )
            ],
            query="food",
        )
        rendered = builder.to_llm_context(packet)
        assert "[inferred]" not in rendered

    def test_context_string_path_also_attributes(self):
        """to_context_string is a third renderer; it must not be the gap."""
        packet = MemoryPacketBuilder().build(
            [_memory("speculation", source=MemorySource.AGENT_INFERRED)],
            query="q",
        )
        assert "[inferred]" in packet.to_context_string()
