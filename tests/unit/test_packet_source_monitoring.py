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


class TestEpisodeDates:
    """When it was said and when it happened are different questions.

    event_date used to *replace* the turn date in the rendered bracket. That was
    invisible while nothing produced event dates, and cost 0.16 on the temporal
    category the moment the write path started producing them — the model lost the
    "when was this said" anchor, and the bracket silently meant one thing on some
    lines and another on others.
    """

    def _episode(self, text: str, said: datetime, event_date: str | None):
        mem = _memory(text)
        mem.record.timestamp = said
        if event_date:
            mem.record.metadata = {"event_date": event_date}
        return mem

    def test_turn_date_is_always_present(self):
        builder = MemoryPacketBuilder()
        said = datetime(2023, 1, 20, tzinfo=UTC)
        packet = builder.build(
            [self._episode("Lost my job yesterday.", said, "2023-01-19T00:00:00")], query="job"
        )
        rendered = builder.to_llm_context(packet)
        assert "said 2023-01-20" in rendered
        assert "refers to 2023-01-19" in rendered

    def test_no_event_date_renders_the_turn_date_alone(self):
        builder = MemoryPacketBuilder()
        said = datetime(2023, 1, 20, tzinfo=UTC)
        packet = builder.build([self._episode("A plain statement.", said, None)], query="q")
        rendered = builder.to_llm_context(packet)
        assert "[2023-01-20]" in rendered
        assert "refers to" not in rendered

    def test_event_date_equal_to_the_turn_date_is_not_repeated(self):
        builder = MemoryPacketBuilder()
        said = datetime(2023, 1, 20, tzinfo=UTC)
        packet = builder.build(
            [self._episode("It happened today.", said, "2023-01-20T00:00:00")], query="q"
        )
        rendered = builder.to_llm_context(packet)
        assert "[2023-01-20]" in rendered
        assert "refers to" not in rendered


def test_episode_threshold_defaults_agree():
    """Two defaults for one knob. They must not drift apart.

    Lowered 0.5 -> 0.4 on measured evidence: episode relevance runs p10 0.462 /
    median 0.555 on a real corpus, so 0.5 discarded ~25% of retrieved episodes and was
    what limited the section rather than max_episodes_default. Frozen-corpus A/B:
    overall 0.513 -> 0.536.
    """
    from src.core.config import RetrievalSettings
    from src.retrieval.packet_builder import EPISODE_RELEVANCE_THRESHOLD

    assert RetrievalSettings().episode_relevance_threshold == EPISODE_RELEVANCE_THRESHOLD
    assert EPISODE_RELEVANCE_THRESHOLD == 0.4
