"""Unit tests for retrieval-time constraint conflict handling in packet builder."""

from datetime import UTC, datetime, timedelta
from uuid import uuid4

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecord, Provenance, RetrievedMemory
from src.retrieval.packet_builder import MemoryPacketBuilder


def _constraint_mem(text: str, key: str, age_days: int) -> RetrievedMemory:
    ts = datetime.now(UTC) - timedelta(days=age_days)
    record = MemoryRecord(
        id=uuid4(),
        tenant_id="t1",
        type=MemoryType.CONSTRAINT,
        text=text,
        key=key,
        timestamp=ts,
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )
    return RetrievedMemory(record=record, relevance_score=0.8, retrieval_source="constraints")


def test_packet_builder_suppresses_older_constraint_for_same_key():
    builder = MemoryPacketBuilder()
    older = _constraint_mem("I never eat shellfish", "user:policy:diet", age_days=10)
    newer = _constraint_mem("I now eat fish occasionally", "user:policy:diet", age_days=1)

    packet = builder.build([older, newer], query="what should I eat?")

    assert len(packet.constraints) == 1
    assert packet.constraints[0].record.text == newer.record.text
    assert any("Suppressed older constraint" in w for w in packet.warnings)
