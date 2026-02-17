"""Unit tests for hippocampal write gate and PII redactor."""

from src.core.enums import MemoryType
from src.memory.hippocampal.redactor import PIIRedactor
from src.memory.hippocampal.write_gate import (
    WriteDecision,
    WriteGate,
    WriteGateConfig,
)
from src.memory.working.models import ChunkType, SemanticChunk


def _chunk(
    chunk_type: ChunkType, text: str = "Some content.", salience: float = 0.8
) -> SemanticChunk:
    return SemanticChunk(id="1", text=text, chunk_type=chunk_type, salience=salience)


class TestWriteGate:
    def test_skip_low_importance(self):
        gate = WriteGate(WriteGateConfig(min_importance=0.5))
        chunk = SemanticChunk(
            id="1",
            text="Just saying hello.",
            chunk_type=ChunkType.STATEMENT,
            salience=0.1,
        )
        result = gate.evaluate(chunk)
        assert result.decision == WriteDecision.SKIP
        assert "threshold" in result.reason.lower()

    def test_store_sync_high_importance(self):
        gate = WriteGate(WriteGateConfig(sync_importance_threshold=0.5))
        chunk = SemanticChunk(
            id="2",
            text="My name is Alice and I live in Paris.",
            chunk_type=ChunkType.FACT,
            salience=0.8,
        )
        result = gate.evaluate(chunk)
        assert result.decision in (
            WriteDecision.STORE,
            WriteDecision.STORE_SYNC,
            WriteDecision.REDACT_AND_STORE,
        )
        assert result.memory_types
        assert result.importance >= 0.5

    def test_skip_secrets(self):
        gate = WriteGate()
        chunk = SemanticChunk(
            id="3",
            text="My api_key=sk-secret123 do not share.",
            chunk_type=ChunkType.STATEMENT,
            salience=0.9,
        )
        result = gate.evaluate(chunk)
        assert result.decision == WriteDecision.SKIP
        assert "contains_secrets" in result.risk_flags

    def test_novelty_low_when_duplicate(self):
        gate = WriteGate()
        chunk = SemanticChunk(
            id="4",
            text="I prefer coffee.",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.8,
        )
        existing = [{"text": "I prefer coffee."}]
        result = gate.evaluate(chunk, existing_memories=existing)
        assert result.novelty == 0.0

    def test_pii_triggers_redaction(self):
        gate = WriteGate()
        chunk = SemanticChunk(
            id="5",
            text="My email is test@example.com",
            chunk_type=ChunkType.FACT,
            salience=0.8,
        )
        result = gate.evaluate(chunk)
        assert result.redaction_required is True
        assert "contains_pii" in result.risk_flags


class TestWriteGateChunkTypeToMemoryType:
    """Flow: each ChunkType maps to expected MemoryType(s); decision store/skip as expected."""

    def test_preference_maps_to_preference(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.PREFERENCE, "I prefer dark mode."))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.PREFERENCE in result.memory_types

    def test_constraint_maps_to_constraint(self):
        gate = WriteGate()
        result = gate.evaluate(
            _chunk(ChunkType.CONSTRAINT, "I never want to be reminded on weekends.")
        )
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.CONSTRAINT in result.memory_types

    def test_fact_maps_to_semantic_fact_and_episodic(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.FACT, "My name is Alice."))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.SEMANTIC_FACT in result.memory_types
        assert MemoryType.EPISODIC_EVENT in result.memory_types

    def test_event_maps_to_episodic_event(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.EVENT, "We met last Tuesday."))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.EPISODIC_EVENT in result.memory_types

    def test_statement_maps_to_episodic_event(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.STATEMENT, "The project is on track."))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.EPISODIC_EVENT in result.memory_types

    def test_instruction_maps_to_task_state(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.INSTRUCTION, "Please send the report by Friday."))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.TASK_STATE in result.memory_types

    def test_question_maps_to_episodic_event(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.QUESTION, "What time is the meeting?"))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.EPISODIC_EVENT in result.memory_types

    def test_opinion_maps_to_hypothesis(self):
        gate = WriteGate()
        result = gate.evaluate(_chunk(ChunkType.OPINION, "I think we should delay launch."))
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert MemoryType.HYPOTHESIS in result.memory_types


class TestWriteGateSalienceThresholds:
    """Importance and novelty combined score and thresholds drive SKIP vs STORE."""

    def test_combined_score_below_min_importance_skips(self):
        gate = WriteGate(WriteGateConfig(min_importance=0.8))
        chunk = SemanticChunk(
            id="1",
            text="Just a casual remark.",
            chunk_type=ChunkType.STATEMENT,
            salience=0.2,
        )
        result = gate.evaluate(chunk)
        assert result.decision == WriteDecision.SKIP
        assert "threshold" in result.reason.lower()

    def test_high_salience_and_novelty_stores(self):
        gate = WriteGate(WriteGateConfig(min_importance=0.3, min_novelty=0.2))
        chunk = SemanticChunk(
            id="1",
            text="Important: the deadline is next Monday.",
            chunk_type=ChunkType.STATEMENT,
            salience=0.9,
        )
        result = gate.evaluate(chunk, existing_memories=None)
        assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
        assert result.importance >= 0.3
        assert result.novelty >= 0.2

    def test_novelty_below_min_skips(self):
        gate = WriteGate(WriteGateConfig(min_novelty=0.5))
        chunk = SemanticChunk(
            id="1",
            text="I prefer coffee.",
            chunk_type=ChunkType.PREFERENCE,
            salience=0.9,
        )
        result = gate.evaluate(chunk, existing_memories=[{"text": "I prefer coffee."}])
        assert result.novelty == 0.0
        assert result.decision == WriteDecision.SKIP
        assert "novelty" in result.reason.lower()


class TestWriteGateConfig:
    """Tests for WriteGateConfig (Phase 10)."""

    def test_default_thresholds(self):
        config = WriteGateConfig()
        assert config.min_importance == 0.3
        assert config.min_novelty == 0.2
        assert config.sync_importance_threshold == 0.7

    def test_custom_thresholds(self):
        config = WriteGateConfig(
            min_importance=0.5,
            min_novelty=0.3,
        )
        assert config.min_importance == 0.5
        assert config.min_novelty == 0.3


class TestPIIRedactor:
    def test_redact_email(self):
        r = PIIRedactor()
        res = r.redact("Contact me at alice@example.com for info.")
        assert res.has_redactions
        assert "alice@example.com" not in res.redacted_text
        assert "EMAIL" in res.redacted_text or "REDACTED" in res.redacted_text

    def test_redact_phone(self):
        r = PIIRedactor()
        res = r.redact("Call 555-123-4567")
        assert res.has_redactions
        assert "555-123-4567" not in res.redacted_text

    def test_no_redaction_clean_text(self):
        r = PIIRedactor()
        res = r.redact("I like pizza.")
        assert not res.has_redactions
        assert res.original_text == res.redacted_text
