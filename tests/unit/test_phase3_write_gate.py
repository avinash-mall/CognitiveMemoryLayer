"""Unit tests for Phase 3: write gate and redactor."""
import pytest

from src.memory.working.models import ChunkType, SemanticChunk
from src.memory.hippocampal.write_gate import (
    WriteDecision,
    WriteGate,
    WriteGateConfig,
    WriteGateResult,
)
from src.memory.hippocampal.redactor import PIIRedactor, RedactionResult


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
