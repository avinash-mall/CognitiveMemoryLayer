"""Unit tests for ChunkType derivation from memory_type in Write Gate."""

from datetime import UTC, datetime
from uuid import uuid4

from src.memory.hippocampal.write_gate import WriteGate
from src.memory.working.models import ChunkType, SemanticChunk


def _chunk(text: str, chunk_type: ChunkType = ChunkType.STATEMENT) -> SemanticChunk:
    return SemanticChunk(
        id=str(uuid4()),
        text=text,
        chunk_type=chunk_type,
        salience=0.5,
        timestamp=datetime.now(UTC),
    )


def test_preference_memory_type_uses_chunk_type_preference_for_importance(monkeypatch):
    """When unified_result.memory_type='preference', gate uses ChunkType.PREFERENCE for importance."""
    from src.extraction.unified_write_extractor import UnifiedExtractionResult

    monkeypatch.setattr(
        "src.memory.hippocampal.write_gate.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_enabled": True,
                        "use_llm_memory_type": True,
                        "use_llm_write_gate_importance": False,
                        "use_llm_salience_refinement": False,
                        "use_llm_pii_redaction": False,
                    },
                )()
            },
        )(),
    )
    gate = WriteGate()
    chunk = _chunk("I prefer dark mode", chunk_type=ChunkType.STATEMENT)
    unified_result = UnifiedExtractionResult(
        entities=[],
        relations=[],
        constraints=[],
        facts=[],
        salience=0.5,
        importance=0.0,
        memory_type="preference",
    )
    result = gate.evaluate(chunk, unified_result=unified_result)
    # PREFERENCE has +0.3 type boost; STATEMENT has 0. With salience 0.5, importance should be higher
    # than a plain STATEMENT chunk because effective_chunk_type is PREFERENCE
    assert result.decision.value == "store"
    assert result.memory_types
    assert "preference" in [m.value for m in result.memory_types]


def test_constraint_memory_type_uses_chunk_type_constraint(monkeypatch):
    """When unified_result.memory_type='constraint', gate uses ChunkType.CONSTRAINT."""
    from src.extraction.unified_write_extractor import UnifiedExtractionResult

    monkeypatch.setattr(
        "src.memory.hippocampal.write_gate.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_enabled": True,
                        "use_llm_memory_type": True,
                        "use_llm_write_gate_importance": False,
                        "use_llm_salience_refinement": False,
                        "use_llm_pii_redaction": False,
                    },
                )()
            },
        )(),
    )
    gate = WriteGate()
    chunk = _chunk("I must avoid gluten", chunk_type=ChunkType.STATEMENT)
    unified_result = UnifiedExtractionResult(
        entities=[],
        relations=[],
        constraints=[],
        facts=[],
        salience=0.5,
        importance=0.0,
        memory_type="constraint",
    )
    result = gate.evaluate(chunk, unified_result=unified_result)
    assert "constraint" in [m.value for m in result.memory_types]


def test_unknown_memory_type_falls_back_to_statement(monkeypatch):
    """Unknown memory_type falls back to ChunkType.STATEMENT."""
    from src.extraction.unified_write_extractor import UnifiedExtractionResult

    monkeypatch.setattr(
        "src.memory.hippocampal.write_gate.get_settings",
        lambda: type(
            "S",
            (),
            {
                "features": type(
                    "F",
                    (),
                    {
                        "use_llm_enabled": True,
                        "use_llm_memory_type": True,
                        "use_llm_write_gate_importance": False,
                        "use_llm_salience_refinement": False,
                        "use_llm_pii_redaction": False,
                    },
                )()
            },
        )(),
    )
    gate = WriteGate()
    chunk = _chunk("Some text", chunk_type=ChunkType.STATEMENT)
    unified_result = UnifiedExtractionResult(
        entities=[],
        relations=[],
        constraints=[],
        facts=[],
        salience=0.5,
        importance=0.0,
        memory_type="tool_result",
    )
    result = gate.evaluate(chunk, unified_result=unified_result)
    # tool_result maps to STATEMENT -> EPISODIC_EVENT
    assert result.memory_types
    assert "episodic_event" in [m.value for m in result.memory_types]
