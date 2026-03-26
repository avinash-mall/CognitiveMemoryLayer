"""Focused write-gate model-hook tests."""

from src.memory.hippocampal.write_gate import WriteDecision, WriteGate
from src.memory.working.models import ChunkType, SemanticChunk


class _SalienceModelPack:
    available = True

    def has_task_model(self, task: str) -> bool:
        return False

    def predict_single(self, task: str, text: str):
        if task == "salience_bin":
            return type("P", (), {"label": "high", "confidence": 0.9})()
        return None


def test_salience_bin_refines_importance_when_importance_model_missing():
    gate = WriteGate(modelpack=_SalienceModelPack())
    chunk = SemanticChunk(
        id="1",
        text="This matters for future planning.",
        chunk_type=ChunkType.STATEMENT,
        salience=0.1,
    )
    result = gate.evaluate(chunk)
    assert result.importance > 0.1
    assert result.decision in (WriteDecision.STORE, WriteDecision.REDACT_AND_STORE)
