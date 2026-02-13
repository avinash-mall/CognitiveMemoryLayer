"""Write gate: decide whether and how to store incoming information."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from ...core.enums import MemoryType
from ..working.models import ChunkType, SemanticChunk


class WriteDecision(StrEnum):
    STORE = "store"
    STORE_SYNC = "store"  # Alias for backward compatibility
    STORE_ASYNC = "store"  # Alias: async path not yet implemented; treated as STORE
    SKIP = "skip"
    REDACT_AND_STORE = "redact_and_store"


@dataclass
class WriteGateResult:
    decision: WriteDecision
    memory_types: list[MemoryType]
    importance: float
    novelty: float
    risk_flags: list[str]
    redaction_required: bool
    reason: str


@dataclass
class WriteGateConfig:
    min_importance: float = 0.3
    min_novelty: float = 0.2
    pii_patterns: list[str] = field(default_factory=list)
    secret_patterns: list[str] = field(default_factory=list)
    sync_importance_threshold: float = 0.7

    def __post_init__(self) -> None:
        if not self.pii_patterns:
            self.pii_patterns = [
                r"\b\d{3}-\d{2}-\d{4}\b",
                r"\b\d{16}\b",
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",
                r"\b\d{3}[-.]?\d{3}[-.]?\d{4}\b",
            ]
        if not self.secret_patterns:
            self.secret_patterns = [
                r"password\s*[:=]\s*\S+",
                r"api[_-]?key\s*[:=]\s*\S+",
                r"secret\s*[:=]\s*\S+",
                r"token\s*[:=]\s*\S+",
            ]


class WriteGate:
    """Decides whether to store information in long-term memory."""

    def __init__(
        self,
        config: WriteGateConfig | None = None,
        known_facts_cache: set[str] | None = None,
    ) -> None:
        self.config = config or WriteGateConfig()
        self._known_facts = known_facts_cache or set()
        self._pii_patterns = [re.compile(p, re.I) for p in self.config.pii_patterns]
        self._secret_patterns = [re.compile(p, re.I) for p in self.config.secret_patterns]

    def evaluate(
        self,
        chunk: SemanticChunk,
        existing_memories: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
    ) -> WriteGateResult:
        risk_flags: list[str] = []
        redaction_required = False

        if self._check_secrets(chunk.text):
            risk_flags.append("contains_secrets")
            return WriteGateResult(
                decision=WriteDecision.SKIP,
                memory_types=[],
                importance=chunk.salience,
                novelty=0.0,
                risk_flags=risk_flags,
                redaction_required=False,
                reason="Contains potential secrets - skipping",
            )

        if self._check_pii(chunk.text):
            risk_flags.append("contains_pii")
            redaction_required = True

        importance = self._compute_importance(chunk, context)
        novelty = self._compute_novelty(chunk, existing_memories)
        memory_types = self._determine_memory_types(chunk)

        # Skip if novelty alone is below minimum threshold (MED-17)
        if novelty < self.config.min_novelty:
            return WriteGateResult(
                decision=WriteDecision.SKIP,
                memory_types=[],
                importance=importance,
                novelty=novelty,
                risk_flags=risk_flags,
                redaction_required=False,
                reason=f"Below novelty threshold: {novelty:.2f} < {self.config.min_novelty:.2f}",
            )

        combined_score = (importance * 0.6) + (novelty * 0.4)

        if combined_score < self.config.min_importance:
            return WriteGateResult(
                decision=WriteDecision.SKIP,
                memory_types=[],
                importance=importance,
                novelty=novelty,
                risk_flags=risk_flags,
                redaction_required=False,
                reason=f"Below importance threshold: {combined_score:.2f}",
            )

        decision = WriteDecision.REDACT_AND_STORE if redaction_required else WriteDecision.STORE

        return WriteGateResult(
            decision=decision,
            memory_types=memory_types,
            importance=importance,
            novelty=novelty,
            risk_flags=risk_flags,
            redaction_required=redaction_required,
            reason=f"Score: {combined_score:.2f}, importance: {importance:.2f}, novelty: {novelty:.2f}",
        )

    def _check_pii(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._pii_patterns)

    def _check_secrets(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._secret_patterns)

    def _compute_importance(
        self, chunk: SemanticChunk, context: dict[str, Any] | None = None
    ) -> float:
        score = chunk.salience
        type_boosts = {
            ChunkType.PREFERENCE: 0.3,
            ChunkType.FACT: 0.2,
            ChunkType.INSTRUCTION: 0.1,
            ChunkType.EVENT: 0.1,
        }
        score += type_boosts.get(chunk.chunk_type, 0.0)
        text_lower = chunk.text.lower()
        if any(m in text_lower for m in ["always", "never", "important", "remember"]):
            score += 0.2
        if any(m in text_lower for m in ["my name", "i am", "i live", "i work"]):
            score += 0.15
        if chunk.entities:
            score += 0.1 * min(len(chunk.entities), 3)
        return min(score, 1.0)

    def _compute_novelty(
        self,
        chunk: SemanticChunk,
        existing_memories: list[dict[str, Any]] | None = None,
    ) -> float:
        # Check known facts cache first (MED-16)
        chunk_text_lower = chunk.text.lower().strip()
        if chunk_text_lower in self._known_facts:
            return 0.0

        if not existing_memories:
            return 1.0
        chunk_words = set(chunk_text_lower.split())
        for mem in existing_memories:
            mem_text = (mem.get("text") or "").lower()
            if chunk_text_lower == mem_text:
                return 0.0
            mem_words = set(mem_text.split())
            if chunk_words and mem_words:
                overlap = len(chunk_words & mem_words) / len(chunk_words | mem_words)
                if overlap > 0.8:
                    return 0.2
                if overlap > 0.5:
                    return 0.5
        return 1.0

    def _determine_memory_types(self, chunk: SemanticChunk) -> list[MemoryType]:
        mapping = {
            ChunkType.PREFERENCE: [MemoryType.PREFERENCE],
            ChunkType.FACT: [MemoryType.SEMANTIC_FACT, MemoryType.EPISODIC_EVENT],
            ChunkType.EVENT: [MemoryType.EPISODIC_EVENT],
            ChunkType.STATEMENT: [MemoryType.EPISODIC_EVENT],
            ChunkType.INSTRUCTION: [MemoryType.TASK_STATE],
            ChunkType.QUESTION: [MemoryType.EPISODIC_EVENT],
            ChunkType.OPINION: [MemoryType.HYPOTHESIS],
        }
        return mapping.get(chunk.chunk_type, [MemoryType.EPISODIC_EVENT])

    def add_known_fact(self, fact_key: str) -> None:
        self._known_facts.add(fact_key)
