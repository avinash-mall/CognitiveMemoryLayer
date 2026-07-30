"""Write gate: decide whether and how to store incoming information."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ...core.config import get_settings
from ...core.enums import MemoryType
from ..working.models import ChunkType, SemanticChunk

if TYPE_CHECKING:
    from ...extraction.unified_write_extractor import UnifiedExtractionResult


class WriteDecision(StrEnum):
    STORE = "store"
    STORE_SYNC = "store"  # Alias for backward compatibility
    STORE_ASYNC = "store"  # Alias: async path uses AsyncStoragePipeline when STORE_ASYNC=true
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
                r"\b\d{3}-\d{2}-\d{4}\b",  # SSN
                r"\b\d{16}\b",  # credit card
                r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b",  # email
                r"(?<!\w)(?:\+?1[\s.\-]?)?(?:\(?\d{3}\)?[\s.\-]?)\d{3}[\s.\-]?\d{4}(?!\w)",
                r"(?<!\w)\+\d{1,3}[\s.\-]?(?:\(?\d{1,4}\)?[\s.\-]?){2,5}\d{2,4}(?!\w)",
                (
                    r"\b\d{1,6}[A-Za-z]?\s+[A-Za-z0-9][A-Za-z0-9\s\.\-]{1,50}?\s+"
                    r"(?:street|st|avenue|ave|road|rd|boulevard|blvd|lane|ln|drive|dr|court|ct|"
                    r"way|terrace|ter|place|pl|parkway|pkwy)\b"
                ),
                (
                    r"\b\d{1,4}[A-Za-z]?\s+[A-Za-z][A-Za-z0-9\s\.\-]{1,50}?\s+"
                    r"(?:road|rd|street|st|lane|ln|avenue|ave|close|cl|drive|dr)\b"
                ),
                r"\b[A-Z]{1,2}\d{6,9}\b",  # passport number
                r"\b\d{7,8}[A-Z]\b",  # driver license (UK-style)
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
        self._secret_patterns = [re.compile(p, re.I) for p in self.config.secret_patterns]
        self._pii_patterns = [re.compile(p, re.I) for p in self.config.pii_patterns]

    def evaluate(
        self,
        chunk: SemanticChunk,
        existing_memories: list[dict[str, Any]] | None = None,
        context: dict[str, Any] | None = None,
        unified_result: "UnifiedExtractionResult | None" = None,
        precomputed_novelty: float | None = None,
        precomputed_pii: bool | None = None,
    ) -> WriteGateResult:
        risk_flags: list[str] = []
        settings = get_settings().features

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

        redaction_required: bool
        # Master PII switch: when disabled, skip all PII detection (eval mode)
        if not settings.pii_redaction_enabled:
            redaction_required = False
        elif settings.use_llm_enabled and unified_result is not None:
            redaction_required = bool(unified_result.pii_spans)
            if redaction_required:
                risk_flags.append("contains_pii")
        else:
            has_pii = (
                precomputed_pii if precomputed_pii is not None else self._predict_pii(chunk.text)
            )
            if has_pii:
                risk_flags.append("contains_pii")
                redaction_required = True
            else:
                redaction_required = False

        effective_chunk_type: ChunkType | None = None
        if unified_result is not None and settings.use_llm_enabled and unified_result.memory_type:
            effective_chunk_type = _MEMORY_TYPE_TO_CHUNK_TYPE.get(
                unified_result.memory_type, ChunkType.STATEMENT
            )

        importance: float
        if settings.use_llm_enabled and unified_result is not None:
            importance = unified_result.importance
        else:
            importance = self._predict_importance(
                chunk, context, effective_chunk_type=effective_chunk_type
            )
        novelty = (
            precomputed_novelty
            if precomputed_novelty is not None
            else self._compute_novelty(chunk, existing_memories)
        )
        memory_types = self._determine_memory_types(
            chunk, effective_chunk_type=effective_chunk_type
        )

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

    def _check_secrets(self, text: str) -> bool:
        return any(pattern.search(text) for pattern in self._secret_patterns)

    def _predict_pii(self, text: str) -> bool:
        if not text.strip():
            return False
        return any(pattern.search(text) for pattern in self._pii_patterns)

    def _predict_importance(
        self,
        chunk: SemanticChunk,
        context: dict[str, Any] | None = None,
        effective_chunk_type: ChunkType | None = None,
    ) -> float:
        _ = context, effective_chunk_type
        # Upstream STM salience is the importance signal when no LLM result exists.
        return max(0.0, min(1.0, chunk.salience))

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

        # ponytail: Jaccard word-overlap novelty; the gate only sees texts today —
        # to upgrade to embedding-cosine, return embeddings from scan_texts_for_gate.
        chunk_words = set(chunk_text_lower.split())
        for mem in existing_memories:
            mem_text = (mem.get("text") or "").lower().strip()
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

    def compute_novelty_batch(
        self,
        chunks: "list[SemanticChunk]",
        existing_memories: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        """Compute novelty for all chunks (Jaccard, cheap enough to loop)."""
        return [self._compute_novelty(chunk, existing_memories) for chunk in chunks]

    def _determine_memory_types(
        self, chunk: SemanticChunk, effective_chunk_type: ChunkType | None = None
    ) -> list[MemoryType]:
        mapping = {
            ChunkType.PREFERENCE: [MemoryType.PREFERENCE],
            ChunkType.CONSTRAINT: [MemoryType.CONSTRAINT],
            ChunkType.FACT: [MemoryType.SEMANTIC_FACT, MemoryType.EPISODIC_EVENT],
            ChunkType.EVENT: [MemoryType.EPISODIC_EVENT],
            ChunkType.STATEMENT: [MemoryType.EPISODIC_EVENT],
            ChunkType.INSTRUCTION: [MemoryType.TASK_STATE],
            ChunkType.QUESTION: [MemoryType.EPISODIC_EVENT],
            ChunkType.OPINION: [MemoryType.HYPOTHESIS],
        }
        ct = effective_chunk_type if effective_chunk_type is not None else chunk.chunk_type
        return mapping.get(ct, [MemoryType.EPISODIC_EVENT])

    def add_known_fact(self, fact_key: str) -> None:
        self._known_facts.add(fact_key)


_MEMORY_TYPE_TO_CHUNK_TYPE: dict[str, ChunkType] = {
    "preference": ChunkType.PREFERENCE,
    "constraint": ChunkType.CONSTRAINT,
    "semantic_fact": ChunkType.FACT,
    "episodic_event": ChunkType.EVENT,
    "task_state": ChunkType.INSTRUCTION,
    "hypothesis": ChunkType.OPINION,
    "procedure": ChunkType.INSTRUCTION,
    "conversation": ChunkType.EVENT,
    "message": ChunkType.EVENT,
}
