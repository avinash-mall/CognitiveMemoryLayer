"""Write gate: decide whether and how to store incoming information."""

import re
from dataclasses import dataclass, field
from enum import StrEnum
from typing import TYPE_CHECKING, Any

from ...core.config import get_settings
from ...core.enums import MemoryType
from ...utils.modelpack import ModelPackRuntime, get_modelpack_runtime
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


_IMPORTANCE_BIN_MAP: dict[str, float] = {
    "very_low": 0.1,
    "low": 0.3,
    "medium": 0.5,
    "high": 0.75,
    "critical": 0.9,
}

_SALIENCE_BIN_MAP: dict[str, float] = {
    "low": 0.2,
    "medium": 0.5,
    "high": 0.8,
}


class WriteGate:
    """Decides whether to store information in long-term memory."""

    def __init__(
        self,
        config: WriteGateConfig | None = None,
        known_facts_cache: set[str] | None = None,
        modelpack: ModelPackRuntime | None = None,
    ) -> None:
        self.config = config or WriteGateConfig()
        self._known_facts = known_facts_cache or set()
        self._secret_patterns = [re.compile(p, re.I) for p in self.config.secret_patterns]
        self._pii_patterns = [re.compile(p, re.I) for p in self.config.pii_patterns]
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

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
        if (
            settings.use_llm_enabled
            and settings.use_llm_pii_redaction
            and unified_result is not None
        ):
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
        if (
            unified_result is not None
            and settings.use_llm_enabled
            and settings.use_llm_memory_type
            and unified_result.memory_type
        ):
            effective_chunk_type = _MEMORY_TYPE_TO_CHUNK_TYPE.get(
                unified_result.memory_type, ChunkType.STATEMENT
            )

        importance: float
        if (
            settings.use_llm_enabled
            and settings.use_llm_write_gate_importance
            and unified_result is not None
        ):
            importance = unified_result.importance
        elif (
            settings.use_llm_enabled
            and settings.use_llm_salience_refinement
            and unified_result is not None
        ):
            importance = unified_result.salience
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

        # TODO: replace fixed weights with learned combination weights once
        # both novelty_pair and write_importance_regression models are trained.
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

    @staticmethod
    def _has_lexical_pii_cue(text: str) -> bool:
        """Require an explicit textual cue before trusting classifier-only PII hits."""
        lowered = text.lower()
        if re.search(r"[@\d]", text):
            return True

        lexical_cues = (
            r"\b(?:email|e-mail)\b",
            r"\b(?:phone|mobile|cell)\b",
            r"\b(?:call|text|sms)\b",
            r"\b(?:contact me|reach me)\b",
            r"\baddress\b",
            r"\b(?:street|st\.?|avenue|ave\.?|road|rd\.?|lane|ln\.?|drive|dr\.?|boulevard|blvd\.?)\b",
            r"\b(?:ssn|social security|passport|license)\b",
        )
        return any(re.search(pattern, lowered) for pattern in lexical_cues)

    def _predict_pii(self, text: str) -> bool:
        if not text.strip():
            return False

        if any(pattern.search(text) for pattern in self._pii_patterns):
            return True

        _has_pii_model = getattr(self.modelpack, "has_task_model", None)
        if _has_pii_model and _has_pii_model("pii_span_detection"):
            try:
                span_pred = self.modelpack.predict_spans("pii_span_detection", text)
                # Trust the task model: spans present = PII, no spans = no PII
                return bool(span_pred is not None and span_pred.spans)
            except Exception:
                pass  # fall through to existing classifier

        if not self.modelpack.available:
            return False

        pred = self.modelpack.predict_single("pii_presence", text)
        if pred is None or not pred.label:
            return False
        label = pred.label.strip().lower()
        if label not in {"pii", "present", "contains_pii", "yes", "true"}:
            return False

        # The current classifier can over-fire on benign sentences, so do not
        # trust classifier-only positives unless the text also contains a
        # concrete lexical cue. Regex and token-span hits above still win.
        lexical_cue = self._has_lexical_pii_cue(text)
        if pred.confidence >= 0.85:
            return lexical_cue
        return lexical_cue and pred.confidence >= 0.5

    def _predict_importance(
        self,
        chunk: SemanticChunk,
        context: dict[str, Any] | None = None,
        effective_chunk_type: ChunkType | None = None,
    ) -> float:
        metadata = {
            "memory_type": (
                (effective_chunk_type or chunk.chunk_type).value
                if hasattr((effective_chunk_type or chunk.chunk_type), "value")
                else str(effective_chunk_type or chunk.chunk_type)
            ),
            "importance": chunk.salience,
            "confidence": chunk.confidence,
            "context_tags": list((context or {}).get("context_tags") or []),
            "namespace": (context or {}).get("namespace"),
        }
        _has = getattr(self.modelpack, "has_task_model", None)
        if _has and _has("write_importance_regression") and chunk.text.strip():
            try:
                score_pred = self.modelpack.predict_score_single(
                    "write_importance_regression",
                    chunk.text,
                    metadata=metadata,
                )
                if score_pred is not None:
                    return max(0.0, min(1.0, score_pred.score))
            except Exception:
                pass  # fall through to existing logic

        # Blend family-level importance/salience signals with STM salience.
        if self.modelpack.available and chunk.text.strip():
            importance_score = self._predict_binned_score(
                "importance_bin",
                chunk.text,
                _IMPORTANCE_BIN_MAP,
            )
            salience_score = self._predict_binned_score(
                "salience_bin",
                chunk.text,
                _SALIENCE_BIN_MAP,
            )
            if importance_score is not None and salience_score is not None:
                return max(
                    0.0,
                    min(
                        1.0,
                        (importance_score * 0.4) + (salience_score * 0.2) + (chunk.salience * 0.4),
                    ),
                )
            if importance_score is not None:
                return max(0.0, min(1.0, (importance_score * 0.4) + (chunk.salience * 0.6)))
            if salience_score is not None:
                return max(0.0, min(1.0, (salience_score * 0.5) + (chunk.salience * 0.5)))

        # No heuristic scoring path: use upstream salience directly.
        return max(0.0, min(1.0, chunk.salience))

    def _predict_binned_score(
        self,
        task: str,
        text: str,
        mapping: dict[str, float],
    ) -> float | None:
        try:
            pred = self.modelpack.predict_single(task, text)
        except Exception:
            return None
        if pred is None or not pred.label:
            return None
        return mapping.get(pred.label.strip().lower())

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

        _has_novelty = getattr(self.modelpack, "has_task_model", None)
        if _has_novelty and _has_novelty("novelty_pair"):
            try:
                mem_texts = [(mem.get("text") or "").strip() for mem in existing_memories]
                valid_pairs = [(chunk.text, mt) for mt in mem_texts if mt]
                if valid_pairs:
                    predict_proba_batch = getattr(self.modelpack, "predict_pair_proba_batch", None)
                    if callable(predict_proba_batch):
                        # Single sklearn call for all N memories — ~10ms vs Nx6ms
                        batch_results = predict_proba_batch("novelty_pair", valid_pairs)
                        min_novelty = 1.0
                        scored_any = False
                        for probs in batch_results:
                            if probs is None:
                                continue
                            pair_novelty = float(probs.get("changed", 0.0)) + float(
                                probs.get("novel", 0.0)
                            )
                            scored_any = True
                            min_novelty = min(min_novelty, max(0.0, min(1.0, pair_novelty)))
                            if min_novelty == 0.0:
                                break  # exact duplicate — no need to check more
                        if scored_any:
                            return min_novelty
                    else:
                        # Fallback: sequential predict_pair_proba calls
                        predict_pair_proba = getattr(self.modelpack, "predict_pair_proba", None)
                        min_novelty = 1.0
                        scored_any = False
                        for text_a, text_b in valid_pairs:
                            probs = (
                                predict_pair_proba("novelty_pair", text_a, text_b)
                                if callable(predict_pair_proba)
                                else None
                            )
                            if probs is not None:
                                pair_novelty = float(probs.get("changed", 0.0)) + float(
                                    probs.get("novel", 0.0)
                                )
                            else:
                                pred = self.modelpack.predict_pair("novelty_pair", text_a, text_b)
                                if pred is None:
                                    continue
                                label = pred.label.strip().lower()
                                confidence = max(0.0, min(1.0, pred.confidence))
                                pair_novelty = (
                                    confidence
                                    if label in {"changed", "novel"}
                                    else (1.0 - confidence)
                                )
                            scored_any = True
                            min_novelty = min(min_novelty, max(0.0, min(1.0, pair_novelty)))
                            if min_novelty == 0.0:
                                break
                        if scored_any:
                            return min_novelty
            except Exception:
                pass  # fall through to heuristic

        # Heuristic fallback: Jaccard word-overlap
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
        """Compute novelty for all chunks in one mega-batch sklearn call.

        Instead of N x predict_pair_proba_batch([M pairs]) = N x ~10ms,
        calls predict_pair_proba_batch([N x M pairs]) once = ~10ms total.
        """
        if not chunks:
            return []
        if not existing_memories:
            return [1.0] * len(chunks)

        # Filter to chunks not in known_facts cache
        needs_model: list[int] = []
        novelties: list[float] = [1.0] * len(chunks)
        for i, chunk in enumerate(chunks):
            key = chunk.text.lower().strip()
            if key in self._known_facts:
                novelties[i] = 0.0
            else:
                needs_model.append(i)

        if not needs_model:
            return novelties

        mem_texts = [(mem.get("text") or "").strip() for mem in existing_memories]
        mem_texts = [t for t in mem_texts if t]
        if not mem_texts:
            return novelties

        predict_proba_batch = getattr(self.modelpack, "predict_pair_proba_batch", None)
        if callable(predict_proba_batch):
            try:
                # Build mega-batch: all (chunk_i, mem_j) pairs in order
                mega_pairs: list[tuple[str, str]] = []
                chunk_starts: list[int] = []
                for idx in needs_model:
                    chunk_starts.append(len(mega_pairs))
                    for mt in mem_texts:
                        mega_pairs.append((chunks[idx].text, mt))

                batch_results = predict_proba_batch("novelty_pair", mega_pairs)

                for pos, idx in enumerate(needs_model):
                    start = chunk_starts[pos]
                    end = start + len(mem_texts)
                    min_nov = 1.0
                    for probs in batch_results[start:end]:
                        if probs is None:
                            continue
                        pair_nov = float(probs.get("changed", 0.0)) + float(probs.get("novel", 0.0))
                        min_nov = min(min_nov, max(0.0, min(1.0, pair_nov)))
                        if min_nov == 0.0:
                            break
                    novelties[idx] = min_nov
                return novelties
            except Exception:
                pass  # fall through to per-chunk fallback

        # Fallback: call _compute_novelty per chunk
        for idx in needs_model:
            novelties[idx] = self._compute_novelty(chunks[idx], existing_memories)
        return novelties

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
