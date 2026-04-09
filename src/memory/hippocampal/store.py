"""Hippocampal store: episodic memory with write gate, embedding, and vector store."""

from __future__ import annotations

import asyncio
import concurrent.futures
import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast

import structlog

from ...core.enums import MemorySource, MemoryStatus, MemoryType
from ...core.schemas import (
    EntityMention,
    MemoryRecord,
    MemoryRecordCreate,
    Provenance,
    Relation,
)
from ...extraction.entity_extractor import EntityExtractor
from ...extraction.relation_extractor import RelationExtractor
from ...storage.base import MemoryStoreBase
from ...utils.embeddings import EmbeddingClient
from ...utils.ner import _SPACY_EXECUTOR
from ...utils.ner import extract_entities as _ner_extract_entities
from ...utils.ner import extract_relations as _ner_extract_relations
from ..working.models import SemanticChunk
from .redactor import PIIRedactor
from .write_gate import WriteDecision, WriteGate, WriteGateResult

if TYPE_CHECKING:
    from ...extraction.constraint_extractor import ConstraintExtractor
    from ...extraction.local_unified_extractor import LocalUnifiedWriteExtractor
    from ...extraction.unified_write_extractor import (
        UnifiedExtractionResult,
        UnifiedWritePathExtractor,
    )


# Dedicated executor for Phase 1 (write-gate novelty checks).
# novelty_pair sklearn model takes ~6ms per call x 50 existing memories = ~300ms per chunk.
# Running in a separate executor frees the event loop for embedding/DeBERTa batching.
# Worker count is configurable via PERFORMANCE__GATE_EXECUTOR_WORKERS;
# 0 = auto-detect based on CPU count (min(cpu_count, 8)).
def _resolve_gate_workers() -> int:
    try:
        from ...core.config import get_settings

        return get_settings().performance.resolved_gate_workers()
    except Exception:
        import os

        return min(os.cpu_count() or 4, 8)


_GATE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_resolve_gate_workers(), thread_name_prefix="write_gate"
)


class _BatchingSpanPredictor:
    """Coalesce concurrent predict_spans_batch calls into one GPU forward pass.

    Works like BatchingEmbeddingClient: texts from concurrent encode_batch calls
    arriving within max_wait_ms are merged into a single modelpack.predict_spans_batch()
    call, turning Nx(1-text batches) into one N-text batch - typically 5-10x faster.

    Each instance owns a dedicated 1-worker ThreadPoolExecutor so that PII and fact
    span batchers run in parallel (not serialized on a shared executor). Batches are
    capped at max_batch_size texts; overflow is re-dispatched immediately, preventing
    super-batches (120 texts -> 800ms) from stalling all concurrent callers.
    """

    def __init__(
        self,
        modelpack: Any,
        task: str,
        max_wait_ms: float | None = None,
        max_batch_size: int | None = None,
    ) -> None:
        self._modelpack = modelpack
        self._task = task
        # Resolve from settings when not explicitly provided
        try:
            from ...core.config import get_settings

            perf = get_settings().performance
            _wait = max_wait_ms if max_wait_ms is not None else perf.resolved_span_batch_wait_ms()
            _batch = (
                max_batch_size
                if max_batch_size is not None
                else perf.resolved_span_max_batch_size()
            )
        except Exception:
            _wait = max_wait_ms if max_wait_ms is not None else 10.0
            _batch = max_batch_size if max_batch_size is not None else 20
        self._max_wait = _wait / 1000.0
        self._max_batch_size = _batch
        self._lock: asyncio.Lock | None = None
        self._pending: list[tuple[str, asyncio.Future[Any]]] = []
        self._dispatch_task: asyncio.Task[None] | None = None
        # Per-instance executor: PII and fact span batchers can run concurrently.
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=1, thread_name_prefix=f"deberta_{task[:8]}"
        )

    def _get_lock(self) -> asyncio.Lock:
        if self._lock is None:
            self._lock = asyncio.Lock()
        return self._lock

    async def predict_batch(self, texts: list[str]) -> list:
        """Queue texts and await their SpanPrediction results."""
        if not texts:
            return []
        loop = asyncio.get_running_loop()
        futures: list[asyncio.Future[Any]] = [loop.create_future() for _ in texts]
        lock = self._get_lock()
        async with lock:
            for text, fut in zip(texts, futures, strict=True):
                self._pending.append((text, fut))
            if self._dispatch_task is None or self._dispatch_task.done():
                self._dispatch_task = loop.create_task(self._dispatch_after_wait())
        return list(await asyncio.gather(*futures))

    async def _dispatch_after_wait(self) -> None:
        await asyncio.sleep(self._max_wait)
        await self._drain()

    async def _drain(self) -> None:
        lock = self._get_lock()
        async with lock:
            if not self._pending:
                return
            # Cap batch size to prevent super-batches (e.g. 120 texts → 800ms).
            # Overflow stays in _pending and is dispatched immediately after this batch.
            batch = self._pending[: self._max_batch_size]
            self._pending = self._pending[self._max_batch_size :]
            if self._pending:
                loop = asyncio.get_running_loop()
                self._dispatch_task = loop.create_task(self._drain())
            else:
                self._dispatch_task = None

        texts_in_batch = [t for t, _ in batch]
        try:
            _mp = self._modelpack
            _task = self._task
            results = await asyncio.get_running_loop().run_in_executor(
                self._executor,
                lambda: _mp.predict_spans_batch(_task, texts_in_batch),
            )
        except Exception as exc:
            for _, fut in batch:
                if not fut.done():
                    fut.set_exception(exc)
            return

        for i, (_, fut) in enumerate(batch):
            if not fut.done():
                fut.set_result(results[i] if i < len(results) else None)


def _gate_result_to_dict(g: WriteGateResult) -> dict:
    """Serialize for API (eval mode)."""
    return {"decision": g.decision.value, "reason": g.reason}


def _ner_entities_for_text(text: str) -> list[EntityMention]:
    return [
        EntityMention(
            text=e.text,
            normalized=e.normalized,
            entity_type=e.entity_type,
            start_char=e.start_char,
            end_char=e.end_char,
        )
        for e in _ner_extract_entities(text)
    ]


# Mapping from fact_extraction_structured span labels to NER entity types.
_FACT_LABEL_TO_ENTITY_TYPE: dict[str, str] = {
    "identity": "PERSON",
    "location": "LOCATION",
    "occupation": "ATTRIBUTE",
    "preference": "CONCEPT",
    "attribute": "ATTRIBUTE",
    "goal": "CONCEPT",
    "value": "CONCEPT",
    "state": "ATTRIBUTE",
    "causal": "CONCEPT",
    "policy": "CONCEPT",
}


def _entities_from_fact_spans(text: str, span_pred: Any) -> list[EntityMention]:
    """Derive EntityMention objects from pre-computed DeBERTa fact spans.

    Avoids a spaCy NER call for texts where Phase 2.5 already ran DeBERTa.
    Uses span label → entity type mapping so downstream graph code still gets
    typed entities; quality is slightly lower than spaCy NER but fast (O(spans)).
    """
    from ...extraction.fact_span_adapter import extract_structured_span_matches

    return [
        EntityMention(
            text=m.value,
            normalized=m.value.lower().strip(),
            entity_type=_FACT_LABEL_TO_ENTITY_TYPE.get(m.label, "CONCEPT"),
        )
        for m in extract_structured_span_matches(text, span_pred)
        if m.value
    ]


def _ner_relations_for_text(text: str) -> list[Relation]:
    return [
        Relation(
            subject=r.subject,
            predicate=r.predicate,
            object=r.object,
            confidence=r.confidence,
        )
        for r in _ner_extract_relations(text)
    ]


class HippocampalStore:
    """
    Fast episodic memory store.
    Coordinates write gate, redaction, embedding, extraction, and vector store.
    """

    def __init__(
        self,
        vector_store: MemoryStoreBase,
        embedding_client: EmbeddingClient,
        entity_extractor: EntityExtractor | None = None,
        relation_extractor: RelationExtractor | None = None,
        write_gate: WriteGate | None = None,
        redactor: PIIRedactor | None = None,
        constraint_extractor: ConstraintExtractor | None = None,
        unified_extractor: UnifiedWritePathExtractor | None = None,
        local_extractor: LocalUnifiedWriteExtractor | None = None,
    ) -> None:
        from ...extraction.constraint_extractor import ConstraintExtractor as _ConstraintExtractor

        self.store = vector_store
        self.embeddings = embedding_client
        self.entity_extractor = entity_extractor
        self.relation_extractor = relation_extractor
        self.write_gate = write_gate or WriteGate()
        self.redactor = redactor or PIIRedactor()
        self.constraint_extractor = constraint_extractor or _ConstraintExtractor()
        self.unified_extractor = unified_extractor
        self.local_extractor = local_extractor
        self._span_batcher: _BatchingSpanPredictor | None = None
        self._pii_span_batcher: _BatchingSpanPredictor | None = None
        # Scan coalescing: avoids redundant DB queries for concurrent writes to the same tenant.
        # _scan_cache: tenant_id -> (monotonic_ts, result_dicts)
        # _scan_futures: tenant_id -> in-flight Future (deduplicate concurrent requests)
        self._scan_cache: dict[str, tuple[float, list[dict]]] = {}
        self._scan_futures: dict[str, asyncio.Future] = {}

    def _use_unified_write_path(self) -> bool:
        """True when use_llm_enabled and any write-path LLM flag is enabled and we have a unified extractor."""
        if self.unified_extractor is None:
            return False
        from ...core.config import get_settings

        s = get_settings().features
        return s.use_llm_enabled and (
            s.use_llm_constraint_extractor
            or s.use_llm_write_time_facts
            or s.use_llm_salience_refinement
            or s.use_llm_pii_redaction
            or s.use_llm_write_gate_importance
        )

    _SCAN_CACHE_TTL = 2.0  # seconds — sufficient to coalesce a burst of concurrent writes

    async def _get_existing_for_gate(self, tenant_id: str) -> list[dict]:
        """Return recent active memories for write-gate novelty check.

        Uses a 2-second TTL cache + in-flight coalescing so N concurrent writes to the
        same tenant issue at most ONE DB scan instead of N, cutting scan_ms from
        19-233 ms down to ~0 ms for all but the first request per burst.
        """
        loop = asyncio.get_running_loop()
        now = loop.time()

        # Fast path: warm cache hit
        cached = self._scan_cache.get(tenant_id)
        if cached and (now - cached[0]) < self._SCAN_CACHE_TTL:
            return list(cached[1])  # return a copy so callers can append freely

        # Coalesce: if a scan is already in-flight for this tenant, wait for it
        if tenant_id in self._scan_futures:
            try:
                return list(await self._scan_futures[tenant_id])
            except Exception:
                pass  # fall through to issue a fresh query

        # Issue a new DB scan and let all concurrent waiters share the result
        fut: asyncio.Future[list[dict]] = loop.create_future()
        self._scan_futures[tenant_id] = fut
        try:
            texts = await self.store.scan_texts_for_gate(tenant_id, limit=10)
            result = [{"text": t} for t in texts]
            self._scan_cache[tenant_id] = (loop.time(), result)
            fut.set_result(result)
            return list(result)
        except Exception as exc:
            if not fut.done():
                fut.set_exception(exc)
            raise
        finally:
            self._scan_futures.pop(tenant_id, None)

    async def encode_chunk(
        self,
        tenant_id: str,
        chunk: SemanticChunk,
        context_tags: list[str] | None = None,
        source_session_id: str | None = None,
        agent_id: str | None = None,
        existing_memories: list[dict[str, Any]] | None = None,
        namespace: str | None = None,
        timestamp: datetime | None = None,
        request_metadata: dict[str, Any] | None = None,
        memory_type_override: MemoryType | None = None,
    ) -> tuple[MemoryRecord | None, WriteGateResult]:
        # S-04: single settings load for the entire method
        from ...core.config import get_settings as _get_settings

        _features = _get_settings().features

        unified_result: UnifiedExtractionResult | None = None
        if self._use_unified_write_path() and self.unified_extractor:
            unified_result = await self.unified_extractor.extract(chunk)

        # Local model fallback: when LLM is disabled, use local extractors
        # for importance scoring if available.
        local_result: dict | None = None
        if unified_result is None and self.local_extractor and self.local_extractor.available:
            text_for_local = getattr(chunk, "text", None)
            if isinstance(text_for_local, str) and text_for_local.strip():
                try:
                    local_result = await self.local_extractor.extract(text_for_local)
                except Exception:
                    pass

        gate_result = self.write_gate.evaluate(
            chunk,
            existing_memories=existing_memories,
            unified_result=unified_result,
        )
        if gate_result.decision == WriteDecision.SKIP:
            return (None, gate_result)

        text = chunk.text
        if gate_result.redaction_required:
            pii_spans = None
            if (
                unified_result
                and unified_result.pii_spans
                and _features.use_llm_enabled
                and _features.use_llm_pii_redaction
            ):
                pii_spans = [(s.start, s.end, s.pii_type) for s in unified_result.pii_spans]
            redaction_result = self.redactor.redact(text, additional_spans=pii_spans)
            text = redaction_result.redacted_text
        elif unified_result and unified_result.pii_spans and self._use_unified_write_path():
            if _features.use_llm_enabled and _features.use_llm_pii_redaction:
                pii_spans = [(s.start, s.end, s.pii_type) for s in unified_result.pii_spans]
                redaction_result = self.redactor.redact(text, additional_spans=pii_spans)
                text = redaction_result.redacted_text

        embedding_result = await self.embeddings.embed(text)

        entities: list[EntityMention] = []
        if self.entity_extractor:
            entities = await self.entity_extractor.extract(text)
        else:
            entities = _ner_entities_for_text(text)

        relations: list[Relation] = []
        if self.relation_extractor:
            entity_texts = [e.normalized for e in entities]
            relations = await self.relation_extractor.extract(text, entities=entity_texts)
        else:
            relations = _ner_relations_for_text(text)

        # Use caller-provided memory_type override, else LLM memory_type, else gate/constraint
        settings = _features
        memory_type = memory_type_override
        if (
            memory_type is None
            and unified_result
            and settings.use_llm_enabled
            and settings.use_llm_memory_type
            and unified_result.memory_type
        ):
            try:
                memory_type = MemoryType(unified_result.memory_type)
            except ValueError:
                structlog.get_logger(__name__).debug(
                    "invalid_llm_memory_type",
                    raw_type=unified_result.memory_type,
                )
        if memory_type is None and local_result and local_result.get("memory_type"):
            try:
                memory_type = MemoryType(local_result["memory_type"])
            except (ValueError, KeyError):
                pass
        if memory_type is None:
            memory_type = (
                gate_result.memory_types[0]
                if gate_result.memory_types
                else MemoryType.EPISODIC_EVENT
            )

        # Constraint extraction: unified LLM or modelpack/NER path
        if unified_result and settings.use_llm_enabled and settings.use_llm_constraint_extractor:
            extracted_constraints = unified_result.constraints
        else:
            extracted_constraints = self.constraint_extractor.extract(chunk)
        constraint_dicts = [c.to_dict() for c in extracted_constraints]

        # If high-confidence constraint extracted and no API/LLM override, override memory type.
        # Do NOT override when chunk is already PREFERENCE — honor caller's classification.
        if (
            memory_type_override is None
            and chunk.chunk_type.value != "preference"
            and not (
                unified_result
                and settings.use_llm_enabled
                and settings.use_llm_memory_type
                and unified_result.memory_type
            )
            and extracted_constraints
            and any(c.confidence >= 0.7 for c in extracted_constraints)
        ):
            memory_type = MemoryType.CONSTRAINT

        if memory_type == MemoryType.CONSTRAINT and extracted_constraints:
            from ...extraction.constraint_extractor import ConstraintExtractor

            key = ConstraintExtractor.constraint_fact_key(extracted_constraints[0])
        else:
            key = self._generate_key(chunk, memory_type) or ""

        # Importance: unified LLM, local model, or gate
        importance = gate_result.importance
        if unified_result and settings.use_llm_enabled and settings.use_llm_write_gate_importance:
            importance = unified_result.importance
        elif local_result and "importance" in local_result:
            importance = local_result["importance"]

        # Merge request-level metadata with system metadata; request metadata wins on conflict
        system_metadata: dict[str, Any] = {
            "chunk_type": chunk.chunk_type.value,
            "source_turn_id": chunk.source_turn_id,
            "source_role": chunk.source_role,
        }
        if constraint_dicts:
            system_metadata["constraints"] = constraint_dicts

        # --- Temporal resolution: resolve relative time refs to absolute dates ---
        if _features.temporal_resolution_enabled:
            try:
                from ...extraction.temporal_resolver import (
                    extract_event_date,
                    resolve_temporal_references,
                )

                session_date = timestamp or chunk.timestamp
                temporal_refs = resolve_temporal_references(text, session_date)
                if temporal_refs:
                    system_metadata["temporal_references"] = [
                        {
                            "original": ref["original"],
                            "resolved_date": ref["resolved_date"].isoformat(),
                            "approximate": ref["approximate"],
                        }
                        for ref in temporal_refs
                    ]
                event_date = extract_event_date(text, session_date)
                if event_date:
                    system_metadata["event_date"] = event_date.isoformat()
            except Exception:
                pass  # Temporal resolution is best-effort

        if request_metadata:
            merged_metadata = {**system_metadata, **request_metadata}
        else:
            merged_metadata = system_metadata

        effective_context_tags = context_tags or []
        if (
            not effective_context_tags
            and unified_result
            and settings.use_llm_enabled
            and settings.use_llm_context_tags
            and hasattr(unified_result, "context_tags")
            and unified_result.context_tags
        ):
            effective_context_tags = unified_result.context_tags
        elif not effective_context_tags and local_result and local_result.get("context_tags"):
            effective_context_tags = local_result["context_tags"]

        conf = chunk.confidence
        if (
            unified_result
            and settings.use_llm_enabled
            and settings.use_llm_confidence
            and hasattr(unified_result, "confidence")
        ):
            conf = unified_result.confidence
        elif local_result and "confidence" in local_result:
            conf = local_result["confidence"]

        decay_rate_val: float | None = None
        dr = getattr(unified_result, "decay_rate", None) if unified_result else None
        if (
            unified_result
            and settings.use_llm_enabled
            and settings.use_llm_decay_rate
            and dr is not None
            and 0.01 <= dr <= 0.5
        ):
            decay_rate_val = dr
        elif local_result and local_result.get("decay_rate") is not None:
            dr_local = local_result["decay_rate"]
            if isinstance(dr_local, (int, float)) and 0.01 <= dr_local <= 0.5:
                decay_rate_val = float(dr_local)

        # --- Enhanced metadata from Improvement Report ---
        if unified_result:
            if unified_result.speaker:
                merged_metadata["speaker"] = unified_result.speaker
            if unified_result.causal_chain:
                merged_metadata["causal_chain"] = unified_result.causal_chain
            if unified_result.event_date:
                merged_metadata["event_date"] = unified_result.event_date

        record = MemoryRecordCreate(
            tenant_id=tenant_id,
            context_tags=effective_context_tags,
            source_session_id=source_session_id,
            agent_id=agent_id,
            namespace=namespace,
            type=memory_type,
            text=text,
            key=key,
            embedding=embedding_result.embedding,
            entities=entities,
            relations=relations,
            metadata=merged_metadata,
            timestamp=timestamp or chunk.timestamp,
            confidence=conf,
            importance=importance,
            decay_rate=decay_rate_val,
            provenance=Provenance(
                source=MemorySource.AGENT_INFERRED,
                evidence_refs=([chunk.source_turn_id] if chunk.source_turn_id else []),
                model_version=embedding_result.model,
            ),
        )
        stored = await self.store.upsert(record)

        # --- Prospective indexing: store implications as linked memories ---
        if stored is not None:
            await self._store_prospective_indexes(
                tenant_id=tenant_id,
                source_record=stored,
                unified_result=unified_result,
                context_tags=effective_context_tags,
                source_session_id=source_session_id,
                agent_id=agent_id,
                namespace=namespace,
                timestamp=timestamp or chunk.timestamp,
            )

        return (stored, gate_result)

    async def encode_batch(
        self,
        tenant_id: str,
        chunks: list[SemanticChunk],
        context_tags: list[str] | None = None,
        source_session_id: str | None = None,
        agent_id: str | None = None,
        namespace: str | None = None,
        timestamp: datetime | None = None,
        request_metadata: dict[str, Any] | None = None,
        memory_type_override: MemoryType | None = None,
        return_gate_results: bool = False,
        unified_results: list[UnifiedExtractionResult | None] | None = None,
    ):
        """Encode chunks using a 4-phase batched pipeline.

        Phase 1: Gate + redact all chunks (CPU only, no network calls).
        Phase 2: Batch-embed surviving texts in ONE API call.
        Phase 3: Batch-extract entities and relations (concurrent).
        Phase 4: Upsert records (bounded concurrency).
        """
        import time as _phase_time

        _t_scan_start = _phase_time.perf_counter()
        existing_dicts = await self._get_existing_for_gate(tenant_id)
        _t_scan_end = _phase_time.perf_counter()

        from ...core.config import get_settings as _cfg_phase1

        # ---- Phase 0.5: Batch PII span detection BEFORE gate (event-loop coalescing) ----
        # Running pii_span_detection inside the gate thread causes N_chunks sequential GPU
        # calls per request x 4 concurrent gate threads = heavy GPU serialization.
        # By running it here via _BatchingSpanPredictor, all 10 concurrent callers'
        # texts are batched into ONE DeBERTa forward pass (~20ms vs 300ms serialized).
        _precomputed_pii_flags: list[bool] = [False] * len(chunks)
        if (
            not self._use_unified_write_path()
            and self.local_extractor
            and self.local_extractor.available
        ):
            _mp_pii = self.local_extractor.modelpack
            if getattr(_mp_pii, "has_task_model", lambda _: False)("pii_span_detection"):
                try:
                    if self._pii_span_batcher is None:
                        self._pii_span_batcher = _BatchingSpanPredictor(
                            _mp_pii, "pii_span_detection"
                        )
                    _wg_ref = self.write_gate
                    # Quick regex pre-check — avoids GPU for obvious PII or empty texts
                    _pii_regex_hit = [
                        (not c.text.strip()) or any(p.search(c.text) for p in _wg_ref._pii_patterns)
                        for c in chunks
                    ]
                    _needs_model_idx = [i for i, hit in enumerate(_pii_regex_hit) if not hit]
                    if _needs_model_idx:
                        _pii_texts = [chunks[i].text for i in _needs_model_idx]
                        _pii_spans = await self._pii_span_batcher.predict_batch(_pii_texts)
                        for _j, _i in enumerate(_needs_model_idx):
                            _pred = _pii_spans[_j] if _j < len(_pii_spans) else None
                            _precomputed_pii_flags[_i] = bool(_pred is not None and _pred.spans)
                    for _i, _hit in enumerate(_pii_regex_hit):
                        if _hit and chunks[_i].text.strip():
                            _precomputed_pii_flags[_i] = True
                except Exception:
                    pass  # fallback: gate will call _predict_pii per chunk

        # ---- Phase 1: Gate + Redact in thread executor ----
        # novelty_pair sklearn model takes ~6ms x 50 existing memories = 300ms per chunk.
        # Running in _GATE_EXECUTOR frees the event loop for embedding/DeBERTa batching.
        gate_results_list: list[dict] = []
        _ur_list = unified_results if unified_results is not None else [None] * len(chunks)
        _cfg = _cfg_phase1().features

        _wg = self.write_gate
        _rd = self.redactor
        _cfg_snap = _cfg
        _pii_flags_for_gate = _precomputed_pii_flags

        def _run_gate() -> list[tuple[int, SemanticChunk, WriteGateResult, str]]:
            _surviving: list[tuple[int, SemanticChunk, WriteGateResult, str]] = []
            # Mega-batch all chunk x memory novelty pairs in ONE sklearn call (~10ms total
            # instead of N_chunks x predict_pair_proba_batch = N_chunks x ~10ms).
            _novelties: list[float] = _wg.compute_novelty_batch(
                list(chunks), existing_memories=existing_dicts
            )
            for _idx, _chunk in enumerate(chunks):
                _ur = _ur_list[_idx] if _idx < len(_ur_list) else None
                _gate = _wg.evaluate(
                    _chunk,
                    existing_memories=existing_dicts,
                    unified_result=_ur,
                    precomputed_novelty=_novelties[_idx],
                    precomputed_pii=(
                        _pii_flags_for_gate[_idx] if _idx < len(_pii_flags_for_gate) else None
                    ),
                )
                if _gate.decision == WriteDecision.SKIP:
                    _surviving.append((_idx, _chunk, _gate, ""))  # empty text signals SKIP
                    continue
                _text = _chunk.text
                if _gate.redaction_required and not (
                    _cfg_snap.use_llm_enabled
                    and _cfg_snap.use_llm_pii_redaction
                    and _ur
                    and getattr(_ur, "pii_spans", None)
                ):
                    _text = _rd.redact(_text).redacted_text
                _surviving.append((_idx, _chunk, _gate, _text))
            return _surviving

        # ---- Phase 0.75: Pre-start span prediction BEFORE gate (overlap with gate) ----
        # The gate runs for ~80ms in a thread; DeBERTa spans take ~130ms on the GPU.
        # Starting spans BEFORE the gate means they complete DURING gate execution,
        # effectively eliminating spans from the critical path (saves ~80ms).
        # We predict spans on ALL pre-gate texts; after gate we map surviving chunk spans.
        # Redacted chunks (text changed after gate) will fall back to spaCy.
        _pre_gate_span_task: asyncio.Task | None = None
        if (
            self.local_extractor
            and self.local_extractor.available
            and not self._use_unified_write_path()
        ):
            _mp_pre = self.local_extractor.modelpack
            if getattr(_mp_pre, "has_task_model", lambda _: False)("fact_extraction_structured"):
                try:
                    if self._span_batcher is None:
                        self._span_batcher = _BatchingSpanPredictor(
                            _mp_pre, "fact_extraction_structured"
                        )
                    _all_chunk_texts = [c.text for c in chunks]
                    _pre_gate_span_task = asyncio.get_running_loop().create_task(
                        self._span_batcher.predict_batch(_all_chunk_texts)
                    )
                except Exception:
                    pass

        _t_gate_start = _phase_time.perf_counter()
        _all_gate_results = await asyncio.get_running_loop().run_in_executor(
            _GATE_EXECUTOR, _run_gate
        )
        _t_gate_end = _phase_time.perf_counter()

        surviving: list[tuple[int, SemanticChunk, WriteGateResult, str]] = []
        for _idx, _chunk, _gate, _text in _all_gate_results:
            if return_gate_results:
                gate_results_list.append(_gate_result_to_dict(_gate))
            if _gate.decision != WriteDecision.SKIP:
                surviving.append((_idx, _chunk, _gate, _text))

        if not surviving:
            return ([], gate_results_list if return_gate_results else None, [], [])

        # ---- Phase 1.5: Unified extraction (when LLM flags enabled) ----
        if unified_results is None:
            unified_results = [None] * len(chunks)
        # Map unified_results (by chunk index) to surviving
        surviving_unified: list[UnifiedExtractionResult | None] = []
        for idx, chunk, _, _ in surviving:
            ur = unified_results[idx] if idx < len(unified_results) else None
            surviving_unified.append(ur)

        if (
            all(r is None for r in surviving_unified)
            and self._use_unified_write_path()
            and self.unified_extractor
        ):
            tasks = [self.unified_extractor.extract(chunk) for _idx, chunk, _gr, _txt in surviving]
            raw_results = await asyncio.gather(*tasks, return_exceptions=True)
            for i, res in enumerate(raw_results):
                if i < len(surviving_unified):
                    surviving_unified[i] = res if not isinstance(res, BaseException) else None

        unified_results = surviving_unified

        # Apply LLM PII spans to texts before embedding (merge with regex redaction)
        from ...core.config import get_settings as _get_settings

        cfg = _get_settings().features
        final_texts: list[str] = []
        for i, (_idx, chunk, gate_result, text) in enumerate(surviving):
            ures = unified_results[i] if i < len(unified_results) else None
            if (
                ures is not None
                and getattr(ures, "pii_spans", None)
                and cfg.use_llm_enabled
                and cfg.use_llm_pii_redaction
            ):
                pii_spans = [(s.start, s.end, s.pii_type) for s in ures.pii_spans]
                text = self.redactor.redact(chunk.text, additional_spans=pii_spans).redacted_text
            final_texts.append(text)

        # ---- Phase 2 + 2.5: Embed + DeBERTa spans CONCURRENTLY ----
        # Pre-gate task (Phase 0.75) predicted spans on original chunk.text.
        # Map non-redacted texts from pre-gate, start DeBERTa for redacted-only.
        # Run DeBERTa redacted + embedding concurrently.

        # Step 1: Map pre-gate spans (fast — already done or near-instant await)
        fact_spans_batch: list = [None] * len(final_texts)
        if _pre_gate_span_task is not None:
            try:
                _pre_gate_spans = await _pre_gate_span_task  # ~0ms if gate took >100ms
                for i, (orig_idx, chunk, _gate_result, text) in enumerate(surviving):
                    if orig_idx < len(_pre_gate_spans) and text == chunk.text:
                        fact_spans_batch[i] = _pre_gate_spans[orig_idx]
            except Exception:
                pass

        # Step 2: Start DeBERTa for redacted texts only (typically ~few texts, not all)
        _redacted_span_task: asyncio.Task | None = None
        _redacted_indices: list[int] = []
        if self._span_batcher is not None:
            _redacted_indices = [i for i in range(len(final_texts)) if fact_spans_batch[i] is None]
            if _redacted_indices:
                _redacted_texts = [final_texts[i] for i in _redacted_indices]
                _redacted_span_task = asyncio.get_running_loop().create_task(
                    self._span_batcher.predict_batch(_redacted_texts)
                )

        # Step 3: Embed (runs concurrently with redacted DeBERTa above)
        _t_p2 = _phase_time.perf_counter()
        texts_to_embed = final_texts
        embedding_results = await self.embeddings.embed_batch(texts_to_embed)
        _t_p3 = _phase_time.perf_counter()

        # Step 4: Await redacted DeBERTa spans
        if _redacted_span_task is not None:
            try:
                _redacted_spans = await _redacted_span_task
                for _j, _i in enumerate(_redacted_indices):
                    if _j < len(_redacted_spans):
                        fact_spans_batch[_i] = _redacted_spans[_j]
            except Exception:
                pass

        _t_p25 = _phase_time.perf_counter()

        # ---- Start Phase 3.5 early (concurrent with Phase 3 entity/relation extraction) ----
        # Phase 3.5 only needs fact_spans_batch and final_texts (ready after Phase 2.5b),
        # NOT entities_batch or relations_batch. Running it concurrently with Phase 3
        # overlaps ~28ms of CPU-bound sklearn work.
        _local_extractor = self.local_extractor
        _surviving_chunks = [s[1] for s in surviving]
        _ce = self.constraint_extractor
        _ev_loop = asyncio.get_running_loop()
        _n = len(surviving)

        if _local_extractor is not None and _local_extractor.available:
            _fb = fact_spans_batch

            def _run_local_batch() -> tuple[list[dict[str, Any] | None], list[list[Any]]]:
                local_out: list[dict[str, Any] | None] = []
                for _i in range(_n):
                    try:
                        local_out.append(
                            _local_extractor.extract_sync_direct(
                                final_texts[_i],
                                precomputed_spans=_fb[_i] if _i < len(_fb) else None,
                                skip_pii=True,
                                skip_slow_router=True,
                            )
                        )
                    except Exception:
                        local_out.append(None)
                constraint_out: list[list[Any]] = []
                for _chunk in _surviving_chunks:
                    try:
                        constraint_out.append(_ce.extract(_chunk))
                    except Exception:
                        constraint_out.append([])
                return local_out, constraint_out

            _local_future = _ev_loop.run_in_executor(None, _run_local_batch)
        else:

            def _run_constraints_only() -> tuple[list[dict[str, Any] | None], list[list[Any]]]:
                constraint_out: list[list[Any]] = []
                for _chunk in _surviving_chunks:
                    try:
                        constraint_out.append(_ce.extract(_chunk))
                    except Exception:
                        constraint_out.append([])
                return ([None] * _n, constraint_out)

            _local_future = _ev_loop.run_in_executor(None, _run_constraints_only)

        # ---- Phase 3: Batch extract entities ----
        # Fast path: derive entities from pre-computed DeBERTa spans (no spaCy call).
        # Fallback to spaCy only for texts where DeBERTa found no spans.
        entities_batch: list[list[EntityMention]] = []
        if self.entity_extractor and getattr(self.entity_extractor, "llm", None):
            entities_batch = await self.entity_extractor.extract_batch(texts_to_embed)
        else:
            _loop = asyncio.get_running_loop()
            _spacy_indices = [i for i, s in enumerate(fact_spans_batch) if s is None]
            entities_batch = [
                _entities_from_fact_spans(texts_to_embed[i], fact_spans_batch[i])
                if fact_spans_batch[i] is not None
                else []
                for i in range(len(texts_to_embed))
            ]
            if _spacy_indices:
                _ent_extractor = self.entity_extractor
                _spacy_fn = (
                    _ent_extractor._spacy_extract if _ent_extractor else _ner_entities_for_text
                )
                _spacy_results = list(
                    await asyncio.gather(
                        *[
                            _loop.run_in_executor(_SPACY_EXECUTOR, _spacy_fn, texts_to_embed[i])
                            for i in _spacy_indices
                        ]
                    )
                )
                for _j, _i in enumerate(_spacy_indices):
                    entities_batch[_i] = _spacy_results[_j]
        _t_p3r = _phase_time.perf_counter()

        relations_batch: list[list[Relation]] = []
        if self.relation_extractor:
            relation_items = [
                (text, [e.normalized for e in entities])
                for text, entities in zip(texts_to_embed, entities_batch, strict=True)
            ]
            if not getattr(self.relation_extractor, "llm", None) and hasattr(
                self.relation_extractor, "extract_batch_with_spans"
            ):
                relations_batch = self.relation_extractor.extract_batch_with_spans(
                    relation_items, fact_spans_batch
                )
            else:
                relations_batch = await self.relation_extractor.extract_batch(relation_items)
        else:
            relations_batch = [_ner_relations_for_text(text) for text in texts_to_embed]
        _t_p35 = _phase_time.perf_counter()

        # ---- Await Phase 3.5 results ----
        local_results_batch, constraint_results_batch = await _local_future
        _t_p4 = _phase_time.perf_counter()

        # ---- Phase 4: Upsert (bounded concurrency) ----
        results: list[MemoryRecord] = []

        async def _process_chunk(idx: int) -> MemoryRecord | None:
            _oi, chunk, gate_result, _ = surviving[idx]
            text = final_texts[idx]
            embedding_result = embedding_results[idx]
            unified_res = unified_results[idx] if idx < len(unified_results) else None
            local_res = local_results_batch[idx] if idx < len(local_results_batch) else None

            from ...core.config import get_settings as _gs

            settings = _gs().features

            # Use unified entities/relations for graph sync when unified path enabled
            if self._use_unified_write_path() and unified_res is not None:
                entities = unified_res.entities if unified_res.entities else entities_batch[idx]
                relations = unified_res.relations if unified_res.relations else relations_batch[idx]
            else:
                entities = entities_batch[idx]
                relations = relations_batch[idx]
            # text from surviving is already redacted (incl. LLM spans applied above)

            memory_type = memory_type_override
            if (
                memory_type is None
                and unified_res
                and settings.use_llm_enabled
                and settings.use_llm_memory_type
                and unified_res.memory_type
            ):
                try:
                    memory_type = MemoryType(unified_res.memory_type)
                except ValueError:
                    structlog.get_logger(__name__).debug(
                        "invalid_llm_memory_type",
                        raw_type=unified_res.memory_type,
                    )
            if memory_type is None and local_res and local_res.get("memory_type"):
                try:
                    memory_type = MemoryType(local_res["memory_type"])
                except (ValueError, KeyError):
                    pass
            if memory_type is None:
                memory_type = (
                    gate_result.memory_types[0]
                    if gate_result.memory_types
                    else MemoryType.EPISODIC_EVENT
                )

            # Constraint extraction: unified LLM or precomputed batch (Phase 3.5 thread)
            if unified_res and settings.use_llm_enabled and settings.use_llm_constraint_extractor:
                extracted_constraints = unified_res.constraints
            else:
                # Use precomputed result from Phase 3.5 thread batch to avoid blocking
                # the event loop with 2 sklearn calls per chunk (saves ~30ms x N_chunks).
                extracted_constraints = (
                    constraint_results_batch[idx]
                    if idx < len(constraint_results_batch)
                    else self.constraint_extractor.extract(chunk)
                )
            constraint_dicts = [c.to_dict() for c in extracted_constraints]

            # If high-confidence constraint extracted and no API/LLM override, override memory type
            if (
                memory_type_override is None
                and not (
                    unified_res
                    and settings.use_llm_enabled
                    and settings.use_llm_memory_type
                    and unified_res.memory_type
                )
                and extracted_constraints
                and any(c.confidence >= 0.7 for c in extracted_constraints)
            ):
                memory_type = MemoryType.CONSTRAINT

            if memory_type == MemoryType.CONSTRAINT and extracted_constraints:
                from ...extraction.constraint_extractor import ConstraintExtractor

                key = ConstraintExtractor.constraint_fact_key(extracted_constraints[0])
            else:
                key = self._generate_key(chunk, memory_type) or ""

            importance = gate_result.importance
            if unified_res and settings.use_llm_enabled and settings.use_llm_write_gate_importance:
                importance = unified_res.importance
            elif local_res and "importance" in local_res:
                importance = local_res["importance"]

            system_metadata: dict[str, Any] = {
                "chunk_type": chunk.chunk_type.value,
                "source_turn_id": chunk.source_turn_id,
                "source_role": chunk.source_role,
            }
            if constraint_dicts:
                system_metadata["constraints"] = constraint_dicts
            merged_metadata = {**system_metadata, **(request_metadata or {})}

            effective_ct = context_tags or []
            if (
                not effective_ct
                and unified_res
                and settings.use_llm_enabled
                and settings.use_llm_context_tags
                and hasattr(unified_res, "context_tags")
                and unified_res.context_tags
            ):
                effective_ct = unified_res.context_tags
            elif not effective_ct and local_res and local_res.get("context_tags"):
                effective_ct = local_res["context_tags"]

            conf = chunk.confidence
            if (
                unified_res
                and settings.use_llm_enabled
                and settings.use_llm_confidence
                and hasattr(unified_res, "confidence")
            ):
                conf = unified_res.confidence
            elif local_res and "confidence" in local_res:
                conf = local_res["confidence"]

            decay_rate_val = None
            dr2 = getattr(unified_res, "decay_rate", None) if unified_res else None
            if (
                unified_res
                and settings.use_llm_enabled
                and settings.use_llm_decay_rate
                and dr2 is not None
                and 0.01 <= dr2 <= 0.5
            ):
                decay_rate_val = dr2
            elif local_res and local_res.get("decay_rate") is not None:
                dr_local = local_res["decay_rate"]
                if isinstance(dr_local, (int, float)) and 0.01 <= dr_local <= 0.5:
                    decay_rate_val = float(dr_local)

            record_create = MemoryRecordCreate(
                tenant_id=tenant_id,
                context_tags=effective_ct,
                source_session_id=source_session_id,
                agent_id=agent_id,
                namespace=namespace,
                type=memory_type,
                text=text,
                key=key,
                embedding=embedding_result.embedding,
                entities=entities,
                relations=relations,
                metadata=merged_metadata,
                timestamp=chunk.timestamp,
                confidence=conf,
                importance=importance,
                decay_rate=decay_rate_val,
                provenance=Provenance(
                    source=MemorySource.AGENT_INFERRED,
                    evidence_refs=([chunk.source_turn_id] if chunk.source_turn_id else []),
                    model_version=embedding_result.model,
                ),
            )
            stored = await self.store.upsert(record_create)
            return stored

        _t_p4_start = _phase_time.perf_counter()
        tasks = [_process_chunk(i) for i in range(len(surviving))]  # type: ignore[misc]
        stored_results = await asyncio.gather(*tasks, return_exceptions=True)
        _t_p4_end = _phase_time.perf_counter()

        for res in stored_results:
            if isinstance(res, BaseException):
                structlog.get_logger(__name__).error("encode_batch_upsert_failed", error=str(res))
                continue
            if res is not None:
                results.append(cast("MemoryRecord", res))
                existing_dicts.append({"text": cast("MemoryRecord", res).text})

        structlog.get_logger("encode_timing").info(
            "encode_batch_full_timing",
            scan_ms=round((_t_scan_end - _t_scan_start) * 1000, 1),
            gate_ms=round((_t_gate_end - _t_gate_start) * 1000, 1),
            embed_ms=round((_t_p3 - _t_p2) * 1000, 1),
            spans_ms=round((_t_p25 - _t_p3) * 1000, 1),
            ner_ms=round((_t_p3r - _t_p25) * 1000, 1),
            rel_ms=round((_t_p35 - _t_p3r) * 1000, 1),
            local_ms=round((_t_p4 - _t_p35) * 1000, 1),
            upsert_ms=round((_t_p4_end - _t_p4_start) * 1000, 1),
        )

        return (
            results,
            (gate_results_list if return_gate_results else None),
            unified_results,
            local_results_batch,
        )

    async def search(
        self,
        tenant_id: str,
        query: str,
        top_k: int = 10,
        context_filter: list[str] | None = None,
        filters: dict[str, Any] | None = None,
        query_embedding: list[float] | None = None,
    ) -> list[MemoryRecord]:
        if query_embedding is None:
            emb_result = await self.embeddings.embed(query)
            embedding = emb_result.embedding
        else:
            embedding = query_embedding
        results = await self.store.vector_search(
            tenant_id,
            embedding=embedding,
            top_k=top_k,
            context_filter=context_filter,
            filters=filters,
        )
        # Batch update access tracking: atomic increment to avoid lost update (BUG-02)
        now = datetime.now(UTC)
        for record in results:
            record.access_count += 1
            record.last_accessed_at = now
        if results:
            if hasattr(self.store, "increment_access_counts"):
                await self.store.increment_access_counts([r.id for r in results], now)
            else:
                import asyncio

                await asyncio.gather(
                    *[
                        self.store.update(
                            record.id,
                            {
                                "access_count": record.access_count,
                                "last_accessed_at": now,
                            },
                            increment_version=False,
                        )
                        for record in results
                    ]
                )
        return results

    async def deactivate_constraints_by_key(
        self,
        tenant_id: str,
        constraint_key: str,
        superseded_by_key: str | None = None,
    ) -> int:
        """Deactivate previous episodic CONSTRAINT records with the same fact key."""
        if hasattr(self.store, "deactivate_constraints_by_key"):
            try:
                return await self.store.deactivate_constraints_by_key(
                    tenant_id,
                    constraint_key,
                    superseded_by_key=superseded_by_key,
                )
            except TypeError:
                # Backward compatibility for stores with legacy two-arg signature.
                return await self.store.deactivate_constraints_by_key(tenant_id, constraint_key)
        return 0

    async def get_recent(
        self,
        tenant_id: str,
        limit: int = 20,
        memory_types: list[MemoryType] | None = None,
    ) -> list[MemoryRecord]:
        filters: dict[str, Any] = {"status": MemoryStatus.ACTIVE.value}
        if memory_types:
            filters["type"] = [t.value for t in memory_types]
        return await self.store.scan(
            tenant_id,
            filters=filters,
            order_by="-timestamp",
            limit=limit,
        )

    async def _store_prospective_indexes(
        self,
        tenant_id: str,
        source_record: MemoryRecord,
        unified_result: UnifiedExtractionResult | None,
        context_tags: list[str],
        source_session_id: str | None,
        agent_id: str | None,
        namespace: str | None,
        timestamp: datetime,
    ) -> None:
        """Store prospective implication indexes as linked memory records.

        Each implication is embedded and stored as a separate record with
        type=EPISODIC_EVENT and metadata linking back to the source memory.
        At retrieval time, queries match against both the original memory
        embedding and the prospective index embeddings.
        """
        from ...core.config import get_settings as _gs

        features = _gs().features
        if not features.prospective_indexing_enabled:
            return

        implications: list[str] = []

        # Try unified LLM result first (already extracted during write)
        if unified_result and unified_result.prospective_implications:
            implications = unified_result.prospective_implications

        # Fallback: use dedicated prospective indexer if LLM is enabled
        if not implications and features.use_llm_enabled:
            try:
                from ...extraction.prospective_indexer import ProspectiveIndexer
                from ...utils.llm import get_internal_llm_client

                llm = get_internal_llm_client()
                if llm is not None:
                    indexer = ProspectiveIndexer(
                        llm,
                        max_implications=features.prospective_index_count,
                    )
                    indexes = await indexer.generate(
                        source_record.text,
                        memory_id=str(source_record.id),
                    )
                    implications = [idx.implication for idx in indexes]
            except Exception as exc:
                structlog.get_logger(__name__).debug(
                    "prospective_indexer_fallback_failed",
                    error=str(exc),
                )

        if not implications:
            return

        # Embed and store each implication
        try:
            texts = implications[: features.prospective_index_count]
            embed_results = await self.embeddings.embed_batch(texts)

            for imp_text, emb_result in zip(texts, embed_results, strict=False):
                imp_key = f"prospective:{source_record.id}:{hashlib.sha256(imp_text.encode()).hexdigest()[:12]}"
                imp_record = MemoryRecordCreate(
                    tenant_id=tenant_id,
                    context_tags=context_tags,
                    source_session_id=source_session_id,
                    agent_id=agent_id,
                    namespace=namespace,
                    type=MemoryType.EPISODIC_EVENT,
                    text=imp_text,
                    key=imp_key,
                    embedding=emb_result.embedding,
                    entities=[],
                    relations=[],
                    metadata={
                        "prospective_source_id": str(source_record.id),
                        "prospective_source_text": source_record.text[:500],
                        "is_prospective_index": True,
                    },
                    timestamp=timestamp,
                    confidence=source_record.confidence * 0.9,
                    importance=source_record.importance * 0.8,
                    provenance=Provenance(
                        source=MemorySource.AGENT_INFERRED,
                        evidence_refs=[str(source_record.id)],
                    ),
                )
                await self.store.upsert(imp_record)
        except Exception as exc:
            structlog.get_logger(__name__).warning(
                "prospective_index_storage_failed",
                error=str(exc),
                source_id=str(source_record.id),
            )

    def _generate_key(self, chunk: SemanticChunk, memory_type: MemoryType) -> str | None:
        """Generate a stable, unique key for deduplication.

        Uses a content-based hash so that distinct facts sharing the same
        first entity (e.g. "Italian food" vs "Italian music") receive
        different keys and are never silently overwritten.
        """
        if memory_type not in (
            MemoryType.PREFERENCE,
            MemoryType.SEMANTIC_FACT,
            MemoryType.CONSTRAINT,
        ):
            return None

        text_normalized = chunk.text.strip().lower()
        content_hash = hashlib.sha256(text_normalized.encode()).hexdigest()[:16]

        # Include first entity for human readability
        entity_prefix = ""
        if chunk.entities:
            entity_prefix = chunk.entities[0].lower().replace(" ", "_") + ":"

        return f"{memory_type.value}:{entity_prefix}{content_hash}"
