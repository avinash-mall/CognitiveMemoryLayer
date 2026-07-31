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
from ..working.models import SemanticChunk
from .redactor import PIIRedactor
from .write_gate import WriteDecision, WriteGate, WriteGateResult

if TYPE_CHECKING:
    from ...extraction.constraint_extractor import ConstraintExtractor
    from ...extraction.unified_write_extractor import (
        UnifiedExtractionResult,
        UnifiedWritePathExtractor,
    )


# Dedicated executor for Phase 1 (write-gate evaluation) so CPU-bound gate work
# doesn't block the event loop during embedding. Worker count configurable via
# PERFORMANCE__GATE_EXECUTOR_WORKERS; 0 = auto (min(cpu_count, 8)).
#
# Imported here rather than at module top and *not* reused by the methods below:
# pool size is read once at import, but the per-call settings reads deliberately
# re-import inside each method so `monkeypatch.setattr("src.core.config.
# get_settings", ...)` reaches them. Binding the name at module scope silently
# pins those reads to the unpatched function.
from ...core.config import get_settings as _settings_for_pool_size

_GATE_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=_settings_for_pool_size().performance.resolved_gate_workers(),
    thread_name_prefix="write_gate",
)


def _gate_result_to_dict(g: WriteGateResult) -> dict:
    """Serialize for API (eval mode)."""
    return {"decision": g.decision.value, "reason": g.reason}


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
        # Scan coalescing: avoids redundant DB queries for concurrent writes to the same tenant.
        # _scan_cache: tenant_id -> (monotonic_ts, result_dicts)
        # _scan_futures: tenant_id -> in-flight Future (deduplicate concurrent requests)
        self._scan_cache: dict[str, tuple[float, list[dict]]] = {}
        self._scan_futures: dict[str, asyncio.Future] = {}

    def _use_unified_write_path(self) -> bool:
        """True when use_llm_enabled and we have a unified extractor."""
        from ...core.config import get_settings

        if self.unified_extractor is None:
            return False

        return get_settings().features.use_llm_enabled

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

    async def encode_batch(
        self,
        tenant_id: str,
        chunks: list[SemanticChunk],
        context_tags: list[str] | None = None,
        source_session_id: str | None = None,
        agent_id: str | None = None,
        namespace: str | None = None,
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

        Each record's timestamp comes from its own ``chunk.timestamp``, which the
        chunker already stamps from the write's timestamp, so there is
        deliberately no per-call override.
        """
        import time as _phase_time

        _t_scan_start = _phase_time.perf_counter()
        existing_dicts = await self._get_existing_for_gate(tenant_id)
        _t_scan_end = _phase_time.perf_counter()

        # ---- Phase 1: Gate + Redact in thread executor ----
        from ...core.config import get_settings

        gate_results_list: list[dict] = []
        _ur_list = unified_results if unified_results is not None else [None] * len(chunks)

        _cfg = get_settings().features

        _wg = self.write_gate
        _rd = self.redactor
        _cfg_snap = _cfg

        def _run_gate() -> list[tuple[int, SemanticChunk, WriteGateResult, str]]:
            _surviving: list[tuple[int, SemanticChunk, WriteGateResult, str]] = []
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
                )
                if _gate.decision == WriteDecision.SKIP:
                    _surviving.append((_idx, _chunk, _gate, ""))  # empty text signals SKIP
                    continue
                _text = _chunk.text
                if _gate.redaction_required and not (
                    _cfg_snap.use_llm_enabled and _ur and getattr(_ur, "pii_spans", None)
                ):
                    _text = _rd.redact(_text).redacted_text
                _surviving.append((_idx, _chunk, _gate, _text))
            return _surviving

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
        cfg = get_settings().features
        final_texts: list[str] = []
        for i, (_idx, chunk, gate_result, text) in enumerate(surviving):
            ures = unified_results[i] if i < len(unified_results) else None
            if ures is not None and getattr(ures, "pii_spans", None) and cfg.use_llm_enabled:
                pii_spans = [(s.start, s.end, s.pii_type) for s in ures.pii_spans]
                text = self.redactor.redact(chunk.text, additional_spans=pii_spans).redacted_text
            final_texts.append(text)

        # ---- Temporal resolution: "yesterday" -> an absolute date ----
        # Pure regex, no LLM, anchored on each chunk's own timestamp. packet_builder
        # renders event_date on the Recent Events section, which is what lets the model
        # answer "when did X happen". Kept as one flat loop rather than folded into
        # _process_chunk so the regex passes stay off the concurrent gather.
        from ...extraction.temporal_resolver import extract_event_date

        event_dates: list[str | None] = []
        for (_idx, chunk, _gr, _txt), etext in zip(surviving, final_texts, strict=True):
            resolved = (
                extract_event_date(etext, chunk.timestamp)
                if cfg.temporal_resolution_enabled
                else None
            )
            event_dates.append(resolved.isoformat() if resolved else None)

        # ---- Phase 2: Embed ----
        _t_p2 = _phase_time.perf_counter()
        texts_to_embed = final_texts
        embedding_results = await self.embeddings.embed_batch(texts_to_embed)
        _t_p3 = _phase_time.perf_counter()

        # ---- Phase 3.5 (concurrent with Phase 3): constraint extraction in a thread ----
        _surviving_chunks = [s[1] for s in surviving]
        _ce = self.constraint_extractor
        _ev_loop = asyncio.get_running_loop()

        def _run_constraints() -> list[list[Any]]:
            constraint_out: list[list[Any]] = []
            for _chunk in _surviving_chunks:
                try:
                    constraint_out.append(_ce.extract(_chunk))
                except Exception:
                    constraint_out.append([])
            return constraint_out

        _constraints_future = _ev_loop.run_in_executor(None, _run_constraints)

        # ---- Phase 3: Batch extract entities/relations (LLM extractors only) ----
        entities_batch: list[list[EntityMention]] = []
        if self.entity_extractor and getattr(self.entity_extractor, "llm", None):
            entities_batch = await self.entity_extractor.extract_batch(texts_to_embed)
        else:
            entities_batch = [[] for _ in texts_to_embed]

        relations_batch: list[list[Relation]] = []
        if self.relation_extractor and getattr(self.relation_extractor, "llm", None):
            relation_items = [
                (text, [e.normalized for e in entities])
                for text, entities in zip(texts_to_embed, entities_batch, strict=True)
            ]
            relations_batch = await self.relation_extractor.extract_batch(relation_items)
        else:
            relations_batch = [[] for _ in texts_to_embed]

        # ---- Await Phase 3.5 results ----
        constraint_results_batch = await _constraints_future
        _t_p4 = _phase_time.perf_counter()

        # ---- Phase 4: Upsert (bounded concurrency) ----
        results: list[MemoryRecord] = []

        async def _process_chunk(idx: int) -> MemoryRecord | None:
            _oi, chunk, gate_result, _ = surviving[idx]
            text = final_texts[idx]
            embedding_result = embedding_results[idx]
            unified_res = unified_results[idx] if idx < len(unified_results) else None
            settings = get_settings().features

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
                and unified_res.memory_type
            ):
                try:
                    memory_type = MemoryType(unified_res.memory_type)
                except ValueError:
                    structlog.get_logger(__name__).debug(
                        "invalid_llm_memory_type",
                        raw_type=unified_res.memory_type,
                    )
            if memory_type is None:
                memory_type = (
                    gate_result.memory_types[0]
                    if gate_result.memory_types
                    else MemoryType.EPISODIC_EVENT
                )

            # Constraint extraction: unified LLM or precomputed batch (Phase 3.5 thread)
            if unified_res and settings.use_llm_enabled:
                extracted_constraints = unified_res.constraints
            else:
                extracted_constraints = (
                    constraint_results_batch[idx]
                    if idx < len(constraint_results_batch)
                    else self.constraint_extractor.extract(chunk)
                )
            constraint_dicts = [c.to_dict() for c in extracted_constraints]

            # If high-confidence constraint extracted and no API/LLM override, override memory type
            if (
                memory_type_override is None
                and not (unified_res and settings.use_llm_enabled and unified_res.memory_type)
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
            if unified_res and settings.use_llm_enabled:
                importance = unified_res.importance

            system_metadata: dict[str, Any] = {
                "chunk_type": chunk.chunk_type.value,
                "source_turn_id": chunk.source_turn_id,
                "source_role": chunk.source_role,
            }
            if constraint_dicts:
                system_metadata["constraints"] = constraint_dicts
            # Precedence, deliberately three-tiered: the regex date goes in before the
            # merge so an explicit caller-supplied event_date wins over it, and the LLM's
            # speaker/event_date go in after so the model wins over both.
            if event_dates[idx]:
                system_metadata["event_date"] = event_dates[idx]
            merged_metadata = {**system_metadata, **(request_metadata or {})}
            if unified_res:
                if unified_res.speaker:
                    merged_metadata["speaker"] = unified_res.speaker
                if unified_res.event_date:
                    merged_metadata["event_date"] = unified_res.event_date

            effective_ct = context_tags or []
            if (
                not effective_ct
                and unified_res
                and settings.use_llm_enabled
                and getattr(unified_res, "context_tags", None)
            ):
                effective_ct = unified_res.context_tags

            conf = chunk.confidence
            if unified_res and settings.use_llm_enabled and hasattr(unified_res, "confidence"):
                conf = unified_res.confidence

            decay_rate_val = None
            dr2 = getattr(unified_res, "decay_rate", None) if unified_res else None
            if unified_res and settings.use_llm_enabled and dr2 is not None and 0.01 <= dr2 <= 0.5:
                decay_rate_val = dr2

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

        # Co-align stored records with their source chunk + extraction results.
        # stored_results[i] corresponds to surviving[i] / unified_results[i].
        # Dropping failed upserts from `results` alone would desync those indices,
        # so build all aligned lists together — the write-time facts/constraints
        # phases rely on stored[k] <-> chunk[k].
        aligned_unified: list[UnifiedExtractionResult | None] = []
        aligned_chunks: list[SemanticChunk] = []
        for i, res in enumerate(stored_results):
            if isinstance(res, BaseException):
                structlog.get_logger(__name__).error("encode_batch_upsert_failed", error=str(res))
                continue
            if res is not None:
                rec = cast("MemoryRecord", res)
                results.append(rec)
                aligned_unified.append(unified_results[i] if i < len(unified_results) else None)
                aligned_chunks.append(surviving[i][1])
                existing_dicts.append({"text": rec.text})

        structlog.get_logger("encode_timing").info(
            "encode_batch_full_timing",
            scan_ms=round((_t_scan_end - _t_scan_start) * 1000, 1),
            gate_ms=round((_t_gate_end - _t_gate_start) * 1000, 1),
            embed_ms=round((_t_p3 - _t_p2) * 1000, 1),
            extract_ms=round((_t_p4 - _t_p3) * 1000, 1),
            upsert_ms=round((_t_p4_end - _t_p4_start) * 1000, 1),
        )

        return (
            results,
            (gate_results_list if return_gate_results else None),
            aligned_unified,
            aligned_chunks,
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
        from ...core.config import get_settings

        features = get_settings().features
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
