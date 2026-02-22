"""Hippocampal store: episodic memory with write gate, embedding, and vector store."""

from __future__ import annotations

import asyncio
import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any

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

    def _use_unified_write_path(self) -> bool:
        """True when any write-path LLM flag is enabled and we have a unified extractor."""
        if self.unified_extractor is None:
            return False
        from ...core.config import get_settings

        s = get_settings().features
        return (
            s.use_llm_constraint_extractor
            or s.use_llm_write_time_facts
            or s.use_llm_salience_refinement
            or s.use_llm_pii_redaction
            or s.use_llm_write_gate_importance
        )

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
        unified_result: UnifiedExtractionResult | None = None
        if self._use_unified_write_path() and self.unified_extractor:
            unified_result = await self.unified_extractor.extract(chunk)

        gate_result = self.write_gate.evaluate(
            chunk,
            existing_memories=existing_memories,
            unified_result=unified_result,
        )
        if gate_result.decision == WriteDecision.SKIP:
            return (None, gate_result)

        text = chunk.text
        if gate_result.redaction_required:
            from ...core.config import get_settings

            pii_spans = None
            if (
                unified_result
                and unified_result.pii_spans
                and get_settings().features.use_llm_pii_redaction
            ):
                pii_spans = [(s.start, s.end, s.pii_type) for s in unified_result.pii_spans]
            redaction_result = self.redactor.redact(text, additional_spans=pii_spans)
            text = redaction_result.redacted_text
        elif unified_result and unified_result.pii_spans and self._use_unified_write_path():
            from ...core.config import get_settings

            if get_settings().features.use_llm_pii_redaction:
                pii_spans = [(s.start, s.end, s.pii_type) for s in unified_result.pii_spans]
                redaction_result = self.redactor.redact(text, additional_spans=pii_spans)
                text = redaction_result.redacted_text

        embedding_result = await self.embeddings.embed(text)

        entities: list[EntityMention] = []
        if self.entity_extractor:
            entities = await self.entity_extractor.extract(text)
        elif chunk.entities:
            entities = [
                EntityMention(text=e, normalized=e, entity_type="CONCEPT") for e in chunk.entities
            ]

        relations: list[Relation] = []
        if self.relation_extractor:
            entity_texts = [e.normalized for e in entities]
            relations = await self.relation_extractor.extract(text, entities=entity_texts)

        # Use caller-provided memory_type override, else LLM memory_type, else gate/constraint
        from ...core.config import get_settings as _get_settings

        settings = _get_settings().features
        memory_type = memory_type_override
        if (
            memory_type is None
            and unified_result
            and settings.use_llm_memory_type
            and unified_result.memory_type
        ):
            try:
                memory_type = MemoryType(unified_result.memory_type)
            except ValueError:
                pass
        if memory_type is None:
            memory_type = (
                gate_result.memory_types[0]
                if gate_result.memory_types
                else MemoryType.EPISODIC_EVENT
            )

        # Constraint extraction: unified or rule-based
        if unified_result and settings.use_llm_constraint_extractor:
            extracted_constraints = unified_result.constraints
        else:
            extracted_constraints = self.constraint_extractor.extract(chunk)
        constraint_dicts = [c.to_dict() for c in extracted_constraints]

        # If high-confidence constraint extracted and no API/LLM override, override memory type
        if (
            memory_type_override is None
            and not (unified_result and settings.use_llm_memory_type and unified_result.memory_type)
            and extracted_constraints
            and any(c.confidence >= 0.7 for c in extracted_constraints)
        ):
            memory_type = MemoryType.CONSTRAINT

        if memory_type == MemoryType.CONSTRAINT and extracted_constraints:
            from ...extraction.constraint_extractor import ConstraintExtractor

            key = ConstraintExtractor.constraint_fact_key(extracted_constraints[0])
        else:
            key = self._generate_key(chunk, memory_type)

        # Importance: unified or gate
        importance = gate_result.importance
        if unified_result and settings.use_llm_write_gate_importance:
            importance = unified_result.importance

        # Merge request-level metadata with system metadata; request metadata wins on conflict
        system_metadata: dict[str, Any] = {
            "chunk_type": chunk.chunk_type.value,
            "source_turn_id": chunk.source_turn_id,
            "source_role": chunk.source_role,
        }
        if constraint_dicts:
            system_metadata["constraints"] = constraint_dicts
        if request_metadata:
            merged_metadata = {**system_metadata, **request_metadata}
        else:
            merged_metadata = system_metadata

        effective_context_tags = context_tags or []
        if (
            not effective_context_tags
            and unified_result
            and settings.use_llm_context_tags
            and hasattr(unified_result, "context_tags")
            and unified_result.context_tags
        ):
            effective_context_tags = unified_result.context_tags

        conf = chunk.confidence
        if unified_result and settings.use_llm_confidence and hasattr(unified_result, "confidence"):
            conf = unified_result.confidence

        decay_rate_val: float | None = None
        if (
            unified_result
            and settings.use_llm_decay_rate
            and getattr(unified_result, "decay_rate", None) is not None
            and 0.01 <= unified_result.decay_rate <= 0.5
        ):
            decay_rate_val = unified_result.decay_rate

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
        existing = await self.store.scan(
            tenant_id,
            filters={"status": MemoryStatus.ACTIVE.value},
            limit=50,
            order_by="-timestamp",
        )
        existing_dicts = [{"text": m.text} for m in existing]

        # ---- Phase 1: Gate + Redact (no network calls) ----
        from ...core.config import get_settings as _cfg_phase1

        surviving: list[tuple[int, SemanticChunk, WriteGateResult, str]] = []
        gate_results_list: list[dict] = []
        _ur_list = unified_results if unified_results is not None else [None] * len(chunks)
        _cfg = _cfg_phase1().features

        for idx, chunk in enumerate(chunks):
            ur = _ur_list[idx] if idx < len(_ur_list) else None
            gate_result = self.write_gate.evaluate(
                chunk, existing_memories=existing_dicts, unified_result=ur
            )
            if return_gate_results:
                gate_results_list.append(_gate_result_to_dict(gate_result))

            if gate_result.decision == WriteDecision.SKIP:
                continue

            text = chunk.text
            if gate_result.redaction_required and not (
                _cfg.use_llm_pii_redaction and ur and getattr(ur, "pii_spans", None)
            ):
                redaction_result = self.redactor.redact(text)
                text = redaction_result.redacted_text

            surviving.append((idx, chunk, gate_result, text))

        if not surviving:
            return ([], gate_results_list if return_gate_results else None, [])

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
                    surviving_unified[i] = res if not isinstance(res, Exception) else None

        unified_results = surviving_unified

        # Apply LLM PII spans to texts before embedding (merge with regex redaction)
        from ...core.config import get_settings as _cfg

        cfg = _cfg().features
        final_texts: list[str] = []
        for i, (_idx, chunk, gate_result, text) in enumerate(surviving):
            if unified_results[i] and unified_results[i].pii_spans and cfg.use_llm_pii_redaction:
                pii_spans = [(s.start, s.end, s.pii_type) for s in unified_results[i].pii_spans]
                text = self.redactor.redact(chunk.text, additional_spans=pii_spans).redacted_text
            final_texts.append(text)

        # ---- Phase 2: Batch embed (ONE API call) ----
        texts_to_embed = final_texts
        embedding_results = await self.embeddings.embed_batch(texts_to_embed)

        # ---- Phase 3: Batch extract entities and relations ----
        entities_batch: list[list[EntityMention]] = []
        if self.entity_extractor:
            entities_batch = await self.entity_extractor.extract_batch(texts_to_embed)
        else:
            entities_batch = [
                [
                    EntityMention(text=e, normalized=e, entity_type="CONCEPT")
                    for e in (chunk.entities or [])
                ]
                for _idx, chunk, _gr, _txt in surviving
            ]

        relations_batch: list[list[Relation]] = []
        if self.relation_extractor:
            relation_items = [
                (text, [e.normalized for e in entities])
                for text, entities in zip(texts_to_embed, entities_batch, strict=True)
            ]
            relations_batch = await self.relation_extractor.extract_batch(relation_items)
        else:
            relations_batch = [[] for _ in surviving]

        # ---- Phase 4: Upsert (bounded concurrency) ----
        results: list[MemoryRecord] = []

        async def _process_chunk(idx: int) -> MemoryRecord | None:
            _oi, chunk, gate_result, _ = surviving[idx]
            text = final_texts[idx]
            embedding_result = embedding_results[idx]
            unified_res = unified_results[idx] if idx < len(unified_results) else None

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
                and settings.use_llm_memory_type
                and unified_res.memory_type
            ):
                try:
                    memory_type = MemoryType(unified_res.memory_type)
                except ValueError:
                    pass
            if memory_type is None:
                memory_type = (
                    gate_result.memory_types[0]
                    if gate_result.memory_types
                    else MemoryType.EPISODIC_EVENT
                )

            # Constraint extraction: unified or rule-based
            if unified_res and settings.use_llm_constraint_extractor:
                extracted_constraints = unified_res.constraints
            else:
                extracted_constraints = self.constraint_extractor.extract(chunk)
            constraint_dicts = [c.to_dict() for c in extracted_constraints]

            # If high-confidence constraint extracted and no API/LLM override, override memory type
            if (
                memory_type_override is None
                and not (unified_res and settings.use_llm_memory_type and unified_res.memory_type)
                and extracted_constraints
                and any(c.confidence >= 0.7 for c in extracted_constraints)
            ):
                memory_type = MemoryType.CONSTRAINT

            if memory_type == MemoryType.CONSTRAINT and extracted_constraints:
                from ...extraction.constraint_extractor import ConstraintExtractor

                key = ConstraintExtractor.constraint_fact_key(extracted_constraints[0])
            else:
                key = self._generate_key(chunk, memory_type)

            importance = gate_result.importance
            if unified_res and settings.use_llm_write_gate_importance:
                importance = unified_res.importance

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
                and settings.use_llm_context_tags
                and hasattr(unified_res, "context_tags")
                and unified_res.context_tags
            ):
                effective_ct = unified_res.context_tags

            conf = chunk.confidence
            if unified_res and settings.use_llm_confidence and hasattr(unified_res, "confidence"):
                conf = unified_res.confidence

            decay_rate_val: float | None = None
            if (
                unified_res
                and settings.use_llm_decay_rate
                and getattr(unified_res, "decay_rate", None) is not None
                and 0.01 <= unified_res.decay_rate <= 0.5
            ):
                decay_rate_val = unified_res.decay_rate

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

        tasks = [_process_chunk(i) for i in range(len(surviving))]
        stored_results = await asyncio.gather(*tasks, return_exceptions=True)

        for res in stored_results:
            if isinstance(res, Exception):
                import structlog

                structlog.get_logger(__name__).error("encode_batch_upsert_failed", error=str(res))
                continue
            if res is not None:
                results.append(res)
                existing_dicts.append({"text": res.text})

        return results, (gate_results_list if return_gate_results else None), unified_results

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

    async def deactivate_constraints_by_key(self, tenant_id: str, constraint_key: str) -> int:
        """Deactivate previous episodic CONSTRAINT records with the same fact key (supersession)."""
        if hasattr(self.store, "deactivate_constraints_by_key"):
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
