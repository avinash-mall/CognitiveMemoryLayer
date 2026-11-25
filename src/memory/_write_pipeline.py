"""Write-path pipeline phases for MemoryOrchestrator."""

from __future__ import annotations

import asyncio
import inspect
from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING, Any

from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from .hippocampal.store import HippocampalStore
    from .neocortical.store import NeocorticalStore
    from .short_term import ShortTermMemory

logger = get_logger(__name__)


@dataclass(frozen=True)
class WritePathConfig:
    """Resolves all write-path feature flags once at the start of write()."""

    use_unified: bool
    write_time_facts: bool
    constraint_extraction: bool
    use_llm_constraints: bool
    use_llm_facts: bool
    use_llm_enabled: bool

    @classmethod
    def from_settings(cls, settings: Any, has_unified_extractor: bool) -> WritePathConfig:
        f = settings.features
        llm = f.use_llm_enabled
        return cls(
            use_unified=llm and has_unified_extractor,
            write_time_facts=f.write_time_facts_enabled,
            constraint_extraction=f.constraint_extraction_enabled,
            use_llm_constraints=llm,
            use_llm_facts=llm,
            use_llm_enabled=llm,
        )


class WritePipelineMixin:
    """Write-path phase methods mixed into MemoryOrchestrator."""

    short_term: ShortTermMemory
    hippocampal: HippocampalStore
    neocortical: NeocorticalStore

    async def _phase_ingest(
        self,
        tenant_id: str,
        content: str,
        session_id: str | None,
        turn_id: str | None,
        timestamp: datetime | None,
    ) -> list:
        """STM ingestion: return chunks for encoding."""
        stm_result = await self.short_term.ingest_turn(
            tenant_id=tenant_id,
            scope_id=session_id or tenant_id,
            text=content,
            turn_id=turn_id,
            role="user",
            timestamp=timestamp,
        )
        return stm_result.get("chunks_for_encoding", [])

    async def _phase_unified_extraction(self, chunks: list, wpc: WritePathConfig) -> list | None:
        """Run unified extractor when LLM path enabled (with optional Thalamic Surprise Triage)."""
        if not (wpc.use_unified and self.hippocampal.unified_extractor):
            return None

        from ..core.config import get_settings as _get_settings

        features = _get_settings().features
        if getattr(features, "thalamic_triage_enabled", False):
            threshold = float(getattr(features, "thalamic_surprise_threshold", 0.45) or 0.45)
            wg = getattr(self.hippocampal, "write_gate", None)
            high_surprise_indices: list[int] = []
            high_surprise_chunks: list[Any] = []
            for idx, chunk in enumerate(chunks):
                surprise = wg.compute_surprise_index(chunk) if wg is not None else chunk.salience
                chunk.metadata["surprise_index"] = surprise
                if surprise >= threshold:
                    high_surprise_indices.append(idx)
                    high_surprise_chunks.append(chunk)
                else:
                    chunk.metadata["thalamic_triaged_system1"] = True

            results: list[Any] = [None] * len(chunks)
            if high_surprise_chunks:
                extracted = await self.hippocampal.unified_extractor.extract_batch(
                    high_surprise_chunks
                )
                for pos, orig_idx in enumerate(high_surprise_indices):
                    if pos < len(extracted):
                        results[orig_idx] = extracted[pos]
            return results

        return await self.hippocampal.unified_extractor.extract_batch(chunks)

    async def _phase_deactivate_constraints(
        self,
        tenant_id: str,
        chunks: list,
        unified_results: list | None,
        wpc: WritePathConfig,
    ) -> None:
        """Deactivate previous episodic constraints by fact key (supersession)."""
        if not wpc.constraint_extraction or not chunks:
            return
        deactivate_constraints_by_key = getattr(
            self.hippocampal, "deactivate_constraints_by_key", None
        )
        if deactivate_constraints_by_key is None:
            return
        from ..extraction.constraint_extractor import ConstraintExtractor, ConstraintObject
        from .neocortical.schemas import FactCategory

        fact_keys: set[str] = set()
        new_constraints: list = []
        if unified_results and wpc.use_llm_constraints:
            for ur in unified_results:
                if ur and hasattr(ur, "constraints"):
                    for c in ur.constraints:
                        fact_keys.add(ConstraintExtractor.constraint_fact_key(c))
                        new_constraints.append(c)
        else:
            extractor = ConstraintExtractor()
            for chunk in chunks:
                for c in extractor.extract(chunk):
                    fact_keys.add(ConstraintExtractor.constraint_fact_key(c))
                    new_constraints.append(c)
        for fk in fact_keys:
            maybe_await = deactivate_constraints_by_key(tenant_id, fk)
            if inspect.isawaitable(maybe_await):
                await maybe_await

        cat_map = {
            "goal": FactCategory.GOAL,
            "value": FactCategory.VALUE,
            "state": FactCategory.STATE,
            "causal": FactCategory.CAUSAL,
            "policy": FactCategory.POLICY,
        }
        llm_client = getattr(
            getattr(self.hippocampal, "unified_extractor", None), "llm_client", None
        )
        for nc in new_constraints:
            cat = cat_map.get((nc.constraint_type or "").lower())
            if not cat:
                continue
            new_fact_key = ConstraintExtractor.constraint_fact_key(nc)
            try:
                existing = await self.neocortical.facts.get_facts_by_category(
                    tenant_id, cat, current_only=True
                )
                for old in existing:
                    if old.key in fact_keys:
                        continue
                    old_obj = ConstraintObject(
                        constraint_type=cat.value,
                        subject="user",
                        description=str(old.value),
                        scope=getattr(old, "context_tags", None) or [],
                    )
                    if await ConstraintExtractor.detect_supersession(
                        old_obj, nc, llm_client=llm_client
                    ):
                        await self.neocortical.facts.invalidate_fact(
                            tenant_id, old.key, reason="superseded"
                        )
                        maybe_await = deactivate_constraints_by_key(
                            tenant_id,
                            old.key,
                            superseded_by_key=new_fact_key,
                        )
                        if inspect.isawaitable(maybe_await):
                            await maybe_await
            except Exception as e:
                logger.warning(
                    "supersession_check_failed",
                    extra={"constraint_type": nc.constraint_type, "error": str(e)},
                    exc_info=True,
                )

    async def _phase_encode_and_store(
        self,
        tenant_id: str,
        chunks: list,
        context_tags: list[str] | None,
        session_id: str | None,
        agent_id: str | None,
        namespace: str | None,
        metadata: dict[str, Any] | None,
        memory_type_override: Any,
        eval_mode: bool,
        unified_results: list | None,
    ) -> tuple[list, list, list | None, list]:
        """Encode chunks and store in hippocampal vector store."""
        return await self.hippocampal.encode_batch(
            tenant_id=tenant_id,
            chunks=chunks,
            context_tags=context_tags,
            source_session_id=session_id,
            agent_id=agent_id,
            namespace=namespace,
            request_metadata=metadata if metadata else None,
            memory_type_override=memory_type_override,
            return_gate_results=eval_mode,
            unified_results=unified_results,
        )

    async def _phase_write_time_facts(
        self,
        tenant_id: str,
        chunks: list,
        stored: list,
        unified_results: list | None,
        timestamp: datetime | None,
        wpc: WritePathConfig,
    ) -> None:
        """Write-time fact extraction: populate semantic store immediately."""
        if not wpc.write_time_facts or not chunks or not stored:
            return
        try:
            fallback_evidence = [str(stored[0].id)] if stored else []
            all_fact_pairs: list[tuple[Any, list[str]]] = []

            if wpc.use_llm_facts and unified_results:
                for idx, ur in enumerate(unified_results):
                    if ur and hasattr(ur, "facts"):
                        evidence = [str(stored[idx].id)] if idx < len(stored) else fallback_evidence
                        for fact in ur.facts:
                            all_fact_pairs.append((fact, evidence))
            else:
                from ..extraction.write_time_facts import WriteTimeFactExtractor

                extractor = WriteTimeFactExtractor()
                for idx, chunk in enumerate(chunks):
                    evidence = [str(stored[idx].id)] if idx < len(stored) else fallback_evidence
                    fact_map: dict[tuple[str, str], Any] = {}

                    for fact in extractor.extract(chunk):
                        key = (str(getattr(fact, "key", "")), str(getattr(fact, "value", "")))
                        if key[0] and key[1] and key not in fact_map:
                            fact_map[key] = fact

                    for fact in fact_map.values():
                        all_fact_pairs.append((fact, evidence))

            if all_fact_pairs:

                async def _store_one(fact: Any, evidence: list[str]) -> None:
                    try:
                        await self.neocortical.store_fact(
                            tenant_id=tenant_id,
                            key=fact.key,
                            value=fact.value,
                            confidence=fact.confidence,
                            evidence_ids=evidence,
                            valid_from=getattr(fact, "valid_from", None) or timestamp,
                            valid_to=getattr(fact, "valid_to", None),
                        )
                    except Exception:
                        logger.warning(
                            "write_time_fact_store_failed",
                            extra={"tenant_id": tenant_id, "fact_key": getattr(fact, "key", None)},
                            exc_info=True,
                        )

                await asyncio.gather(*[_store_one(f, e) for f, e in all_fact_pairs])
        except Exception:
            logger.warning("write_time_facts_skipped", exc_info=True)

    async def _phase_write_constraints(
        self,
        tenant_id: str,
        chunks: list,
        stored: list,
        unified_results: list | None,
        timestamp: datetime | None,
        wpc: WritePathConfig,
    ) -> None:
        """Write-time constraint extraction: store constraints as semantic facts."""
        if not wpc.constraint_extraction or not chunks or not stored:
            return
        try:
            from ..extraction.constraint_extractor import ConstraintExtractor

            extractor = ConstraintExtractor()
            fallback_evidence_c = [str(stored[0].id)] if stored else []
            constraint_pairs: list[tuple[Any, str, list[str]]] = []

            if wpc.use_llm_constraints and unified_results:
                for idx, ur in enumerate(unified_results):
                    if ur and hasattr(ur, "constraints"):
                        evidence_c = (
                            [str(stored[idx].id)] if idx < len(stored) else fallback_evidence_c
                        )
                        for constraint in ur.constraints:
                            fact_key = ConstraintExtractor.constraint_fact_key(constraint)
                            constraint_pairs.append((constraint, fact_key, evidence_c))
            else:
                for idx, chunk in enumerate(chunks):
                    evidence_c = [str(stored[idx].id)] if idx < len(stored) else fallback_evidence_c
                    for constraint in extractor.extract(chunk):
                        fact_key = ConstraintExtractor.constraint_fact_key(constraint)
                        constraint_pairs.append((constraint, fact_key, evidence_c))

            if constraint_pairs:

                async def _store_constraint(c: Any, fk: str, ev: list[str]) -> bool:
                    try:
                        await self.neocortical.store_fact(
                            tenant_id=tenant_id,
                            key=fk,
                            value=c.description,
                            confidence=c.confidence,
                            evidence_ids=ev,
                            context_tags=c.scope,
                            valid_from=timestamp,
                        )
                        return True
                    except Exception:
                        logger.warning(
                            "constraint_fact_store_failed",
                            extra={"tenant_id": tenant_id, "constraint_type": c.constraint_type},
                            exc_info=True,
                        )
                        return False

                results = await asyncio.gather(
                    *[_store_constraint(c, fk, ev) for c, fk, ev in constraint_pairs]
                )
                constraints_stored = sum(1 for r in results if r)
                if constraints_stored > 0:
                    logger.info(
                        "constraints_extracted",
                        extra={"tenant_id": tenant_id, "count": constraints_stored},
                    )
        except Exception:
            logger.warning("constraint_extraction_skipped", exc_info=True)

    def _empty_write_response(self, eval_mode: bool) -> dict[str, Any]:
        """Build response when no chunks to store."""
        out: dict[str, Any] = {
            "memory_id": None,
            "chunks_created": 0,
            "message": "No significant information to store",
        }
        if eval_mode:
            out["eval_outcome"] = "skipped"
            out["eval_reason"] = "No significant information to store"
        return out

    def _build_write_response(
        self, stored: list, gate_results: list, eval_mode: bool
    ) -> dict[str, Any]:
        """Build write response dict from stored records and gate results."""
        n_stored = len(stored)
        if eval_mode:
            n_skipped = sum(1 for g in gate_results if g.get("decision") == "skip")
            if n_stored == 0:
                eval_outcome = "skipped"
                eval_reason = (
                    gate_results[0].get("reason", "unknown") if gate_results else "all skipped"
                )
            else:
                eval_outcome = "stored"
                if n_skipped:
                    reasons = [
                        g.get("reason", "") for g in gate_results if g.get("decision") == "skip"
                    ]
                    eval_reason = f"{n_stored} stored, {n_skipped} skipped: " + (
                        reasons[0] if reasons else ""
                    )
                else:
                    eval_reason = f"{n_stored} chunk(s) stored"
            return {
                "memory_id": stored[0].id if stored else None,
                "chunks_created": n_stored,
                "message": f"Stored {n_stored} memory chunks",
                "eval_outcome": eval_outcome,
                "eval_reason": eval_reason,
            }
        return {
            "memory_id": stored[0].id if stored else None,
            "chunks_created": n_stored,
            "message": f"Stored {n_stored} memory chunks",
        }

    async def _sync_to_graph(self, tenant_id: str, records: list) -> None:
        """Sync extracted entities and relations from stored records to Neo4j."""
        all_entities: list[dict[str, Any]] = []
        relation_tasks: list[Any] = []

        for record in records:
            for entity in getattr(record, "entities", None) or []:
                all_entities.append(
                    {
                        "entity": entity.normalized,
                        "entity_type": entity.entity_type,
                    }
                )
            relations = getattr(record, "relations", None) or []
            if relations:
                relation_tasks.append(self._sync_relations_to_graph(tenant_id, record, relations))

        tasks: list[Any] = []
        if all_entities:
            tasks.append(self._sync_entities_batch_to_graph(tenant_id, all_entities))
        tasks.extend(relation_tasks)

        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True)
            for r in results:
                if isinstance(r, Exception):
                    logger.warning(
                        "neo4j_sync_task_failed",
                        extra={"error": str(r)},
                        exc_info=(type(r), r, r.__traceback__),
                    )

    async def _sync_entities_batch_to_graph(
        self, tenant_id: str, entities: list[dict[str, Any]]
    ) -> None:
        """Batch-merge entity nodes to Neo4j in a single call."""
        await self.neocortical.graph.merge_nodes_batch(
            tenant_id=tenant_id,
            scope_id=tenant_id,
            nodes=entities,
        )

    async def _sync_relations_to_graph(self, tenant_id: str, record: Any, relations: list) -> None:
        """Merge relation edges for a record to Neo4j."""
        await self.neocortical.store_relations_batch(
            tenant_id=tenant_id,
            relations=relations,
            evidence_ids=[str(record.id)],
        )
