"""Hybrid retriever executing plans across memory sources."""

import asyncio
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ..core.enums import MemorySource, MemoryType
from ..core.schemas import MemoryRecord, Provenance, RetrievedMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.logging_config import get_logger
from .planner import RetrievalPlan, RetrievalSource, RetrievalStep
from .query_types import QueryAnalysis


@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""

    source: RetrievalSource
    items: list[dict[str, Any]]
    elapsed_ms: float
    success: bool
    error: str | None = None


class HybridRetriever:
    """Executes retrieval plans across multiple memory sources."""

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        cache: Any | None = None,
    ):
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.cache = cache

    async def retrieve(
        self,
        tenant_id: str,
        plan: RetrievalPlan,
        context_filter: list[str] | None = None,
    ) -> list[RetrievedMemory]:
        """Execute a retrieval plan and return results. Holistic: tenant-only."""
        all_results: list[dict[str, Any]] = []
        for group_indices in plan.parallel_steps:
            group_steps = [plan.steps[i] for i in group_indices if i < len(plan.steps)]
            group_results = await asyncio.gather(
                *[self._execute_step(tenant_id, step, context_filter) for step in group_steps],
                return_exceptions=True,
            )
            for step, result in zip(group_steps, group_results, strict=False):
                if isinstance(result, Exception):
                    continue
                if result.success and result.items:
                    all_results.extend(result.items)
                    if step.skip_if_found and result.items:
                        break
        return self._to_retrieved_memories(all_results, plan.analysis)

    async def _execute_step(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
    ) -> RetrievalResult:
        """Execute a single retrieval step. Holistic: tenant-only."""
        start = datetime.now(UTC)
        try:
            if step.source == RetrievalSource.FACTS:
                items = await self._retrieve_facts(tenant_id, step)
            elif step.source == RetrievalSource.VECTOR:
                items = await self._retrieve_vector(tenant_id, step, context_filter)
            elif step.source == RetrievalSource.GRAPH:
                items = await self._retrieve_graph(tenant_id, step)
            elif step.source == RetrievalSource.CACHE:
                items = await self._retrieve_cache(tenant_id, step)
            else:
                items = []
            elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
            return RetrievalResult(
                source=step.source,
                items=items,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (datetime.now(UTC) - start).total_seconds() * 1000
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=elapsed,
                success=False,
                error=str(e),
            )

    async def _retrieve_facts(
        self,
        tenant_id: str,
        step: RetrievalStep,
    ) -> list[dict[str, Any]]:
        """Retrieve from semantic fact store. Holistic: tenant-only."""
        results: list[dict[str, Any]] = []
        if step.key:
            fact = await self.neocortical.get_fact(tenant_id, step.key)
            if fact:
                results.append(
                    {
                        "type": MemoryType.SEMANTIC_FACT.value,
                        "source": "facts",
                        "key": fact.key,
                        "text": f"{fact.predicate}: {fact.value}",
                        "value": fact.value,
                        "confidence": fact.confidence,
                        "relevance": 1.0,
                        "record": fact,
                    }
                )
        if step.query:
            facts = await self.neocortical.text_search(tenant_id, step.query, limit=step.top_k)
            for f in facts:
                results.append(
                    {
                        "type": MemoryType.SEMANTIC_FACT.value,
                        "source": "facts",
                        "key": f.get("key"),
                        "text": f"{f.get('key', '')}: {f.get('value', '')}",
                        "value": f.get("value"),
                        "confidence": f.get("confidence", 0.5),
                        "relevance": 0.8,
                        "record": f,
                    }
                )
        return results[: step.top_k]

    async def _retrieve_vector(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve via vector similarity search. Holistic: tenant-only."""
        filters = None
        if step.time_filter or step.memory_types or step.min_confidence > 0:
            filters = {}
            if step.time_filter:
                filters.update(step.time_filter)
            if step.memory_types:
                filters["type"] = step.memory_types
            if step.min_confidence > 0:
                filters["min_confidence"] = step.min_confidence
        records = await self.hippocampal.search(
            tenant_id,
            query=step.query or "",
            top_k=step.top_k,
            context_filter=context_filter,
            filters=filters,
        )
        return [
            {
                "type": "episode",
                "source": "vector",
                "text": r.text,
                "confidence": r.confidence,
                "relevance": r.metadata.get("_similarity", 0.5),
                "timestamp": r.timestamp,
                "record": r,
            }
            for r in records
        ]

    async def _retrieve_graph(
        self,
        tenant_id: str,
        step: RetrievalStep,
    ) -> list[dict[str, Any]]:
        """Retrieve via knowledge graph PPR. Holistic: tenant-only."""
        if not step.seeds:
            return []
        results = await self.neocortical.multi_hop_query(
            tenant_id, seed_entities=step.seeds, max_hops=3
        )
        return [
            {
                "type": "graph",
                "source": "graph",
                "entity": r.get("entity"),
                "text": self._format_entity_info(r),
                "relevance": r.get("relevance_score", 0.5),
                "record": r,
            }
            for r in results
        ]

    async def _retrieve_cache(
        self,
        tenant_id: str,
        step: RetrievalStep,
    ) -> list[dict[str, Any]]:
        """Retrieve from hot cache. Holistic: tenant-only."""
        if not self.cache:
            return []
        cache_key = f"hot:{tenant_id}"
        try:
            cached = await self.cache.get(cache_key)
        except Exception:
            return []
        if cached:
            import json

            try:
                return json.loads(cached) if isinstance(cached, str) else cached
            except (TypeError, json.JSONDecodeError) as e:
                get_logger(__name__).warning(
                    "retrieval_cache_decode_error",
                    cache_key=cache_key,
                    error=str(e),
                )
        return []

    def _to_retrieved_memories(
        self,
        items: list[dict[str, Any]],
        analysis: QueryAnalysis,
    ) -> list[RetrievedMemory]:
        """Convert raw items to RetrievedMemory objects."""
        seen_texts: set = set()
        retrieved: list[RetrievedMemory] = []
        for item in items:
            text = item.get("text", "")
            if text in seen_texts:
                continue
            seen_texts.add(text)
            record = item.get("record")
            if isinstance(record, MemoryRecord):
                mem_record = record
            elif hasattr(record, "key") and hasattr(record, "value"):
                mem_record = self._fact_to_record(record, item)
            else:
                mem_record = self._dict_to_record(item)
            retrieved.append(
                RetrievedMemory(
                    record=mem_record,
                    relevance_score=item.get("relevance", 0.5),
                    retrieval_source=item.get("source", "unknown"),
                )
            )
        retrieved.sort(key=lambda x: x.relevance_score, reverse=True)
        return retrieved

    def _format_entity_info(self, entity_data: dict[str, Any]) -> str:
        """Format entity data as readable text."""
        lines = [f"Entity: {entity_data.get('entity', 'Unknown')}"]
        for rel in entity_data.get("relations", [])[:5]:
            lines.append(f"  - {rel.get('predicate', '')}: {rel.get('related_entity', '')}")
        return "\n".join(lines)

    def _fact_to_record(self, fact: Any, item: dict[str, Any]) -> MemoryRecord:
        """Build MemoryRecord from SemanticFact-like object. Holistic: context_tags."""
        from uuid import UUID

        text = (
            item.get("text", "")
            or f"{getattr(fact, 'predicate', '')}: {getattr(fact, 'value', '')}"
        )
        fid = getattr(fact, "id", None)
        context_tags = getattr(fact, "context_tags", None) or []
        return MemoryRecord(
            id=UUID(fid) if fid else uuid4(),
            tenant_id=getattr(fact, "tenant_id", ""),
            context_tags=list(context_tags),
            source_session_id=None,
            type=MemoryType.SEMANTIC_FACT,
            text=text,
            key=getattr(fact, "key", None),
            confidence=getattr(fact, "confidence", 0.5),
            importance=0.5,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=getattr(fact, "updated_at", datetime.now(UTC)),
            written_at=getattr(fact, "created_at", datetime.now(UTC)),
        )

    def _dict_to_record(self, item: dict[str, Any]) -> MemoryRecord:
        """Create minimal MemoryRecord from dict. Holistic: context_tags."""
        mem_type = item.get("type", "episodic_event")
        try:
            mtype = MemoryType(mem_type)
        except ValueError:
            mtype = MemoryType.EPISODIC_EVENT
        return MemoryRecord(
            id=uuid4(),
            tenant_id=item.get("tenant_id", ""),
            context_tags=item.get("context_tags", []),
            source_session_id=item.get("source_session_id"),
            type=mtype,
            text=item.get("text", ""),
            key=item.get("key"),
            confidence=item.get("confidence", 0.5),
            importance=0.5,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=item.get("timestamp", datetime.now(UTC)),
            written_at=datetime.now(UTC),
        )
