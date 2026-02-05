"""Hybrid retriever executing plans across memory sources."""
import asyncio
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Dict, List, Optional
from uuid import uuid4

from ..core.enums import MemoryScope, MemorySource, MemoryType
from ..core.schemas import MemoryRecord, Provenance, RetrievedMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from .planner import RetrievalPlan, RetrievalSource, RetrievalStep
from .query_types import QueryAnalysis


@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""

    source: RetrievalSource
    items: List[Dict[str, Any]]
    elapsed_ms: float
    success: bool
    error: Optional[str] = None


class HybridRetriever:
    """Executes retrieval plans across multiple memory sources."""

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        cache: Optional[Any] = None,
    ):
        self.hippocampal = hippocampal
        self.neocortical = neocortical
        self.cache = cache

    async def retrieve(
        self,
        tenant_id: str,
        scope_id: str,
        plan: RetrievalPlan,
    ) -> List[RetrievedMemory]:
        """Execute a retrieval plan and return results."""
        all_results: List[Dict[str, Any]] = []
        for group_indices in plan.parallel_steps:
            group_steps = [
                plan.steps[i] for i in group_indices if i < len(plan.steps)
            ]
            group_results = await asyncio.gather(
                *[
                    self._execute_step(tenant_id, scope_id, step)
                    for step in group_steps
                ],
                return_exceptions=True,
            )
            for step, result in zip(group_steps, group_results):
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
        scope_id: str,
        step: RetrievalStep,
    ) -> RetrievalResult:
        """Execute a single retrieval step."""
        start = datetime.utcnow()
        try:
            if step.source == RetrievalSource.FACTS:
                items = await self._retrieve_facts(tenant_id, scope_id, step)
            elif step.source == RetrievalSource.VECTOR:
                items = await self._retrieve_vector(tenant_id, scope_id, step)
            elif step.source == RetrievalSource.GRAPH:
                items = await self._retrieve_graph(tenant_id, scope_id, step)
            elif step.source == RetrievalSource.CACHE:
                items = await self._retrieve_cache(tenant_id, scope_id, step)
            else:
                items = []
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
            return RetrievalResult(
                source=step.source,
                items=items,
                elapsed_ms=elapsed,
                success=True,
            )
        except Exception as e:
            elapsed = (datetime.utcnow() - start).total_seconds() * 1000
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
        scope_id: str,
        step: RetrievalStep,
    ) -> List[Dict[str, Any]]:
        """Retrieve from semantic fact store."""
        results: List[Dict[str, Any]] = []
        if step.key:
            fact = await self.neocortical.get_fact(tenant_id, scope_id, step.key)
            if fact:
                results.append({
                    "type": "fact",
                    "source": "facts",
                    "key": fact.key,
                    "text": f"{fact.predicate}: {fact.value}",
                    "value": fact.value,
                    "confidence": fact.confidence,
                    "relevance": 1.0,
                    "record": fact,
                })
        if step.query:
            facts = await self.neocortical.text_search(
                tenant_id, scope_id, step.query, limit=step.top_k
            )
            for f in facts:
                results.append({
                    "type": "fact",
                    "source": "facts",
                    "key": f.get("key"),
                    "text": f"{f.get('key', '')}: {f.get('value', '')}",
                    "value": f.get("value"),
                    "confidence": f.get("confidence", 0.5),
                    "relevance": 0.8,
                    "record": f,
                })
        return results[: step.top_k]

    async def _retrieve_vector(
        self,
        tenant_id: str,
        scope_id: str,
        step: RetrievalStep,
    ) -> List[Dict[str, Any]]:
        """Retrieve via vector similarity search."""
        filters = None
        if step.time_filter or step.memory_types or step.min_confidence > 0:
            filters = {}
            if step.time_filter:
                filters.update(step.time_filter)
            if step.memory_types:
                filters["type"] = step.memory_types
        records = await self.hippocampal.search(
            tenant_id,
            scope_id,
            query=step.query or "",
            top_k=step.top_k,
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
        scope_id: str,
        step: RetrievalStep,
    ) -> List[Dict[str, Any]]:
        """Retrieve via knowledge graph PPR."""
        if not step.seeds:
            return []
        results = await self.neocortical.multi_hop_query(
            tenant_id, scope_id, seed_entities=step.seeds, max_hops=3
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
        scope_id: str,
        step: RetrievalStep,
    ) -> List[Dict[str, Any]]:
        """Retrieve from hot cache."""
        if not self.cache:
            return []
        cache_key = f"hot:{tenant_id}:{scope_id}"
        try:
            cached = await self.cache.get(cache_key)
        except Exception:
            return []
        if cached:
            import json
            try:
                return json.loads(cached) if isinstance(cached, str) else cached
            except (TypeError, json.JSONDecodeError):
                pass
        return []

    def _to_retrieved_memories(
        self,
        items: List[Dict[str, Any]],
        analysis: QueryAnalysis,
    ) -> List[RetrievedMemory]:
        """Convert raw items to RetrievedMemory objects."""
        seen_texts: set = set()
        retrieved: List[RetrievedMemory] = []
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

    def _format_entity_info(self, entity_data: Dict[str, Any]) -> str:
        """Format entity data as readable text."""
        lines = [f"Entity: {entity_data.get('entity', 'Unknown')}"]
        for rel in entity_data.get("relations", [])[:5]:
            lines.append(
                f"  - {rel.get('predicate', '')}: {rel.get('related_entity', '')}"
            )
        return "\n".join(lines)

    def _fact_to_record(self, fact: Any, item: Dict[str, Any]) -> MemoryRecord:
        """Build MemoryRecord from SemanticFact-like object."""
        from uuid import UUID
        text = item.get("text", "") or f"{getattr(fact, 'predicate', '')}: {getattr(fact, 'value', '')}"
        fid = getattr(fact, "id", None)
        return MemoryRecord(
            id=UUID(fid) if fid else uuid4(),
            tenant_id=getattr(fact, "tenant_id", ""),
            scope=MemoryScope.SESSION,
            scope_id=getattr(fact, "scope_id", ""),
            type=MemoryType.SEMANTIC_FACT,
            text=text,
            key=getattr(fact, "key", None),
            confidence=getattr(fact, "confidence", 0.5),
            importance=0.5,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=getattr(fact, "updated_at", datetime.utcnow()),
            written_at=getattr(fact, "created_at", datetime.utcnow()),
        )

    def _dict_to_record(self, item: Dict[str, Any]) -> MemoryRecord:
        """Create minimal MemoryRecord from dict."""
        mem_type = item.get("type", "episodic_event")
        try:
            mtype = MemoryType(mem_type)
        except ValueError:
            mtype = MemoryType.EPISODIC_EVENT
        return MemoryRecord(
            id=uuid4(),
            tenant_id=item.get("tenant_id", ""),
            scope=MemoryScope.SESSION,
            scope_id=item.get("scope_id", ""),
            type=mtype,
            text=item.get("text", ""),
            key=item.get("key"),
            confidence=item.get("confidence", 0.5),
            importance=0.5,
            provenance=Provenance(source=MemorySource.AGENT_INFERRED),
            timestamp=item.get("timestamp", datetime.utcnow()),
            written_at=datetime.utcnow(),
        )
