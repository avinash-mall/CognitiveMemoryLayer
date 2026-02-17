"""Hybrid retriever executing plans across memory sources."""

import asyncio
import time
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

logger = get_logger(__name__)


@dataclass
class RetrievalResult:
    """Result from a single retrieval step."""

    source: RetrievalSource
    items: list[dict[str, Any]]
    elapsed_ms: float
    success: bool
    error: str | None = None


class HybridRetriever:
    """Executes retrieval plans across multiple memory sources.

    Enforces per-step and total retrieval timeouts so that slow sources
    (e.g. Neo4j GDS) don't dominate tail latency.
    """

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
        """Execute a retrieval plan with enforced timeouts. Holistic: tenant-only."""
        from ..core.config import get_settings

        settings = get_settings()
        timeouts_enabled = settings.features.retrieval_timeouts_enabled
        cross_skip = settings.features.skip_if_found_cross_group

        all_results: list[dict[str, Any]] = []
        plan_start = time.perf_counter()
        plan_budget = plan.total_timeout_ms / 1000.0
        skip_remaining = False

        for group_indices in plan.parallel_steps:
            if skip_remaining:
                break

            # Check remaining plan budget
            if timeouts_enabled:
                elapsed = time.perf_counter() - plan_start
                remaining = plan_budget - elapsed
                if remaining <= 0:
                    logger.info(
                        "retrieval_plan_budget_exceeded", extra={"elapsed_ms": elapsed * 1000}
                    )
                    break
            else:
                remaining = None  # No cap

            group_steps = [plan.steps[i] for i in group_indices if i < len(plan.steps)]

            # Execute group in parallel with per-step timeouts
            if timeouts_enabled:
                coros = [
                    self._execute_step_with_timeout(tenant_id, step, context_filter)
                    for step in group_steps
                ]
                try:
                    group_results = await asyncio.wait_for(
                        asyncio.gather(*coros, return_exceptions=True),
                        timeout=remaining,
                    )
                except TimeoutError:
                    logger.info("retrieval_group_timeout", extra={"group": group_indices})
                    break
            else:
                group_results = await asyncio.gather(
                    *[self._execute_step(tenant_id, step, context_filter) for step in group_steps],
                    return_exceptions=True,
                )

            for step, result in zip(group_steps, group_results, strict=False):
                if isinstance(result, Exception):
                    continue
                if result.success and result.items:
                    all_results.extend(result.items)
                    # Phase 3.2: cross-group skip
                    if step.skip_if_found and result.items and cross_skip:
                        skip_remaining = True

        return self._to_retrieved_memories(all_results, plan.analysis)

    async def _execute_step_with_timeout(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
    ) -> RetrievalResult:
        """Execute a step wrapped in its own timeout."""
        timeout_s = step.timeout_ms / 1000.0
        try:
            return await asyncio.wait_for(
                self._execute_step(tenant_id, step, context_filter),
                timeout=timeout_s,
            )
        except TimeoutError:
            logger.info(
                "retrieval_step_timeout",
                extra={"source": step.source.value, "timeout_ms": step.timeout_ms},
            )
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=float(step.timeout_ms),
                success=False,
                error=f"Timeout after {step.timeout_ms}ms",
            )

    async def _execute_step(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
    ) -> RetrievalResult:
        """Execute a single retrieval step. Holistic: tenant-only."""
        start = time.perf_counter()
        try:
            if step.source == RetrievalSource.FACTS:
                items = await self._retrieve_facts(tenant_id, step)
            elif step.source == RetrievalSource.VECTOR:
                items = await self._retrieve_vector(tenant_id, step, context_filter)
            elif step.source == RetrievalSource.GRAPH:
                items = await self._retrieve_graph(tenant_id, step)
            elif step.source == RetrievalSource.CONSTRAINTS:
                items = await self._retrieve_constraints(tenant_id, step, context_filter)
            elif step.source == RetrievalSource.CACHE:
                items = await self._retrieve_cache(tenant_id, step)
            else:
                items = []
            elapsed_ms = (time.perf_counter() - start) * 1000

            # Phase 6.2: emit step metrics
            self._record_step_metrics(step.source, elapsed_ms, len(items))

            return RetrievalResult(
                source=step.source,
                items=items,
                elapsed_ms=elapsed_ms,
                success=True,
            )
        except Exception as e:
            elapsed_ms = (time.perf_counter() - start) * 1000
            return RetrievalResult(
                source=step.source,
                items=[],
                elapsed_ms=elapsed_ms,
                success=False,
                error=str(e),
            )

    @staticmethod
    def _record_step_metrics(source: RetrievalSource, elapsed_ms: float, count: int) -> None:
        """Emit Prometheus metrics for a retrieval step (best-effort)."""
        try:
            from ..utils.metrics import (
                RETRIEVAL_STEP_DURATION,
                RETRIEVAL_STEP_RESULT_COUNT,
            )

            RETRIEVAL_STEP_DURATION.labels(source=source.value).observe(elapsed_ms)
            RETRIEVAL_STEP_RESULT_COUNT.labels(source=source.value).observe(count)
        except Exception:
            pass  # Metrics are optional

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

    async def _retrieve_constraints(
        self,
        tenant_id: str,
        step: RetrievalStep,
        context_filter: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve active constraints via vector search + semantic fact lookup.

        Two-pronged approach:
        1. Vector search filtered to MemoryType.CONSTRAINT (episodic constraint records).
        2. Semantic fact lookup for cognitive categories (goal, value, state, causal, policy).
        """
        results: list[dict[str, Any]] = []

        # 1. Vector search for episodic constraint records
        constraint_filters: dict[str, Any] = {
            "type": [MemoryType.CONSTRAINT.value],
            "status": "active",
        }
        try:
            records = await self.hippocampal.search(
                tenant_id,
                query=step.query or "",
                top_k=step.top_k,
                context_filter=context_filter,
                filters=constraint_filters,
            )
            for r in records:
                results.append(
                    {
                        "type": MemoryType.CONSTRAINT.value,
                        "source": "constraints",
                        "text": r.text,
                        "confidence": r.confidence,
                        "relevance": r.metadata.get("_similarity", 0.7),
                        "timestamp": r.timestamp,
                        "record": r,
                    }
                )
        except Exception:
            logger.warning("constraint_vector_search_failed", exc_info=True)

        # 2. Semantic fact lookup for cognitive constraint categories
        from ..memory.neocortical.schemas import FactCategory

        category_map = {
            "goal": FactCategory.GOAL,
            "value": FactCategory.VALUE,
            "state": FactCategory.STATE,
            "causal": FactCategory.CAUSAL,
            "policy": FactCategory.POLICY,
        }
        all_cognitive = [
            FactCategory.GOAL,
            FactCategory.VALUE,
            FactCategory.STATE,
            FactCategory.CAUSAL,
            FactCategory.POLICY,
        ]
        if step.constraint_categories:
            cognitive_categories = [
                category_map[c.lower()]
                for c in step.constraint_categories
                if c.lower() in category_map
            ]
        else:
            cognitive_categories = all_cognitive
        if not cognitive_categories:
            cognitive_categories = all_cognitive
        try:
            for category in cognitive_categories:
                facts = await self.neocortical.facts.get_facts_by_category(
                    tenant_id, category, current_only=True
                )
                for fact in facts:
                    fact_text = f"[{category.value.title()}] {fact.value}"
                    if fact_text not in {r["text"] for r in results}:
                        results.append(
                            {
                                "type": MemoryType.CONSTRAINT.value,
                                "source": "constraints",
                                "key": fact.key,
                                "text": fact_text,
                                "value": fact.value,
                                "confidence": fact.confidence,
                                "relevance": 0.75,
                                "record": fact,
                            }
                        )
        except Exception:
            logger.warning("constraint_fact_search_failed", exc_info=True)

        return results[: step.top_k]

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
