"""Explainability workbench helper mixed into MemoryRetriever."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .packet_builder import MemoryPacketBuilder
    from .reranker import MemoryReranker
    from .retriever import HybridRetriever


class RetrievalExplainMixin:
    """Dashboard retrieval explainability builder."""

    retriever: HybridRetriever
    reranker: MemoryReranker
    packet_builder: MemoryPacketBuilder

    async def _prepare_plan_and_embedding(
        self,
        tenant_id: str,
        query: str,
        *,
        recent_context: str | None = None,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        user_timezone: str | None = None,
        source_session_id: str | None = None,
    ) -> tuple[Any, list[float] | None]:
        raise NotImplementedError

    async def explain(
        self,
        tenant_id: str,
        query: str,
        max_results: int = 20,
        context_filter: list[str] | None = None,
        recent_context: str | None = None,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        user_timezone: str | None = None,
        source_session_id: str | None = None,
    ) -> dict[str, Any]:
        """Run retrieval with execution and reranker breakdowns for dashboard explain mode."""
        plan, query_embedding = await self._prepare_plan_and_embedding(
            tenant_id,
            query,
            recent_context=recent_context,
            memory_types=memory_types,
            since=since,
            until=until,
            user_timezone=user_timezone,
            source_session_id=source_session_id,
        )
        raw_results = await self.retriever.retrieve(
            tenant_id,
            plan,
            context_filter=context_filter,
            query_embedding=query_embedding,
            debug=True,
        )
        reranked, rerank_breakdown = await self.reranker.rerank_with_breakdown(
            raw_results,
            query,
            max_results=max_results,
        )
        packet = self.packet_builder.build(reranked, query)
        packet.retrieval_meta = plan.analysis.metadata.get("retrieval_meta")
        llm_context = self.packet_builder.to_llm_context(packet, max_tokens=3000)

        results = [
            {
                "id": mem.record.id,
                "text": mem.record.text,
                "type": mem.record.type.value,
                "confidence": mem.record.confidence,
                "relevance_score": mem.relevance_score,
                "retrieval_source": mem.retrieval_source,
                "timestamp": mem.record.timestamp,
                "supersedes_id": mem.record.supersedes_id,
                "metadata": mem.record.metadata,
            }
            for mem in packet.all_memories
        ]

        plan_steps = [
            {
                "source": step.source.value,
                "priority": step.priority,
                "key": step.key,
                "query": step.query,
                "seeds": list(step.seeds),
                "memory_types": list(step.memory_types),
                "time_filter": dict(step.time_filter or {}) if step.time_filter else None,
                "min_confidence": step.min_confidence,
                "top_k": step.top_k,
                "timeout_ms": step.timeout_ms,
                "skip_if_found": step.skip_if_found,
                "associative_expansion": step.associative_expansion,
                "constraint_categories": list(step.constraint_categories or []),
            }
            for step in plan.steps
        ]

        analysis = plan.analysis
        return {
            "query": query,
            "analysis": {
                "intent": analysis.intent.value,
                "confidence": analysis.confidence,
                "entities": list(analysis.entities),
                "key_phrases": list(analysis.key_phrases),
                "time_reference": analysis.time_reference,
                "time_start": analysis.time_start,
                "time_end": analysis.time_end,
                "suggested_sources": list(analysis.suggested_sources),
                "suggested_top_k": analysis.suggested_top_k,
                "constraint_dimensions": list(analysis.constraint_dimensions or []),
                "is_decision_query": analysis.is_decision_query,
                "metadata": dict(analysis.metadata),
            },
            "plan_steps": plan_steps,
            "parallel_groups": [list(group) for group in plan.parallel_steps],
            "execution_steps": list(plan.analysis.metadata.get("retrieval_steps", [])),
            "retrieval_meta": dict(plan.analysis.metadata.get("retrieval_meta") or {}),
            "packet_warnings": list(packet.warnings),
            "open_questions": list(packet.open_questions),
            "llm_context": llm_context,
            "results": results,
            "rerank": rerank_breakdown,
            "total_count": len(results),
        }
