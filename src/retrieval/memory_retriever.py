"""Main memory retriever facade."""

from datetime import datetime
from typing import Any

from ..core.config import get_settings
from ..core.schemas import MemoryPacket
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.llm import LLMClient
from ..utils.modelpack import get_modelpack_runtime
from ..utils.tracing import async_trace_span
from .classifier import QueryClassifier
from .packet_builder import MemoryPacketBuilder
from .planner import RetrievalPlanner, RetrievalSource, RetrievalStep
from .reranker import MemoryReranker, RerankerConfig
from .retriever import HybridRetriever


def _reranker_config_from_settings() -> RerankerConfig:
    """Build RerankerConfig from application settings."""
    settings = get_settings()
    r = settings.retrieval.reranker
    return RerankerConfig(
        recency_weight=r.recency_weight,
        relevance_weight=r.relevance_weight,
        confidence_weight=r.confidence_weight,
    )


class MemoryRetriever:
    """Main entry point for memory retrieval. Coordinates classification, planning, retrieval, rerank, and formatting. Holistic: tenant-only."""

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        llm_client: LLMClient | None = None,
        cache: object | None = None,
    ):
        modelpack = get_modelpack_runtime()
        self.classifier = QueryClassifier(llm_client, modelpack=modelpack)
        self.planner = RetrievalPlanner()
        self.retriever = HybridRetriever(hippocampal, neocortical, cache, modelpack=modelpack)
        self.reranker = MemoryReranker(
            config=_reranker_config_from_settings(),
            llm_client=llm_client,
            modelpack=modelpack,
        )
        self.packet_builder = MemoryPacketBuilder()

    @staticmethod
    def _apply_session_scope_restrictions(
        plan,
        *,
        source_session_id: str | None,
    ) -> None:
        """Restrict retrieval sources for session-scoped reads."""
        if not source_session_id:
            return
        kept_steps: list[RetrievalStep] = []
        idx_map: dict[int, int] = {}
        for old_idx, step in enumerate(plan.steps):
            if step.source in {RetrievalSource.VECTOR, RetrievalSource.CONSTRAINTS}:
                idx_map[old_idx] = len(kept_steps)
                kept_steps.append(step)
        kept_groups: list[list[int]] = []
        for group in plan.parallel_steps:
            new_group = [idx_map[i] for i in group if i in idx_map]
            if new_group:
                kept_groups.append(new_group)
        plan.steps = kept_steps
        plan.parallel_steps = kept_groups or (
            [[i for i in range(len(plan.steps))]] if plan.steps else []
        )

    @staticmethod
    def _inject_step_filters(
        plan,
        *,
        memory_types: list[str] | None,
        since: datetime | None,
        until: datetime | None,
        source_session_id: str | None,
    ) -> None:
        """Inject API-level filters into planned retrieval steps."""
        if not (memory_types or since or until or source_session_id):
            return
        for step in plan.steps:
            if memory_types and not step.memory_types:
                step.memory_types = memory_types
            if since or until or source_session_id:
                time_filter = step.time_filter or {}
                if since:
                    time_filter["since"] = since
                if until:
                    time_filter["until"] = until
                if source_session_id:
                    time_filter["source_session_id"] = source_session_id
                step.time_filter = time_filter

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
        """Classify, plan, filter, and embed the query once."""
        analysis = await self.classifier.classify(query, recent_context=recent_context)
        if user_timezone is not None:
            analysis.user_timezone = user_timezone
        plan = self.planner.plan(analysis)
        self._apply_session_scope_restrictions(plan, source_session_id=source_session_id)
        self._inject_step_filters(
            plan,
            memory_types=memory_types,
            since=since,
            until=until,
            source_session_id=source_session_id,
        )

        query_embedding = None
        if plan.steps:
            emb_result = await self.retriever.hippocampal.embeddings.embed(
                plan.analysis.original_query
            )
            query_embedding = emb_result.embedding
        return plan, query_embedding

    async def retrieve(
        self,
        tenant_id: str,
        query: str,
        max_results: int = 20,
        context_filter: list[str] | None = None,
        recent_context: str | None = None,
        return_packet: bool = True,
        memory_types: list[str] | None = None,
        since: datetime | None = None,
        until: datetime | None = None,
        user_timezone: str | None = None,
        source_session_id: str | None = None,
    ) -> MemoryPacket:
        """Retrieve relevant memories for a query. Holistic: tenant-only."""
        async with async_trace_span("retrieval.pipeline", tenant_id=tenant_id):
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
            )
            reranked = await self.reranker.rerank(raw_results, query, max_results=max_results)
            retrieval_meta = plan.analysis.metadata.get("retrieval_meta")
            if return_packet:
                packet = self.packet_builder.build(reranked, query)
                packet.retrieval_meta = retrieval_meta
                return packet
            return MemoryPacket(
                query=query, recent_episodes=reranked, retrieval_meta=retrieval_meta
            )

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
        async with async_trace_span("retrieval.explain", tenant_id=tenant_id):
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
            llm_context = self.packet_builder.to_llm_context(packet, max_tokens=2000)

            results = []
            for mem in packet.all_memories:
                results.append(
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
                )

            plan_steps = []
            for step in plan.steps:
                plan_steps.append(
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
                        "query_domain": step.query_domain,
                        "constraint_categories": list(step.constraint_categories or []),
                    }
                )

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
                    "query_domain": analysis.query_domain,
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

    async def retrieve_for_llm(
        self,
        tenant_id: str,
        query: str,
        max_tokens: int = 2000,
        format: str = "markdown",
        context_filter: list[str] | None = None,
        recent_context: str | None = None,
    ) -> str:
        """Retrieve and format memories for LLM context. Holistic: tenant-only."""
        packet = await self.retrieve(
            tenant_id,
            query,
            context_filter=context_filter,
            recent_context=recent_context,
        )
        return self.packet_builder.to_llm_context(packet, max_tokens, format)
