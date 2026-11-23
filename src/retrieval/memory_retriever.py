"""Main memory retriever facade."""

from datetime import datetime
from typing import Any

from ..core.config import get_settings
from ..core.schemas import MemoryPacket, RetrievedMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from .classifier import QueryClassifier
from .packet_builder import MemoryPacketBuilder
from .planner import RetrievalPlanner, RetrievalSource, RetrievalStep
from .reranker import MemoryReranker, RerankerConfig
from .retriever import HybridRetriever
from .rrf import rrf_merge

_logger = get_logger(__name__)


GRAPH_RETRIEVAL_SOURCE = "graph"


def drop_graph_results(memories: list[RetrievedMemory], *, enabled: bool) -> list[RetrievedMemory]:
    """Remove knowledge-graph hits from the candidate set.

    Called before the rerank cap, not after: filtering at packet-build time would leave
    the slots a graph hit consumed already spent, which defeats the point. Everything
    else — vector, facts, constraints — passes through untouched.

    ``enabled`` is inverted from the feature flag at the call site so the flag can stay
    positively named (graph_results_in_packet) while this function stays a filter.
    """
    if not enabled:
        return memories
    return [m for m in memories if m.retrieval_source != GRAPH_RETRIEVAL_SOURCE]


# Feeling-of-knowing thresholds. A packet is "sufficient" when at least one memory is a
# strong match and the retrieval is not merely a flat field of weak ones. Calibrated to sit
# just under the vector prong's typical direct-answer band rather than to any single query.
SUFFICIENCY_STRENGTH_FLOOR = 0.55
SUFFICIENCY_SUPPORT_FLOOR = 0.45


def assess_sufficiency(memories: list[RetrievedMemory]) -> dict[str, Any]:
    """Score whether retrieval found anything worth answering from.

    Distinct from relevance rank on purpose. Ranking answers "which of these is best",
    which is defined even when every candidate is junk; this answers "is there anything
    here at all", which is what a caller needs in order to decline. Models are
    demonstrably bad at making that call for themselves — abstention does not improve
    with scale, and reasoning fine-tuning measurably degrades it — so the memory layer
    computing it from its own score distribution is more trustworthy than asking the
    answering model to introspect.

    Cheap by design: mean/max of the first-pass scores, no LLM and no extra query.
    """
    scores = sorted((m.relevance_score for m in memories), reverse=True)
    strength = scores[0] if scores else 0.0
    support = sum(1 for s in scores if s >= SUFFICIENCY_SUPPORT_FLOOR)
    return {
        "evidence_strength": round(strength, 4),
        "supporting_memories": support,
        "retrieved": len(scores),
        "sufficient": bool(scores) and strength >= SUFFICIENCY_STRENGTH_FLOOR,
    }


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
        short_term: Any | None = None,
    ):
        self.classifier = QueryClassifier(llm_client)
        self.planner = RetrievalPlanner()
        self.retriever = HybridRetriever(hippocampal, neocortical)
        self.reranker = MemoryReranker(config=_reranker_config_from_settings())
        self.packet_builder = MemoryPacketBuilder()
        self.llm_client = llm_client
        self.short_term = short_term

        # --- Improvement Report: HyDE ---
        self._hyde_generator = None
        settings = get_settings()

        if llm_client and settings.features.hyde_retrieval_enabled:
            from .hyde import HyDEGenerator

            self._hyde_generator = HyDEGenerator(llm_client)

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

    async def _hyde_augment(
        self,
        tenant_id: str,
        query: str,
        plan: Any,
        context_filter: list[str] | None,
    ) -> list[list[Any]]:
        """Run HyDE: generate hypothetical memory cues and retrieve with them.

        On Cognitive dialogue cues (zero-keyword-overlap triggers), generates 3
        diverse hypothetical past-experience/constraint cues and runs parallel
        vector retrievals for RRF fusion. Returns a list of result lists.
        """
        if not self._hyde_generator:
            return []
        try:
            is_cognitive = False
            try:
                from ..core.laya_system1 import LayaSystem1Engine

                is_cognitive = LayaSystem1Engine.classify_query_routing(query).is_cognitive_cue
            except Exception:
                is_cognitive = False

            if is_cognitive and hasattr(self._hyde_generator, "generate_cognitive_cues"):
                hypos = await self._hyde_generator.generate_cognitive_cues(query)
            else:
                hypo = await self._hyde_generator.generate_hypothetical_memory(query)
                hypos = [hypo] if hypo else []

            if not hypos:
                return []

            emb_client = self.retriever.hippocampal.embeddings
            if len(hypos) > 1 and hasattr(emb_client, "embed_batch"):
                batch_res = await emb_client.embed_batch(hypos)
                embeddings: list[list[float]] = [r.embedding for r in batch_res]
            else:
                embeddings = [(await emb_client.embed(h)).embedding for h in hypos]

            result_lists: list[list[Any]] = []
            for emb in embeddings:
                res = await self.retriever.retrieve(
                    tenant_id,
                    plan,
                    context_filter=context_filter,
                    query_embedding=emb,
                )
                if res:
                    result_lists.append(res)
            return result_lists
        except Exception as exc:
            _logger.debug("hyde_augment_failed", error=str(exc))
            return []

    async def _persona_profile_augment(
        self,
        tenant_id: str,
        query: str,
        plan: Any,
        context_filter: list[str] | None,
    ) -> list[RetrievedMemory]:
        """Augment common-sense/hypothetical queries with speaker persona & trait turns."""
        import re

        stopwords = {
            "would",
            "could",
            "should",
            "might",
            "likely",
            "what",
            "when",
            "where",
            "who",
            "which",
            "why",
            "how",
            "does",
            "did",
            "will",
            "can",
            "has",
            "have",
            "the",
            "this",
            "that",
            "still",
            "want",
        }
        names = [
            m.group(0)
            for m in re.finditer(r"\b[A-Z][a-z]{2,15}\b", query)
            if m.group(0).lower() not in stopwords
        ]
        if not names:
            return []
        person = names[0]
        persona_q = f"{person} career job hobbies interests personality values background likes goals family"
        try:
            emb = await self.retriever.hippocampal.embeddings.embed(persona_q)
            return await self.retriever.retrieve(
                tenant_id,
                plan,
                context_filter=context_filter,
                query_embedding=emb.embedding,
            )
        except Exception as exc:
            _logger.debug("persona_profile_augment_failed", error=str(exc))
            return []

    async def _bridge_hop_augment(
        self,
        tenant_id: str,
        query: str,
        first_pass: list[RetrievedMemory],
        plan: Any,
        context_filter: list[str] | None,
    ) -> list[RetrievedMemory]:
        """Iterative 2-hop recollection: ask LLM_INTERNAL for missing bridge fact and retrieve."""
        if not self._hyde_generator or not hasattr(self._hyde_generator, "generate_bridge_query"):
            return []
        try:
            snippets = [
                str(getattr(r.record, "text", "") or "").strip()
                for r in first_pass[:6]
                if getattr(r.record, "text", None)
            ]
            if not snippets:
                return []
            bridge_q = await self._hyde_generator.generate_bridge_query(query, snippets)
            if not bridge_q:
                return []
            emb = await self.retriever.hippocampal.embeddings.embed(bridge_q)
            return await self.retriever.retrieve(
                tenant_id,
                plan,
                context_filter=context_filter,
                query_embedding=emb.embedding,
            )
        except Exception as exc:
            _logger.debug("bridge_hop_augment_failed", error=str(exc))
            return []

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

        # --- Improvement Report: HyDE, Persona-Profile & Iterative 2-Hop Augmentation ---
        settings = get_settings()
        routing_intent = "single_hop"
        is_cognitive_cue = False
        try:
            from ..core.laya_system1 import LayaSystem1Engine

            routing = LayaSystem1Engine.classify_query_routing(query)
            routing_intent = routing.intent
            is_cognitive_cue = routing.is_cognitive_cue
        except Exception:
            pass

        extra_result_lists: list[list[RetrievedMemory]] = []
        hyde_hit_ids: set[str] = set()

        if settings.features.hyde_retrieval_enabled and self._hyde_generator:
            hyde_result_lists = await self._hyde_augment(tenant_id, query, plan, context_filter)
            for rl in hyde_result_lists:
                if rl:
                    extra_result_lists.append(rl)
                    for r in rl:
                        hyde_hit_ids.add(str(r.record.id))

        # Speaker Persona-Profile Augmentation for common-sense / hypothetical queries
        if routing_intent == "common_sense":
            persona_hits = await self._persona_profile_augment(
                tenant_id, query, plan, context_filter
            )
            if persona_hits:
                extra_result_lists.append(persona_hits)

        # Iterative 2-Hop Bridge Recollection for non-cognitive factual/multi-hop queries
        if (
            not is_cognitive_cue
            and getattr(settings.features, "multi_hop_recollection_enabled", True)
            and self._hyde_generator is not None
            and raw_results
        ):
            bridge_hits = await self._bridge_hop_augment(
                tenant_id, query, raw_results, plan, context_filter
            )
            if bridge_hits:
                extra_result_lists.append(bridge_hits)

        if extra_result_lists:
            main_dicts = [{"id": str(r.record.id), "mem": r} for r in raw_results]
            all_lists = [main_dicts] + [
                [{"id": str(r.record.id), "mem": r} for r in rl] for rl in extra_result_lists
            ]
            merged = rrf_merge(all_lists, k=60)
            seen_ids: set[str] = set()
            deduped_results: list[RetrievedMemory] = []
            for item in merged:
                rid = item["id"]
                if rid not in seen_ids:
                    seen_ids.add(rid)
                    mem: RetrievedMemory = item["mem"]
                    rank = len(deduped_results)
                    # Preserve RRF rank through downstream reranker: semantically distant
                    # HyDE / 2-hop / persona hits would otherwise be buried by low raw cosine.
                    if rank < 12 and (is_cognitive_cue or rid in hyde_hit_ids or rank < 8):
                        rrf_floor = max(0.55, 0.84 - 0.018 * rank)
                        if mem.relevance_score < rrf_floor:
                            mem.relevance_score = rrf_floor
                    if is_cognitive_cue and rid in hyde_hit_ids:
                        meta = dict(mem.record.metadata or {})
                        meta["_hyde_promoted"] = True
                        mem.record.metadata = meta
                    deduped_results.append(mem)
            raw_results = deduped_results

        raw_results = drop_graph_results(
            raw_results, enabled=not settings.features.graph_results_in_packet
        )

        if self.short_term is not None and getattr(
            settings.features, "working_memory_context_binding_enabled", True
        ):
            raw_results = self._overlay_working_memory(
                tenant_id, query, raw_results, source_session_id=source_session_id
            )

        reranked = await self.reranker.rerank(raw_results, query, max_results=max_results)
        retrieval_meta = plan.analysis.metadata.get("retrieval_meta")
        # Assessed on the reranked set, not the raw one: raw carries every prong's flat
        # per-source constant (facts 0.8, constraints 0.75) which would report confident
        # evidence for a query nothing actually matched.
        sufficiency = assess_sufficiency(reranked)
        if return_packet:
            packet = self.packet_builder.build(reranked, query)
            packet.retrieval_meta = retrieval_meta
            packet.sufficiency = sufficiency
            if packet.all_memories and getattr(settings.features, "intrinsic_memory_enabled", True):
                spec = self.compute_intrinsic_spec(
                    packet, tenant_id=tenant_id, intent=plan.analysis.intent.value
                )
                meta_copy = (
                    dict(packet.retrieval_meta) if isinstance(packet.retrieval_meta, dict) else {}
                )
                meta_copy["intrinsic_injection"] = spec.to_dict()
                packet.retrieval_meta = meta_copy
            return packet
        return MemoryPacket(
            query=query,
            recent_episodes=reranked,
            retrieval_meta=retrieval_meta,
            sufficiency=sufficiency,
        )

    def _overlay_working_memory(
        self,
        tenant_id: str,
        query: str,
        raw_results: list[RetrievedMemory],
        *,
        source_session_id: str | None = None,
    ) -> list[RetrievedMemory]:
        """Overlay active Baddeley Working Memory slots with zero DB latency."""
        if self.short_term is None or not hasattr(self.short_term, "search_working_memory"):
            return raw_results
        try:
            from uuid import uuid4

            from ..core.enums import MemorySource, MemoryType
            from ..core.schemas import MemoryRecord, Provenance

            wm_hits = self.short_term.search_working_memory(
                tenant_id, query, scope_id=source_session_id, top_k=3
            )
            if not wm_hits:
                return raw_results

            by_text = {r.record.text: r for r in raw_results}
            merged = list(raw_results)
            for chunk, rel in wm_hits:
                if chunk.text in by_text:
                    existing = by_text[chunk.text]
                    existing.relevance_score = max(existing.relevance_score, rel)
                    meta = existing.record.metadata
                    if isinstance(meta, dict):
                        srcs = set(meta.get("_converging_sources", [existing.retrieval_source]))
                        srcs.add("working_memory")
                        meta["_converging_sources"] = sorted(srcs)
                        meta["_coactivation_count"] = len(srcs)
                else:
                    rec = MemoryRecord(
                        id=uuid4(),
                        tenant_id=tenant_id,
                        context_tags=[],
                        source_session_id=source_session_id,
                        type=MemoryType.EPISODIC_EVENT,
                        text=chunk.text,
                        confidence=chunk.confidence,
                        importance=chunk.salience,
                        timestamp=chunk.timestamp,
                        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
                        metadata={
                            **dict(chunk.metadata or {}),
                            "working_memory_overlay": True,
                        },
                    )
                    rm = RetrievedMemory(
                        record=rec,
                        relevance_score=rel,
                        retrieval_source="working_memory",
                    )
                    merged.append(rm)
                    by_text[chunk.text] = rm
            return merged
        except Exception as exc:
            _logger.debug("working_memory_overlay_failed", error=str(exc))
            return raw_results

    def compute_intrinsic_spec(
        self,
        packet: MemoryPacket,
        *,
        tenant_id: str | None = None,
        intent: str | None = None,
        target_hidden_dim: int | None = None,
    ) -> Any:
        """Synthesize ActiveCML IntrinsicInjectionSpec for the given MemoryPacket."""
        from ..intrinsic.controller import get_intrinsic_controller

        return get_intrinsic_controller().compute_spec(
            packet,
            tenant_id=tenant_id,
            intent=intent,
            target_hidden_dim=target_hidden_dim,
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
