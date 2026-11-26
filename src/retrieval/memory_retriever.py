"""Main memory retriever facade."""

from datetime import datetime
from typing import Any

from ..core.config import get_settings
from ..core.schemas import MemoryPacket, RetrievedMemory
from ..memory.hippocampal.store import HippocampalStore
from ..memory.neocortical.store import NeocorticalStore
from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from .augmenters import RetrievalAugmentersMixin
from .classifier import QueryClassifier
from .explain import RetrievalExplainMixin
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


class MemoryRetriever(RetrievalAugmentersMixin, RetrievalExplainMixin):
    """Main entry point for memory retrieval. Coordinates classification, planning, retrieval, rerank, and formatting. Holistic: tenant-only."""

    def __init__(
        self,
        hippocampal: HippocampalStore,
        neocortical: NeocorticalStore,
        llm_client: LLMClient | None = None,
        short_term: Any | None = None,
        dmn_store: Any | None = None,
        procedural_engine: Any | None = None,
    ):
        self.classifier = QueryClassifier(llm_client)
        self.planner = RetrievalPlanner()
        self.retriever = HybridRetriever(hippocampal, neocortical)
        self.reranker = MemoryReranker(config=_reranker_config_from_settings())
        self.packet_builder = MemoryPacketBuilder()
        self.llm_client = llm_client
        self.short_term = short_term
        self.dmn_store = dmn_store
        self.procedural_engine = procedural_engine

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
        """Retrieve relevant memories across all Whole-Brain pathways. Holistic: tenant-only."""
        import asyncio

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

        # Neuromodulatory Gain (ACh retrieval mode + LC-NE exploration breadth)
        ce_temperature = 1.0
        ppr_damping = 0.82
        neuro_dict: dict[str, Any] = {}
        if getattr(settings.features, "neuromodulation_enabled", True):
            try:
                from ..memory.neuromodulation import NeuromodulatoryController

                nstate = NeuromodulatoryController.for_read(
                    query,
                    routing_intent=routing_intent,
                    is_cognitive_cue=is_cognitive_cue,
                    recent_context=recent_context,
                )
                ce_temperature = nstate.cross_encoder_temperature
                ppr_damping = nstate.ppr_damping
                neuro_dict = {
                    "ach_mode": nstate.ach_mode,
                    "ne_arousal_gain": nstate.ne_arousal_gain,
                    "candidate_pool": nstate.recommended_candidate_pool,
                }
            except Exception:
                pass

        # Execute Dense/Plan + Dentate Gyrus Sparse BM25 + CA3 Hypergraph PPR concurrently
        coros: list[Any] = [
            self.retriever.retrieve(
                tenant_id,
                plan,
                context_filter=context_filter,
                query_embedding=query_embedding,
            )
        ]
        run_bm25 = bool(getattr(settings.features, "sparse_bm25_enabled", True))
        run_ppr = bool(
            getattr(settings.features, "hypergraph_ppr_enabled", True) and not source_session_id
        )
        if run_bm25:
            coros.append(
                self._sparse_bm25_augment(
                    tenant_id,
                    query,
                    top_k=20,
                    context_filter=context_filter,
                    memory_types=memory_types,
                    since=since,
                    until=until,
                    source_session_id=source_session_id,
                )
            )
        if run_ppr:
            coros.append(
                self._hypergraph_ppr_augment(
                    tenant_id,
                    query,
                    getattr(plan.analysis, "entities", None),
                    damping=ppr_damping,
                    memory_types=memory_types,
                    source_session_id=source_session_id,
                )
            )

        parallel_out = await asyncio.gather(*coros)
        raw_results: list[RetrievedMemory] = parallel_out[0]
        extra_result_lists: list[list[RetrievedMemory]] = []
        idx_cursor = 1
        if run_bm25:
            bm25_hits = parallel_out[idx_cursor]
            idx_cursor += 1
            if bm25_hits:
                extra_result_lists.append(bm25_hits)
        if run_ppr:
            ppr_hits = parallel_out[idx_cursor]
            if ppr_hits:
                extra_result_lists.append(ppr_hits)

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

        # ACC Feeling-of-Knowing Iterative Recollection (up to 2 hops on multi-hop gaps)
        if (
            not is_cognitive_cue
            and getattr(settings.features, "multi_hop_recollection_enabled", True)
            and self._hyde_generator is not None
            and (raw_results or extra_result_lists)
        ):
            acc_enabled = bool(getattr(settings.features, "acc_iterative_loop_enabled", True))
            max_hops = 2 if (acc_enabled and routing_intent == "multi_hop") else 1
            seed_pool = list(raw_results)
            for rl in extra_result_lists[:2]:
                seed_pool.extend(rl[:6])
            seen_subq: set[str] = set()
            for _hop in range(max_hops):
                snippets = [
                    str(getattr(r.record, "text", "") or "").strip()
                    for r in seed_pool[:14]
                    if getattr(r.record, "text", None)
                ]
                if not snippets:
                    break
                bridge_q = await self._hyde_generator.generate_bridge_query(query, snippets)
                if not bridge_q or bridge_q.lower() in seen_subq:
                    break
                seen_subq.add(bridge_q.lower())
                try:
                    emb = await self.retriever.hippocampal.embeddings.embed(bridge_q)
                    hop_dense = await self.retriever.retrieve(
                        tenant_id,
                        plan,
                        context_filter=context_filter,
                        query_embedding=emb.embedding,
                    )
                    if hop_dense:
                        extra_result_lists.append(hop_dense)
                        seed_pool = hop_dense[:6] + seed_pool
                    if run_bm25:
                        hop_lex = await self._sparse_bm25_augment(
                            tenant_id,
                            bridge_q,
                            top_k=10,
                            context_filter=context_filter,
                            source_session_id=source_session_id,
                        )
                        if hop_lex:
                            extra_result_lists.append(hop_lex)
                except Exception as exc:
                    _logger.debug("acc_hop_failed", error=str(exc))
                    break

        if extra_result_lists:
            # Track multi-pathway Synaptic Co-Activation across all parallel prongs
            source_tracker: dict[str, set[str]] = {}
            for r in raw_results:
                source_tracker.setdefault(str(r.record.id), set()).add(r.retrieval_source)
            for rl in extra_result_lists:
                for r in rl:
                    source_tracker.setdefault(str(r.record.id), set()).add(r.retrieval_source)

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
                    srcs = source_tracker.get(rid, {mem.retrieval_source})
                    meta = dict(mem.record.metadata or {})
                    if len(srcs) > 1:
                        meta["_converging_sources"] = sorted(srcs)
                        meta["_coactivation_count"] = len(srcs)
                        # Synaptic co-activation resonance bonus
                        mem.relevance_score = min(
                            0.97, mem.relevance_score + 0.06 * (len(srcs) - 1)
                        )
                    if rank < 12 and (is_cognitive_cue or rid in hyde_hit_ids or rank < 8):
                        rrf_floor = max(0.55, 0.84 - 0.018 * rank)
                        if mem.relevance_score < rrf_floor:
                            mem.relevance_score = rrf_floor
                    if is_cognitive_cue and rid in hyde_hit_ids:
                        meta["_hyde_promoted"] = True
                    mem.record.metadata = meta
                    deduped_results.append(mem)
            raw_results = deduped_results

        # Resolve Grounded Proustian Cue Pointers -> verbatim parent episodes
        if getattr(settings.features, "conway_hierarchy_enabled", True) and raw_results:
            raw_results = await self._resolve_proustian_and_conway_parents(
                raw_results, is_cognitive_cue=is_cognitive_cue
            )

        raw_results = drop_graph_results(
            raw_results, enabled=not settings.features.graph_results_in_packet
        )

        if self.short_term is not None and getattr(
            settings.features, "working_memory_context_binding_enabled", True
        ):
            raw_results = self._overlay_working_memory(
                tenant_id, query, raw_results, source_session_id=source_session_id
            )

        # Enforce explicit caller filters across all merged pathways
        # (including facts, working-memory overlay, Conway parents, and PPR/BM25).
        if memory_types:
            allowed_types = {
                str(mt.value if hasattr(mt, "value") else mt).lower() for mt in memory_types
            }
            raw_results = [
                r
                for r in raw_results
                if str(
                    r.record.type.value if hasattr(r.record.type, "value") else r.record.type
                ).lower()
                in allowed_types
            ]
        if since is not None or until is not None:
            from ..storage.utils import naive_utc

            since_utc = naive_utc(since)
            until_utc = naive_utc(until)
            filtered_time: list[RetrievedMemory] = []
            for r in raw_results:
                rec_ts = naive_utc(getattr(r.record, "timestamp", None))
                if rec_ts is None:
                    continue
                if since_utc is not None and rec_ts < since_utc:
                    continue
                if until_utc is not None and rec_ts > until_utc:
                    continue
                filtered_time.append(r)
            raw_results = filtered_time

        # Rerank a wider pool so Neocortical facts/constraints never starve Hippocampal episodes,
        # then enforce Complementary Learning Systems stratified slot reservation.
        wide_cap = max(max_results + 25, 45)
        wide_reranked = await self.reranker.rerank(
            raw_results, query, max_results=wide_cap, ce_temperature=ce_temperature
        )
        if len(wide_reranked) <= max_results:
            reranked = wide_reranked
        else:
            from ..core.enums import MemoryType

            def _is_pure_episode(m: RetrievedMemory) -> bool:
                meta = m.record.metadata or {}
                if meta.get("proustian_cue_pointer") or meta.get("conway_tier") == "chapter_gist":
                    return False
                if m.retrieval_source in ("facts", "constraints"):
                    return False
                if m.record.type in (
                    MemoryType.SEMANTIC_FACT,
                    MemoryType.PREFERENCE,
                    MemoryType.PROCEDURE,
                ):
                    return False
                if m.record.type == MemoryType.CONSTRAINT:
                    # On Cognitive cues (or when promoted / high relevance), verbatim CONSTRAINT
                    # turns are genuine hippocampal dialogue episodes and must never be capped out.
                    return bool(
                        is_cognitive_cue
                        or meta.get("_proustian_promoted")
                        or meta.get("_hyde_promoted")
                        or m.relevance_score >= 0.72
                    )
                return True

            pure_eps = [m for m in wide_reranked if _is_pure_episode(m)]
            constraints_pool = [
                m
                for m in wide_reranked
                if m.record.type == MemoryType.CONSTRAINT and m.relevance_score >= 0.55
            ]
            facts_pool = [
                m
                for m in wide_reranked
                if not _is_pure_episode(m)
                and m.record.type != MemoryType.CONSTRAINT
                and m.relevance_score >= 0.58
            ]
            c_cap = 4 if is_cognitive_cue else 2
            f_cap = 3
            chosen_aux = constraints_pool[:c_cap] + facts_pool[:f_cap]
            ep_cap = max(1, max_results - len(chosen_aux))
            chosen_eps = pure_eps[:ep_cap]
            keep_ids = {id(m) for m in (chosen_eps + chosen_aux)}
            reranked = [m for m in wide_reranked if id(m) in keep_ids][:max_results]

        # Round 2 Cognitive Cue Selection: explicitly identify 1-3 verbatim episodes with hidden link
        if (
            is_cognitive_cue
            and self._hyde_generator is not None
            and hasattr(self._hyde_generator, "select_cognitive_cues")
            and len(reranked) > 3
        ):
            try:
                verbatim_pool = [
                    (idx_r, m)
                    for idx_r, m in enumerate(reranked[:24])
                    if not (m.record.metadata or {}).get("proustian_cue_pointer")
                ]
                cand_texts = [
                    f"({(m.record.metadata or {}).get('speaker', '')}) {m.record.text}".strip()
                    for _, m in verbatim_pool
                ]
                chosen_local = await self._hyde_generator.select_cognitive_cues(query, cand_texts)
                if chosen_local:
                    chosen_orig = [
                        verbatim_pool[c][0] for c in chosen_local if c < len(verbatim_pool)
                    ]
                    promoted = []
                    chosen_set = set(chosen_orig)
                    for rank_i, idx in enumerate(chosen_orig):
                        m = reranked[idx]
                        m.relevance_score = max(m.relevance_score, 0.99 - 0.01 * rank_i)
                        meta = dict(m.record.metadata or {})
                        meta["_hyde_promoted"] = True
                        m.record.metadata = meta
                        promoted.append(m)
                    rest = [m for i, m in enumerate(reranked) if i not in chosen_set]
                    reranked = promoted + rest
            except Exception as exc:
                _logger.debug("cognitive_cue_selection_failed", error=str(exc))

        raw_meta = plan.analysis.metadata.get("retrieval_meta")
        retrieval_meta: dict[str, Any] | None = (
            dict(raw_meta) if isinstance(raw_meta, dict) else None
        )
        if neuro_dict and reranked:
            if retrieval_meta is None:
                retrieval_meta = {}
            retrieval_meta["neuromodulation"] = neuro_dict

        # Attach Conway Tier-3 Persistent DMN Self-Profiles & Procedural Skill Playbooks
        dmn_lines: list[str] = []
        if self.dmn_store is not None and getattr(
            settings.features, "conway_hierarchy_enabled", True
        ):
            try:
                dmn_lines = await self.dmn_store.get_formatted_profiles(tenant_id, query)
                if dmn_lines:
                    if retrieval_meta is None:
                        retrieval_meta = {}
                    retrieval_meta["dmn_self_profiles"] = dmn_lines
            except Exception:
                pass

        proc_lines: list[str] = []
        if self.procedural_engine is not None and getattr(
            settings.features, "procedural_rpe_enabled", True
        ):
            try:
                skills = await self.procedural_engine.retrieve_skills(
                    tenant_id, query, query_embedding=query_embedding, top_k=3
                )
                if skills:
                    proc_lines = [s.format_playbook_line() for s in skills]
                    if retrieval_meta is None:
                        retrieval_meta = {}
                    retrieval_meta["procedural_skills"] = proc_lines
            except Exception:
                pass

        sufficiency = assess_sufficiency(reranked)
        if return_packet:
            packet = self.packet_builder.build(reranked, query)
            if getattr(settings.features, "preplay_simulation_enabled", True) and (
                is_cognitive_cue or proc_lines
            ):
                try:
                    from .preplay import HippocampalPreplaySimulator

                    alerts = HippocampalPreplaySimulator.inline_packet_check(
                        query, packet, dmn_lines=dmn_lines, procedural_lines=proc_lines
                    )
                    if alerts:
                        if retrieval_meta is None:
                            retrieval_meta = {}
                        retrieval_meta["preplay_alerts"] = alerts
                except Exception:
                    pass

            packet.retrieval_meta = retrieval_meta
            packet.sufficiency = sufficiency
            if packet.all_memories and getattr(settings.features, "intrinsic_memory_enabled", True):
                spec = self.compute_intrinsic_spec(
                    packet, tenant_id=tenant_id, intent=plan.analysis.intent.value
                )
                meta_copy = dict(packet.retrieval_meta or {})
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
