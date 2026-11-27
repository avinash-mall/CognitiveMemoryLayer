"""Multi-pathway retrieval augmenters (HyDE, Persona, Bridge, Sparse BM25, Hypergraph PPR, Proustian)."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..core.schemas import RetrievedMemory
from ..utils.logging_config import get_logger

if TYPE_CHECKING:
    from .retriever import HybridRetriever

_logger = get_logger(__name__)

_PERSONA_STOPWORDS = frozenset(
    {
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
)


class RetrievalAugmentersMixin:
    """4-Pathway Ecphory & associative augmentation helpers mixed into MemoryRetriever."""

    retriever: HybridRetriever
    _hyde_generator: Any | None

    async def _hyde_augment(
        self,
        tenant_id: str,
        query: str,
        plan: Any,
        context_filter: list[str] | None,
        *,
        run_bm25: bool = False,
        run_ppr: bool = False,
        ppr_damping: float = 0.82,
        memory_types: list[str] | None = None,
        source_session_id: str | None = None,
    ) -> list[list[Any]]:
        """Run Stage-1 multi-angle query reformulation / HyDE across dense, BM25, and PPR."""
        import asyncio

        if not self._hyde_generator:
            return []
        try:
            is_cognitive = False
            routing_intent = "single_hop"
            try:
                from ..core.laya_system1 import LayaSystem1Engine

                rdec = LayaSystem1Engine.classify_query_routing(query)
                is_cognitive = rdec.is_cognitive_cue
                routing_intent = rdec.intent
            except Exception:
                is_cognitive = False

            use_multi_angle = False
            try:
                from ..core.config import get_settings

                use_multi_angle = bool(
                    getattr(get_settings().features, "llm_multi_angle_expansion_enabled", True)
                )
            except Exception:
                use_multi_angle = True

            hypos: list[str] = []
            if use_multi_angle and hasattr(self._hyde_generator, "expand_multi_angle_queries"):
                hypos = await self._hyde_generator.expand_multi_angle_queries(
                    query,
                    is_cognitive=is_cognitive,
                    routing_intent=routing_intent,
                )
            if not hypos:
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

            from .planner import RetrievalPlan, RetrievalSource, RetrievalStep

            base_tf = None
            base_mt: list[str] = []
            for s in getattr(plan, "steps", None) or []:
                if getattr(s, "source", None) == RetrievalSource.VECTOR:
                    base_tf = getattr(s, "time_filter", None)
                    base_mt = list(getattr(s, "memory_types", None) or [])
                    break

            dense_coros: list[Any] = []
            bm25_coros: list[Any] = []
            for h_text, emb in zip(hypos, embeddings, strict=False):
                vec_step = RetrievalStep(
                    source=RetrievalSource.VECTOR,
                    query=h_text,
                    top_k=18,
                    priority=1,
                    timeout_ms=5000,
                    time_filter=dict(base_tf) if isinstance(base_tf, dict) else None,
                    memory_types=list(base_mt),
                )
                vec_plan = RetrievalPlan(
                    query=h_text,
                    analysis=plan.analysis,
                    steps=[vec_step],
                    parallel_steps=[[0]],
                    total_timeout_ms=max(
                        8000, int(getattr(plan, "total_timeout_ms", 5000) or 5000)
                    ),
                    max_results=18,
                )
                dense_coros.append(
                    self.retriever.retrieve(
                        tenant_id,
                        vec_plan,
                        context_filter=context_filter,
                        query_embedding=emb,
                    )
                )
                if run_bm25:
                    bm25_coros.append(
                        self._sparse_bm25_augment(
                            tenant_id,
                            h_text,
                            top_k=12,
                            context_filter=context_filter,
                            memory_types=memory_types,
                            source_session_id=source_session_id,
                            keep_proustian=is_cognitive,
                        )
                    )

            _ = (run_ppr, ppr_damping)
            all_gathered = await asyncio.gather(*(dense_coros + bm25_coros), return_exceptions=True)
            n_dense = len(dense_coros)
            dense_outs = all_gathered[:n_dense]
            bm25_outs = all_gathered[n_dense:]

            def _union_by_best_score(lists: list[Any]) -> list[RetrievedMemory]:
                best: dict[str, RetrievedMemory] = {}
                for lst in lists:
                    if isinstance(lst, BaseException) or not lst:
                        continue
                    for m in lst:
                        mid = str(m.record.id)
                        prev = best.get(mid)
                        if prev is None or m.relevance_score > prev.relevance_score:
                            best[mid] = m
                return sorted(best.values(), key=lambda x: x.relevance_score, reverse=True)[:28]

            result_lists: list[list[Any]] = []
            u_dense = _union_by_best_score(list(dense_outs))
            if u_dense:
                result_lists.append(u_dense)
            u_bm25 = _union_by_best_score(list(bm25_outs))
            if u_bm25:
                result_lists.append(u_bm25)
            return result_lists
        except Exception as exc:
            _logger.debug("hyde_augment_failed", error=str(exc))
            return []

    @staticmethod
    def _apply_entity_bridge_boost(
        candidates: list[RetrievedMemory],
        angle_lists: list[list[RetrievedMemory]],
        query: str,
    ) -> None:
        """Non-hub Entity-Bridge Join (Kumiho-inspired): boost memories sharing discriminative
        entities across >=2 distinct retrieval angles while ignoring ubiquitous speaker hubs."""
        if len(angle_lists) < 2 or not candidates:
            return
        q_low = (query or "").lower()
        # Count entity frequency across candidate pool to filter out conversation-wide speaker hubs
        doc_freq: dict[str, int] = {}
        total_docs = max(1, len(candidates))
        for m in candidates:
            txt = str(getattr(m.record, "text", "") or "")
            ents = {
                tok.lower()
                for tok in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", txt)
                if tok.lower() not in _PERSONA_STOPWORDS
            }
            for e in ents:
                doc_freq[e] = doc_freq.get(e, 0) + 1

        hub_cutoff = max(6, int(total_docs * 0.38))
        angle_entity_hits: dict[str, set[int]] = {}
        for a_idx, rlist in enumerate(angle_lists):
            for m in rlist[:8]:
                txt = str(getattr(m.record, "text", "") or "")
                for tok in re.findall(r"\b[A-Z][a-zA-Z0-9]{2,}\b", txt):
                    e_low = tok.lower()
                    if e_low in _PERSONA_STOPWORDS or e_low in q_low:
                        continue
                    if doc_freq.get(e_low, 0) > hub_cutoff:
                        continue
                    angle_entity_hits.setdefault(e_low, set()).add(a_idx)

        bridge_entities = {e for e, idxs in angle_entity_hits.items() if len(idxs) >= 2}
        if not bridge_entities:
            return

        for m in candidates:
            t_low = str(getattr(m.record, "text", "") or "").lower()
            matched_bridges = [b for b in bridge_entities if b in t_low]
            if matched_bridges:
                m.relevance_score = min(0.96, m.relevance_score + 0.08)
                meta = dict(m.record.metadata or {})
                meta["_bridge_entities"] = matched_bridges[:4]
                m.record.metadata = meta

    async def _persona_profile_augment(
        self,
        tenant_id: str,
        query: str,
        plan: Any,
        context_filter: list[str] | None,
    ) -> list[RetrievedMemory]:
        """Augment common-sense/hypothetical queries with speaker persona & trait turns."""
        names = [
            m.group(0)
            for m in re.finditer(r"\b[A-Z][a-z]{2,15}\b", query)
            if m.group(0).lower() not in _PERSONA_STOPWORDS
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
                for r in first_pass[:12]
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

    async def _sparse_bm25_augment(
        self,
        tenant_id: str,
        query: str,
        *,
        top_k: int = 18,
        context_filter: list[str] | None = None,
        memory_types: list[str] | None = None,
        since: Any | None = None,
        until: Any | None = None,
        source_session_id: str | None = None,
        keep_proustian: bool = False,
    ) -> list[RetrievedMemory]:
        """Dentate Gyrus sparse lexical search (Postgres tsvector GIN / SQLite FTS5)."""
        store = getattr(self.retriever.hippocampal, "store", None)
        if store is None or not hasattr(store, "sparse_lexical_search"):
            return []
        filters: dict[str, Any] = {"status": "active"}
        if source_session_id:
            filters["source_session_id"] = source_session_id
        if memory_types:
            filters["type"] = memory_types
        if since is not None:
            filters["since"] = since
        if until is not None:
            filters["until"] = until
        try:
            hits = await store.sparse_lexical_search(
                tenant_id=tenant_id,
                query=query,
                top_k=top_k + 6,
                context_filter=context_filter,
                filters=filters,
            )
            out_bm25: list[RetrievedMemory] = []
            for rec, score in hits:
                if not keep_proustian and (rec.metadata or {}).get("proustian_cue_pointer"):
                    continue
                out_bm25.append(
                    RetrievedMemory(
                        record=rec,
                        relevance_score=score,
                        retrieval_source="sparse_bm25",
                    )
                )
                if len(out_bm25) >= top_k:
                    break
            return out_bm25
        except Exception as exc:
            _logger.debug("sparse_bm25_augment_failed", error=str(exc))
            return []

    async def _hypergraph_ppr_augment(
        self,
        tenant_id: str,
        query: str,
        seed_entities: list[str] | None,
        *,
        damping: float = 0.82,
        memory_types: list[str] | None = None,
        source_session_id: str | None = None,
    ) -> list[RetrievedMemory]:
        """Sub-5ms CA3 Bipartite Hypergraph Personalized PageRank spreading activation."""
        hypergraph = getattr(self.retriever.neocortical, "hypergraph", None)
        store = getattr(self.retriever.hippocampal, "store", None)
        if hypergraph is None or store is None:
            return []
        try:
            ep_scores, _engrams = await hypergraph.spreading_activation(
                tenant_id=tenant_id,
                query=query,
                seed_entities=seed_entities,
                top_k_episodes=16,
                damping=damping,
            )
            if not ep_scores:
                return []
            score_map = dict(ep_scores)
            uuids: list[UUID] = []
            for eid, _ in ep_scores:
                try:
                    uuids.append(UUID(str(eid)))
                except ValueError:
                    continue
            if not uuids:
                return []
            allowed_types = set(memory_types) if memory_types else None
            records = await store.get_by_ids_batch(uuids)
            out: list[RetrievedMemory] = []
            for rec in records:
                if source_session_id and rec.source_session_id != source_session_id:
                    continue
                if allowed_types is not None and rec.type.value not in allowed_types:
                    continue
                sc = score_map.get(str(rec.id), 0.65)
                out.append(
                    RetrievedMemory(
                        record=rec,
                        relevance_score=sc,
                        retrieval_source="hypergraph_ppr",
                    )
                )
            return out
        except Exception as exc:
            _logger.debug("hypergraph_ppr_augment_failed", error=str(exc))
            return []

    async def _resolve_proustian_and_conway_parents(
        self,
        candidates: list[RetrievedMemory],
        *,
        is_cognitive_cue: bool,
    ) -> list[RetrievedMemory]:
        """Resolve Grounded Proustian Cue Pointers directly to verbatim parent episodes and strip pointers."""
        store = getattr(self.retriever.hippocampal, "store", None)
        if not candidates:
            return candidates

        min_ptr_rel = 0.42 if is_cognitive_cue else 0.72
        ptr_hits: list[tuple[float, str]] = []
        cleaned_candidates: list[RetrievedMemory] = []
        for m in candidates:
            meta = m.record.metadata or {}
            if meta.get("proustian_cue_pointer"):
                if meta.get("parent_episode_id") and m.relevance_score >= min_ptr_rel:
                    ptr_hits.append((float(m.relevance_score), str(meta["parent_episode_id"])))
                continue
            cleaned_candidates.append(m)

        ptr_hits.sort(key=lambda item: item[0], reverse=True)
        parent_scores: dict[str, float] = {}
        for rel, pid in ptr_hits:
            if len(parent_scores) < 8 or pid in parent_scores:
                parent_scores[pid] = max(parent_scores.get(pid, 0.0), min(0.94, rel + 0.08))

        if not parent_scores or store is None:
            return cleaned_candidates

        existing_by_id = {str(m.record.id): m for m in cleaned_candidates}
        missing_uuids: list[UUID] = []
        for pid, pscore in parent_scores.items():
            if pid in existing_by_id:
                em = existing_by_id[pid]
                em.relevance_score = max(em.relevance_score, pscore)
                em_meta = dict(em.record.metadata or {})
                em_meta["_proustian_promoted"] = True
                if is_cognitive_cue:
                    em_meta["_hyde_promoted"] = True
                em.record.metadata = em_meta
            else:
                try:
                    missing_uuids.append(UUID(pid))
                except ValueError:
                    continue

        if not missing_uuids:
            return cleaned_candidates

        try:
            fetched = await store.get_by_ids_batch(missing_uuids)
            resolved_list: list[RetrievedMemory] = []
            for rec in fetched:
                pscore = parent_scores.get(str(rec.id), 0.85)
                rmeta = dict(rec.metadata or {})
                rmeta["_proustian_promoted"] = True
                if is_cognitive_cue:
                    rmeta["_hyde_promoted"] = True
                rec.metadata = rmeta
                resolved_list.append(
                    RetrievedMemory(
                        record=rec,
                        relevance_score=pscore,
                        retrieval_source="proustian_ecphory",
                    )
                )
            return resolved_list + cleaned_candidates
        except Exception as exc:
            _logger.debug("proustian_parent_resolve_failed", error=str(exc))
            return cleaned_candidates

    async def _expand_post_merge_contiguity(
        self,
        tenant_id: str,
        memories: list[RetrievedMemory],
        *,
        k: int = 1,
        max_seeds: int = 8,
    ) -> list[RetrievedMemory]:
        """Expand ±k immediate conversational turns around top merged hits (including HyDE/BM25/PPR/Proustian)."""
        from datetime import timedelta

        from .graph_expansion import _neighbours_of

        store = getattr(getattr(self.retriever, "hippocampal", None), "store", None)
        scan_fn = getattr(store, "scan", None) if store is not None else None
        if not callable(scan_fn) or not memories:
            return memories

        seeds: list[RetrievedMemory] = []
        for m in memories:
            rec = m.record
            rmeta = rec.metadata or {}
            if rmeta.get("proustian_cue_pointer") or rmeta.get("conway_tier") == "chapter_gist":
                continue
            if str(rec.text or "").startswith(("[Chapter Summary", "[Prospective", "[Proustian")):
                continue
            if not getattr(rec, "source_session_id", None) or not getattr(rec, "timestamp", None):
                continue
            if "turn_idx" not in rmeta and m.retrieval_source in ("facts", "constraints"):
                continue
            seeds.append(m)
            if len(seeds) >= max_seeds:
                break

        if not seeds:
            return memories

        existing_ids = {str(m.record.id) for m in memories}
        session_cache: dict[str, list[Any]] = {}
        added: list[RetrievedMemory] = []
        window = timedelta(minutes=30)

        for seed_mem in seeds:
            rec = seed_mem.record
            ts = rec.timestamp
            sess_key = f"{rec.source_session_id}:{ts.isoformat()[:16]}"
            rows = session_cache.get(sess_key)
            if rows is None:
                try:
                    maybe_coro = scan_fn(
                        tenant_id,
                        filters={
                            "source_session_id": rec.source_session_id,
                            "status": "active",
                            "since": ts - window,
                            "until": ts + window,
                        },
                        limit=200,
                    )
                    if not hasattr(maybe_coro, "__await__"):
                        return memories
                    rows = await maybe_coro
                    session_cache[sess_key] = rows or []
                except Exception as exc:
                    _logger.debug("post_merge_contiguity_failed", error=str(exc))
                    return memories
            for nb in _neighbours_of(rec, rows, k):
                nb_id = str(getattr(nb, "id", ""))
                if not nb_id or nb_id in existing_ids:
                    continue
                if str(getattr(nb, "text", "") or "").startswith(
                    ("[Chapter Summary", "[Prospective", "[Proustian")
                ):
                    continue
                existing_ids.add(nb_id)
                nb_meta = dict(getattr(nb, "metadata", None) or {})
                if (rec.metadata or {}).get("_proustian_promoted") or (rec.metadata or {}).get(
                    "_hyde_promoted"
                ):
                    nb_meta["_hyde_promoted"] = True
                nb.metadata = nb_meta
                added.append(
                    RetrievedMemory(
                        record=nb,
                        relevance_score=round(seed_mem.relevance_score * 0.88, 4),
                        retrieval_source="contiguity",
                    )
                )

        return memories + added
