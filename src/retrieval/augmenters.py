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
    ) -> list[list[Any]]:
        """Run HyDE: generate hypothetical memory cues and retrieve with them."""
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
                if (rec.metadata or {}).get("proustian_cue_pointer"):
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

        min_ptr_rel = 0.52 if is_cognitive_cue else 0.72
        parent_scores: dict[str, float] = {}
        cleaned_candidates: list[RetrievedMemory] = []
        for m in candidates:
            meta = m.record.metadata or {}
            if meta.get("proustian_cue_pointer"):
                if meta.get("parent_episode_id") and m.relevance_score >= min_ptr_rel:
                    pid = str(meta["parent_episode_id"])
                    if len(parent_scores) < 5 or pid in parent_scores:
                        parent_scores[pid] = max(
                            parent_scores.get(pid, 0.0), min(0.92, m.relevance_score + 0.08)
                        )
                continue
            cleaned_candidates.append(m)

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
