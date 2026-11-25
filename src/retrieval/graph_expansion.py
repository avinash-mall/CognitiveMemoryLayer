"""Associative, temporal contiguity, BA10 prospective, and graph PPR expansion for HybridRetriever."""

from __future__ import annotations

import inspect
import time
from datetime import timedelta
from typing import TYPE_CHECKING, Any
from uuid import UUID

from ..utils.logging_config import get_logger
from ..utils.similarity import cosine_similarity
from .planner import RetrievalPlan, RetrievalSource, RetrievalStep

if TYPE_CHECKING:
    from ..memory.hippocampal.store import HippocampalStore
    from ..memory.neocortical.store import NeocorticalStore

logger = get_logger(__name__)

# Band that graph hits are rank-normalised into.
GRAPH_RELEVANCE_FLOOR = 0.55
GRAPH_RELEVANCE_CEILING = 0.85

# Temporal contiguity expansion parameters.
_CONTIGUITY_MAX_SEEDS = 3
_CONTIGUITY_WINDOW_MINUTES = 30
_CONTIGUITY_RELEVANCE_FACTOR = 0.7
_CONTIGUITY_SESSION_SCAN_LIMIT = 200


def _neighbours_of(seed: Any, rows: list[Any], k: int) -> list[Any]:
    """The k rows either side of ``seed`` in encoding order, seed excluded."""
    ordered = GraphExpansionMixin._encoding_order(rows)
    position = next((i for i, r in enumerate(ordered) if r.id == seed.id), None)
    if position is None:
        return []
    return ordered[max(0, position - k) : position] + ordered[position + 1 : position + 1 + k]


class GraphExpansionMixin:
    """Graph traversal, temporal contiguity, and BA10 prospective expansion methods."""

    hippocampal: HippocampalStore
    neocortical: NeocorticalStore

    @staticmethod
    def _derive_graph_seeds(
        all_results: list[dict[str, Any]],
        max_seeds: int = 10,
    ) -> list[str]:
        """Derive entity seeds from top constraint/fact retrieval results."""
        seeds: list[str] = []
        seen: set[str] = set()

        def _add(name: str) -> None:
            normalised = name.strip().lower()
            if normalised and normalised not in seen and len(normalised) > 1:
                seen.add(normalised)
                seeds.append(name.strip())

        for item in all_results:
            if len(seeds) >= max_seeds:
                break
            rec = item.get("record")
            if rec is None:
                continue

            for ent in getattr(rec, "entities", None) or []:
                name = getattr(ent, "normalized", None) or getattr(ent, "text", None) or str(ent)
                _add(name)

            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                for c in meta.get("constraints", []):
                    if isinstance(c, dict):
                        for field_name in ("subject", "scope"):
                            val = c.get(field_name)
                            if isinstance(val, str):
                                _add(val)
                subj = meta.get("subject")
                if isinstance(subj, str):
                    _add(subj)

        return seeds[:max_seeds]

    @staticmethod
    def _encoding_order(rows: list[Any]) -> list[Any]:
        """Sort a session's records into the order they were encoded."""
        indices = [(r.metadata or {}).get("turn_idx") for r in rows]
        if indices and all(isinstance(i, int) for i in indices):
            return sorted(rows, key=lambda r: (r.metadata or {})["turn_idx"])
        return sorted(rows, key=lambda r: r.timestamp)

    async def _expand_temporal_contiguity(
        self,
        tenant_id: str,
        all_results: list[dict[str, Any]],
    ) -> None:
        """Pull the turns encoded around each top vector hit into the candidate set."""
        from ..core.config import get_settings

        settings = get_settings()
        if not settings.features.temporal_contiguity_enabled:
            return

        k = settings.features.temporal_contiguity_neighbours
        seeds = [
            r
            for r in all_results
            if r.get("source") == "vector"
            and getattr(r.get("record"), "source_session_id", None)
            and getattr(r.get("record"), "timestamp", None)
        ]
        if not seeds:
            return
        seeds.sort(key=lambda r: r.get("relevance", 0.0), reverse=True)

        existing_ids = {
            str(getattr(x.get("record"), "id", "")) for x in all_results if x.get("record")
        }
        for seed in seeds[:_CONTIGUITY_MAX_SEEDS]:
            record = seed["record"]
            ts = record.timestamp
            try:
                window = timedelta(minutes=_CONTIGUITY_WINDOW_MINUTES)
                rows = await self.hippocampal.store.scan(
                    tenant_id,
                    filters={
                        "source_session_id": record.source_session_id,
                        "status": "active",
                        "since": ts - window,
                        "until": ts + window,
                    },
                    limit=_CONTIGUITY_SESSION_SCAN_LIMIT,
                )
            except Exception as exc:
                logger.warning("temporal_contiguity_failed", extra={"error": str(exc)})
                return
            for row in _neighbours_of(record, rows, k):
                rid = str(getattr(row, "id", ""))
                if not rid or rid in existing_ids:
                    continue
                existing_ids.add(rid)
                all_results.append(
                    {
                        "type": row.type.value,
                        "source": "contiguity",
                        "text": row.text,
                        "confidence": getattr(row, "confidence", 0.7),
                        "relevance": seed.get("relevance", 0.5) * _CONTIGUITY_RELEVANCE_FACTOR,
                        "timestamp": row.timestamp,
                        "record": row,
                    }
                )

    async def _resolve_prospective_parents(
        self,
        tenant_id: str,
        all_results: list[dict[str, Any]],
        query_embedding: list[float] | None = None,
    ) -> None:
        """BA10 Prospective Ecphory: resolve prospective implication hits to their parent episode."""
        parent_scores: dict[str, float] = {}
        for item in all_results:
            rec = item.get("record")
            meta = getattr(rec, "metadata", None) if rec is not None else None
            if not isinstance(meta, dict) or not meta.get("is_prospective_index"):
                continue
            pid = meta.get("parent_id") or meta.get("prospective_source_id")
            if pid:
                rel = float(item.get("relevance", 0.5) or 0.5)
                parent_scores[str(pid)] = max(parent_scores.get(str(pid), 0.0), rel)

        if not parent_scores:
            return

        existing_ids = {
            str(getattr(x.get("record"), "id", "")) for x in all_results if x.get("record")
        }
        to_fetch: list[UUID] = []
        for pid_str in parent_scores:
            if pid_str in existing_ids:
                continue
            try:
                to_fetch.append(UUID(pid_str))
            except (ValueError, TypeError, AttributeError):
                continue

        if not to_fetch:
            return

        get_batch = getattr(getattr(self.hippocampal, "store", None), "get_by_ids_batch", None)
        if not callable(get_batch):
            return
        try:
            maybe_coro = get_batch(to_fetch[:10])
            if not inspect.isawaitable(maybe_coro):
                return
            parents = await maybe_coro
            for parent in parents or []:
                if getattr(parent, "tenant_id", None) != tenant_id:
                    continue
                pid_str = str(parent.id)
                if pid_str in existing_ids:
                    continue
                existing_ids.add(pid_str)
                cue_rel = parent_scores.get(pid_str, 0.6)
                direct_rel = self._graph_relevance(parent, cue_rel, query_embedding)
                all_results.append(
                    {
                        "type": parent.type.value,
                        "source": "prospective_parent",
                        "text": parent.text,
                        "confidence": getattr(parent, "confidence", 0.8),
                        "relevance": max(direct_rel, cue_rel * 0.92),
                        "timestamp": parent.timestamp,
                        "record": parent,
                    }
                )
        except Exception as exc:
            logger.debug("prospective_parent_resolution_failed", extra={"error": str(exc)})

    async def _expand_second_hop_recollection(
        self,
        tenant_id: str,
        plan: RetrievalPlan,
        all_results: list[dict[str, Any]],
        query_embedding: list[float] | None,
        plan_start: float,
        plan_budget: float,
        timeouts_enabled: bool,
    ) -> None:
        """Feeling-of-Knowing Gated 2nd-Hop Cortical Recollection."""
        from ..core.config import get_settings

        settings = get_settings()
        if not getattr(settings.features, "multi_hop_recollection_enabled", True):
            return
        if not all_results:
            return
        if timeouts_enabled and (plan_budget - (time.perf_counter() - plan_start)) < 0.15:
            return

        initial_graph_steps = [s for s in plan.steps if s.source == RetrievalSource.GRAPH]
        if not initial_graph_steps:
            return

        queried_seeds = {
            s.strip().lower()
            for step in initial_graph_steps
            for s in (step.seeds or [])
            if s and s.strip()
        }
        top_first_pass = sorted(
            all_results, key=lambda r: float(r.get("relevance", 0.0) or 0.0), reverse=True
        )[:4]

        bridge_seeds: list[str] = []
        for item in top_first_pass:
            rec = item.get("record")
            if rec is None:
                continue
            for ent in getattr(rec, "entities", None) or []:
                name = getattr(ent, "normalized", None) or getattr(ent, "text", None) or str(ent)
                norm = name.strip().lower()
                if norm and len(norm) > 1 and norm not in queried_seeds:
                    queried_seeds.add(norm)
                    bridge_seeds.append(name.strip())
            meta = getattr(rec, "metadata", None)
            if isinstance(meta, dict):
                for focus_ent in meta.get("working_memory_focus") or []:
                    norm = str(focus_ent).strip().lower()
                    if norm and len(norm) > 1 and norm not in queried_seeds:
                        queried_seeds.add(norm)
                        bridge_seeds.append(str(focus_ent).strip())

        if not bridge_seeds:
            return

        hop2_step = RetrievalStep(
            source=RetrievalSource.GRAPH,
            seeds=bridge_seeds[:6],
            top_k=6,
            priority=1,
            timeout_ms=200,
        )
        try:
            hop2_items = await self._retrieve_graph(tenant_id, hop2_step, query_embedding)
            if hop2_items:
                all_results.extend(hop2_items)
        except Exception as exc:
            logger.debug("second_hop_recollection_failed", extra={"error": str(exc)})

    async def _retrieve_graph(
        self,
        tenant_id: str,
        step: RetrievalStep,
        query_embedding: list[float] | None = None,
    ) -> list[dict[str, Any]]:
        """Retrieve via knowledge graph PPR. Holistic: tenant-only."""
        if not step.seeds:
            return []
        results = await self.neocortical.multi_hop_query(tenant_id, seed_entities=step.seeds)
        if not results:
            return []

        raw = [float(r.get("relevance_score", 0.5) or 0.0) for r in results]
        hi = max(raw, default=0.0)
        lo = min(raw, default=0.0)
        span = hi - lo

        best: dict[str, float] = {}
        for r, score in zip(results, raw, strict=True):
            normalised = (
                GRAPH_RELEVANCE_CEILING
                if span <= 0
                else GRAPH_RELEVANCE_FLOOR
                + (score - lo) / span * (GRAPH_RELEVANCE_CEILING - GRAPH_RELEVANCE_FLOOR)
            )
            for rel in r.get("relations", []) or []:
                props = rel.get("relation_properties") or {}
                for rid in props.get("evidence_ids") or []:
                    if normalised > best.get(str(rid), 0.0):
                        best[str(rid)] = normalised
        if not best:
            return []

        ranked = sorted(best.items(), key=lambda kv: kv[1], reverse=True)[: step.top_k]
        ids: list[UUID] = []
        for rid, _score in ranked:
            try:
                ids.append(UUID(rid))
            except (ValueError, AttributeError, TypeError):
                continue
        records = await self.hippocampal.store.get_by_ids_batch(ids)

        items = []
        for record in records:
            if record.tenant_id != tenant_id:
                continue
            items.append(
                {
                    "type": record.type.value,
                    "source": "graph",
                    "text": record.text,
                    "relevance": self._graph_relevance(
                        record, best.get(str(record.id), GRAPH_RELEVANCE_FLOOR), query_embedding
                    ),
                    "record": record,
                }
            )
        return items

    @staticmethod
    def _graph_relevance(
        record: Any,
        graph_score: float,
        query_embedding: list[float] | None,
    ) -> float:
        """Rank a graph-resolved episode by how well it answers *this* query."""
        if query_embedding and record.embedding:
            return cosine_similarity(query_embedding, record.embedding)
        return graph_score
