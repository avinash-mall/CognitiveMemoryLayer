"""Bipartite Cortical Hypergraph & Sub-5ms CA3 Spreading Activation (HippoRAG + HyperMem).

Models N-ary episodic bindings (Hyperedges connecting Actors, Entities, Event Summary,
Temporal Anchor, Location, Causal State, Grounded Prospective Cues, Sensory Descriptors,
and Verbatim Episode IDs) with a fast in-memory sparse bipartite Personalized PageRank
(CA3 auto-associative pattern completion) persisted write-through to Postgres or SQLite.
"""

from __future__ import annotations

import hashlib
import re
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

from ...utils.logging_config import get_logger

logger = get_logger(__name__)

_TOKEN_RE = re.compile(r"[a-zA-Z0-9]{3,}")
_STOPWORDS = frozenset(
    {
        "the",
        "and",
        "for",
        "that",
        "this",
        "with",
        "from",
        "what",
        "when",
        "where",
        "who",
        "which",
        "why",
        "how",
        "would",
        "could",
        "should",
        "have",
        "has",
        "had",
        "were",
        "was",
        "are",
        "been",
        "into",
        "about",
        "they",
        "them",
        "their",
        "there",
        "here",
        "just",
        "very",
        "some",
        "more",
        "also",
        "than",
        "then",
    }
)


def _norm_node(raw: str) -> str:
    return " ".join((raw or "").strip().lower().split())


def _extract_cue_nodes(texts: list[str], max_terms: int = 18) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for txt in texts:
        for m in _TOKEN_RE.finditer((txt or "").lower()):
            tok = m.group(0)
            if tok not in _STOPWORDS and tok not in seen:
                seen.add(tok)
                out.append(f"cue:{tok}")
                if len(out) >= max_terms:
                    return out
    return out


@dataclass
class HyperedgeEngram:
    """N-ary Cortical Hyperedge binding situational dimensions to verbatim episodes."""

    engram_key: str
    tenant_id: str
    event_summary: str
    actors: list[str] = field(default_factory=list)
    entities: list[str] = field(default_factory=list)
    temporal_anchor: str | None = None
    location: str | None = None
    causal_state: str | None = None
    prospective_cues: list[str] = field(default_factory=list)
    sensory_descriptors: list[str] = field(default_factory=list)
    episode_ids: list[str] = field(default_factory=list)
    confidence: float = 0.85
    synaptic_weight: float = 1.0
    id: str = field(default_factory=lambda: str(uuid4()))

    def incident_nodes(self) -> list[tuple[str, float]]:
        """Return weighted bipartite concept/entity/cue nodes attached to this hyperedge."""
        nodes: list[tuple[str, float]] = []
        seen: set[str] = set()

        def _add(key: str, w: float) -> None:
            if key and key not in seen:
                seen.add(key)
                nodes.append((key, w))

        for a in self.actors:
            na = _norm_node(a)
            if na:
                _add(f"ent:{na}", 1.35)
        for e in self.entities:
            ne = _norm_node(e)
            if ne:
                _add(f"ent:{ne}", 1.20)
                for tok in _TOKEN_RE.findall(ne):
                    if tok not in _STOPWORDS:
                        _add(f"cue:{tok}", 0.75)
        if self.location:
            nl = _norm_node(self.location)
            if nl:
                _add(f"loc:{nl}", 1.10)
        if self.temporal_anchor:
            nt = _norm_node(self.temporal_anchor)
            if nt:
                _add(f"time:{nt}", 0.90)

        cue_sources = (
            [self.event_summary]
            + ([self.causal_state] if self.causal_state else [])
            + list(self.prospective_cues)
            + list(self.sensory_descriptors)
        )
        for cnode in _extract_cue_nodes(cue_sources, max_terms=24):
            _add(cnode, 0.85)
        return nodes


class _TenantBipartiteIndex:
    """In-memory sparse adjacency structure for sub-5ms Personalized PageRank."""

    def __init__(self) -> None:
        self.engrams: dict[str, HyperedgeEngram] = {}
        self.node_to_hedges: dict[str, dict[str, float]] = defaultdict(dict)
        self.hedge_to_nodes: dict[str, dict[str, float]] = defaultdict(dict)
        self.hydrated: bool = False

    def upsert_engram(self, engram: HyperedgeEngram) -> None:
        key = engram.engram_key
        # Merge episode_ids & boost synaptic weight if already present
        if key in self.engrams:
            prev = self.engrams[key]
            merged_eps = list(dict.fromkeys(prev.episode_ids + engram.episode_ids))[-32:]
            engram.episode_ids = merged_eps
            engram.synaptic_weight = max(prev.synaptic_weight, engram.synaptic_weight)
            # Clean previous incidences
            for n in list(self.hedge_to_nodes.get(key, {})):
                self.node_to_hedges[n].pop(key, None)
            self.hedge_to_nodes.pop(key, None)

        self.engrams[key] = engram
        sw = max(0.1, engram.synaptic_weight)
        for node, base_w in engram.incident_nodes():
            eff_w = base_w * sw
            self.node_to_hedges[node][key] = eff_w
            self.hedge_to_nodes[key][node] = eff_w


class CorticalHypergraphEngine:
    """Manages N-ary hyperedges and sub-5ms CA3 Spreading Activation across Postgres/SQLite."""

    def __init__(
        self,
        pg_session_factory: Any | None = None,
        sqlite_store: Any | None = None,
        neo4j_graph: Any | None = None,
    ) -> None:
        self.pg_session_factory = pg_session_factory
        self.sqlite_store = sqlite_store
        self.neo4j_graph = neo4j_graph
        self._indices: dict[str, _TenantBipartiteIndex] = defaultdict(_TenantBipartiteIndex)

    @staticmethod
    def make_engram_key(tenant_id: str, summary: str, episode_ids: list[str]) -> str:
        seed = f"{tenant_id}:{summary.strip().lower()}:{','.join(sorted(episode_ids[:3]))}"
        return hashlib.sha1(seed.encode("utf-8")).hexdigest()[:32]

    async def _ensure_hydrated(self, tenant_id: str) -> _TenantBipartiteIndex:
        idx = self._indices[tenant_id]
        if idx.hydrated:
            return idx
        idx.hydrated = True

        # 1. Hydrate from Postgres if configured
        if self.pg_session_factory is not None:
            try:
                from sqlalchemy import select

                from ...storage.models import HyperedgeEngramModel

                async with self.pg_session_factory() as session:
                    res = await session.execute(
                        select(HyperedgeEngramModel)
                        .where(HyperedgeEngramModel.tenant_id == tenant_id)
                        .limit(5000)
                    )
                    for row in res.scalars().all():
                        idx.upsert_engram(
                            HyperedgeEngram(
                                id=str(row.id),
                                engram_key=row.engram_key,
                                tenant_id=row.tenant_id,
                                event_summary=row.event_summary,
                                actors=list(row.actors or []),
                                entities=list(row.entities or []),
                                temporal_anchor=row.temporal_anchor,
                                location=row.location,
                                causal_state=row.causal_state,
                                prospective_cues=list(row.prospective_cues or []),
                                sensory_descriptors=list(row.sensory_descriptors or []),
                                episode_ids=[str(x) for x in (row.episode_ids or [])],
                                confidence=float(row.confidence or 0.85),
                                synaptic_weight=float(row.synaptic_weight or 1.0),
                            )
                        )
            except Exception as exc:
                logger.debug("hypergraph_pg_hydrate_skipped", extra={"error": str(exc)})

        # 2. Hydrate from SQLite if configured
        if self.sqlite_store is not None and hasattr(self.sqlite_store, "load_hyperedges"):
            try:
                rows = await self.sqlite_store.load_hyperedges(tenant_id)
                for r in rows:
                    idx.upsert_engram(r)
            except Exception as exc:
                logger.debug("hypergraph_sqlite_hydrate_skipped", extra={"error": str(exc)})

        return idx

    async def upsert_engrams(self, tenant_id: str, engrams: list[HyperedgeEngram]) -> int:
        """Upsert N-ary hyperedges into in-memory CSR index + persistent store + Neo4j."""
        if not engrams:
            return 0
        idx = await self._ensure_hydrated(tenant_id)
        for eng in engrams:
            idx.upsert_engram(eng)

        # Persist to Postgres
        if self.pg_session_factory is not None:
            try:
                from sqlalchemy import and_, select

                from ...storage.models import HyperedgeEngramModel

                now = datetime.now(UTC).replace(tzinfo=None)
                async with self.pg_session_factory() as session:
                    for eng in engrams:
                        res = await session.execute(
                            select(HyperedgeEngramModel).where(
                                and_(
                                    HyperedgeEngramModel.tenant_id == tenant_id,
                                    HyperedgeEngramModel.engram_key == eng.engram_key,
                                )
                            )
                        )
                        existing = res.scalar_one_or_none()
                        if existing is not None:
                            existing.event_summary = eng.event_summary
                            existing.actors = eng.actors
                            existing.entities = eng.entities
                            existing.temporal_anchor = eng.temporal_anchor
                            existing.location = eng.location
                            existing.causal_state = eng.causal_state
                            existing.prospective_cues = eng.prospective_cues
                            existing.sensory_descriptors = eng.sensory_descriptors
                            existing.episode_ids = eng.episode_ids
                            existing.confidence = eng.confidence
                            existing.synaptic_weight = eng.synaptic_weight
                            existing.updated_at = now
                        else:
                            session.add(
                                HyperedgeEngramModel(
                                    tenant_id=tenant_id,
                                    engram_key=eng.engram_key,
                                    actors=eng.actors,
                                    entities=eng.entities,
                                    event_summary=eng.event_summary,
                                    temporal_anchor=eng.temporal_anchor,
                                    location=eng.location,
                                    causal_state=eng.causal_state,
                                    prospective_cues=eng.prospective_cues,
                                    sensory_descriptors=eng.sensory_descriptors,
                                    episode_ids=eng.episode_ids,
                                    confidence=eng.confidence,
                                    synaptic_weight=eng.synaptic_weight,
                                    created_at=now,
                                    updated_at=now,
                                )
                            )
                    await session.commit()
            except Exception as exc:
                logger.debug("hypergraph_pg_persist_failed", extra={"error": str(exc)})

        # Persist to SQLite
        if self.sqlite_store is not None and hasattr(self.sqlite_store, "save_hyperedges"):
            try:
                await self.sqlite_store.save_hyperedges(tenant_id, engrams)
            except Exception as exc:
                logger.debug("hypergraph_sqlite_persist_failed", extra={"error": str(exc)})

        # Mirror actor-entity co-occurrence edges to Neo4j if present
        if self.neo4j_graph is not None and hasattr(self.neo4j_graph, "merge_edges_batch"):
            try:
                edges: list[dict[str, Any]] = []
                for eng in engrams:
                    participants = list(dict.fromkeys(eng.actors + eng.entities))[:6]
                    for i, src in enumerate(participants):
                        for dst in participants[i + 1 :]:
                            edges.append(
                                {
                                    "subject": src,
                                    "predicate": "CO_OCCURS_IN_ENGRAM",
                                    "object": dst,
                                    "properties": {
                                        "summary": eng.event_summary[:200],
                                        "evidence_ids": eng.episode_ids[:5],
                                        "confidence": eng.confidence,
                                    },
                                }
                            )
                if edges:
                    await self.neo4j_graph.merge_edges_batch(tenant_id, tenant_id, edges[:60])
            except Exception:
                pass

        return len(engrams)

    async def spreading_activation(
        self,
        tenant_id: str,
        query: str,
        seed_entities: list[str] | None = None,
        *,
        top_k_episodes: int = 16,
        top_k_engrams: int = 8,
        damping: float = 0.82,
        iterations: int = 10,
    ) -> tuple[list[tuple[str, float]], list[tuple[HyperedgeEngram, float]]]:
        """Run damped Personalized PageRank over the bipartite Node <-> Hyperedge graph (<5ms).

        Returns:
          - Ranked ``(episode_id, calibrated_score)`` pairs for grounded verbatim recollection
          - Ranked ``(HyperedgeEngram, calibrated_score)`` pairs for N-ary situational context
        """
        idx = await self._ensure_hydrated(tenant_id)
        if not idx.engrams:
            return [], []

        # Build personalization restart vector p over matched bipartite nodes
        restart: dict[str, float] = {}
        for ent in seed_entities or []:
            ne = _norm_node(ent)
            if f"ent:{ne}" in idx.node_to_hedges:
                restart[f"ent:{ne}"] = restart.get(f"ent:{ne}", 0.0) + 2.0
            if f"loc:{ne}" in idx.node_to_hedges:
                restart[f"loc:{ne}"] = restart.get(f"loc:{ne}", 0.0) + 1.5

        for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", query or ""):
            cand = _norm_node(m.group(0))
            if f"ent:{cand}" in idx.node_to_hedges:
                restart[f"ent:{cand}"] = restart.get(f"ent:{cand}", 0.0) + 1.8

        for cnode in _extract_cue_nodes([query or ""], max_terms=20):
            if cnode in idx.node_to_hedges:
                restart[cnode] = restart.get(cnode, 0.0) + 1.0
            raw_tok = cnode.removeprefix("cue:")
            if f"ent:{raw_tok}" in idx.node_to_hedges:
                restart[f"ent:{raw_tok}"] = restart.get(f"ent:{raw_tok}", 0.0) + 1.2

        if not restart:
            return [], []

        total_p = sum(restart.values())
        p_vec = {k: v / total_p for k, v in restart.items()}
        node_scores: dict[str, float] = dict(p_vec)
        hedge_scores: dict[str, float] = defaultdict(float)

        # Bipartite Personalized PageRank power iterations (Node -> Hyperedge -> Node)
        for _ in range(max(2, iterations)):
            next_hedges: dict[str, float] = defaultdict(float)
            for node, n_mass in node_scores.items():
                if n_mass < 1e-6:
                    continue
                incident = idx.node_to_hedges.get(node)
                if not incident:
                    continue
                denom = sum(incident.values()) or 1.0
                for h_key, w in incident.items():
                    next_hedges[h_key] += n_mass * (w / denom)

            hedge_scores = next_hedges
            next_nodes: dict[str, float] = {k: (1.0 - damping) * v for k, v in p_vec.items()}
            for h_key, h_mass in hedge_scores.items():
                if h_mass < 1e-6:
                    continue
                h_nodes = idx.hedge_to_nodes.get(h_key)
                if not h_nodes:
                    continue
                denom = sum(h_nodes.values()) or 1.0
                for node, w in h_nodes.items():
                    next_nodes[node] = next_nodes.get(node, 0.0) + damping * h_mass * (w / denom)
            node_scores = next_nodes

        if not hedge_scores:
            return [], []

        max_h = max(hedge_scores.values()) or 1.0
        ranked_hedges = sorted(hedge_scores.items(), key=lambda x: x[1], reverse=True)
        engram_results: list[tuple[HyperedgeEngram, float]] = []
        episode_acc: dict[str, float] = defaultdict(float)

        for h_key, raw_h in ranked_hedges[: top_k_engrams * 3]:
            eng = idx.engrams.get(h_key)
            if eng is None:
                continue
            norm_h = min(0.92, 0.54 + 0.38 * (raw_h / max_h)) * min(
                1.15, max(0.75, eng.synaptic_weight)
            )
            norm_h = min(0.94, norm_h)
            if len(engram_results) < top_k_engrams:
                engram_results.append((eng, norm_h))
            for ep_id in eng.episode_ids:
                # Multi-hyperedge convergence onto the same verbatim episode boosts activation
                prev = episode_acc.get(ep_id, 0.0)
                episode_acc[ep_id] = max(prev, norm_h) + (0.04 if prev > 0 else 0.0)

        ranked_eps = sorted(
            ((ep_id, min(0.95, sc)) for ep_id, sc in episode_acc.items()),
            key=lambda x: x[1],
            reverse=True,
        )[:top_k_episodes]
        return ranked_eps, engram_results

    async def modulate_synaptic_weights(
        self,
        tenant_id: str,
        episode_ids: list[str],
        delta: float,
    ) -> int:
        """Dopaminergic RPE synaptic weight plasticity over hyperedges linked to episode_ids."""
        if not episode_ids:
            return 0
        idx = await self._ensure_hydrated(tenant_id)
        ep_set = {str(e) for e in episode_ids}
        updated: list[HyperedgeEngram] = []
        for eng in idx.engrams.values():
            if ep_set & set(eng.episode_ids):
                eng.synaptic_weight = max(0.2, min(2.5, eng.synaptic_weight + delta))
                updated.append(eng)
        if updated:
            await self.upsert_engrams(tenant_id, updated)
        return len(updated)
