"""Reranker for retrieved memories."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..core.enums import MemoryType
from ..core.schemas import RetrievedMemory
from ..utils.retention import frequency_score, retention
from ..utils.similarity import jaccard, word_set

# Recency weights by memory type stability
_STABLE_TYPES = {MemoryType.CONSTRAINT}
_STABLE_CONSTRAINT_TYPES = {"value", "policy", "identity"}
_SEMI_STABLE_CONSTRAINT_TYPES = {"state", "goal"}


@dataclass
class RerankerConfig:
    """Reranker configuration."""

    relevance_weight: float = 0.5
    recency_weight: float = 0.1  # BUG-04: reduced from 0.2 to limit recency bias
    confidence_weight: float = 0.2
    diversity_weight: float = 0.1
    frequency_weight: float = 0.05  # retrieval practice; smallest term by design
    diversity_threshold: float = 0.8
    max_results: int = 20
    # Fraction of its own score a consolidated episode loses when the gist fact it
    # was folded into is also in the result set. Demotion, not exclusion.
    consolidated_penalty: float = 0.4


class MemoryReranker:
    """Reranks retrieved memories by relevance, recency, confidence, and diversity."""

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()

    async def rerank(
        self,
        memories: list[RetrievedMemory],
        query: str,
        max_results: int | None = None,
    ) -> list[RetrievedMemory]:
        """Rerank memories by combined score and diversity."""
        ranked, _ = await self.rerank_with_breakdown(memories, query, max_results=max_results)
        return ranked

    async def rerank_with_breakdown(
        self,
        memories: list[RetrievedMemory],
        query: str,
        max_results: int | None = None,
    ) -> tuple[list[RetrievedMemory], list[dict[str, Any]]]:
        """Rerank memories and return score breakdowns for explain mode."""
        if not memories:
            return [], []
        max_results = max_results or self.config.max_results

        word_sets = {i: word_set(mem.record.text) for i, mem in enumerate(memories)}
        breakdowns = {
            i: self._score_components(mem, memories, word_sets, i, query=query)
            for i, mem in enumerate(memories)
        }
        base_scores = {i: breakdowns[i]["base_score"] for i in breakdowns}

        # ponytail: constraint boost from vector-search cosine relevance. If ranking
        # quality ever demands it, a pairwise scorer could replace this — but note the
        # scale trap in _score_components before wiring one in.
        for idx, mem in enumerate(memories):
            if mem.record.type == MemoryType.CONSTRAINT:
                boost = max(0.0, min(1.0, mem.relevance_score)) * 2.0
                breakdowns[idx]["constraint_boost"] = boost
                base_scores[idx] += boost

        # Consolidation is additive: migrating a cluster of episodes into a gist fact
        # leaves every source episode active and embedded, so a query can match both
        # halves and the packet carries the same content twice. Demote the source when
        # its gist is present in this same result set. Demote, not drop — detail that
        # survives only in the episode would otherwise become unreachable.
        gist_keys = {
            mem.record.key for mem in memories if mem.retrieval_source == "facts" and mem.record.key
        }
        if gist_keys:
            for idx, mem in enumerate(memories):
                fact_key = (mem.record.metadata or {}).get("consolidated_into_fact_key")
                if fact_key and fact_key in gist_keys:
                    penalty = base_scores[idx] * self.config.consolidated_penalty
                    base_scores[idx] -= penalty
                    breakdowns[idx]["notes"].append("demoted_superseded_by_gist")

        scored = [(score, memories[idx], idx) for idx, score in base_scores.items()]

        scored.sort(key=lambda x: x[0], reverse=True)
        diverse = self._apply_diversity_with_indices(scored, max_results)

        ranked_memories: list[RetrievedMemory] = []
        ranked_breakdowns: list[dict[str, Any]] = []
        for rank, (score, mem, idx) in enumerate(diverse, start=1):
            ranked_memories.append(mem)
            ranked_breakdowns.append(
                {
                    # Keys must match RetrievalExplainRerankItem (src/api/schemas.py),
                    # which is also what the dashboard's rerank list renders.
                    "rank": rank,
                    "id": mem.record.id,
                    "text": mem.record.text,
                    "source_type": mem.record.type.value,
                    "retrieval_source": mem.retrieval_source,
                    "final_score": score,
                    "breakdown": {
                        "relevance": breakdowns[idx]["relevance"],
                        "recency": breakdowns[idx]["recency"],
                        "confidence": breakdowns[idx]["confidence"],
                        "diversity": breakdowns[idx]["diversity"],
                        "frequency": breakdowns[idx]["frequency"],
                        "recency_weight": breakdowns[idx]["recency_weight"],
                        "constraint_boost": breakdowns[idx].get("constraint_boost", 0.0),
                    },
                    "notes": list(breakdowns[idx].get("notes", [])),
                }
            )
        return ranked_memories, ranked_breakdowns

    def _get_recency_weight(self, memory: RetrievedMemory) -> float:
        """Determine recency weight based on memory type stability.

        Stable constraints (value/policy) and preference/value facts (BUG-04)
        should not be heavily penalised for age.
        """
        recency_weight = self.config.recency_weight
        key = getattr(memory.record, "key", None) or ""
        if key.startswith(("user:preference:", "user:value:")):
            recency_weight = min(recency_weight, 0.1)  # Semi-stable: preferences/values age slowly
        if memory.record.type in _STABLE_TYPES:
            # Check constraint sub-type from metadata
            meta = memory.record.metadata or {}
            constraints_meta = meta.get("constraints", [])
            if constraints_meta and isinstance(constraints_meta, list):
                ctype = constraints_meta[0].get("constraint_type", "")
                if ctype in _STABLE_CONSTRAINT_TYPES:
                    recency_weight = 0.0  # Stable: age does not affect score
                if ctype in _SEMI_STABLE_CONSTRAINT_TYPES:
                    recency_weight = min(recency_weight, 0.15)
            else:
                recency_weight = min(recency_weight, 0.10)  # Generic constraint: moderate stability

        return recency_weight

    def _score_components(
        self,
        memory: RetrievedMemory,
        all_memories: list[RetrievedMemory],
        word_sets: dict[int, frozenset[str]] | None = None,
        mem_index: int = -1,
        query: str = "",
    ) -> dict[str, Any]:
        """Calculate score components for a memory."""
        _ = query
        notes: list[str] = []
        # Clamp to [0,1]: retrieval sources do not share a scale. Vector/fact prongs
        # emit cosine-like 0..1, but the graph prong passes through a raw Neo4j
        # co-occurrence score that is unbounded (observed 245.5), which would let a
        # single graph hit dominate every other signal in the weighted sum below.
        relevance = max(0.0, min(1.0, memory.relevance_score))
        if memory.relevance_score > 1.0:
            notes.append(f"relevance_clamped_from_{memory.relevance_score:.3g}")
        ts = memory.record.timestamp
        if isinstance(ts, datetime):
            now = datetime.now(UTC)
            tz_ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
            age_days = (now - tz_ts).days
        else:
            age_days = 0
        # Same curve the forgetting scorer uses, reading the same per-record rate.
        # These were two different functions until now, so a memory aged at one speed
        # when deciding what to delete and another when deciding what to show.
        recency = retention(age_days, getattr(memory.record, "decay_rate", None))
        confidence = memory.record.confidence
        # Retrieval practice: what you use often gets easier to find, not merely
        # harder to delete. Deliberately the smallest weight in the sum — this is a
        # rich-get-richer loop, and only the vector prong increments access_count,
        # so a large weight would systematically bury fact- and graph-sourced hits.
        frequency = frequency_score(getattr(memory.record, "access_count", 0) or 0)

        diversity_cap = min(len(all_memories), 20)
        if len(all_memories) <= 5:
            diversity = 1.0
        elif len(all_memories) > 1:
            my_ws = word_sets[mem_index] if word_sets and mem_index >= 0 else None
            total_sim = 0.0
            count = 0
            for j, other in enumerate(all_memories[:diversity_cap]):
                if other is memory:
                    continue
                if my_ws is not None and word_sets:
                    total_sim += jaccard(my_ws, word_sets.get(j, frozenset()))
                else:
                    total_sim += jaccard(memory.record.text, other.record.text)
                count += 1
            avg_sim = total_sim / count if count > 0 else 0.0
            diversity = 1.0 - avg_sim
        else:
            diversity = 1.0

        recency_weight = self._get_recency_weight(memory)
        score = (
            self.config.relevance_weight * relevance
            + self.config.confidence_weight * confidence
            + self.config.diversity_weight * diversity
            + self.config.frequency_weight * frequency
        )
        if recency_weight > 0:
            score += recency_weight * recency
        return {
            "relevance": relevance,
            "recency": recency,
            "confidence": confidence,
            "diversity": diversity,
            "frequency": frequency,
            "recency_weight": recency_weight,
            "base_score": score,
            "notes": notes,
        }

    def _apply_diversity(
        self,
        scored: list[tuple[float, RetrievedMemory]],
        max_results: int,
    ) -> list[tuple[float, RetrievedMemory]]:
        """Apply MMR-style diversity selection."""
        if len(scored) <= max_results:
            return scored
        selected: list[tuple[float, RetrievedMemory]] = []
        candidates = list(scored)
        while len(selected) < max_results and candidates:
            if not selected:
                selected.append(candidates.pop(0))
            else:
                best_idx = 0
                best_mmr = float("-inf")
                for i, (score, mem) in enumerate(candidates):
                    max_sim = max(jaccard(mem.record.text, s[1].record.text) for s in selected)
                    mmr = score - self.config.diversity_threshold * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                selected.append(candidates.pop(best_idx))
        return selected

    def _apply_diversity_with_indices(
        self,
        scored: list[tuple[float, RetrievedMemory, int]],
        max_results: int,
    ) -> list[tuple[float, RetrievedMemory, int]]:
        """Apply the same diversity strategy while preserving original indices."""
        if len(scored) <= max_results:
            return scored
        selected: list[tuple[float, RetrievedMemory, int]] = []
        candidates = list(scored)
        while len(selected) < max_results and candidates:
            if not selected:
                selected.append(candidates.pop(0))
            else:
                best_idx = 0
                best_mmr = float("-inf")
                for i, (score, mem, idx) in enumerate(candidates):
                    _ = idx
                    max_sim = max(jaccard(mem.record.text, s[1].record.text) for s in selected)
                    mmr = score - self.config.diversity_threshold * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                selected.append(candidates.pop(best_idx))
        return selected
