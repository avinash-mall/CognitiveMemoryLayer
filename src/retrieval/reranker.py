"""Reranker for retrieved memories."""

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

from ..core.enums import MemoryType
from ..core.schemas import RetrievedMemory

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
    diversity_threshold: float = 0.8
    max_results: int = 20


class MemoryReranker:
    """Reranks retrieved memories by relevance, recency, confidence, and diversity."""

    def __init__(
        self,
        config: RerankerConfig | None = None,
        llm_client=None,
    ):
        self.config = config or RerankerConfig()
        self.llm_client = llm_client

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

        word_sets = {i: self._word_set(mem.record.text) for i, mem in enumerate(memories)}
        breakdowns = {
            i: self._score_components(mem, memories, word_sets, i, query=query)
            for i, mem in enumerate(memories)
        }
        base_scores = {i: breakdowns[i]["base_score"] for i in breakdowns}

        # ponytail: constraint boost from vector-search cosine relevance; a pairwise
        # scorer (cross-encoder or LLM) could replace it if ranking quality demands.
        for idx, mem in enumerate(memories):
            if mem.record.type == MemoryType.CONSTRAINT:
                boost = max(0.0, min(1.0, mem.relevance_score)) * 2.0
                breakdowns[idx]["constraint_boost"] = boost
                base_scores[idx] += boost

        scored = [(score, memories[idx], idx) for idx, score in base_scores.items()]

        scored.sort(key=lambda x: x[0], reverse=True)
        diverse = self._apply_diversity_with_indices(scored, max_results)

        ranked_memories: list[RetrievedMemory] = []
        ranked_breakdowns: list[dict[str, Any]] = []
        for rank, (score, mem, idx) in enumerate(diverse, start=1):
            ranked_memories.append(mem)
            ranked_breakdowns.append(
                {
                    "rank": rank,
                    "memory_id": mem.record.id,
                    "text": mem.record.text,
                    "type": mem.record.type.value,
                    "retrieval_source": mem.retrieval_source,
                    "final_score": score,
                    "breakdown": {
                        "relevance": breakdowns[idx]["relevance"],
                        "recency": breakdowns[idx]["recency"],
                        "confidence": breakdowns[idx]["confidence"],
                        "diversity": breakdowns[idx]["diversity"],
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
        relevance = memory.relevance_score
        ts = memory.record.timestamp
        if isinstance(ts, datetime):
            now = datetime.now(UTC)
            tz_ts = ts if ts.tzinfo else ts.replace(tzinfo=UTC)
            age_days = (now - tz_ts).days
        else:
            age_days = 0
        recency = 1.0 / (1.0 + age_days * 0.1)
        confidence = memory.record.confidence

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
                    total_sim += self._text_similarity(my_ws, word_sets.get(j, frozenset()))
                else:
                    total_sim += self._text_similarity(memory.record.text, other.record.text)
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
        )
        if recency_weight > 0:
            score += recency_weight * recency
        return {
            "relevance": relevance,
            "recency": recency,
            "confidence": confidence,
            "diversity": diversity,
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
                    max_sim = max(
                        self._text_similarity(mem.record.text, s[1].record.text) for s in selected
                    )
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
                    max_sim = max(
                        self._text_similarity(mem.record.text, s[1].record.text) for s in selected
                    )
                    mmr = score - self.config.diversity_threshold * max_sim
                    if mmr > best_mmr:
                        best_mmr = mmr
                        best_idx = i
                selected.append(candidates.pop(best_idx))
        return selected

    @staticmethod
    def _word_set(text: str) -> frozenset[str]:
        """Return a cached-friendly word set for Jaccard computation."""
        return frozenset(text.lower().split())

    def _text_similarity(self, text1: str | frozenset[str], text2: str | frozenset[str]) -> float:
        """Jaccard word-overlap similarity (accepts pre-computed frozensets)."""
        words1 = text1 if isinstance(text1, frozenset) else self._word_set(text1)
        words2 = text2 if isinstance(text2, frozenset) else self._word_set(text2)
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
