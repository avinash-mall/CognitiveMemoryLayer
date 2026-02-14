"""Reranker for retrieved memories."""

from dataclasses import dataclass
from datetime import UTC, datetime

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
    recency_weight: float = 0.2
    confidence_weight: float = 0.2
    diversity_weight: float = 0.1
    diversity_threshold: float = 0.8
    max_results: int = 20


class MemoryReranker:
    """Reranks retrieved memories by relevance, recency, confidence, and diversity."""

    def __init__(self, config: RerankerConfig | None = None):
        self.config = config or RerankerConfig()

    def rerank(
        self,
        memories: list[RetrievedMemory],
        query: str,
        max_results: int | None = None,
    ) -> list[RetrievedMemory]:
        """Rerank memories by combined score and diversity."""
        if not memories:
            return []
        max_results = max_results or self.config.max_results
        scored: list[tuple[float, RetrievedMemory]] = []
        for mem in memories:
            score = self._calculate_score(mem, memories)
            scored.append((score, mem))
        scored.sort(key=lambda x: x[0], reverse=True)
        diverse = self._apply_diversity(scored, max_results)
        return [mem for _, mem in diverse]

    def _get_recency_weight(self, memory: RetrievedMemory) -> float:
        """Determine recency weight based on memory type stability.

        Stable constraints (value/policy) should not be penalised for age.
        """
        if memory.record.type in _STABLE_TYPES:
            # Check constraint sub-type from metadata
            meta = memory.record.metadata or {}
            constraints_meta = meta.get("constraints", [])
            if constraints_meta and isinstance(constraints_meta, list):
                ctype = constraints_meta[0].get("constraint_type", "")
                if ctype in _STABLE_CONSTRAINT_TYPES:
                    return 0.05
                if ctype in _SEMI_STABLE_CONSTRAINT_TYPES:
                    return 0.15
            return 0.10  # Generic constraint: moderate stability
        return self.config.recency_weight

    def _calculate_score(
        self,
        memory: RetrievedMemory,
        all_memories: list[RetrievedMemory],
    ) -> float:
        """Calculate combined score for a memory."""
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

        # Compute actual diversity: average dissimilarity to other memories (LOW-13)
        # Cap pairwise comparisons to avoid O(n^2) on large result sets
        diversity_cap = min(len(all_memories), 50)
        if len(all_memories) > 1:
            total_sim = 0.0
            count = 0
            for other in all_memories[:diversity_cap]:
                if other is memory:
                    continue
                total_sim += self._text_similarity(memory.record.text, other.record.text)
                count += 1
            avg_sim = total_sim / count if count > 0 else 0.0
            diversity = 1.0 - avg_sim  # Higher diversity = less similar to others
        else:
            diversity = 1.0

        recency_weight = self._get_recency_weight(memory)
        score = (
            self.config.relevance_weight * relevance
            + recency_weight * recency
            + self.config.confidence_weight * confidence
            + self.config.diversity_weight * diversity
        )
        return score

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

    def _text_similarity(self, text1: str, text2: str) -> float:
        """Simple word overlap similarity."""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0
