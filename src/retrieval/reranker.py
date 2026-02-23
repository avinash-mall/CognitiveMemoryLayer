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
    recency_weight: float = 0.1  # BUG-04: reduced from 0.2 to limit recency bias
    confidence_weight: float = 0.2
    diversity_weight: float = 0.1
    diversity_threshold: float = 0.8
    max_results: int = 20


class MemoryReranker:
    """Reranks retrieved memories by relevance, recency, confidence, and diversity."""

    def __init__(self, config: RerankerConfig | None = None, llm_client=None):
        self.config = config or RerankerConfig()
        self.llm_client = llm_client

    async def rerank(
        self,
        memories: list[RetrievedMemory],
        query: str,
        max_results: int | None = None,
    ) -> list[RetrievedMemory]:
        """Rerank memories by combined score and diversity."""
        if not memories:
            return []
        max_results = max_results or self.config.max_results

        base_scores = {i: self._calculate_score(mem, memories) for i, mem in enumerate(memories)}

        constraints = [
            (i, m) for i, m in enumerate(memories) if m.record.type == MemoryType.CONSTRAINT
        ]
        if constraints:
            constraint_texts = [m.record.text for _, m in constraints]
            boosts = await self._score_constraints_batch(query, constraint_texts)
            for (idx, mem), boost in zip(constraints, boosts, strict=True):
                base_scores[idx] += boost * 2.0

        scored = [(score, memories[idx]) for idx, score in base_scores.items()]

        scored.sort(key=lambda x: x[0], reverse=True)
        diverse = self._apply_diversity(scored, max_results)
        return [mem for _, mem in diverse]

    def _get_recency_weight(self, memory: RetrievedMemory) -> float:
        """Determine recency weight based on memory type stability.

        Stable constraints (value/policy) and preference/value facts (BUG-04)
        should not be heavily penalised for age.
        """
        key = getattr(memory.record, "key", None) or ""
        if key.startswith(("user:preference:", "user:value:")):
            return 0.1  # Semi-stable: preferences/values age slowly
        if memory.record.type in _STABLE_TYPES:
            # Check constraint sub-type from metadata
            meta = memory.record.metadata or {}
            constraints_meta = meta.get("constraints", [])
            if constraints_meta and isinstance(constraints_meta, list):
                ctype = constraints_meta[0].get("constraint_type", "")
                if ctype in _STABLE_CONSTRAINT_TYPES:
                    return 0.0  # Stable: age does not affect score
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

        # Compute actual diversity: average dissimilarity to other memories (LOW-13, BUG-07)
        # Cap pairwise comparisons to avoid O(n^2); skip for small N
        diversity_cap = min(len(all_memories), 20)
        if len(all_memories) <= 5:
            diversity = 1.0
        elif len(all_memories) > 1:
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
            + self.config.confidence_weight * confidence
            + self.config.diversity_weight * diversity
        )
        if recency_weight > 0:
            score += recency_weight * recency
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

    async def _score_constraints_batch(
        self, query: str, constraint_texts: list[str]
    ) -> list[float]:
        """Score multiple constraints against a query using a single LLM batch call.

        The LLM call is skipped (falling back to text similarity) when:
        - The query is empty (context-only reads with query='')
        - No llm_client is configured
        - FEATURES__USE_LLM_CONSTRAINT_RERANKER is False (default)
        """
        if not self.llm_client or not constraint_texts:
            return [self._text_similarity(query, text) for text in constraint_texts]

        # Skip expensive LLM scoring when query is empty (e.g. context reads)
        if not query.strip():
            return [self._text_similarity(query, text) for text in constraint_texts]

        # Gate behind feature flags (use_llm_enabled and use_llm_constraint_reranker)
        from ..core.config import get_settings

        feat = get_settings().features
        if not (feat.use_llm_enabled and feat.use_llm_constraint_reranker):
            return [self._text_similarity(query, text) for text in constraint_texts]

        import asyncio

        batch_size = 10
        results = [self._text_similarity(query, text) for text in constraint_texts]

        for i in range(0, len(constraint_texts), batch_size):
            batch = constraint_texts[i : i + batch_size]
            prompt = f"""Evaluate the relevance of multiple constraints to the given query.
Score each constraint from 0.0 to 1.0 based on how logically it applies to fulfilling the query.
Return the results in a JSON array of objects, where each object has "index" (the integer ID of the constraint) and "score" (a float between 0.0 and 1.0).

Query: "{query}"

Constraints:
"""
            for idx, text in enumerate(batch):
                prompt += f'[{idx}] "{text}"\n'

            try:
                resp = await asyncio.wait_for(
                    self.llm_client.complete_json(prompt, temperature=0.0), timeout=5.0
                )
                items = (
                    resp
                    if isinstance(resp, list)
                    else resp.get("results", [])
                    if isinstance(resp, dict)
                    else []
                )
                for item in items:
                    if isinstance(item, dict):
                        idx_raw = item.get("index")
                        score = item.get("score")
                        if (
                            isinstance(idx_raw, int)
                            and 0 <= idx_raw < len(batch)
                            and isinstance(score, (int, float))
                        ):
                            results[i + idx_raw] = min(1.0, max(0.0, float(score)))
            except Exception:
                pass

        return results
