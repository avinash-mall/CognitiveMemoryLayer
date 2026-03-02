"""Reranker for retrieved memories."""

from dataclasses import dataclass
from datetime import UTC, datetime

from ..core.enums import MemoryType
from ..core.schemas import RetrievedMemory
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime

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
        modelpack: ModelPackRuntime | None = None,
    ):
        self.config = config or RerankerConfig()
        self.llm_client = llm_client
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

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

        word_sets = {i: self._word_set(mem.record.text) for i, mem in enumerate(memories)}
        base_scores = {
            i: self._calculate_score(mem, memories, word_sets, i) for i, mem in enumerate(memories)
        }

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

            # Model-backed stability fallback when structured type is missing/weak.
            if self.modelpack.available:
                stability = self.modelpack.predict_single(
                    "constraint_stability", memory.record.text
                )
                if stability and stability.confidence >= 0.55:
                    if stability.label == "stable":
                        recency_weight = 0.0
                    elif stability.label == "semi_stable":
                        recency_weight = min(recency_weight, 0.12)
                    elif stability.label == "volatile":
                        recency_weight = max(recency_weight, 0.25)

        return recency_weight

    def _calculate_score(
        self,
        memory: RetrievedMemory,
        all_memories: list[RetrievedMemory],
        word_sets: dict[int, frozenset[str]] | None = None,
        mem_index: int = -1,
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

    async def _score_constraints_batch(
        self, query: str, constraint_texts: list[str]
    ) -> list[float]:
        """Score multiple constraints against a query using modelpack and/or LLM."""
        if not constraint_texts:
            return []

        model_scores = self._score_constraints_with_modelpack(query, constraint_texts)
        if model_scores is not None:
            return model_scores

        if not self.llm_client:
            return [0.0 for _ in constraint_texts]

        if not query.strip():
            return [0.0 for _ in constraint_texts]

        from ..core.config import get_settings

        feat = get_settings().features
        if not (feat.use_llm_enabled and feat.use_llm_constraint_reranker):
            return [0.0 for _ in constraint_texts]

        import asyncio

        batch_size = 10
        results = [0.0 for _ in constraint_texts]

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
            except Exception as exc:
                from ..utils.logging_config import get_logger

                get_logger(__name__).debug("llm_constraint_scoring_failed", error=str(exc))

        return results

    def _score_constraints_with_modelpack(
        self,
        query: str,
        constraint_texts: list[str],
    ) -> list[float] | None:
        if not query.strip():
            return None
        if not self.modelpack.available:
            return None

        out: list[float] = []
        used = False
        for text in constraint_texts:
            signals: list[float] = []

            rel_pred = self.modelpack.predict_pair("constraint_rerank", query, text)
            if rel_pred:
                used = True
                rel_signal = (
                    rel_pred.confidence
                    if rel_pred.label == "relevant"
                    else (1.0 - rel_pred.confidence)
                )
                signals.append(rel_signal)

            scope_pred = self.modelpack.predict_pair("scope_match", query, text)
            if scope_pred:
                used = True
                scope_signal = (
                    scope_pred.confidence
                    if scope_pred.label == "match"
                    else (1.0 - scope_pred.confidence)
                )
                signals.append(scope_signal)

            score = sum(signals) / len(signals) if signals else 0.0
            out.append(max(0.0, min(1.0, score)))

        return out if used else None
