"""Schema alignment for consolidated gists."""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

from .summarizer import ExtractedGist
from ..memory.neocortical.fact_store import SemanticFactStore
from ..memory.neocortical.schemas import FactCategory


@dataclass
class AlignmentResult:
    """Result of schema alignment."""

    gist: ExtractedGist

    matched_schema: Optional[str] = None
    schema_similarity: float = 0.0

    can_integrate_rapidly: bool = False
    integration_key: Optional[str] = None

    suggested_schema: Optional[Dict[str, Any]] = None


class SchemaAligner:
    """Aligns extracted gists with existing semantic schemas."""

    def __init__(
        self,
        fact_store: SemanticFactStore,
        rapid_integration_threshold: float = 0.7,
    ):
        self.fact_store = fact_store
        self.threshold = rapid_integration_threshold

    async def align(
        self,
        tenant_id: str,
        user_id: str,
        gist: ExtractedGist,
    ) -> AlignmentResult:
        """Align a gist with existing schemas."""
        if gist.key:
            existing = await self.fact_store.get_fact(tenant_id, gist.key)
            if existing:
                return AlignmentResult(
                    gist=gist,
                    matched_schema=gist.key,
                    schema_similarity=0.9,
                    can_integrate_rapidly=True,
                    integration_key=gist.key,
                )

        if gist.gist_type == "preference" and gist.predicate:
            key = f"user:preference:{gist.predicate}"
            existing = await self.fact_store.get_fact(tenant_id, key)
            if existing:
                return AlignmentResult(
                    gist=gist,
                    matched_schema=key,
                    schema_similarity=0.8,
                    can_integrate_rapidly=True,
                    integration_key=key,
                )

        similar_facts = await self.fact_store.search_facts(
            tenant_id, gist.text, limit=5
        )
        if similar_facts:
            best_match = similar_facts[0]
            similarity = self._calculate_similarity(gist.text, best_match.value)
            if similarity >= self.threshold:
                return AlignmentResult(
                    gist=gist,
                    matched_schema=best_match.key,
                    schema_similarity=similarity,
                    can_integrate_rapidly=True,
                    integration_key=best_match.key,
                )

        suggested = self._suggest_schema(gist)
        return AlignmentResult(
            gist=gist,
            schema_similarity=0.0,
            can_integrate_rapidly=False,
            suggested_schema=suggested,
        )

    async def align_batch(
        self,
        tenant_id: str,
        user_id: str,
        gists: List[ExtractedGist],
    ) -> List[AlignmentResult]:
        """Align multiple gists."""
        import asyncio

        return await asyncio.gather(*[self.align(tenant_id, user_id, g) for g in gists])

    def _calculate_similarity(self, text1: str, text2: Any) -> float:
        text2_str = str(text2)
        words1 = set(text1.lower().split())
        words2 = set(text2_str.lower().split())
        if not words1 or not words2:
            return 0.0
        intersection = len(words1 & words2)
        union = len(words1 | words2)
        return intersection / union if union > 0 else 0.0

    def _suggest_schema(self, gist: ExtractedGist) -> Dict[str, Any]:
        if gist.gist_type == "preference":
            category = FactCategory.PREFERENCE
        elif gist.gist_type == "fact":
            category = FactCategory.ATTRIBUTE
        else:
            category = FactCategory.CUSTOM
        key = gist.key or f"user:{category.value}:{gist.predicate or 'unknown'}"
        return {
            "category": category.value,
            "key": key,
            "value_type": type(gist.value).__name__ if gist.value else "string",
            "source": "consolidation",
        }
