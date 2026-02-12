"""Retrieval planning from query analysis."""

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

from .query_types import QueryAnalysis, QueryIntent


class RetrievalSource(str, Enum):
    """Source for a retrieval step."""

    FACTS = "facts"
    VECTOR = "vector"
    GRAPH = "graph"
    CACHE = "cache"


@dataclass
class RetrievalStep:
    """Single step in a retrieval plan."""

    source: RetrievalSource
    priority: int = 0
    key: Optional[str] = None
    query: Optional[str] = None
    seeds: List[str] = field(default_factory=list)
    memory_types: List[str] = field(default_factory=list)
    time_filter: Optional[Dict[str, Any]] = None
    min_confidence: float = 0.0
    top_k: int = 10
    timeout_ms: int = 100
    skip_if_found: bool = False


@dataclass
class RetrievalPlan:
    """Complete retrieval plan."""

    query: str
    analysis: QueryAnalysis
    steps: List[RetrievalStep]
    total_timeout_ms: int = 500
    max_results: int = 20
    parallel_steps: List[List[int]] = field(default_factory=list)


class RetrievalPlanner:
    """Generates retrieval plans based on query analysis."""

    def plan(self, analysis: QueryAnalysis) -> RetrievalPlan:
        """Generate a retrieval plan for the analyzed query."""
        steps: List[RetrievalStep] = []
        parallel_groups: List[List[int]] = []

        if analysis.intent in (
            QueryIntent.PREFERENCE_LOOKUP,
            QueryIntent.IDENTITY_LOOKUP,
            QueryIntent.TASK_STATUS,
        ):
            steps.append(self._create_fact_lookup_step(analysis))
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.VECTOR,
                    query=analysis.original_query,
                    top_k=5,
                    priority=1,
                    skip_if_found=True,
                )
            )
            parallel_groups = [[0], [1]]

        elif analysis.intent == QueryIntent.MULTI_HOP:
            if analysis.entities:
                steps.append(
                    RetrievalStep(
                        source=RetrievalSource.GRAPH,
                        seeds=analysis.entities,
                        top_k=15,
                        priority=2,
                    )
                )
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.VECTOR,
                    query=analysis.original_query,
                    top_k=10,
                    priority=1,
                )
            )
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.FACTS,
                    query=analysis.original_query,
                    top_k=5,
                    priority=0,
                )
            )
            parallel_groups = [[0, 1, 2]] if len(steps) == 3 else [[0, 1]]

        elif analysis.intent == QueryIntent.TEMPORAL_QUERY:
            time_filter = self._build_time_filter(analysis)
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.VECTOR,
                    query=analysis.original_query,
                    time_filter=time_filter,
                    top_k=15,
                    priority=2,
                )
            )
            parallel_groups = [[0]]

        else:
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.VECTOR,
                    query=analysis.original_query,
                    top_k=analysis.suggested_top_k,
                    priority=2,
                )
            )
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.FACTS,
                    query=analysis.original_query,
                    top_k=5,
                    priority=1,
                )
            )
            if analysis.entities:
                steps.append(
                    RetrievalStep(
                        source=RetrievalSource.GRAPH,
                        seeds=analysis.entities,
                        top_k=10,
                        priority=1,
                    )
                )
            parallel_groups = [[0, 1, 2]] if len(steps) == 3 else [[0, 1]]

        return RetrievalPlan(
            query=analysis.original_query,
            analysis=analysis,
            steps=steps,
            parallel_steps=parallel_groups,
            total_timeout_ms=self._calculate_timeout(steps),
            max_results=analysis.suggested_top_k,
        )

    def _create_fact_lookup_step(self, analysis: QueryAnalysis) -> RetrievalStep:
        """Create a fast fact lookup step."""
        key_prefix = {
            QueryIntent.PREFERENCE_LOOKUP: "user:preference:",
            QueryIntent.IDENTITY_LOOKUP: "user:identity:",
            QueryIntent.TASK_STATUS: "user:task:",
        }.get(analysis.intent, "user:")
        key = None
        if analysis.entities:
            key = f"{key_prefix}{analysis.entities[0].lower()}"
        return RetrievalStep(
            source=RetrievalSource.FACTS,
            key=key,
            query=analysis.original_query if not key else None,
            top_k=3,
            priority=3,
            timeout_ms=50,
        )

    def _build_time_filter(self, analysis: QueryAnalysis) -> Optional[Dict[str, Any]]:
        """Build time filter from analysis."""
        if not analysis.time_reference:
            return None
        now = datetime.now(timezone.utc)
        ref = (analysis.time_reference or "").lower()
        if "today" in ref:
            return {"since": now.replace(hour=0, minute=0, second=0, microsecond=0)}
        if "yesterday" in ref:
            yesterday = now - timedelta(days=1)
            return {
                "since": yesterday.replace(hour=0, minute=0, second=0, microsecond=0),
                "until": yesterday.replace(hour=23, minute=59, second=59, microsecond=999999),
            }
        if "week" in ref:
            return {"since": now - timedelta(days=7)}
        if "month" in ref:
            return {"since": now - timedelta(days=30)}
        if "recent" in ref:
            return {"since": now - timedelta(days=3)}
        return None

    def _calculate_timeout(self, steps: List[RetrievalStep]) -> int:
        """Calculate total timeout based on steps."""
        return sum(s.timeout_ms for s in steps) // 2 + 100
