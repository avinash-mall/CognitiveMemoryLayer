"""Retrieval planning from query analysis."""

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from enum import StrEnum
from typing import Any

from .query_types import QueryAnalysis, QueryIntent


class RetrievalSource(StrEnum):
    """Source for a retrieval step."""

    FACTS = "facts"
    VECTOR = "vector"
    GRAPH = "graph"
    CACHE = "cache"
    CONSTRAINTS = "constraints"


@dataclass
class RetrievalStep:
    """Single step in a retrieval plan."""

    source: RetrievalSource
    priority: int = 0
    key: str | None = None
    query: str | None = None
    seeds: list[str] = field(default_factory=list)
    memory_types: list[str] = field(default_factory=list)
    time_filter: dict[str, Any] | None = None
    min_confidence: float = 0.0
    top_k: int = 10
    timeout_ms: int = 100
    skip_if_found: bool = False
    constraint_categories: list[str] | None = None


@dataclass
class RetrievalPlan:
    """Complete retrieval plan."""

    query: str
    analysis: QueryAnalysis
    steps: list[RetrievalStep]
    total_timeout_ms: int = 500
    max_results: int = 20
    parallel_steps: list[list[int]] = field(default_factory=list)


class RetrievalPlanner:
    """Generates retrieval plans based on query analysis."""

    def plan(self, analysis: QueryAnalysis) -> RetrievalPlan:
        """Generate a retrieval plan for the analyzed query."""
        steps: list[RetrievalStep] = []
        parallel_groups: list[list[int]] = []

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

        elif (
            analysis.intent == QueryIntent.CONSTRAINT_CHECK
            or analysis.is_decision_query
            or analysis.constraint_dimensions
        ):
            # Constraints-first retrieval: prioritise active constraints
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.CONSTRAINTS,
                    query=analysis.original_query,
                    memory_types=["constraint"],
                    min_confidence=0.0,
                    top_k=10,
                    priority=0,
                    timeout_ms=200,
                    constraint_categories=analysis.constraint_dimensions or None,
                )
            )
            steps.append(
                RetrievalStep(
                    source=RetrievalSource.VECTOR,
                    query=analysis.original_query,
                    top_k=analysis.suggested_top_k,
                    priority=1,
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
            parallel_groups = [[0, 1, 2]]

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

        self._apply_retrieval_timeouts(steps)

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

    def _build_time_filter(self, analysis: QueryAnalysis) -> dict[str, Any] | None:
        """Build time filter from analysis.

        When the analysis carries a ``user_timezone``, "today" and
        "yesterday" are computed in the user's local time, then
        converted to UTC for the database query.
        """
        if not analysis.time_reference:
            return None

        # Resolve user-local "now"
        tz = UTC
        if analysis.user_timezone:
            try:
                from zoneinfo import ZoneInfo

                tz = ZoneInfo(analysis.user_timezone)
            except Exception:
                tz = UTC
        user_now = datetime.now(tz)

        ref = (analysis.time_reference or "").lower()
        if "today" in ref:
            start = user_now.replace(hour=0, minute=0, second=0, microsecond=0)
            return {"since": start.astimezone(UTC).replace(tzinfo=None)}
        if "yesterday" in ref:
            yesterday = user_now - timedelta(days=1)
            start = yesterday.replace(hour=0, minute=0, second=0, microsecond=0)
            end = yesterday.replace(hour=23, minute=59, second=59, microsecond=999999)
            return {
                "since": start.astimezone(UTC).replace(tzinfo=None),
                "until": end.astimezone(UTC).replace(tzinfo=None),
            }
        if "week" in ref:
            return {"since": (user_now - timedelta(days=7)).astimezone(UTC).replace(tzinfo=None)}
        if "month" in ref:
            return {"since": (user_now - timedelta(days=30)).astimezone(UTC).replace(tzinfo=None)}
        if "recent" in ref:
            return {"since": (user_now - timedelta(days=3)).astimezone(UTC).replace(tzinfo=None)}
        return None

    def _apply_retrieval_timeouts(self, steps: list[RetrievalStep]) -> None:
        """Apply config-based timeouts so vector/embedding steps don't use hardcoded 100ms."""
        try:
            from ..core.config import get_settings

            settings = get_settings()
            if not getattr(settings.features, "retrieval_timeouts_enabled", True):
                return
            r = settings.retrieval
            for step in steps:
                if step.source == RetrievalSource.VECTOR:
                    step.timeout_ms = r.default_step_timeout_ms
                elif step.source == RetrievalSource.GRAPH:
                    step.timeout_ms = r.graph_timeout_ms
                elif step.source == RetrievalSource.FACTS and step.timeout_ms == 100:
                    step.timeout_ms = r.fact_timeout_ms
                elif step.source == RetrievalSource.CONSTRAINTS and step.timeout_ms == 100:
                    step.timeout_ms = r.default_step_timeout_ms
        except Exception:
            pass

    def _calculate_timeout(self, steps: list[RetrievalStep]) -> int:
        """Calculate total timeout based on steps."""
        try:
            from ..core.config import get_settings

            return get_settings().retrieval.total_timeout_ms
        except Exception:
            return sum(s.timeout_ms for s in steps) // 2 + 100
