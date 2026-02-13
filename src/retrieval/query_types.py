"""Query classification types for retrieval."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from typing import Any


class QueryIntent(StrEnum):
    """Classified intent of a user query."""

    PREFERENCE_LOOKUP = "preference_lookup"
    IDENTITY_LOOKUP = "identity_lookup"
    TASK_STATUS = "task_status"
    EPISODIC_RECALL = "episodic_recall"
    GENERAL_QUESTION = "general_question"
    MULTI_HOP = "multi_hop"
    TEMPORAL_QUERY = "temporal_query"
    PROCEDURAL = "procedural"
    CONSTRAINT_CHECK = "constraint_check"
    UNKNOWN = "unknown"


@dataclass
class QueryAnalysis:
    """Analysis result of a query."""

    original_query: str
    intent: QueryIntent
    confidence: float
    entities: list[str] = field(default_factory=list)
    key_phrases: list[str] = field(default_factory=list)
    time_reference: str | None = None
    time_start: datetime | None = None
    time_end: datetime | None = None
    suggested_sources: list[str] = field(default_factory=list)
    suggested_top_k: int = 10
    metadata: dict[str, Any] = field(default_factory=dict)
