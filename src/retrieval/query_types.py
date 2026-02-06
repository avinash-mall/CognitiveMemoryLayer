"""Query classification types for retrieval."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional


class QueryIntent(str, Enum):
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
    entities: List[str] = field(default_factory=list)
    key_phrases: List[str] = field(default_factory=list)
    time_reference: Optional[str] = None
    time_start: Optional[datetime] = None
    time_end: Optional[datetime] = None
    suggested_sources: List[str] = field(default_factory=list)
    suggested_top_k: int = 10
    metadata: Dict[str, Any] = field(default_factory=dict)
