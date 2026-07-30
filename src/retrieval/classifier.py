"""Query classifier for retrieval strategy selection."""

import re

from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from .query_types import QueryAnalysis, QueryIntent

logger = get_logger(__name__)

_ALL_CONSTRAINT_DIMENSIONS = ["goal", "state", "value", "causal", "policy"]
_DECISION_QUERY_PATTERNS = (
    r"\bshould i\b",
    r"\bcan i\b",
    r"\bam i allowed\b",
    r"\bis it (?:ok|okay)\b",
    r"\bwould it be (?:ok|okay)\b",
    r"\brecommend\b",
    r"\bwhat gift should\b",
    r"\bwhat packaging should\b",
)
_POLICY_QUERY_PATTERNS = (
    r"\brules?\b",
    r"\bpolicy\b",
    r"\bpolicies\b",
    r"\bethics?\b",
    r"\ballerg",
    r"\bpersonal rules?\b",
    r"\bplastics?\b",
    r"\bsingle-use\b",
    r"\bpackaging\b",
    r"\brestaurant\b",
)
_GOAL_QUERY_PATTERNS = (
    r"\bgoal\b",
    r"\bgoals\b",
    r"\btarget\b",
    r"\bworking toward\b",
    r"\bpublication\b",
    r"\bcareer\b",
    r"\bjob offer\b",
    r"\btenure\b",
)
_VALUE_QUERY_PATTERNS = (
    r"\bvalues?\b",
    r"\bimportant to me\b",
    r"\bpriorit",
    r"\bbelieve in\b",
)
_STATE_QUERY_PATTERNS = (
    r"\bright now\b",
    r"\bcurrently\b",
    r"\bdealing with\b",
    r"\bchallenges?\b",
    r"\bstress",
    r"\bworried\b",
)
_CAUSAL_QUERY_PATTERNS = (
    r"^\s*why\b",
    r"\breason\b",
    r"\bdue to\b",
    r"\bbecause\b",
)
_PREFERENCE_ENTITY_HINTS = (
    (r"\bhobb(?:y|ies)\b", "hobby"),
    (r"\bcuisine\b|\bfood\b|\beat\b", "cuisine"),
    (r"\blanguage\b", "language"),
    (r"\bbook\b", "book"),
)

CLASSIFICATION_PROMPT = """Classify this query for a memory retrieval system.

Query: {query}

Determine:
1. Intent (one of: preference_lookup, identity_lookup, task_status, episodic_recall,
   general_question, multi_hop, temporal_query, procedural, constraint_check, unknown)
2. Key entities mentioned
3. Time reference if any (recent, specific date, always, etc.)
4. Confidence (0.0-1.0)

Return JSON only:
{{"intent": "preference_lookup", "entities": ["cuisine"], "time_reference": null, "confidence": 0.9, "constraint_dimensions": ["dietary"], "suggested_top_k": 10}}

Rules:
- preference_lookup: asking about likes/dislikes/preferences
- identity_lookup: asking about personal info (name, email, etc.)
- task_status: asking about current progress on something
- episodic_recall: asking about past conversations/events
- general_question: broad questions about topics
- multi_hop: questions requiring connecting multiple pieces of info
- temporal_query: questions with specific time references
- procedural: how-to questions
- constraint_check: checking rules/policies

5. constraint_dimensions (optional): array of strings when query implies checking constraints, e.g. ["goal","value","dietary","state","causal","policy"]
6. suggested_top_k (optional): integer 5-20, how many memories to retrieve for this query"""


class QueryClassifier:
    """Classifies queries using the LLM plus regex heuristics."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client

    async def classify(
        self,
        query: str,
        recent_context: str | None = None,
    ) -> QueryAnalysis:
        """Classify a query and extract relevant information."""
        from ..core.config import get_settings

        settings = get_settings().features

        result: QueryAnalysis | None = None
        if self.llm and settings.use_llm_enabled:
            result = await self._llm_classify(query, recent_context=recent_context)

        if result is None:
            result = QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10,
            )

        # Constraint/decision heuristics first: "Can I eat X?" is a constraint
        # check even when a preference entity ("cuisine") is also mentioned.
        self._apply_constraint_heuristics(result)
        self._apply_preference_heuristics(result)
        result.is_decision_query = result.is_decision_query or (
            result.intent == QueryIntent.CONSTRAINT_CHECK
        )
        if result.constraint_dimensions and result.intent in {
            QueryIntent.GENERAL_QUESTION,
            QueryIntent.UNKNOWN,
            QueryIntent.EPISODIC_RECALL,
            QueryIntent.PROCEDURAL,
        }:
            result.intent = QueryIntent.CONSTRAINT_CHECK
            result.is_decision_query = True
            result.suggested_sources = self._get_sources_for_intent(QueryIntent.CONSTRAINT_CHECK)
            result.suggested_top_k = self._get_top_k_for_intent(QueryIntent.CONSTRAINT_CHECK)
        return result

    def _apply_constraint_heuristics(self, analysis: QueryAnalysis) -> None:
        query = (analysis.original_query or "").strip().lower()
        if not query:
            return

        heuristic_dims: list[str] = []
        decision_like = any(re.search(pattern, query) for pattern in _DECISION_QUERY_PATTERNS)

        if decision_like:
            heuristic_dims.extend(_ALL_CONSTRAINT_DIMENSIONS)
            analysis.is_decision_query = True
        if any(re.search(pattern, query) for pattern in _POLICY_QUERY_PATTERNS):
            heuristic_dims.append("policy")
        if any(re.search(pattern, query) for pattern in _GOAL_QUERY_PATTERNS):
            heuristic_dims.append("goal")
        if any(re.search(pattern, query) for pattern in _VALUE_QUERY_PATTERNS):
            heuristic_dims.append("value")
        if any(re.search(pattern, query) for pattern in _STATE_QUERY_PATTERNS):
            heuristic_dims.append("state")
        if any(re.search(pattern, query) for pattern in _CAUSAL_QUERY_PATTERNS):
            heuristic_dims.append("causal")

        if not heuristic_dims:
            return

        merged_dims = list(dict.fromkeys((analysis.constraint_dimensions or []) + heuristic_dims))
        analysis.constraint_dimensions = merged_dims

        if analysis.intent in {
            QueryIntent.GENERAL_QUESTION,
            QueryIntent.UNKNOWN,
            QueryIntent.EPISODIC_RECALL,
            QueryIntent.PROCEDURAL,
        }:
            analysis.intent = QueryIntent.CONSTRAINT_CHECK
            analysis.suggested_sources = self._get_sources_for_intent(QueryIntent.CONSTRAINT_CHECK)
            analysis.suggested_top_k = self._get_top_k_for_intent(QueryIntent.CONSTRAINT_CHECK)

    def _apply_preference_heuristics(self, analysis: QueryAnalysis) -> None:
        query = (analysis.original_query or "").strip().lower()
        if not query:
            return

        inferred_entity = next(
            (entity for pattern, entity in _PREFERENCE_ENTITY_HINTS if re.search(pattern, query)),
            None,
        )
        if inferred_entity is None:
            return

        entities = analysis.entities or []
        if inferred_entity not in entities:
            analysis.entities = [inferred_entity, *entities]

        if analysis.intent in {QueryIntent.GENERAL_QUESTION, QueryIntent.UNKNOWN}:
            analysis.intent = QueryIntent.PREFERENCE_LOOKUP
            analysis.suggested_sources = self._get_sources_for_intent(QueryIntent.PREFERENCE_LOOKUP)
            analysis.suggested_top_k = self._get_top_k_for_intent(QueryIntent.PREFERENCE_LOOKUP)

    async def _llm_classify(
        self,
        query: str,
        recent_context: str | None = None,
    ) -> QueryAnalysis:
        """LLM-based classification for complex queries."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        if recent_context:
            prompt = f"Recent conversation context:\n{recent_context}\n\n{prompt}"
        try:
            if self.llm is None:
                return QueryAnalysis(
                    original_query=query,
                    intent=QueryIntent.UNKNOWN,
                    confidence=0.5,
                    suggested_sources=["vector", "facts"],
                    suggested_top_k=10,
                )
            data = await self.llm.complete_json(prompt, temperature=0.0)
            intent_str = data.get("intent", "unknown")
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.UNKNOWN

            constraint_dimensions: list[str] | None = None
            raw_cd = data.get("constraint_dimensions")
            if isinstance(raw_cd, list) and all(isinstance(x, str) for x in raw_cd):
                constraint_dimensions = [s for s in raw_cd if s.strip()]

            suggested_top_k: int | None = None
            raw_tk = data.get("suggested_top_k")
            if isinstance(raw_tk, int) and 5 <= raw_tk <= 20:
                suggested_top_k = raw_tk

            analysis = QueryAnalysis(
                original_query=query,
                intent=intent,
                confidence=float(data.get("confidence", 0.7)),
                entities=data.get("entities", []),
                time_reference=data.get("time_reference"),
                suggested_sources=self._get_sources_for_intent(intent),
                suggested_top_k=(
                    suggested_top_k
                    if suggested_top_k is not None
                    else self._get_top_k_for_intent(intent)
                ),
            )
            if constraint_dimensions is not None:
                analysis.constraint_dimensions = constraint_dimensions
                analysis.constraint_dimensions_from_llm = True
            return analysis
        except Exception as e:
            logger.warning("llm_classify_failed", extra={"error": str(e)})
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10,
            )

    def _get_sources_for_intent(self, intent: QueryIntent) -> list[str]:
        """Map intent to retrieval sources."""
        mapping = {
            QueryIntent.PREFERENCE_LOOKUP: ["facts"],
            QueryIntent.IDENTITY_LOOKUP: ["facts"],
            QueryIntent.TASK_STATUS: ["facts", "vector"],
            QueryIntent.EPISODIC_RECALL: ["vector"],
            QueryIntent.GENERAL_QUESTION: ["vector", "facts"],
            QueryIntent.MULTI_HOP: ["graph", "vector"],
            QueryIntent.TEMPORAL_QUERY: ["vector"],
            QueryIntent.PROCEDURAL: ["facts", "vector"],
            QueryIntent.CONSTRAINT_CHECK: ["constraints", "facts", "vector"],
            QueryIntent.UNKNOWN: ["vector", "facts", "graph"],
        }
        return mapping.get(intent, ["vector"])

    def _get_top_k_for_intent(self, intent: QueryIntent) -> int:
        """Map intent to suggested top_k."""
        mapping = {
            QueryIntent.PREFERENCE_LOOKUP: 3,
            QueryIntent.IDENTITY_LOOKUP: 1,
            QueryIntent.TASK_STATUS: 5,
            QueryIntent.EPISODIC_RECALL: 10,
            QueryIntent.GENERAL_QUESTION: 10,
            QueryIntent.MULTI_HOP: 15,
            QueryIntent.TEMPORAL_QUERY: 15,
            QueryIntent.PROCEDURAL: 5,
            QueryIntent.CONSTRAINT_CHECK: 5,
        }
        return mapping.get(intent, 10)
