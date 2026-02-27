"""Query classifier for retrieval strategy selection."""

import json

from ..utils.llm import LLMClient
from ..utils.logging_config import get_logger
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime
from ..utils.ner import extract_entity_texts
from .query_types import QueryAnalysis, QueryIntent

logger = get_logger(__name__)

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
    """Classifies queries using modelpack and/or LLM."""

    _MODELPACK_INTENT_HINTS: dict[str, QueryIntent] = {
        "constraint_check": QueryIntent.CONSTRAINT_CHECK,
        "preference_lookup": QueryIntent.PREFERENCE_LOOKUP,
        "identity_lookup": QueryIntent.IDENTITY_LOOKUP,
        "task_status": QueryIntent.TASK_STATUS,
        "episodic_recall": QueryIntent.EPISODIC_RECALL,
        "general_question": QueryIntent.GENERAL_QUESTION,
        "multi_hop": QueryIntent.MULTI_HOP,
        "temporal_query": QueryIntent.TEMPORAL_QUERY,
        "procedural": QueryIntent.PROCEDURAL,
        "unknown": QueryIntent.UNKNOWN,
        "planning": QueryIntent.PROCEDURAL,
        "conversation": QueryIntent.EPISODIC_RECALL,
        "factual": QueryIntent.GENERAL_QUESTION,
        "tool_query": QueryIntent.GENERAL_QUESTION,
    }

    def __init__(
        self,
        llm_client: LLMClient | None = None,
        modelpack: ModelPackRuntime | None = None,
    ):
        self.llm = llm_client
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()
        self._direct_intent_map = {intent.value: intent for intent in QueryIntent}

    async def classify(
        self,
        query: str,
        recent_context: str | None = None,
    ) -> QueryAnalysis:
        """Classify a query and extract relevant information."""
        from ..core.config import get_settings

        settings = get_settings().features
        llm_forced = settings.use_llm_enabled and settings.use_llm_query_classifier_only

        result: QueryAnalysis | None = None
        if not llm_forced:
            result = self._modelpack_classify(query)

        if result is None and self.llm and settings.use_llm_enabled:
            result = await self._llm_classify(query, recent_context=recent_context)

        if result is None:
            result = QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10,
            )

        if not result.entities:
            result.entities = self._extract_entities(query)

        self._enrich_with_modelpack(result)
        result.is_decision_query = result.intent == QueryIntent.CONSTRAINT_CHECK
        if result.constraint_dimensions and result.intent in {
            QueryIntent.GENERAL_QUESTION,
            QueryIntent.UNKNOWN,
        }:
            result.intent = QueryIntent.CONSTRAINT_CHECK
            result.is_decision_query = True
            result.suggested_sources = self._get_sources_for_intent(QueryIntent.CONSTRAINT_CHECK)
            result.suggested_top_k = self._get_top_k_for_intent(QueryIntent.CONSTRAINT_CHECK)
        return result

    def _modelpack_classify(self, query: str) -> QueryAnalysis | None:
        if not self.modelpack.available:
            return None
        if not query.strip():
            return None

        intent_pred = self.modelpack.predict_single("query_intent", query)
        if intent_pred is None or not intent_pred.label:
            return None
        intent = self._intent_from_label(intent_pred.label)
        if intent is None:
            return None

        return QueryAnalysis(
            original_query=query,
            intent=intent,
            confidence=max(0.0, min(1.0, intent_pred.confidence)),
            entities=self._extract_entities(query),
            suggested_sources=self._get_sources_for_intent(intent),
            suggested_top_k=self._get_top_k_for_intent(intent),
        )

    def _enrich_with_modelpack(self, analysis: QueryAnalysis) -> None:
        """Augment query analysis with router model signals when available."""
        if not self.modelpack.available:
            return

        query = analysis.original_query or ""
        if not query.strip():
            return

        domain_pred = self.modelpack.predict_single("query_domain", query)
        if domain_pred and domain_pred.label:
            analysis.query_domain = domain_pred.label
            analysis.metadata["query_domain_confidence"] = domain_pred.confidence

        if not analysis.constraint_dimensions_from_llm:
            dim_pred = self.modelpack.predict_single("constraint_dimension", query)
            if dim_pred and dim_pred.label and dim_pred.label != "other":
                dims = analysis.constraint_dimensions
                if dims is None:
                    analysis.constraint_dimensions = [dim_pred.label]
                elif dim_pred.label not in dims:
                    dims.append(dim_pred.label)

        intent_pred = self.modelpack.predict_single("query_intent", query)
        if intent_pred and intent_pred.label:
            hint_intent = self._intent_from_label(intent_pred.label)
            if (
                hint_intent is not None
                and intent_pred.confidence >= 0.55
                and analysis.intent in {QueryIntent.GENERAL_QUESTION, QueryIntent.UNKNOWN}
            ):
                analysis.intent = hint_intent
                analysis.suggested_sources = self._get_sources_for_intent(hint_intent)
                analysis.suggested_top_k = self._get_top_k_for_intent(hint_intent)

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
        except (json.JSONDecodeError, ValueError, TypeError, Exception) as e:
            logger.warning("llm_classify_failed", extra={"error": str(e)})
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10,
            )

    def _extract_entities(self, query: str) -> list[str]:
        return extract_entity_texts(query, max_entities=12)

    def _intent_from_label(self, raw_label: str) -> QueryIntent | None:
        label = raw_label.strip().lower()
        if label in self._direct_intent_map:
            return self._direct_intent_map[label]
        return self._MODELPACK_INTENT_HINTS.get(label)

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
