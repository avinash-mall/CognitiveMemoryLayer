"""Query classifier for retrieval strategy selection."""

import json
import re
from typing import List, Optional

from ..utils.llm import LLMClient
from .query_types import QueryAnalysis, QueryIntent

FAST_PATTERNS = {
    QueryIntent.PREFERENCE_LOOKUP: [
        r"what (do|does) (i|my) (like|prefer|want|enjoy)",
        r"(my|i) (favorite|preferred)",
        r"do i (like|prefer|enjoy)",
    ],
    QueryIntent.IDENTITY_LOOKUP: [
        r"what('s| is) my (name|email|phone|address|job|title)",
        r"who am i",
        r"my (name|email|phone)",
    ],
    QueryIntent.TASK_STATUS: [
        r"where (am i|are we) (in|on|at)",
        r"what('s| is) (the|my) (status|progress)",
        r"(current|next) (step|task)",
    ],
    QueryIntent.TEMPORAL_QUERY: [
        r"(last|past) (week|month|day|year)",
        r"(yesterday|today|recently)",
        r"when did (i|we)",
        r"what happened",
    ],
    QueryIntent.PROCEDURAL: [
        r"how (do|can|should) (i|we)",
        r"what('s| are) the steps",
        r"(procedure|process) for",
    ],
}

CLASSIFICATION_PROMPT = """Classify this query for a memory retrieval system.

Query: {query}

Determine:
1. Intent (one of: preference_lookup, identity_lookup, task_status, episodic_recall,
   general_question, multi_hop, temporal_query, procedural, constraint_check, unknown)
2. Key entities mentioned
3. Time reference if any (recent, specific date, always, etc.)
4. Confidence (0.0-1.0)

Return JSON only:
{{"intent": "preference_lookup", "entities": ["cuisine"], "time_reference": null, "confidence": 0.9}}

Rules:
- preference_lookup: asking about likes/dislikes/preferences
- identity_lookup: asking about personal info (name, email, etc.)
- task_status: asking about current progress on something
- episodic_recall: asking about past conversations/events
- general_question: broad questions about topics
- multi_hop: questions requiring connecting multiple pieces of info
- temporal_query: questions with specific time references
- procedural: how-to questions
- constraint_check: checking rules/policies"""


class QueryClassifier:
    """Classifies queries to determine retrieval strategy. Fast patterns first, then LLM."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in FAST_PATTERNS.items()
        }

    async def classify(
        self,
        query: str,
        recent_context: Optional[str] = None,
    ) -> QueryAnalysis:
        """Classify a query and extract relevant information. Uses recent_context when
        query is vague (e.g. 'Any suggestions?')."""
        # If query is vague and we have context, use it to infer intent
        effective_query = query
        if recent_context and self._is_vague(query):
            effective_query = f"{recent_context}\nUser now asks: {query}"
        fast_result = self._fast_classify(effective_query)
        if fast_result and fast_result.confidence > 0.8:
            return fast_result
        if self.llm:
            return await self._llm_classify(effective_query, recent_context=recent_context)
        return fast_result or QueryAnalysis(
            original_query=query,
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.5,
            suggested_sources=["vector", "facts"],
            suggested_top_k=10,
        )

    def _is_vague(self, query: str) -> bool:
        """Heuristic: query is too short or lacks clear intent."""
        q = query.strip().lower()
        if len(q) < 15:
            return True
        vague_starts = (
            "any ",
            "what about",
            "suggestions?",
            "thoughts?",
            "and?",
            "so?",
            "what do you think",
        )
        return any(q.startswith(p) or q == p for p in vague_starts)

    def _fast_classify(self, query: str) -> Optional[QueryAnalysis]:
        """Fast pattern-based classification."""
        query_lower = query.lower()
        for intent, patterns in self._compiled_patterns.items():
            for pattern in patterns:
                if pattern.search(query_lower):
                    return QueryAnalysis(
                        original_query=query,
                        intent=intent,
                        confidence=0.85,
                        entities=self._extract_entities_simple(query),
                        suggested_sources=self._get_sources_for_intent(intent),
                        suggested_top_k=self._get_top_k_for_intent(intent),
                    )
        return None

    async def _llm_classify(
        self,
        query: str,
        recent_context: Optional[str] = None,
    ) -> QueryAnalysis:
        """LLM-based classification for complex queries."""
        prompt = CLASSIFICATION_PROMPT.format(query=query)
        if recent_context:
            prompt = f"Recent conversation context:\n{recent_context}\n\n{prompt}"
        try:
            data = await self.llm.complete_json(prompt, temperature=0.0)
            intent_str = data.get("intent", "unknown")
            try:
                intent = QueryIntent(intent_str)
            except ValueError:
                intent = QueryIntent.UNKNOWN
            return QueryAnalysis(
                original_query=query,
                intent=intent,
                confidence=float(data.get("confidence", 0.7)),
                entities=data.get("entities", []),
                time_reference=data.get("time_reference"),
                suggested_sources=self._get_sources_for_intent(intent),
                suggested_top_k=self._get_top_k_for_intent(intent),
            )
        except (json.JSONDecodeError, ValueError, TypeError):
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10,
            )

    def _extract_entities_simple(self, query: str) -> List[str]:
        """Simple entity extraction using capitalization."""
        words = query.split()
        entities = []
        for i, word in enumerate(words):
            w = word.strip("?.,!")
            prev_ends = i == 0 or words[i - 1][-1] in ".!?"
            if w and w[0].isupper() and len(w) > 1 and (prev_ends or i > 0):
                entities.append(w)
        return entities

    def _get_sources_for_intent(self, intent: QueryIntent) -> List[str]:
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
            QueryIntent.CONSTRAINT_CHECK: ["facts"],
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
