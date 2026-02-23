"""Query classifier for retrieval strategy selection."""

import json
import re

from ..utils.llm import LLMClient
from .query_types import QueryAnalysis, QueryIntent

FAST_PATTERNS = {
    QueryIntent.PREFERENCE_LOOKUP: [
        r"what (do|does) (i|my) (like|prefer|want|enjoy)",
        r"(my|i) (favorite|preferred)",
        r"do i (like|prefer|enjoy)",
        r"what (food|cuisine|music|movie|book|sport|hobby|color|drink) do (i|they)",
    ],
    QueryIntent.IDENTITY_LOOKUP: [
        r"what('s| is) my (name|email|phone|address|job|title)",
        r"who am i",
        r"my (name|email|phone)",
        r"what (is|was) (\w+'s|their|his|her) (name|job|role|title)",
    ],
    QueryIntent.TASK_STATUS: [
        r"where (am i|are we) (in|on|at)",
        r"what('s| is) (the|my) (status|progress)",
        r"(current|next) (step|task)",
    ],
    QueryIntent.TEMPORAL_QUERY: [
        r"(last|past) (week|month|day|year)",
        r"(yesterday|today|recently)",
        r"when did (i|we|he|she|they)",
        r"what happened",
        r"(first|last) time",
        r"how long (ago|have|has)",
    ],
    QueryIntent.PROCEDURAL: [
        r"how (do|can|should|did) (i|we)",
        r"what('s| are) the steps",
        r"(procedure|process) for",
    ],
    QueryIntent.EPISODIC_RECALL: [
        r"what did (\w+ )?(say|talk|mention|discuss|tell|share|bring up)",
        r"what (was|were) (\w+ )?(talking|saying|discussing|doing)",
        r"tell me (about|what)",
        r"do (you|i) remember",
        r"(who|what|where|which) (was|were|did|is|are) (\w+ )?(said|told|mentioned|discussed|shared)",
        r"(describe|summarize) (the|our|my|their) (conversation|discussion|talk|meeting)",
        r"what (topics?|subjects?) (came up|were covered|did we discuss)",
    ],
    QueryIntent.CONSTRAINT_CHECK: [
        r"should i",
        r"can i",
        r"is it ok (to|if)",
        r"would it be (ok|fine|good|bad)",
        r"what if i",
        r"do you think i should",
        r"recommend",
        r"is (this|that|it) (consistent|aligned|compatible)",
        r"would (this|that) (conflict|contradict|go against)",
    ],
    QueryIntent.GENERAL_QUESTION: [
        r"(who|what|where|when|why|which|how) (is|are|was|were|did|do|does|has|have|had)",
        r"tell me (more )?(about|what|how|why|when|where)",
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
    """Classifies queries to determine retrieval strategy. Fast patterns first, then LLM."""

    def __init__(self, llm_client: LLMClient | None = None):
        self.llm = llm_client
        self._compiled_patterns = {
            intent: [re.compile(p, re.IGNORECASE) for p in patterns]
            for intent, patterns in FAST_PATTERNS.items()
        }

    # Patterns that indicate a decision/temptation/recommendation query
    _DECISION_PATTERNS = [
        re.compile(r"\bshould i\b", re.I),
        re.compile(r"\bcan i\b", re.I),
        re.compile(r"\bis it ok\b", re.I),
        re.compile(r"\bwould it be\b", re.I),
        re.compile(r"\bwhat if i\b", re.I),
        re.compile(r"\brecommend\b", re.I),
        re.compile(r"\bstart (watching|doing|eating|buying)\b", re.I),
        re.compile(r"\btry (this|that|the)\b", re.I),
        re.compile(r"\bgo (out|for|to)\b", re.I),
    ]
    # Patterns for constraint dimension classification
    _CONSTRAINT_DIM_PATTERNS: dict[str, list[re.Pattern]] = {
        "goal": [re.compile(r"\b(goal|objective|target|aim|ambition|milestone)\b", re.I)],
        "value": [re.compile(r"\b(value|principle|ethic|belief|priority)\b", re.I)],
        "state": [
            re.compile(r"\b(feel|mood|stress|anxious|tired|busy|sick)\b", re.I),
            re.compile(r"\b(energy|improve|energetic)\b", re.I),  # BUG-03: energy/improve -> state
        ],
        "causal": [re.compile(r"\b(because|reason|consequence|result|cause)\b", re.I)],
        "policy": [re.compile(r"\b(rule|policy|restriction|limit|boundary)\b", re.I)],
    }

    async def classify(
        self,
        query: str,
        recent_context: str | None = None,
    ) -> QueryAnalysis:
        """Classify a query and extract relevant information.

        Strategy:
        1. Always try fast regex patterns first.
        2. If a pattern matches with confidence > 0.8, return immediately.
        3. Only call the LLM when ``use_llm_query_classifier_only`` is
           explicitly enabled in settings — never as a silent fallback for
           unmatched queries.  This avoids one LLM call per read request for
           the common case of queries that simply don't match any pattern.
        4. For unmatched queries (fast_result is None or low-confidence)
           without LLM, default to GENERAL_QUESTION with confidence 0.5,
           which triggers vector + facts retrieval — a safe, broad baseline.
        """
        # If query is vague and we have context, use it to infer intent
        effective_query = query
        if recent_context and self._is_vague(query):
            effective_query = f"{recent_context}\nUser now asks: {query}"

        from ..core.config import get_settings

        settings = get_settings().features

        # Fast path (always attempted when use_llm_enabled=false or not force-LLM)
        fast_result: QueryAnalysis | None = None
        if not (settings.use_llm_enabled and settings.use_llm_query_classifier_only):
            fast_result = self._fast_classify(effective_query)

        if fast_result and fast_result.confidence > 0.8:
            self._enrich_constraint_dimensions(fast_result)
            return fast_result

        # LLM path: only when use_llm_enabled and explicitly requested via feature flag
        if self.llm and settings.use_llm_enabled and settings.use_llm_query_classifier_only:
            result = await self._llm_classify(effective_query, recent_context=recent_context)
            self._enrich_constraint_dimensions(result)
            return result

        # Default: use the partial fast result when available, else GENERAL_QUESTION
        result = fast_result or QueryAnalysis(
            original_query=query,
            intent=QueryIntent.GENERAL_QUESTION,
            confidence=0.5,
            suggested_sources=["vector", "facts"],
            suggested_top_k=10,
        )
        self._enrich_constraint_dimensions(result)
        return result

    def _enrich_constraint_dimensions(self, analysis: QueryAnalysis) -> None:
        """Detect constraint dimensions and decision-query nature from the query text."""
        q = analysis.original_query
        # Check if this is a decision/temptation query
        if any(p.search(q) for p in self._DECISION_PATTERNS):
            analysis.is_decision_query = True
        # Skip constraint_dimensions if LLM already provided them
        if analysis.constraint_dimensions_from_llm:
            return
        # BUG-03: When confidence low, retrieve all categories (constraint_dimensions=None)
        if analysis.confidence < 0.8:
            analysis.constraint_dimensions = None
            return
        # Detect constraint dimensions
        dims: list[str] = []
        for dim, patterns in self._CONSTRAINT_DIM_PATTERNS.items():
            if any(p.search(q) for p in patterns):
                dims.append(dim)
        analysis.constraint_dimensions = dims
        # If decision query detected but no specific intent was CONSTRAINT_CHECK,
        # and the intent is general/unknown, upgrade it
        if analysis.is_decision_query and analysis.intent in (
            QueryIntent.GENERAL_QUESTION,
            QueryIntent.UNKNOWN,
        ):
            analysis.intent = QueryIntent.CONSTRAINT_CHECK
            analysis.suggested_sources = self._get_sources_for_intent(QueryIntent.CONSTRAINT_CHECK)
            analysis.suggested_top_k = self._get_top_k_for_intent(QueryIntent.CONSTRAINT_CHECK)

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

    def _fast_classify(self, query: str) -> QueryAnalysis | None:
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
            # Catch all errors including LLM network failures, rate limits, etc. (MED-26)
            import logging

            logging.getLogger(__name__).warning("llm_classify_failed: %s", str(e))
            return QueryAnalysis(
                original_query=query,
                intent=QueryIntent.GENERAL_QUESTION,
                confidence=0.5,
                suggested_sources=["vector", "facts"],
                suggested_top_k=10,
            )

    def _extract_entities_simple(self, query: str) -> list[str]:
        """Simple entity extraction using capitalization.

        Only treats capitalized words as entities if they do NOT appear at a
        sentence boundary (i.e., after a period/exclamation/question mark or at
        position 0). This avoids false positives like 'The', 'What', 'How'
        (MED-23).
        """
        words = query.split()
        entities = []
        for i, word in enumerate(words):
            w = word.strip("?.,!")
            # Skip sentence-initial words: position 0 or preceded by sentence-ending punctuation
            prev_ends = i == 0 or words[i - 1][-1] in ".!?"
            if w and w[0].isupper() and len(w) > 1 and not prev_ends:
                entities.append(w)
        return entities

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
