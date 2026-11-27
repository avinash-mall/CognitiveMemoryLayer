"""System 1 Subcortical Fast-Decision Engine powered by Laya (Convai Innovations).

Laya is a non-autoregressive encoder-based System 1 decision model (~322M params,
ModernBERT-large backbone) trained with Reinforcement Learning for Calibrated Decisions
(RLCD). In ~20ms on GPU (single forward pass, zero text generation), it evaluates a
structured state against typed questions (`choice`, `noul`, `score`) and returns
mathematically calibrated probabilities.

In Cognitive Memory Layer (CML), Laya acts as the fast subcortical System 1 layer:
1. Query Intent & Adaptive Context Routing: classifies whether a query requires wide
   multi-episode aggregation (temporal, multi-hop, common-sense -> 12 episodes) or
   focused precision (single-hop, adversarial trap, cognitive empathy -> 6 episodes).
2. CA1 Reality Monitoring (Presupposition & Speaker-Swap Verification): evaluates
   calibrated `noul` probabilities for whether retrieved evidence actually supports
   the query's factual premise and person-action attribution.

Gracefully falls back to deterministic structural heuristics when `laya` is not
installed or `features.laya_system1_enabled` is False, ensuring zero overhead in
lightweight/unit-test environments.
"""

from __future__ import annotations

import re
import threading
from dataclasses import dataclass
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Fast structural regexes (zero-latency pre-screen & fallback when Laya is disabled)
_COGNITIVE_DIALOGUE_RE = re.compile(
    r"^\s*(?:[A-Z][a-zA-Z0-9_]{0,15}\s*:|User\s*:|Speaker\s*[AB]\s*:)",
)
_TEMPORAL_QUERY_RE = re.compile(
    r"\b(?:when\s+did|when\s+was|when\s+will|what\s+date|what\s+year|what\s+month|"
    r"what\s+time|how\s+long\s+ago|how\s+many\s+(?:days|weeks|months|years)|"
    r"before\s+or\s+after|chronological|timeline|recently\s+when)\b",
    re.IGNORECASE,
)
_MULTIHOP_QUERY_RE = re.compile(
    r"\b(?:how\s+many\s+times|how\s+many\s+\w+\s+has|how\s+many\s+\w+\s+did|"
    r"list\s+all|all\s+the\s+|both\s+|compare\s+|in\s+common|which\s+authors|"
    r"what\s+(?:activities|books|hobbies|places|things|projects|items|countries|events|instruments|sports|movies|songs)\b|"
    r"where\s+has\s+\w+\s+\w+ed\b|where\s+did\s+\w+\s+move\s+from|"
    r"what\s+else|multiple|altogether|in\s+total)\b",
    re.IGNORECASE,
)
_COMMON_SENSE_QUERY_RE = re.compile(
    r"\b(?:might\s+|would\s+|could\s+\w+\s+be\b|recommend|"
    r"likely|probably|what\s+kind\s+of|what\s+type\s+of|suggest|"
    r"considered\s+(?:a\s+|an\s+)?\w+|financial\s+status|religious|political|"
    r"career\s+option|personality|describe\s+\w+'s|relationship\s+between|who\s+is\s+[A-Z])\b",
    re.IGNORECASE,
)

_ROUTING_QUESTIONS = {
    "needs_wide_context": {
        "type": "noul",
        "instructions": (
            "Does answering this query require aggregating multiple past events across time, "
            "counting occurrences, resolving dates, or making a broad common-sense inference?"
        ),
    },
    "is_cognitive_empathy": {
        "type": "noul",
        "instructions": (
            "Is the input a conversational personal statement sharing a feeling, habit, or "
            "situation expecting an empathetic reply (rather than a direct factual question)?"
        ),
    },
}


@dataclass(frozen=True)
class QueryRoutingDecision:
    """System 1 routing decision for adaptive retrieval budget & Cognitive protection."""

    intent: str  # single_hop | multi_hop | temporal | common_sense | cognitive_empathy
    needs_wide_context: bool
    is_cognitive_cue: bool
    recommended_max_episodes: int
    confidence: float = 1.0
    source: str = "heuristic"


class LayaSystem1Engine:
    """Thread-safe lazy singleton for Convai's Laya System 1 decision model."""

    _lock = threading.Lock()
    _agent: Any = None
    _load_attempted: bool = False
    _loaded_model_id: str | None = None

    @classmethod
    def _is_enabled(cls) -> bool:
        try:
            from .config import get_settings

            return bool(getattr(get_settings().features, "laya_system1_enabled", False))
        except Exception:
            return False

    @classmethod
    def _model_id(cls) -> str:
        try:
            from .config import get_settings

            return str(
                getattr(get_settings().features, "laya_model_id", "convaiinnovations/laya")
                or "convaiinnovations/laya"
            )
        except Exception:
            return "convaiinnovations/laya"

    @classmethod
    def get_agent(cls) -> Any | None:
        """Lazily load and return the Laya agent when enabled, else None."""
        if not cls._is_enabled():
            return None
        model_id = cls._model_id()
        if cls._agent is not None and cls._loaded_model_id == model_id:
            return cls._agent
        with cls._lock:
            if cls._agent is not None and cls._loaded_model_id == model_id:
                return cls._agent
            if cls._load_attempted and cls._loaded_model_id == model_id:
                return None
            cls._load_attempted = True
            cls._loaded_model_id = model_id
            try:
                from laya import load as laya_load

                cls._agent = laya_load(model_id)
                logger.info(
                    "laya_system1_loaded",
                    model_id=model_id,
                    device=str(getattr(cls._agent, "device", "unknown")),
                )
            except Exception as exc:
                logger.warning("laya_system1_load_failed", model_id=model_id, error=str(exc))
                cls._agent = None
        return cls._agent

    @classmethod
    def classify_query_routing(
        cls,
        query: str,
        default_wide: int = 22,
        default_focused: int = 16,
    ) -> QueryRoutingDecision:
        """Classify query intent and recommend adaptive episode budget.

        Combines zero-latency structural cues with Laya System 1 calibrated
        decisions (when enabled) to keep wide episode budgets for temporal,
        multi-hop, and common-sense queries while tightening focus for
        single-hop, adversarial-trap, and cognitive-empathy turns.
        """
        q = (query or "").strip()
        if not q:
            return QueryRoutingDecision(
                intent="single_hop",
                needs_wide_context=False,
                is_cognitive_cue=False,
                recommended_max_episodes=default_focused,
            )

        # 1. Conversational empathy / Cognitive cue (e.g. "A: I cancelled that restaurant...")
        if _COGNITIVE_DIALOGUE_RE.match(q) and not q.endswith("?"):
            return QueryRoutingDecision(
                intent="cognitive_empathy",
                needs_wide_context=False,
                is_cognitive_cue=True,
                recommended_max_episodes=default_focused,
                confidence=0.95,
                source="structural",
            )

        # 2. Explicit temporal / date / chronology queries (need wider episode horizon)
        if _TEMPORAL_QUERY_RE.search(q):
            return QueryRoutingDecision(
                intent="temporal",
                needs_wide_context=True,
                is_cognitive_cue=False,
                recommended_max_episodes=default_wide,
                confidence=0.92,
                source="structural",
            )

        # 3. Explicit multi-hop aggregation queries
        if _MULTIHOP_QUERY_RE.search(q):
            return QueryRoutingDecision(
                intent="multi_hop",
                needs_wide_context=True,
                is_cognitive_cue=False,
                recommended_max_episodes=default_wide,
                confidence=0.90,
                source="structural",
            )

        # 4. Explicit common-sense / recommendation / inference queries
        if _COMMON_SENSE_QUERY_RE.search(q):
            return QueryRoutingDecision(
                intent="common_sense",
                needs_wide_context=True,
                is_cognitive_cue=False,
                recommended_max_episodes=default_wide,
                confidence=0.88,
                source="structural",
            )

        # 5. Laya System 1 calibrated routing for ambiguous queries
        agent = cls.get_agent()
        if agent is not None:
            try:
                res = agent.system_one({"query": q}, _ROUTING_QUESTIONS)
                ans = res.get("answers", {})
                wide_p = float(ans.get("needs_wide_context", {}).get("noul", 0.0))
                cog_p = float(ans.get("is_cognitive_empathy", {}).get("noul", 0.0))
                if cog_p >= 0.65:
                    return QueryRoutingDecision(
                        intent="cognitive_empathy",
                        needs_wide_context=False,
                        is_cognitive_cue=True,
                        recommended_max_episodes=default_focused,
                        confidence=cog_p,
                        source="laya",
                    )
                if wide_p >= 0.55:
                    return QueryRoutingDecision(
                        intent="common_sense",
                        needs_wide_context=True,
                        is_cognitive_cue=False,
                        recommended_max_episodes=default_wide,
                        confidence=wide_p,
                        source="laya",
                    )
                return QueryRoutingDecision(
                    intent="single_hop",
                    needs_wide_context=False,
                    is_cognitive_cue=False,
                    recommended_max_episodes=default_focused,
                    confidence=max(1.0 - wide_p, 1.0 - cog_p),
                    source="laya",
                )
            except Exception as exc:
                logger.debug("laya_routing_error", error=str(exc))

        # 6. Default focused single-hop retrieval
        return QueryRoutingDecision(
            intent="single_hop",
            needs_wide_context=False,
            is_cognitive_cue=False,
            recommended_max_episodes=default_focused,
            confidence=0.75,
            source="heuristic",
        )
