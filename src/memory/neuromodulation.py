"""Locus Coeruleus (NE) & Basal Forebrain Acetylcholine (ACh) Neuromodulatory Gain Control.

Models two foundational ascending neuromodulatory systems of the mammalian brain:
1. **Acetylcholine (ACh)**: switches hippocampal circuitry between **Encoding Mode**
   (high novelty/sensory surprise -> high DG/EC input gain, suppressed CA3 interference)
   and **Retrieval Mode** (query-driven -> high CA3 recurrent pattern completion).
2. **Norepinephrine (LC-NE)**: tracks phasic surprise, query ambiguity, and topic shift
   to modulate **Exploration Breadth** (retrieval candidate pool size) and **Neural Gain**
   (Cross-Encoder inverse temperature).
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..utils.similarity import jaccard, word_set


@dataclass(frozen=True)
class NeuromodulatoryState:
    """Instantaneous ACh + LC-NE neuromodulatory state for a read or write operation."""

    ach_mode: str  # "encoding" | "balanced" | "retrieval"
    ach_level: float  # [0.0, 1.0]: 1.0 = max encoding plasticity, 0.0 = max CA3 recall
    ne_arousal_gain: float  # [0.75, 1.60]: phasic LC-NE exploration gain
    recommended_candidate_pool: int
    cross_encoder_temperature: float
    ppr_damping: float


class NeuromodulatoryController:
    """Stateless + session-shift-aware ACh and LC-NE gain computation."""

    _BROAD_EXPLORATORY_RE = re.compile(
        r"\b(why|how would|might|likely|overall|all|every|compare|between|change|evolve|difference|recommend|suggest)\b",
        re.IGNORECASE,
    )

    @classmethod
    def for_write(cls, text: str, recent_texts: list[str] | None = None) -> NeuromodulatoryState:
        """Compute ACh encoding drive & NE surprise from incoming turn vs recent working context."""
        ws = word_set(text or "")
        novelty = 1.0
        if ws and recent_texts:
            max_sim = max((jaccard(ws, word_set(rt)) for rt in recent_texts[-5:]), default=0.0)
            novelty = max(0.0, 1.0 - max_sim)

        # High novelty elevates ACh -> favors crisp hippocampal Dentate Gyrus encoding
        ach = min(1.0, max(0.25, 0.35 + 0.65 * novelty))
        ne_gain = round(0.85 + 0.60 * novelty, 3)
        mode = "encoding" if ach >= 0.60 else "balanced"
        return NeuromodulatoryState(
            ach_mode=mode,
            ach_level=round(ach, 3),
            ne_arousal_gain=ne_gain,
            recommended_candidate_pool=40,
            cross_encoder_temperature=1.0,
            ppr_damping=0.82,
        )

    @classmethod
    def for_read(
        cls,
        query: str,
        *,
        routing_intent: str = "single_hop",
        is_cognitive_cue: bool = False,
        recent_context: str | None = None,
    ) -> NeuromodulatoryState:
        """Compute low-ACh retrieval mode & LC-NE exploration breadth for an incoming query."""
        q_ws = word_set(query or "")
        topic_shift = 0.35
        if q_ws and recent_context:
            ctx_ws = word_set(recent_context)
            if ctx_ws:
                topic_shift = max(0.0, 1.0 - jaccard(q_ws, ctx_ws))

        exploratory = bool(cls._BROAD_EXPLORATORY_RE.search(query or ""))
        complexity_bonus = 0.0
        if is_cognitive_cue:
            complexity_bonus = 0.42
        elif routing_intent in ("multi_hop", "common_sense"):
            complexity_bonus = 0.32
        elif routing_intent == "temporal":
            complexity_bonus = 0.22
        elif exploratory:
            complexity_bonus = 0.18

        ne_gain = min(1.55, max(0.80, 0.85 + 0.35 * topic_shift + complexity_bonus))
        # Candidate pool expands under high LC-NE gain (broad exploration) and narrows on crisp lookups
        pool_size = round(32 * ne_gain)
        pool_size = max(28, min(64, pool_size))

        # High NE on cognitive/exploratory queries softens Cross-Encoder temperature slightly
        # so indirect causal bridges aren't prematurely crushed before RRF/Conway fusion;
        # crisp single-hop / adversarial queries use sharper discrimination.
        if is_cognitive_cue or routing_intent in ("common_sense", "multi_hop"):
            ce_temp = 1.18
            damping = 0.85
        elif routing_intent == "adversarial":
            ce_temp = 0.88
            damping = 0.78
        else:
            ce_temp = 0.95
            damping = 0.82

        return NeuromodulatoryState(
            ach_mode="retrieval",
            ach_level=0.18,
            ne_arousal_gain=round(ne_gain, 3),
            recommended_candidate_pool=pool_size,
            cross_encoder_temperature=ce_temp,
            ppr_damping=damping,
        )
