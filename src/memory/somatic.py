"""Amygdala Somatic Markers (Damasio's Somatic Marker Hypothesis).

Tags episodic and semantic memories with affective valence [-1.0, +1.0],
arousal [0.0, 1.0], and dominant emotional categories. Modulates hippocampal
long-term potentiation (flashbulb memory retention and importance) and supports
state-congruent affective retrieval in the reranker.
"""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field
from typing import Any

_TOKEN_RE = re.compile(r"[a-z']+|[!?]+")

_EMOTION_LEXICON: dict[str, tuple[float, float, str]] = {
    # (valence, arousal, emotion_category)
    # Joy / Excitement / Achievement
    "love": (0.85, 0.70, "joy"),
    "loved": (0.85, 0.70, "joy"),
    "loving": (0.80, 0.65, "joy"),
    "thrilled": (0.95, 0.90, "excitement"),
    "excited": (0.85, 0.85, "excitement"),
    "ecstatic": (0.95, 0.95, "excitement"),
    "overjoyed": (0.95, 0.90, "joy"),
    "happy": (0.75, 0.55, "joy"),
    "happiest": (0.90, 0.75, "joy"),
    "delighted": (0.85, 0.70, "joy"),
    "grateful": (0.80, 0.50, "trust"),
    "thankful": (0.75, 0.45, "trust"),
    "proud": (0.80, 0.65, "joy"),
    "promoted": (0.85, 0.75, "excitement"),
    "promotion": (0.80, 0.70, "excitement"),
    "engaged": (0.85, 0.80, "joy"),
    "wedding": (0.80, 0.75, "joy"),
    "married": (0.80, 0.75, "joy"),
    "graduated": (0.85, 0.75, "joy"),
    "graduation": (0.80, 0.70, "joy"),
    "won": (0.80, 0.75, "excitement"),
    "award": (0.80, 0.70, "joy"),
    "celebrate": (0.80, 0.70, "joy"),
    "celebrating": (0.80, 0.70, "joy"),
    "amazing": (0.80, 0.65, "joy"),
    "wonderful": (0.80, 0.60, "joy"),
    "fantastic": (0.80, 0.65, "excitement"),
    "relieved": (0.65, 0.55, "trust"),
    "relief": (0.65, 0.55, "trust"),
    "passionate": (0.80, 0.80, "excitement"),
    "inspired": (0.75, 0.65, "excitement"),
    "dream": (0.65, 0.60, "excitement"),
    # Stress / Anxiety / Fear
    "stressed": (-0.75, 0.80, "stress"),
    "stress": (-0.70, 0.75, "stress"),
    "stressful": (-0.75, 0.80, "stress"),
    "overwhelmed": (-0.80, 0.85, "stress"),
    "anxious": (-0.75, 0.80, "fear"),
    "anxiety": (-0.75, 0.80, "fear"),
    "worried": (-0.65, 0.70, "fear"),
    "worry": (-0.65, 0.70, "fear"),
    "terrified": (-0.95, 0.95, "fear"),
    "scared": (-0.80, 0.85, "fear"),
    "afraid": (-0.75, 0.75, "fear"),
    "panic": (-0.90, 0.95, "fear"),
    "panicked": (-0.90, 0.95, "fear"),
    "burnout": (-0.85, 0.80, "stress"),
    "exhausted": (-0.70, 0.65, "stress"),
    "deadline": (-0.45, 0.70, "stress"),
    "emergency": (-0.85, 0.95, "fear"),
    "hospital": (-0.80, 0.90, "fear"),
    "surgery": (-0.75, 0.85, "fear"),
    "accident": (-0.85, 0.90, "fear"),
    "injured": (-0.80, 0.80, "fear"),
    "injury": (-0.75, 0.75, "fear"),
    "broke": (-0.65, 0.70, "stress"),
    "broken": (-0.65, 0.65, "sadness"),
    "allergic": (-0.50, 0.75, "fear"),
    "allergy": (-0.45, 0.70, "fear"),
    "anaphylaxis": (-0.95, 0.98, "fear"),
    # Sadness / Grief / Loss
    "sad": (-0.75, 0.55, "sadness"),
    "devastated": (-0.95, 0.90, "sadness"),
    "heartbroken": (-0.95, 0.90, "sadness"),
    "grief": (-0.90, 0.85, "sadness"),
    "grieving": (-0.90, 0.85, "sadness"),
    "died": (-0.95, 0.90, "sadness"),
    "passed": (-0.60, 0.65, "sadness"),
    "funeral": (-0.90, 0.85, "sadness"),
    "depressed": (-0.85, 0.65, "sadness"),
    "lonely": (-0.75, 0.60, "sadness"),
    "disappointed": (-0.70, 0.60, "sadness"),
    "failed": (-0.75, 0.70, "sadness"),
    "fired": (-0.85, 0.85, "sadness"),
    "laid": (-0.65, 0.75, "stress"),
    "divorce": (-0.85, 0.85, "sadness"),
    "breakup": (-0.80, 0.80, "sadness"),
    "miss": (-0.50, 0.55, "sadness"),
    "regret": (-0.70, 0.65, "sadness"),
    # Anger / Frustration / Conflict
    "angry": (-0.80, 0.85, "anger"),
    "furious": (-0.95, 0.95, "anger"),
    "frustrated": (-0.70, 0.75, "anger"),
    "frustrating": (-0.70, 0.75, "anger"),
    "annoyed": (-0.55, 0.60, "anger"),
    "upset": (-0.70, 0.75, "anger"),
    "hate": (-0.85, 0.80, "anger"),
    "hated": (-0.85, 0.80, "anger"),
    "argued": (-0.70, 0.80, "anger"),
    "argument": (-0.70, 0.80, "anger"),
    "fight": (-0.75, 0.85, "anger"),
    "betrayed": (-0.90, 0.90, "anger"),
    "unfair": (-0.65, 0.70, "anger"),
}

_INTENSIFIERS = frozenset(
    {"very", "extremely", "so", "really", "deeply", "incredibly", "totally", "absolutely", "super"}
)
_NEGATIONS = frozenset(
    {
        "not",
        "never",
        "no",
        "hardly",
        "barely",
        "don't",
        "didn't",
        "wasn't",
        "isn't",
        "aren't",
        "won't",
    }
)

_AFFECTIVE_QUERY_CUES: dict[str, set[str]] = {
    "stress": {
        "stress",
        "stressed",
        "overwhelmed",
        "burnout",
        "pressure",
        "struggling",
        "challenge",
        "challenges",
        "worried",
        "anxious",
        "anxiety",
        "coping",
        "dealing",
    },
    "joy": {
        "happy",
        "joy",
        "enjoy",
        "love",
        "cheer",
        "gift",
        "appreciate",
        "celebrate",
        "proud",
        "comfort",
        "excited",
        "passion",
    },
    "fear": {
        "afraid",
        "scared",
        "fear",
        "worry",
        "worried",
        "risk",
        "danger",
        "emergency",
        "medical",
        "allergy",
        "health",
    },
    "sadness": {
        "sad",
        "grief",
        "loss",
        "upset",
        "hurt",
        "disappointed",
        "support",
        "feel",
        "feeling",
        "emotional",
    },
    "anger": {
        "angry",
        "frustrated",
        "conflict",
        "argument",
        "annoyed",
        "bother",
        "disagree",
        "tension",
    },
}


@dataclass(frozen=True)
class SomaticMarker:
    """Damasio somatic marker attached to an episodic or semantic trace."""

    valence: float = 0.0  # [-1.0, +1.0]
    arousal: float = 0.0  # [0.0, 1.0]
    dominant_emotion: str = "neutral"
    emotional_tags: list[str] = field(default_factory=list)
    retention_modifier: float = 1.0  # Multiplier on decay_rate (<1.0 = slower decay)
    importance_boost: float = 0.0  # Additive boost to memory importance

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SomaticMarkerAnalyzer:
    """Computes affective valence, arousal, and flashbulb retention modulation."""

    @classmethod
    def analyze(cls, text: str) -> SomaticMarker:
        """Analyze text for affective valence, arousal, and emotional categories."""
        if not text or not text.strip():
            return SomaticMarker()

        tokens = _TOKEN_RE.findall(text.lower())
        if not tokens:
            return SomaticMarker()

        valences: list[float] = []
        arousals: list[float] = []
        emotion_counts: dict[str, float] = {}
        exclamations = sum(1 for t in tokens if "!" in t)

        for idx, tok in enumerate(tokens):
            entry = _EMOTION_LEXICON.get(tok)
            if entry is None:
                continue
            val, aro, cat = entry

            # Look back 1-2 tokens for negation or intensifier
            window_prev = tokens[max(0, idx - 2) : idx]
            if any(p in _NEGATIONS for p in window_prev):
                val = -val * 0.6
                aro = max(0.2, aro * 0.7)
            if any(p in _INTENSIFIERS for p in window_prev):
                val = max(-1.0, min(1.0, val * 1.25))
                aro = min(1.0, aro * 1.25)

            valences.append(val)
            arousals.append(aro)
            emotion_counts[cat] = emotion_counts.get(cat, 0.0) + aro

        if not valences:
            if exclamations > 0:
                return SomaticMarker(
                    valence=0.1,
                    arousal=min(0.45, 0.20 * exclamations),
                    dominant_emotion="excitement",
                    emotional_tags=["emphasis"],
                    retention_modifier=0.9,
                    importance_boost=0.05,
                )
            return SomaticMarker()

        mean_valence = sum(valences) / len(valences)
        # Peak-end rule (Kahneman): affective memory arousal is dominated by the peak
        # emotional intensity + density of emotional cues.
        peak_arousal = max(arousals)
        mean_arousal = sum(arousals) / len(arousals)
        density_bonus = min(0.20, 0.05 * (len(arousals) - 1))
        exclaim_bonus = min(0.15, 0.05 * exclamations)
        final_arousal = min(
            1.0, 0.65 * peak_arousal + 0.35 * mean_arousal + density_bonus + exclaim_bonus
        )
        final_valence = max(-1.0, min(1.0, mean_valence))

        sorted_cats = sorted(emotion_counts.items(), key=lambda kv: kv[1], reverse=True)
        dominant = sorted_cats[0][0] if sorted_cats else "neutral"
        tags = [cat for cat, _ in sorted_cats[:3]]

        # Amygdala modulation of hippocampal consolidation (flashbulb retention):
        # High arousal slows decay by up to 70% and boosts importance by up to +0.25.
        retention_mod = (
            round(max(0.30, 1.0 - 0.70 * final_arousal), 3) if final_arousal >= 0.35 else 1.0
        )
        imp_boost = round(min(0.25, 0.25 * final_arousal), 3) if final_arousal >= 0.35 else 0.0

        return SomaticMarker(
            valence=round(final_valence, 3),
            arousal=round(final_arousal, 3),
            dominant_emotion=dominant,
            emotional_tags=tags,
            retention_modifier=retention_mod,
            importance_boost=imp_boost,
        )

    @classmethod
    def query_affective_resonance(cls, query: str, record_metadata: dict[str, Any] | None) -> float:
        """Compute state-congruent affective resonance bonus [0.0, 0.15] between query and memory."""
        if not query or not record_metadata:
            return 0.0
        somatic = record_metadata.get("somatic_marker")
        if not isinstance(somatic, dict):
            return 0.0

        arousal = float(somatic.get("arousal", 0.0) or 0.0)
        if arousal < 0.3:
            return 0.0

        q_words = set(_TOKEN_RE.findall(query.lower()))
        target_emotions: set[str] = set()
        for emo, cues in _AFFECTIVE_QUERY_CUES.items():
            if q_words & cues:
                target_emotions.add(emo)

        # Also check general psychological/empathetic query framing
        is_psych_query = bool(
            q_words
            & {
                "feel",
                "feeling",
                "feelings",
                "mood",
                "react",
                "reaction",
                "emotional",
                "emotion",
                "happy",
                "stressed",
                "stress",
                "worried",
                "support",
                "gift",
                "appreciate",
                "care",
                "comfort",
                "mental",
                "wellbeing",
            }
        )
        if not target_emotions and not is_psych_query:
            return 0.0

        mem_emotion = str(somatic.get("dominant_emotion", "neutral"))
        mem_tags = set(somatic.get("emotional_tags") or [])
        if target_emotions and (mem_emotion in target_emotions or (mem_tags & target_emotions)):
            return round(min(0.15, 0.12 * arousal + 0.03), 4)
        if is_psych_query and arousal >= 0.5:
            return round(min(0.10, 0.08 * arousal), 4)
        return 0.0
