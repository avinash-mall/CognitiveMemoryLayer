"""Default Mode Network (DMN / mPFC & TPJ) Theory-of-Mind & Autobiographical Self-Model.

Synthesizes a latent psychological & motivational model of the user/participants
from retrieved episodic traces, somatic markers, preferences, and constraints.
Directly powers 'Cognitive', 'Common-Sense', and interpersonal reasoning queries.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

_MENTALIZING_PATTERNS = (
    r"\bhow would\b",
    r"\bwould (?:he|she|they|[a-z]+) (?:like|enjoy|prefer|feel|react|think|appreciate|choose|want)\b",
    r"\bwhat (?:gift|present|activity|advice|suggestion|recommendation|book|movie|place)\b",
    r"\bwhy did\b",
    r"\bwhat motivated\b",
    r"\bpersonality\b",
    r"\battitude\b",
    r"\bfeel about\b",
    r"\blikely to\b",
    r"\bcoping with\b",
    r"\bstressed about\b",
    r"\bcare about\b",
    r"\bimportant to\b",
    r"\bmindset\b",
)
_MENTALIZING_RE = re.compile("|".join(_MENTALIZING_PATTERNS), re.IGNORECASE)


@dataclass
class MentalizingProfile:
    """Inferred Theory-of-Mind snapshot for a participant or user."""

    subject: str
    emotional_baseline: str = "neutral"
    active_affective_states: list[str] = field(default_factory=list)
    core_motivations_and_goals: list[str] = field(default_factory=list)
    preferences_and_affinities: list[str] = field(default_factory=list)
    active_stressors_or_constraints: list[str] = field(default_factory=list)

    def to_summary_lines(self) -> list[str]:
        lines: list[str] = []
        if self.active_affective_states:
            lines.append(
                f"Affective State ({self.subject}): {', '.join(self.active_affective_states[:3])}"
            )
        if self.core_motivations_and_goals:
            lines.append(
                f"Motivations & Goals ({self.subject}): {'; '.join(self.core_motivations_and_goals[:3])}"
            )
        if self.active_stressors_or_constraints:
            lines.append(
                f"Active Constraints/Stressors ({self.subject}): {'; '.join(self.active_stressors_or_constraints[:3])}"
            )
        if self.preferences_and_affinities:
            lines.append(
                f"Affinities ({self.subject}): {'; '.join(self.preferences_and_affinities[:3])}"
            )
        return lines


class DefaultModeNetwork:
    """Constructs Theory-of-Mind (mPFC/TPJ) mentalizing context from retrieved memories."""

    @classmethod
    def is_mentalizing_query(cls, query: str) -> bool:
        """Return True if the query requires Theory-of-Mind, cognitive, or motivational inference."""
        if not query:
            return False
        return bool(_MENTALIZING_RE.search(query))

    @classmethod
    def synthesize_mental_model(cls, packet: Any) -> list[str]:
        """Build Theory-of-Mind summary lines from a MemoryPacket's retrieved memories."""
        profiles: dict[str, MentalizingProfile] = {}

        def _get_profile(subj: str) -> MentalizingProfile:
            norm = (subj or "user").strip() or "user"
            if norm not in profiles:
                profiles[norm] = MentalizingProfile(subject=norm)
            return profiles[norm]

        # 1. Inspect constraints for goals, values, states, and causal rules
        for c in getattr(packet, "constraints", [])[:6]:
            rec = c.record
            meta = rec.metadata or {}
            speaker = str(meta.get("speaker") or "user")
            prof = _get_profile(speaker)
            cmeta = meta.get("constraints", [])
            ctype = (
                cmeta[0].get("constraint_type", "") if cmeta and isinstance(cmeta[0], dict) else ""
            )
            snippet = rec.text.strip()
            if ctype in ("goal", "value"):
                if snippet not in prof.core_motivations_and_goals:
                    prof.core_motivations_and_goals.append(snippet)
            else:
                if snippet not in prof.active_stressors_or_constraints:
                    prof.active_stressors_or_constraints.append(snippet)

        # 2. Inspect preferences
        for p in getattr(packet, "preferences", [])[:5]:
            rec = p.record
            meta = rec.metadata or {}
            speaker = str(meta.get("speaker") or "user")
            prof = _get_profile(speaker)
            snippet = rec.text.strip()
            if snippet not in prof.preferences_and_affinities:
                prof.preferences_and_affinities.append(snippet)

        # 3. Inspect episodes for high-arousal somatic markers & prospective implications
        for e in getattr(packet, "recent_episodes", [])[:8]:
            rec = e.record
            meta = rec.metadata or {}
            speaker = str(meta.get("speaker") or "user")
            prof = _get_profile(speaker)
            somatic = meta.get("somatic_marker")
            if isinstance(somatic, dict):
                arousal = float(somatic.get("arousal", 0.0) or 0.0)
                dom = str(somatic.get("dominant_emotion", "neutral"))
                if arousal >= 0.45 and dom != "neutral":
                    state_desc = f'{dom} (arousal={arousal:.2f}: "{rec.text[:90]}")'
                    if state_desc not in prof.active_affective_states:
                        prof.active_affective_states.append(state_desc)
            if meta.get("is_prospective_index"):
                snippet = rec.text.strip()
                if snippet not in prof.active_stressors_or_constraints:
                    prof.active_stressors_or_constraints.append(snippet)

        summary_lines: list[str] = []
        for prof in profiles.values():
            summary_lines.extend(prof.to_summary_lines())
        return summary_lines[:6]
