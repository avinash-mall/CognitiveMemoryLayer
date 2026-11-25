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


@dataclass
class ConceptualSelfProfile:
    """Conway Tier-3 persistent Conceptual-Self profile for a speaker/actor."""

    subject: str
    traits: list[str] = field(default_factory=list)
    values: list[str] = field(default_factory=list)
    boundaries_fears: list[str] = field(default_factory=list)
    goals: list[str] = field(default_factory=list)
    habits: list[str] = field(default_factory=list)
    career_identity: list[str] = field(default_factory=list)
    evidence_count: int = 1

    def merge_delta(self, delta: dict[str, Any]) -> None:
        def _merge_list(curr: list[str], incoming: Any, cap: int = 12) -> list[str]:
            if not isinstance(incoming, list):
                return curr
            seen = {x.strip().lower() for x in curr if x.strip()}
            out = list(curr)
            for item in incoming:
                s = str(item or "").strip()
                if s and s.lower() not in seen:
                    seen.add(s.lower())
                    out.append(s)
            return out[-cap:]

        self.traits = _merge_list(self.traits, delta.get("traits"))
        self.values = _merge_list(self.values, delta.get("values"))
        self.boundaries_fears = _merge_list(self.boundaries_fears, delta.get("boundaries_fears"))
        self.goals = _merge_list(self.goals, delta.get("goals"))
        self.habits = _merge_list(self.habits, delta.get("habits"))
        self.career_identity = _merge_list(self.career_identity, delta.get("career_identity"))
        self.evidence_count += 1

    def to_dict(self) -> dict[str, Any]:
        return {
            "subject": self.subject,
            "traits": self.traits,
            "values": self.values,
            "boundaries_fears": self.boundaries_fears,
            "goals": self.goals,
            "habits": self.habits,
            "career_identity": self.career_identity,
            "evidence_count": self.evidence_count,
        }

    @classmethod
    def from_dict(cls, subject: str, data: dict[str, Any]) -> ConceptualSelfProfile:
        return cls(
            subject=subject,
            traits=list(data.get("traits") or []),
            values=list(data.get("values") or []),
            boundaries_fears=list(data.get("boundaries_fears") or []),
            goals=list(data.get("goals") or []),
            habits=list(data.get("habits") or []),
            career_identity=list(data.get("career_identity") or []),
            evidence_count=int(data.get("evidence_count") or 1),
        )

    def format_compact(self) -> str:
        parts: list[str] = []
        if self.career_identity:
            parts.append(f"Identity/Background: {'; '.join(self.career_identity[:4])}")
        if self.traits:
            parts.append(f"Traits: {', '.join(self.traits[:5])}")
        if self.values:
            parts.append(f"Values: {'; '.join(self.values[:4])}")
        if self.boundaries_fears:
            parts.append(f"Boundaries/Sensitivities: {'; '.join(self.boundaries_fears[:5])}")
        if self.goals:
            parts.append(f"Goals: {'; '.join(self.goals[:4])}")
        if self.habits:
            parts.append(f"Habits: {'; '.join(self.habits[:4])}")
        if not parts:
            return ""
        return f"**{self.subject}** — " + " | ".join(parts)


class PersistentDMNStore:
    """In-memory + Postgres/SQLite store for Conway Tier-3 Conceptual-Self profiles."""

    def __init__(
        self,
        pg_session_factory: Any | None = None,
        sqlite_store: Any | None = None,
    ) -> None:
        self.pg_session_factory = pg_session_factory
        self.sqlite_store = sqlite_store
        self._cache: dict[str, dict[str, ConceptualSelfProfile]] = {}
        self._hydrated: set[str] = set()

    async def _ensure_hydrated(self, tenant_id: str) -> dict[str, ConceptualSelfProfile]:
        if tenant_id in self._hydrated:
            return self._cache.setdefault(tenant_id, {})
        self._hydrated.add(tenant_id)
        bucket = self._cache.setdefault(tenant_id, {})

        if self.pg_session_factory is not None:
            try:
                from sqlalchemy import select

                from ..storage.models import DMNSelfProfileModel

                async with self.pg_session_factory() as session:
                    res = await session.execute(
                        select(DMNSelfProfileModel).where(
                            DMNSelfProfileModel.tenant_id == tenant_id
                        )
                    )
                    for row in res.scalars().all():
                        prof = ConceptualSelfProfile.from_dict(
                            row.subject, dict(row.profile_json or {})
                        )
                        prof.evidence_count = int(row.evidence_count or 1)
                        bucket[row.subject.lower()] = prof
            except Exception:
                pass

        if self.sqlite_store is not None and hasattr(self.sqlite_store, "load_dmn_profiles"):
            try:
                loaded = await self.sqlite_store.load_dmn_profiles(tenant_id)
                for subj, data in loaded.items():
                    bucket[subj.lower()] = ConceptualSelfProfile.from_dict(subj, data)
            except Exception:
                pass

        return bucket

    async def apply_deltas(self, tenant_id: str, deltas: list[dict[str, Any]]) -> int:
        if not deltas:
            return 0
        bucket = await self._ensure_hydrated(tenant_id)
        touched: list[ConceptualSelfProfile] = []

        for d in deltas:
            subj = str(d.get("speaker") or d.get("subject") or "").strip()
            if not subj:
                continue
            key = subj.lower()
            if key not in bucket:
                bucket[key] = ConceptualSelfProfile(subject=subj)
            bucket[key].merge_delta(d)
            touched.append(bucket[key])

        if not touched:
            return 0

        if self.pg_session_factory is not None:
            try:
                from datetime import UTC, datetime

                from sqlalchemy import and_, select

                from ..storage.models import DMNSelfProfileModel

                now = datetime.now(UTC).replace(tzinfo=None)
                async with self.pg_session_factory() as session:
                    for prof in touched:
                        res = await session.execute(
                            select(DMNSelfProfileModel).where(
                                and_(
                                    DMNSelfProfileModel.tenant_id == tenant_id,
                                    DMNSelfProfileModel.subject == prof.subject,
                                )
                            )
                        )
                        existing = res.scalar_one_or_none()
                        if existing is not None:
                            existing.profile_json = prof.to_dict()
                            existing.evidence_count = prof.evidence_count
                            existing.updated_at = now
                        else:
                            session.add(
                                DMNSelfProfileModel(
                                    tenant_id=tenant_id,
                                    subject=prof.subject,
                                    profile_json=prof.to_dict(),
                                    evidence_count=prof.evidence_count,
                                    updated_at=now,
                                )
                            )
                    await session.commit()
            except Exception:
                pass

        if self.sqlite_store is not None and hasattr(self.sqlite_store, "save_dmn_profile"):
            for prof in touched:
                try:
                    await self.sqlite_store.save_dmn_profile(
                        tenant_id, prof.subject, prof.to_dict(), prof.evidence_count
                    )
                except Exception:
                    pass

        return len(touched)

    async def get_formatted_profiles(self, tenant_id: str, query: str = "") -> list[str]:
        bucket = await self._ensure_hydrated(tenant_id)
        if not bucket:
            return []
        q_low = (query or "").lower()
        matched = [
            prof
            for key, prof in bucket.items()
            if key in q_low or any(part in q_low for part in key.split() if len(part) >= 3)
        ]
        selected = matched if matched else list(bucket.values())[:3]
        lines = [prof.format_compact() for prof in selected[:3]]
        return [ln for ln in lines if ln]
