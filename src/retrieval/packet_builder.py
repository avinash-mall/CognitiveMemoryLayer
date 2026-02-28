"""Memory packet builder for LLM consumption."""

import json
from datetime import datetime

from ..core.enums import MemoryType
from ..core.schemas import MemoryPacket, RetrievedMemory
from ..utils.modelpack import get_modelpack_runtime

# Fallback constants when config unavailable (BUG-02: avoid diluting constraints)
EPISODE_RELEVANCE_THRESHOLD = 0.5
MAX_EPISODES_WHEN_CONSTRAINTS = 3
MAX_EPISODES_DEFAULT = 5
MAX_CONSTRAINT_TOKENS = 400


class MemoryPacketBuilder:
    """Builds structured memory packets from retrieved memories."""

    def build(
        self,
        memories: list[RetrievedMemory],
        query: str,
        include_provenance: bool = True,
    ) -> MemoryPacket:
        """Build a memory packet from retrieved memories."""
        packet = MemoryPacket(query=query)
        for mem in memories:
            mem_type = mem.record.type
            if mem_type == MemoryType.SEMANTIC_FACT:
                packet.facts.append(mem)
            elif mem_type == MemoryType.PREFERENCE:
                packet.preferences.append(mem)
            elif mem_type == MemoryType.PROCEDURE:
                packet.procedures.append(mem)
            elif mem_type == MemoryType.CONSTRAINT:
                packet.constraints.append(mem)
            else:
                packet.recent_episodes.append(mem)
        packet.constraints, constraint_warnings = self._resolve_constraint_conflicts(
            packet.constraints
        )
        conflicts = self._detect_conflicts(packet)
        if conflicts:
            packet.warnings.extend(conflicts)
        if constraint_warnings:
            packet.warnings.extend(constraint_warnings)
        for mem in memories:
            if mem.record.confidence < 0.5:
                packet.open_questions.append(
                    f"Uncertain: {mem.record.text} (confidence: {mem.record.confidence:.2f})"
                )
        return packet

    def _resolve_constraint_conflicts(
        self,
        constraints: list[RetrievedMemory],
    ) -> tuple[list[RetrievedMemory], list[str]]:
        """Resolve retrieval-time supersession conflicts among constraints.

        Keeps the most recent surviving constraints and drops older constraints
        when a newer one supersedes them (same key or model-backed supersession).
        """
        if len(constraints) <= 1:
            return constraints, []

        def _ts(mem: RetrievedMemory) -> datetime:
            ts = mem.record.timestamp
            return ts if isinstance(ts, datetime) else datetime.min

        ordered = sorted(constraints, key=_ts, reverse=True)
        kept: list[RetrievedMemory] = []
        warnings: list[str] = []
        modelpack = get_modelpack_runtime()

        for candidate in ordered:
            cand_key = getattr(candidate.record, "key", None) or ""
            suppressed = False
            for newer in kept:
                newer_key = getattr(newer.record, "key", None) or ""
                if cand_key and newer_key and cand_key == newer_key:
                    warnings.append(
                        f"Suppressed older constraint for key '{cand_key}' in favor of newer statement."
                    )
                    suppressed = True
                    break

                if modelpack.available:
                    pred = modelpack.predict_pair(
                        "supersession", candidate.record.text, newer.record.text
                    )
                    if pred and pred.label == "supersedes" and pred.confidence >= 0.6:
                        warnings.append(
                            f"Suppressed likely superseded constraint: '{candidate.record.text[:80]}...'"
                        )
                        suppressed = True
                        break

            if not suppressed:
                kept.append(candidate)

        return kept, warnings

    def _detect_conflicts(self, packet: MemoryPacket) -> list[str]:
        """Detect potential conflicts in retrieved memories."""
        conflicts: list[str] = []
        preference_values: dict = {}
        for pref in packet.preferences:
            key = getattr(pref.record, "key", None)
            if key in preference_values:
                if preference_values[key] != pref.record.text:
                    conflicts.append(
                        f"Conflicting preferences for {key}: "
                        f"'{preference_values[key]}' vs '{pref.record.text}'"
                    )
            elif key:
                preference_values[key] = pref.record.text
        fact_values: dict = {}
        for fact in packet.facts:
            key = getattr(fact.record, "key", None)
            if key and key in fact_values:
                if fact_values[key] != fact.record.text:
                    conflicts.append(f"Conflicting facts for {key}")
            elif key:
                fact_values[key] = fact.record.text
        return conflicts

    def to_llm_context(
        self,
        packet: MemoryPacket,
        max_tokens: int = 2000,
        format: str = "markdown",
    ) -> str:
        """Format packet for LLM context injection."""
        if format == "markdown":
            return self._format_markdown(packet, max_tokens)
        if format == "json":
            return self._format_json(packet, max_tokens)
        return packet.to_context_string(max_chars=max_tokens * 4)

    @staticmethod
    def _constraint_provenance(mem: RetrievedMemory) -> str:
        """Extract compact provenance string from a constraint memory."""
        meta = mem.record.metadata or {}
        # Check for structured constraints in metadata
        constraints_meta = meta.get("constraints", [])
        if constraints_meta and isinstance(constraints_meta, list):
            first = constraints_meta[0] if constraints_meta else {}
            ctype = first.get("constraint_type", "")
            prov = first.get("provenance", [])
            prov_str = ", ".join(prov) if prov else ""
            label = f"[{ctype.title()}]" if ctype else ""
            src = f" (from {prov_str})" if prov_str else ""
            return f"{label}{src}"
        # Fallback: use source_turn_id from evidence refs
        turn_id = meta.get("source_turn_id", "")
        if turn_id:
            return f"(from {turn_id})"
        return ""

    def _format_markdown(self, packet: MemoryPacket, max_tokens: int) -> str:
        """Format as markdown with constraint-first token budget.

        Constraints get a reserved budget so they are never truncated.
        Facts, preferences, and episodes share the remaining budget.
        """
        max_chars = max_tokens * 4
        main_header = "# Retrieved Memory Context\n\n"
        sections_budget = max_chars - len(main_header)
        try:
            from ..core.config import get_settings

            settings = get_settings()
            constraint_budget = min(settings.retrieval.max_constraint_tokens * 4, sections_budget)
            threshold = settings.retrieval.episode_relevance_threshold
            episode_limit = (
                settings.retrieval.max_episodes_when_constraints
                if packet.constraints
                else settings.retrieval.max_episodes_default
            )
        except Exception:
            constraint_budget = min(MAX_CONSTRAINT_TOKENS * 4, sections_budget)
            threshold = EPISODE_RELEVANCE_THRESHOLD
            episode_limit = (
                MAX_EPISODES_WHEN_CONSTRAINTS if packet.constraints else MAX_EPISODES_DEFAULT
            )

        sections: list[str] = []
        used = 0

        # 1. Constraints first (reserved budget; never truncate mid-constraint)
        if packet.constraints:
            must_follow = []
            consider = []
            for c in packet.constraints[:6]:
                meta = c.record.metadata or {}
                cmeta = meta.get("constraints", [])
                ctype = cmeta[0].get("constraint_type", "") if cmeta else ""
                if ctype in ("value", "policy", "preference"):
                    must_follow.append(c)
                else:
                    consider.append(c)

            constraint_lines: list[str] = []
            if must_follow:
                constraint_lines.append("## Constraints (Must Follow)\n")
                for c in must_follow:
                    prov = self._constraint_provenance(c)
                    line = f"- Earlier you said: \"{c.record.text}\" {prov}\n"
                    if (
                        used + sum(len(x) for x in constraint_lines) + len(line)
                        <= constraint_budget
                    ):
                        constraint_lines.append(line)
            if consider:
                constraint_lines.append("## Other Constraints to Consider\n")
                for c in consider:
                    prov = self._constraint_provenance(c)
                    line = f"- You also mentioned: \"{c.record.text}\" {prov}\n"
                    if (
                        used + sum(len(x) for x in constraint_lines) + len(line)
                        <= constraint_budget
                    ):
                        constraint_lines.append(line)

            if constraint_lines:
                block = "".join(constraint_lines) + "\n"
                sections.append(block)
                used += len(block)

        remaining = sections_budget - used

        # 2. Facts
        if packet.facts and remaining > 100:
            header = "## Known Facts\n"
            fact_lines: list[str] = []
            for f in packet.facts[:5]:
                conf = f"[{f.record.confidence:.0%}]" if f.record.confidence < 1.0 else ""
                line = f"- {f.record.text} {conf}\n"
                if len(header) + sum(len(x) for x in fact_lines) + len(line) <= remaining:
                    fact_lines.append(line)
                else:
                    break
            if fact_lines:
                s = header + "".join(fact_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)

        # 3. Preferences
        if packet.preferences and remaining > 100:
            header = "## User Preferences\n"
            pref_lines: list[str] = []
            for p in packet.preferences[:5]:
                line = f"- {p.record.text}\n"
                if len(header) + sum(len(x) for x in pref_lines) + len(line) <= remaining:
                    pref_lines.append(line)
                else:
                    break
            if pref_lines:
                s = header + "".join(pref_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)

        # 4. Recent episodes
        relevant_episodes = [
            e for e in packet.recent_episodes if getattr(e, "relevance_score", 1.0) > threshold
        ]
        if relevant_episodes and remaining > 100:
            header = "## Recent Events\n"
            ep_lines: list[str] = []
            for e in relevant_episodes[:episode_limit]:
                line = f"- {e.record.text} (confidence: {e.record.confidence:.2f})\n"
                if len(header) + sum(len(x) for x in ep_lines) + len(line) <= remaining:
                    ep_lines.append(line)
                else:
                    break
            if ep_lines:
                s = header + "".join(ep_lines) + "\n"
                sections.append(s)
                used += len(s)
                remaining -= len(s)

        # 5. Warnings
        if packet.warnings and remaining > 50:
            header = "## Warnings\n"
            warn_lines = [f"- {w}\n" for w in packet.warnings]
            s = header + "".join(warn_lines) + "\n"
            if len(s) <= remaining:
                sections.append(s)

        result = "# Retrieved Memory Context\n\n" + "".join(sections)
        if len(result) > max_chars:
            result = result[:max_chars] + "\n... (truncated)"
        return result

    def _format_json(self, packet: MemoryPacket, max_tokens: int) -> str:
        """Format as JSON string."""
        data = {
            "facts": [
                {"text": f.record.text, "confidence": f.record.confidence} for f in packet.facts[:5]
            ],
            "preferences": [{"text": p.record.text} for p in packet.preferences[:5]],
            "recent": [
                {
                    "text": e.record.text,
                    "date": (
                        e.record.timestamp.isoformat()
                        if hasattr(e.record.timestamp, "isoformat")
                        else str(e.record.timestamp)
                    ),
                }
                for e in packet.recent_episodes[:5]
            ],
            "constraints": [
                {
                    "text": c.record.text,
                    "confidence": c.record.confidence,
                    "provenance": self._constraint_provenance(c),
                }
                for c in packet.constraints[:6]
            ],
            "warnings": packet.warnings,
        }
        return json.dumps(data, indent=2)
