"""Memory packet builder for LLM consumption."""

import json

from ..core.enums import MemoryType
from ..core.schemas import MemoryPacket, RetrievedMemory

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
        conflicts = self._detect_conflicts(packet)
        if conflicts:
            packet.warnings.extend(conflicts)
        for mem in memories:
            if mem.record.confidence < 0.5:
                packet.open_questions.append(
                    f"Uncertain: {mem.record.text} (confidence: {mem.record.confidence:.2f})"
                )
        return packet

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
            header = "## Active Constraints (Must Follow)\n"
            constraint_lines: list[str] = []
            for c in packet.constraints[:6]:
                prov = self._constraint_provenance(c)
                line = f"- [!IMPORTANT] **{c.record.text}** {prov}".rstrip() + "\n"
                if (
                    used + len(header) + sum(len(x) for x in constraint_lines) + len(line)
                    <= constraint_budget
                ):
                    constraint_lines.append(line)
                else:
                    break
            if constraint_lines:
                sections.append(header + "".join(constraint_lines) + "\n")
                used += len(sections[-1])

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
            header = "## Recent Context\n"
            ep_lines: list[str] = []
            for e in relevant_episodes[:episode_limit]:
                ts = e.record.timestamp
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
                line = f"- [{date_str}] {e.record.text}\n"
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
