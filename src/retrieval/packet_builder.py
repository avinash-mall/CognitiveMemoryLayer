"""Memory packet builder for LLM consumption."""

import json

from ..core.enums import MemoryType
from ..core.schemas import MemoryPacket, RetrievedMemory


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
        """Format as markdown."""
        lines = ["# Retrieved Memory Context\n"]
        if packet.constraints:
            lines.append("## Active Constraints (Must Follow)")
            for c in packet.constraints[:6]:
                prov = self._constraint_provenance(c)
                lines.append(f"- **{c.record.text}** {prov}".rstrip())
            lines.append("")
        if packet.facts:
            lines.append("## Known Facts")
            for f in packet.facts[:5]:
                conf = f"[{f.record.confidence:.0%}]" if f.record.confidence < 1.0 else ""
                lines.append(f"- {f.record.text} {conf}")
            lines.append("")
        if packet.preferences:
            lines.append("## User Preferences")
            for p in packet.preferences[:5]:
                lines.append(f"- {p.record.text}")
            lines.append("")
        if packet.recent_episodes:
            lines.append("## Recent Context")
            for e in packet.recent_episodes[:5]:
                ts = e.record.timestamp
                date_str = ts.strftime("%Y-%m-%d") if hasattr(ts, "strftime") else str(ts)
                lines.append(f"- [{date_str}] {e.record.text}")
            lines.append("")
        if packet.warnings:
            lines.append("## Warnings")
            for w in packet.warnings:
                lines.append(f"- ⚠️ {w}")
            lines.append("")
        result = "\n".join(lines)
        max_chars = max_tokens * 4
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
