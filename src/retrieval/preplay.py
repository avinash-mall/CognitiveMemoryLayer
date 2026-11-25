"""Hippocampal Forward Preplay & Counterfactual Simulation Engine.

Models "Episodic Future Thinking" (Schacter & Addis; Pfeiffer & Foster hippocampal
preplay): before acting or responding, simulates a proposed action/remark against
stored Constraints, Conway DMN Conceptual-Self boundaries, Procedural pitfalls,
and CA1 Reality Monitoring premise checks.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Any

from ..utils.similarity import word_set

_PREPLAY_PROMPT = """Evaluate this proposed action or conversational scenario against the person's stored memory profile and constraints.

Proposed Action / Scenario:
"{scenario}"

Stored Memory Context & Constraints:
{context}

Return ONLY valid JSON:
{{
  "compatible": true,
  "risk_score": 0.15,
  "violated_constraints": [],
  "predicted_reaction": "1-sentence prediction grounded in past memories",
  "recommended_adjustment": "1-sentence advice to respect implicit boundaries/preferences"
}}"""


@dataclass
class SimulationVerdict:
    """Structured counterfactual outcome from Hippocampal Forward Preplay."""

    scenario: str
    compatible: bool = True
    risk_score: float = 0.0
    violated_constraints: list[str] = field(default_factory=list)
    predicted_reaction: str = ""
    recommended_adjustment: str = ""
    alerts: list[str] = field(default_factory=list)
    supporting_memory_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "scenario": self.scenario,
            "compatible": self.compatible,
            "risk_score": round(self.risk_score, 4),
            "violated_constraints": self.violated_constraints,
            "predicted_reaction": self.predicted_reaction,
            "recommended_adjustment": self.recommended_adjustment,
            "alerts": self.alerts,
            "supporting_memory_ids": self.supporting_memory_ids,
        }


class HippocampalPreplaySimulator:
    """Runs fast inline constraint preplay and full counterfactual simulation."""

    def __init__(
        self,
        retriever: Any,
        dmn_store: Any | None = None,
        procedural_engine: Any | None = None,
        llm_client: Any | None = None,
    ) -> None:
        self.retriever = retriever
        self.dmn_store = dmn_store
        self.procedural_engine = procedural_engine
        self.llm_client = llm_client

    @staticmethod
    def inline_packet_check(
        query: str,
        packet: Any,
        dmn_lines: list[str] | None = None,
        procedural_lines: list[str] | None = None,
    ) -> list[str]:
        """Zero-LLM (<2ms) preplay alert synthesis for Cognitive & planning reads."""
        alerts: list[str] = []
        q_ws = word_set(query or "")

        # Surface active Proustian / Constraint reminders if present
        for c in getattr(packet, "constraints", [])[:4]:
            rec = c.record
            meta = rec.metadata or {}
            txt = (rec.text or "").strip()
            if meta.get("rpe_correction"):
                alerts.append(f"Prior user correction: {txt[:180]}")
            elif meta.get("proustian_cue_pointer") and len(alerts) < 2:
                alerts.append(txt[:200])

        # Surface matching procedural pitfalls
        for pl in procedural_lines or []:
            if "Avoid:" in pl and q_ws & word_set(pl):
                alerts.append(f"Procedural guardrail — {pl[:180]}")

        # Surface DMN sensitivities if query touches sensitive/decision domain
        for dl in dmn_lines or []:
            if "Boundaries/Sensitivities:" in dl and len(alerts) < 3:
                seg = dl.split("Boundaries/Sensitivities:", 1)[-1].split("|")[0].strip()
                subj = dl.split("**")[1] if "**" in dl else "Speaker"
                if seg:
                    alerts.append(f"Respect {subj}'s sensitivities: {seg[:160]}")

        return alerts[:3]

    async def simulate(
        self,
        tenant_id: str,
        scenario: str,
        *,
        max_results: int = 12,
        context_filter: list[str] | None = None,
    ) -> SimulationVerdict:
        """Execute full Hippocampal Forward Preplay simulation for a proposed action/scenario."""
        packet = await self.retriever.retrieve(
            tenant_id=tenant_id,
            query=scenario,
            max_results=max_results,
            context_filter=context_filter,
        )
        rmeta = packet.retrieval_meta or {}
        dmn_lines = list(rmeta.get("dmn_self_profiles") or [])
        if not dmn_lines and self.dmn_store is not None:
            dmn_lines = await self.dmn_store.get_formatted_profiles(tenant_id, scenario)

        proc_lines = list(rmeta.get("procedural_skills") or [])
        if not proc_lines and self.procedural_engine is not None:
            skills = await self.procedural_engine.retrieve_skills(tenant_id, scenario, top_k=3)
            proc_lines = [s.format_playbook_line() for s in skills]

        alerts = self.inline_packet_check(scenario, packet, dmn_lines, proc_lines)
        for w in packet.warnings[:3]:
            if w not in alerts:
                alerts.append(w)

        violated: list[str] = []
        for c in packet.constraints[:5]:
            txt = (c.record.text or "").strip()
            if txt and (
                (c.record.metadata or {}).get("rpe_correction")
                or (c.record.metadata or {}).get("proustian_cue_pointer")
                or c.relevance_score >= 0.55
            ):
                violated.append(txt[:180])

        sup_ids = [str(m.record.id) for m in packet.ranked_memories[:8]]
        compatible = len(packet.warnings) == 0 and not any(
            (c.record.metadata or {}).get("rpe_correction") for c in packet.constraints[:5]
        )
        risk = min(
            0.95,
            0.15 * len(violated) + 0.35 * len(packet.warnings) + (0.0 if compatible else 0.35),
        )
        predicted = (
            "Likely positive/aligned with stored history."
            if compatible and not violated
            else f"Potential friction with {len(violated) or len(alerts)} stored constraint(s)/premise(s)."
        )
        adjustment = (
            alerts[0] if alerts else "Proceed normally; no conflicting constraints detected."
        )

        if self.llm_client is not None:
            ctx_str = self.retriever.packet_builder.to_llm_context(packet, max_tokens=1200)
            try:
                raw = await self.llm_client.complete(
                    _PREPLAY_PROMPT.format(scenario=scenario.strip(), context=ctx_str),
                    temperature=0.0,
                    max_tokens=320,
                )
                txt = raw.strip()
                if "<think>" in txt and "</think>" in txt:
                    txt = txt.split("</think>")[-1].strip()
                s_idx, e_idx = txt.find("{"), txt.rfind("}")
                if s_idx != -1 and e_idx > s_idx:
                    parsed = json.loads(re.sub(r",\s*([}\]])", r"\1", txt[s_idx : e_idx + 1]))
                    if isinstance(parsed, dict):
                        compatible = bool(parsed.get("compatible", compatible))
                        risk = max(0.0, min(1.0, float(parsed.get("risk_score", risk))))
                        llm_v = [str(x) for x in (parsed.get("violated_constraints") or []) if x]
                        if llm_v:
                            violated = llm_v[:5]
                        if parsed.get("predicted_reaction"):
                            predicted = str(parsed["predicted_reaction"])
                        if parsed.get("recommended_adjustment"):
                            adjustment = str(parsed["recommended_adjustment"])
            except Exception:
                pass

        return SimulationVerdict(
            scenario=scenario,
            compatible=compatible,
            risk_score=risk,
            violated_constraints=violated,
            predicted_reaction=predicted,
            recommended_adjustment=adjustment,
            alerts=alerts,
            supporting_memory_ids=sup_ids,
        )
