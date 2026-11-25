"""Ventral Tegmental Area (VTA) Dopaminergic Reward Prediction Error (RPE) Engine.

Computes phasic prediction error ``delta = outcome_reward - expected_reward`` on agent
outcomes or explicit user feedback:
- **Positive RPE (delta > +0.05)**: strengthens relied-on episodic confidence/importance,
  potentiates linked Cortical Hyperedge synaptic weights (LTP), and reinforces Procedural Skills.
- **Negative RPE (delta < -0.05 or explicit correction)**: destabilizes relied-on traces
  into ``labile = True`` for reconsolidation, depresses hyperedge weights (LTD), stores
  the corrective memory, and compiles a Cerebellar procedural guardrail pitfall.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from uuid import UUID

from ..core.enums import MemorySource, MemoryType
from ..core.schemas import MemoryRecordCreate, Provenance
from ..memory.procedural import ProceduralMemoryEngine


@dataclass
class DopamineRPEResult:
    """Outcome of processing a Dopaminergic RPE feedback signal."""

    rpe_delta: float
    memories_reinforced: int
    memories_marked_labile: int
    hyperedges_modulated: int
    procedural_skill_key: str | None = None
    correction_memory_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "rpe_delta": round(self.rpe_delta, 4),
            "memories_reinforced": self.memories_reinforced,
            "memories_marked_labile": self.memories_marked_labile,
            "hyperedges_modulated": self.hyperedges_modulated,
            "procedural_skill_key": self.procedural_skill_key,
            "correction_memory_id": self.correction_memory_id,
        }


class DopamineRPEEngine:
    """Applies VTA Dopaminergic RPE synaptic updates across Episodic, Hypergraph & Procedural stores."""

    def __init__(
        self,
        episodic_store: Any,
        neocortical_store: Any,
        procedural_engine: ProceduralMemoryEngine,
        embedding_client: Any | None = None,
    ) -> None:
        self.episodic_store = episodic_store
        self.neocortical_store = neocortical_store
        self.procedural = procedural_engine
        self.embedding_client = embedding_client

    async def process_feedback(
        self,
        tenant_id: str,
        *,
        outcome_reward: float,
        expected_reward: float = 0.5,
        query: str | None = None,
        memory_ids: list[str] | None = None,
        correction_text: str | None = None,
        preferred_steps: list[str] | None = None,
        session_id: str | None = None,
    ) -> DopamineRPEResult:
        reward_clamped = max(-1.0, min(1.0, float(outcome_reward)))
        if correction_text and reward_clamped > -0.2:
            reward_clamped = -0.5
        delta = reward_clamped - float(expected_reward)

        reinforced = 0
        labile_marked = 0
        mem_id_strs = [str(m) for m in (memory_ids or []) if m]

        for raw_id in mem_id_strs:
            try:
                uid = UUID(raw_id)
                rec = await self.episodic_store.get_by_id(uid)
                if rec is None or rec.tenant_id != tenant_id:
                    continue
                if delta >= 0.05:
                    new_conf = min(1.0, round(rec.confidence + 0.08 * delta, 4))
                    new_imp = min(1.0, round(rec.importance + 0.06 * delta, 4))
                    await self.episodic_store.update(
                        uid,
                        {"confidence": new_conf, "importance": new_imp, "labile": False},
                        increment_version=False,
                    )
                    reinforced += 1
                elif delta <= -0.05:
                    new_conf = max(0.15, round(rec.confidence + 0.12 * delta, 4))
                    meta = dict(rec.metadata or {})
                    if correction_text:
                        meta["rpe_conflict_note"] = correction_text[:240]
                    await self.episodic_store.update(
                        uid,
                        {"confidence": new_conf, "labile": True, "metadata": meta},
                        increment_version=False,
                    )
                    labile_marked += 1
            except Exception:
                continue

        hyperedges_mod = 0
        hypergraph = getattr(self.neocortical_store, "hypergraph", None)
        if hypergraph is not None and mem_id_strs and abs(delta) >= 0.05:
            hyperedges_mod = await hypergraph.modulate_synaptic_weights(
                tenant_id, mem_id_strs, delta=0.25 * delta
            )

        skill_key: str | None = None
        if query and (abs(delta) >= 0.1 or correction_text or preferred_steps):
            is_pos = delta >= 0.0 and not correction_text
            pitfalls = [correction_text.strip()] if correction_text else []
            skill = await self.procedural.upsert_skill(
                tenant_id=tenant_id,
                trigger_pattern=query.strip(),
                steps=preferred_steps,
                pitfalls=pitfalls,
                weight_delta=0.20 * delta if is_pos else 0.15,
                is_success=is_pos,
            )
            skill_key = skill.skill_key

        corr_mem_id: str | None = None
        if correction_text and correction_text.strip():
            emb: list[float] | None = None
            if self.embedding_client is not None:
                try:
                    emb = (await self.embedding_client.embed(correction_text.strip())).embedding
                except Exception:
                    emb = None
            created = await self.episodic_store.upsert(
                MemoryRecordCreate(
                    tenant_id=tenant_id,
                    context_tags=["conversation", "rpe_correction"],
                    source_session_id=session_id,
                    type=MemoryType.CONSTRAINT,
                    text=correction_text.strip(),
                    embedding=emb,
                    confidence=0.95,
                    importance=0.92,
                    metadata={
                        "rpe_correction": True,
                        "supersedes_memory_ids": mem_id_strs,
                        "constraints": [
                            {
                                "constraint_type": "policy",
                                "provenance": mem_id_strs[:4],
                            }
                        ],
                    },
                    provenance=Provenance(
                        source=MemorySource.USER_EXPLICIT,
                        evidence_refs=mem_id_strs[:4],
                    ),
                )
            )
            corr_mem_id = str(created.id)

        return DopamineRPEResult(
            rpe_delta=delta,
            memories_reinforced=reinforced,
            memories_marked_labile=labile_marked,
            hyperedges_modulated=hyperedges_mod,
            procedural_skill_key=skill_key,
            correction_memory_id=corr_mem_id,
        )
