"""Basal Ganglia & Cerebellar Procedural Skill Memory ("Knowing How").

Compiles repeated agent action-outcome trajectories, tool workflows, and user
corrections into reusable ``ProceduralSkill`` playbooks stored in Postgres or SQLite,
with synaptic weights modulated by Dopaminergic Reward Prediction Error (RPE).
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from ..utils.similarity import cosine_similarity, jaccard, word_set


@dataclass
class ProceduralSkill:
    """Reusable procedural skill / guardrail playbook."""

    skill_key: str
    tenant_id: str
    trigger_pattern: str
    steps: list[str] = field(default_factory=list)
    pitfalls: list[str] = field(default_factory=list)
    success_count: int = 1
    failure_count: int = 0
    synaptic_weight: float = 1.0
    embedding: list[float] | None = None

    def format_playbook_line(self) -> str:
        parts = [f"When: {self.trigger_pattern}"]
        if self.steps:
            parts.append(f"Do: {'; '.join(self.steps[:3])}")
        if self.pitfalls:
            parts.append(f"Avoid: {'; '.join(self.pitfalls[:3])}")
        total = max(1, self.success_count + self.failure_count)
        rate = self.success_count / total
        parts.append(f"(weight={self.synaptic_weight:.2f}, reliability={rate:.0%})")
        return " | ".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "skill_key": self.skill_key,
            "trigger_pattern": self.trigger_pattern,
            "steps": self.steps,
            "pitfalls": self.pitfalls,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "synaptic_weight": self.synaptic_weight,
            "embedding": self.embedding,
        }


class ProceduralMemoryEngine:
    """Manages Basal Ganglia / Cerebellar Procedural Skills across Postgres & SQLite."""

    def __init__(
        self,
        embedding_client: Any | None = None,
        pg_session_factory: Any | None = None,
        sqlite_store: Any | None = None,
    ) -> None:
        self.embedding_client = embedding_client
        self.pg_session_factory = pg_session_factory
        self.sqlite_store = sqlite_store
        self._cache: dict[str, dict[str, ProceduralSkill]] = {}
        self._hydrated: set[str] = set()

    @staticmethod
    def make_skill_key(trigger_pattern: str) -> str:
        norm = " ".join((trigger_pattern or "").strip().lower().split())
        return f"skill:{hashlib.sha1(norm.encode('utf-8')).hexdigest()[:20]}"

    async def _ensure_hydrated(self, tenant_id: str) -> dict[str, ProceduralSkill]:
        if tenant_id in self._hydrated:
            return self._cache.setdefault(tenant_id, {})
        self._hydrated.add(tenant_id)
        bucket = self._cache.setdefault(tenant_id, {})

        if self.pg_session_factory is not None:
            try:
                from sqlalchemy import select

                from ..storage.models import ProceduralSkillModel

                async with self.pg_session_factory() as session:
                    res = await session.execute(
                        select(ProceduralSkillModel)
                        .where(ProceduralSkillModel.tenant_id == tenant_id)
                        .limit(500)
                    )
                    for row in res.scalars().all():
                        bucket[row.skill_key] = ProceduralSkill(
                            skill_key=row.skill_key,
                            tenant_id=tenant_id,
                            trigger_pattern=row.trigger_pattern,
                            steps=list(row.steps or []),
                            pitfalls=list(row.pitfalls or []),
                            success_count=int(row.success_count or 1),
                            failure_count=int(row.failure_count or 0),
                            synaptic_weight=float(row.synaptic_weight or 1.0),
                            embedding=list(row.embedding_json) if row.embedding_json else None,
                        )
            except Exception:
                pass

        if self.sqlite_store is not None and hasattr(self.sqlite_store, "load_procedural_skills"):
            try:
                rows = await self.sqlite_store.load_procedural_skills(tenant_id)
                for r in rows:
                    bucket[r["skill_key"]] = ProceduralSkill(
                        skill_key=r["skill_key"],
                        tenant_id=tenant_id,
                        trigger_pattern=r["trigger_pattern"],
                        steps=list(r.get("steps") or []),
                        pitfalls=list(r.get("pitfalls") or []),
                        success_count=int(r.get("success_count", 1)),
                        failure_count=int(r.get("failure_count", 0)),
                        synaptic_weight=float(r.get("synaptic_weight", 1.0)),
                        embedding=r.get("embedding"),
                    )
            except Exception:
                pass

        return bucket

    async def upsert_skill(
        self,
        tenant_id: str,
        trigger_pattern: str,
        *,
        steps: list[str] | None = None,
        pitfalls: list[str] | None = None,
        weight_delta: float = 0.0,
        is_success: bool = True,
        skill_key: str | None = None,
    ) -> ProceduralSkill:
        bucket = await self._ensure_hydrated(tenant_id)
        skey = skill_key or self.make_skill_key(trigger_pattern)
        existing = bucket.get(skey)

        if existing is not None:
            for st in steps or []:
                if st and st not in existing.steps:
                    existing.steps.append(st)
            for pf in pitfalls or []:
                if pf and pf not in existing.pitfalls:
                    existing.pitfalls.append(pf)
            if is_success:
                existing.success_count += 1
            else:
                existing.failure_count += 1
            existing.synaptic_weight = max(0.2, min(2.5, existing.synaptic_weight + weight_delta))
            skill = existing
        else:
            emb: list[float] | None = None
            if self.embedding_client is not None:
                try:
                    emb_txt = (
                        f"{trigger_pattern} {' '.join(steps or [])} {' '.join(pitfalls or [])}"
                    )
                    emb = (await self.embedding_client.embed(emb_txt.strip())).embedding
                except Exception:
                    emb = None
            skill = ProceduralSkill(
                skill_key=skey,
                tenant_id=tenant_id,
                trigger_pattern=trigger_pattern.strip(),
                steps=list(steps or []),
                pitfalls=list(pitfalls or []),
                success_count=1 if is_success else 0,
                failure_count=0 if is_success else 1,
                synaptic_weight=max(0.2, min(2.5, 1.0 + weight_delta)),
                embedding=emb,
            )
            bucket[skey] = skill

        await self._persist_skill(tenant_id, skill)
        return skill

    async def _persist_skill(self, tenant_id: str, skill: ProceduralSkill) -> None:
        if self.pg_session_factory is not None:
            try:
                from sqlalchemy import and_, select

                from ..storage.models import ProceduralSkillModel

                now = datetime.now(UTC).replace(tzinfo=None)
                async with self.pg_session_factory() as session:
                    res = await session.execute(
                        select(ProceduralSkillModel).where(
                            and_(
                                ProceduralSkillModel.tenant_id == tenant_id,
                                ProceduralSkillModel.skill_key == skill.skill_key,
                            )
                        )
                    )
                    row = res.scalar_one_or_none()
                    if row is not None:
                        row.trigger_pattern = skill.trigger_pattern
                        row.steps = skill.steps
                        row.pitfalls = skill.pitfalls
                        row.success_count = skill.success_count
                        row.failure_count = skill.failure_count
                        row.synaptic_weight = skill.synaptic_weight
                        row.embedding_json = skill.embedding
                        row.updated_at = now
                    else:
                        session.add(
                            ProceduralSkillModel(
                                tenant_id=tenant_id,
                                skill_key=skill.skill_key,
                                trigger_pattern=skill.trigger_pattern,
                                steps=skill.steps,
                                pitfalls=skill.pitfalls,
                                success_count=skill.success_count,
                                failure_count=skill.failure_count,
                                synaptic_weight=skill.synaptic_weight,
                                embedding_json=skill.embedding,
                                updated_at=now,
                            )
                        )
                    await session.commit()
            except Exception:
                pass

        if self.sqlite_store is not None and hasattr(self.sqlite_store, "save_procedural_skill"):
            try:
                await self.sqlite_store.save_procedural_skill(tenant_id, skill.to_dict())
            except Exception:
                pass

    async def retrieve_skills(
        self,
        tenant_id: str,
        query: str,
        query_embedding: list[float] | None = None,
        top_k: int = 3,
    ) -> list[ProceduralSkill]:
        bucket = await self._ensure_hydrated(tenant_id)
        if not bucket:
            return []
        q_ws = word_set(query or "")
        scored: list[tuple[float, ProceduralSkill]] = []
        for sk in bucket.values():
            s_ws = word_set(f"{sk.trigger_pattern} {' '.join(sk.pitfalls)}")
            inter = len(q_ws & s_ws)
            lex = (
                0.70 * (inter / len(q_ws)) + 0.30 * jaccard(q_ws, s_ws) if (q_ws and s_ws) else 0.0
            )
            cos = (
                cosine_similarity(query_embedding, sk.embedding)
                if (query_embedding and sk.embedding and len(query_embedding) == len(sk.embedding))
                else 0.0
            )
            rel = max(lex, cos)
            if rel >= 0.22:
                scored.append((rel * sk.synaptic_weight, sk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [sk for _, sk in scored[:top_k]]
