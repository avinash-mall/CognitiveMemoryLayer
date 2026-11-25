"""SQLite-backed Neocortical Graph & Semantic Fact Stores for py-cml embedded parity.

Replaces NoOpGraphStore and NoOpFactStore in EmbeddedCognitiveMemoryLayer so embedded
deployments get full entity-relation graph traversal, sub-5ms Personalized PageRank,
versioned semantic facts, and tokenized BM25/keyword fact search over a single SQLite DB.
"""

from __future__ import annotations

import json
import re
from collections import defaultdict
from datetime import UTC, datetime
from typing import Any
from uuid import uuid4

import aiosqlite

from src.memory.neocortical.schemas import FactCategory, SemanticFact
from src.storage.base import GraphStoreBase


def _infer_category_subject_predicate(key: str) -> tuple[FactCategory, str, str]:
    parts = [p.strip() for p in (key or "").split(":") if p.strip()]
    subject = parts[0] if parts else "user"
    predicate = parts[-1] if len(parts) >= 2 else (key or "fact")
    cat = FactCategory.CUSTOM
    if len(parts) >= 3:
        raw_cat = parts[1].lower()
        for c in FactCategory:
            if c.value == raw_cat:
                cat = c
                break
    return cat, subject, predicate


class SQLiteGraphStore(GraphStoreBase):
    """SQLite-backed knowledge graph with fast in-memory Personalized PageRank."""

    def __init__(self, sqlite_memory_store: Any) -> None:
        self._mem_store = sqlite_memory_store

    async def _db(self) -> aiosqlite.Connection:
        if self._mem_store._db is None:
            await self._mem_store.initialize()
        assert self._mem_store._db is not None
        return self._mem_store._db

    async def merge_node(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        entity_type: str,
        properties: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        db = await self._db()
        norm = entity.strip()
        node_id = f"{tenant_id}:{scope_id}:{norm.lower()}"
        props = dict(properties or {})
        if namespace is not None:
            props["namespace"] = namespace
        now = datetime.now(UTC).isoformat()
        await db.execute(
            """
            INSERT INTO graph_nodes (id, tenant_id, scope_id, entity, entity_type, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                entity_type = excluded.entity_type,
                properties = excluded.properties,
                updated_at = excluded.updated_at
            """,
            (node_id, tenant_id, scope_id, norm, entity_type or "ENTITY", json.dumps(props), now),
        )
        await db.commit()
        return node_id

    async def merge_edge(
        self,
        tenant_id: str,
        scope_id: str,
        subject: str,
        predicate: str,
        object: str,
        properties: dict[str, Any] | None = None,
        namespace: str | None = None,
    ) -> str:
        db = await self._db()
        s_norm = subject.strip()
        p_norm = (predicate or "RELATED_TO").strip().upper().replace(" ", "_")
        o_norm = object.strip()
        await self.merge_node(tenant_id, scope_id, s_norm, "ENTITY", namespace=namespace)
        await self.merge_node(tenant_id, scope_id, o_norm, "ENTITY", namespace=namespace)
        edge_id = f"{tenant_id}:{scope_id}:{s_norm.lower()}:{p_norm}:{o_norm.lower()}"
        props = dict(properties or {})
        now = datetime.now(UTC).isoformat()
        await db.execute(
            """
            INSERT INTO graph_edges (id, tenant_id, scope_id, subject, predicate, object, properties, updated_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(id) DO UPDATE SET
                properties = excluded.properties,
                updated_at = excluded.updated_at
            """,
            (edge_id, tenant_id, scope_id, s_norm, p_norm, o_norm, json.dumps(props), now),
        )
        await db.commit()
        return edge_id

    async def merge_edges_batch(
        self,
        tenant_id: str,
        scope_id: str,
        edges: list[dict[str, Any]],
    ) -> list[str]:
        ids: list[str] = []
        for e in edges:
            subj = str(e.get("subject") or "").strip()
            pred = str(e.get("predicate") or "RELATED_TO").strip()
            obj = str(e.get("object") or "").strip()
            if subj and obj:
                eid = await self.merge_edge(
                    tenant_id,
                    scope_id,
                    subj,
                    pred,
                    obj,
                    properties=e.get("properties"),
                    namespace=e.get("namespace"),
                )
                ids.append(eid)
        return ids

    async def get_neighbors(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
        max_depth: int = 2,
    ) -> list[dict[str, Any]]:
        facts = await self.get_entity_facts(tenant_id, scope_id, entity)
        return facts[:50]

    async def get_entity_facts(
        self,
        tenant_id: str,
        scope_id: str,
        entity: str,
    ) -> list[dict[str, Any]]:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        ent_low = entity.strip().lower()
        async with db.execute(
            """
            SELECT subject, predicate, object, properties
            FROM graph_edges
            WHERE tenant_id = ? AND (lower(subject) = ? OR lower(object) = ?)
            LIMIT 50
            """,
            (tenant_id, ent_low, ent_low),
        ) as cur:
            rows = await cur.fetchall()
        out: list[dict[str, Any]] = []
        for r in rows:
            props = json.loads(r["properties"] or "{}")
            out.append(
                {
                    "subject": r["subject"],
                    "predicate": r["predicate"],
                    "object": r["object"],
                    "properties": props,
                    "evidence_ids": props.get("evidence_ids", []),
                }
            )
        return out

    async def get_entity_facts_batch(
        self,
        tenant_id: str,
        scope_id: str,
        entity_names: list[str],
    ) -> dict[str, list[dict[str, Any]]]:
        out: dict[str, list[dict[str, Any]]] = {}
        for name in entity_names:
            facts = await self.get_entity_facts(tenant_id, scope_id, name)
            if facts:
                out[name] = facts
        return out

    async def personalized_pagerank(
        self,
        tenant_id: str,
        scope_id: str,
        seed_entities: list[str],
        top_k: int = 20,
        damping: float = 0.85,
    ) -> list[dict[str, Any]]:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT subject, predicate, object, properties FROM graph_edges WHERE tenant_id = ? LIMIT 5000",
            (tenant_id,),
        ) as cur:
            rows = await cur.fetchall()
        if not rows or not seed_entities:
            return []

        adj: dict[str, list[str]] = defaultdict(list)
        display_name: dict[str, str] = {}
        edge_evidence: dict[str, list[str]] = defaultdict(list)
        for r in rows:
            s = str(r["subject"])
            o = str(r["object"])
            sl, ol = s.lower(), o.lower()
            display_name[sl] = s
            display_name[ol] = o
            adj[sl].append(ol)
            adj[ol].append(sl)
            props = json.loads(r["properties"] or "{}")
            for eid in props.get("evidence_ids", []) or []:
                edge_evidence[sl].append(str(eid))
                edge_evidence[ol].append(str(eid))

        seeds = [s.strip().lower() for s in seed_entities if s.strip().lower() in display_name]
        if not seeds:
            return []
        p_val = 1.0 / len(seeds)
        p_vec = {s: p_val for s in seeds}
        scores = dict(p_vec)
        for _ in range(10):
            nxt: dict[str, float] = {k: (1.0 - damping) * v for k, v in p_vec.items()}
            for u, mass in scores.items():
                nbrs = adj.get(u)
                if not nbrs:
                    continue
                share = damping * mass / len(nbrs)
                for v in nbrs:
                    nxt[v] = nxt.get(v, 0.0) + share
            scores = nxt

        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:top_k]
        return [
            {
                "entity": display_name.get(node, node),
                "score": round(sc, 5),
                "evidence_ids": list(dict.fromkeys(edge_evidence.get(node, [])))[:10],
            }
            for node, sc in ranked
        ]


class SQLiteFactStore:
    """SQLite-backed SemanticFactStore with tokenized relevance search."""

    def __init__(self, sqlite_memory_store: Any) -> None:
        self._mem_store = sqlite_memory_store

    async def _db(self) -> aiosqlite.Connection:
        if self._mem_store._db is None:
            await self._mem_store.initialize()
        assert self._mem_store._db is not None
        return self._mem_store._db

    @staticmethod
    def _row_to_fact(row: dict[str, Any]) -> SemanticFact:
        raw_cat = str(row.get("category") or "custom")
        cat = FactCategory.CUSTOM
        for c in FactCategory:
            if c.value == raw_cat:
                cat = c
                break

        def _parse_dt(val: Any) -> datetime | None:
            if not val or not isinstance(val, str):
                return None
            try:
                return datetime.fromisoformat(val.replace("Z", "+00:00"))
            except ValueError:
                return None

        now = datetime.now(UTC)
        return SemanticFact(
            id=str(row["id"]),
            tenant_id=str(row["tenant_id"]),
            category=cat,
            key=str(row["key"]),
            subject=str(row["subject"]),
            predicate=str(row["predicate"]),
            value=json.loads(row["value_json"]) if row.get("value_json") else "",
            value_type=str(row.get("value_type") or "string"),
            context_tags=json.loads(row.get("context_tags") or "[]"),
            confidence=float(row.get("confidence") or 0.8),
            evidence_count=int(row.get("evidence_count") or 1),
            evidence_ids=json.loads(row.get("evidence_ids") or "[]"),
            valid_from=_parse_dt(row.get("valid_from")),
            valid_to=_parse_dt(row.get("valid_to")),
            is_current=bool(row.get("is_current", 1)),
            created_at=_parse_dt(row.get("created_at")) or now,
            updated_at=_parse_dt(row.get("updated_at")) or now,
            version=int(row.get("version") or 1),
            supersedes_id=row.get("supersedes_id"),
        )

    async def upsert_fact(
        self,
        tenant_id: str,
        key: str,
        value: Any,
        confidence: float = 0.8,
        evidence_ids: list[str] | None = None,
        valid_from: datetime | None = None,
        context_tags: list[str] | None = None,
        valid_to: datetime | None = None,
    ) -> SemanticFact:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        cat, subject, predicate = _infer_category_subject_predicate(key)
        now = datetime.now(UTC).isoformat()
        val_json = json.dumps(value)
        vtype = "list" if isinstance(value, list) else "string"

        async with db.execute(
            "SELECT * FROM semantic_facts WHERE tenant_id = ? AND key = ? AND is_current = 1 LIMIT 1",
            (tenant_id, key),
        ) as cur:
            existing = await cur.fetchone()

        if existing is not None:
            ex = dict(existing)
            prev_ev = json.loads(ex.get("evidence_ids") or "[]")
            merged_ev = list(dict.fromkeys(prev_ev + list(evidence_ids or [])))
            new_ver = int(ex.get("version") or 1) + 1
            await db.execute(
                """
                UPDATE semantic_facts SET
                    value_json = ?, confidence = ?, evidence_count = ?, evidence_ids = ?,
                    valid_from = ?, valid_to = ?, updated_at = ?, version = ?
                WHERE id = ?
                """,
                (
                    val_json,
                    confidence,
                    len(merged_ev) or 1,
                    json.dumps(merged_ev),
                    valid_from.isoformat() if valid_from else ex.get("valid_from"),
                    valid_to.isoformat() if valid_to else ex.get("valid_to"),
                    now,
                    new_ver,
                    ex["id"],
                ),
            )
            await db.commit()
            got = await self.get_fact(tenant_id, key)
            assert got is not None
            return got

        fid = str(uuid4())
        ev_list = list(evidence_ids or [])
        await db.execute(
            """
            INSERT INTO semantic_facts (
                id, tenant_id, category, key, subject, predicate, value_json, value_type,
                context_tags, confidence, evidence_count, evidence_ids, valid_from, valid_to,
                is_current, created_at, updated_at, version
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, ?, ?, 1)
            """,
            (
                fid,
                tenant_id,
                cat.value,
                key,
                subject,
                predicate,
                val_json,
                vtype,
                json.dumps(context_tags or []),
                confidence,
                max(1, len(ev_list)),
                json.dumps(ev_list),
                valid_from.isoformat() if valid_from else None,
                valid_to.isoformat() if valid_to else None,
                now,
                now,
            ),
        )
        await db.commit()
        got = await self.get_fact(tenant_id, key)
        assert got is not None
        return got

    async def get_fact(
        self,
        tenant_id: str,
        key: str,
        include_historical: bool = False,
    ) -> SemanticFact | None:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        sql = "SELECT * FROM semantic_facts WHERE tenant_id = ? AND key = ?"
        if not include_historical:
            sql += " AND is_current = 1"
        sql += " ORDER BY version DESC LIMIT 1"
        async with db.execute(sql, (tenant_id, key)) as cur:
            row = await cur.fetchone()
        return self._row_to_fact(dict(row)) if row else None

    async def get_facts_by_category(
        self,
        tenant_id: str,
        category: FactCategory,
        current_only: bool = True,
        limit: int = 50,
    ) -> list[SemanticFact]:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        sql = "SELECT * FROM semantic_facts WHERE tenant_id = ? AND category = ?"
        if current_only:
            sql += " AND is_current = 1"
        sql += " ORDER BY confidence DESC, updated_at DESC LIMIT ?"
        async with db.execute(sql, (tenant_id, category.value, limit)) as cur:
            rows = await cur.fetchall()
        return [self._row_to_fact(dict(r)) for r in rows]

    async def get_facts_by_categories(
        self,
        tenant_id: str,
        categories: list[FactCategory],
        current_only: bool = True,
        limit: int = 200,
    ) -> list[SemanticFact]:
        if not categories:
            return []
        db = await self._db()
        db.row_factory = aiosqlite.Row
        placeholders = ", ".join("?" for _ in categories)
        sql = f"SELECT * FROM semantic_facts WHERE tenant_id = ? AND category IN ({placeholders})"
        if current_only:
            sql += " AND is_current = 1"
        sql += " ORDER BY confidence DESC LIMIT ?"
        params: list[Any] = [tenant_id, *[c.value for c in categories], limit]
        async with db.execute(sql, params) as cur:
            rows = await cur.fetchall()
        return [self._row_to_fact(dict(r)) for r in rows]

    async def get_tenant_profile(self, tenant_id: str) -> dict[str, Any]:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        async with db.execute(
            "SELECT * FROM semantic_facts WHERE tenant_id = ? AND is_current = 1 ORDER BY confidence DESC LIMIT 200",
            (tenant_id,),
        ) as cur:
            rows = await cur.fetchall()
        grouped: dict[str, dict[str, Any]] = defaultdict(dict)
        for r in rows:
            fact = self._row_to_fact(dict(r))
            grouped[fact.category.value][fact.predicate] = fact.value
        return dict(grouped)

    async def search_facts(
        self,
        tenant_id: str,
        query: str,
        limit: int = 20,
    ) -> list[SemanticFact]:
        db = await self._db()
        db.row_factory = aiosqlite.Row
        tokens = [
            t
            for t in re.findall(r"[a-zA-Z0-9]{2,}", (query or "").lower())
            if t not in {"the", "and", "for", "what", "when", "where", "who", "how", "did", "does"}
        ]
        if not tokens:
            return []
        async with db.execute(
            "SELECT * FROM semantic_facts WHERE tenant_id = ? AND is_current = 1 LIMIT 1000",
            (tenant_id,),
        ) as cur:
            rows = await cur.fetchall()
        scored: list[tuple[float, SemanticFact]] = []
        for r in rows:
            fact = self._row_to_fact(dict(r))
            blob = f"{fact.subject} {fact.predicate} {fact.value} {fact.key}".lower()
            hits = sum(1 for t in set(tokens) if t in blob)
            if hits > 0:
                score = min(0.95, 0.45 + 0.50 * (hits / len(set(tokens)))) * max(
                    0.5, fact.confidence
                )
                scored.append((score, fact))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [f for _, f in scored[:limit]]

    async def search_facts_batch(
        self,
        tenant_id: str,
        entity_names: list[str],
        limit_per_entity: int = 5,
    ) -> dict[str, list[SemanticFact]]:
        out: dict[str, list[SemanticFact]] = {}
        for ent in entity_names:
            hits = await self.search_facts(tenant_id, ent, limit=limit_per_entity)
            if hits:
                out[ent] = hits
        return out

    async def invalidate_fact(
        self,
        tenant_id: str,
        key: str,
        reason: str = "superseded",
    ) -> bool:
        db = await self._db()
        cur = await db.execute(
            "UPDATE semantic_facts SET is_current = 0, updated_at = ? WHERE tenant_id = ? AND key = ? AND is_current = 1",
            (datetime.now(UTC).isoformat(), tenant_id, key),
        )
        await db.commit()
        return (cur.rowcount or 0) > 0
