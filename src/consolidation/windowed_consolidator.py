"""Windowed Micro-Sleep Consolidator (Complementary Learning Systems + Conway 3-Tier SMS).

Processes ordered windows of verbatim hippocampal turns (default 18 turns/call = ~18x fewer
LLM calls than per-turn System-2 extraction while preserving full cross-turn coreference)
to synthesize:
1. Coref-resolved Semantic Facts (Neocortical FactStore)
2. N-ary Cortical Hyperedges (CorticalHypergraphEngine)
3. Grounded Proustian Situational Cue Pointers (cue record -> verbatim parent_episode_id)
4. Conway Tier-2 Autobiographical Chapter Gist Nodes
5. Conway Tier-3 Persistent DMN Conceptual-Self Deltas
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import os
import re
import sqlite3
import threading
import zlib
from datetime import UTC, datetime
from typing import Any

from ..core.enums import MemorySource, MemoryType
from ..core.schemas import EntityMention, MemoryRecord, MemoryRecordCreate, Provenance
from ..extraction.temporal_resolver import resolve_temporal_references
from ..memory.dmn import PersistentDMNStore
from ..memory.neocortical.hypergraph import CorticalHypergraphEngine, HyperedgeEngram
from ..memory.prospective import ProspectiveMemoryEngine
from ..utils.logging_config import get_logger

logger = get_logger(__name__)

_PROMPT_VERSION = "v3.0-conway-hypergraph-2"

_WINDOW_CONSOLIDATION_PROMPT = """Analyze this chronological dialogue window and extract structured memory representations.
Resolve all pronouns ("I", "he", "she", "they", "that", "there") to exact speaker/entity names, and resolve relative dates ("yesterday", "last month", "last year", "two weeks ago") against each turn's [YYYY-MM-DD] spoken date using accurate granularity (e.g. "2022", "April 2023", or "2023-05-07").
Keep output compact and high-signal: at most 5 facts, at most 3 hyperedges, at most 2 proustian_pointers, 1 chapter_gist, and at most 2 persona_deltas (max 3 short items per trait list).

Dialogue Window:
{window_text}

Return ONLY valid JSON matching this exact schema (no markdown fences, concise strings):
{{
  "facts": [
    {{"subject": "Name", "predicate": "concise_attribute_or_relation", "value": "exact resolved fact", "category": "identity|occupation|preference|relationship|goal|state|value|causal|attribute", "turn_indices": [0], "event_date_display": ""}}
  ],
  "hyperedges": [
    {{"actors": ["Name"], "entities": ["Item/Org/Topic"], "event_summary": "1-sentence coref-resolved event or state", "temporal_anchor": "resolved date/period or empty", "location": "place or empty", "causal_state": "why it happened or active constraint", "prospective_cues": ["concrete future remark or decision topic where this matters"], "turn_indices": [0, 1]}}
  ],
  "proustian_pointers": [
    {{"speaker": "Name", "turn_index": 0, "implicit_lesson_or_boundary": "hidden fear, allergy, habit, trauma, rule, value, or goal from that turn", "future_cue_scenarios": ["2-3 everyday remarks or situations with ZERO word overlap where this past experience explains their reaction"]}}
  ],
  "chapter_gist": {{
    "title": "Short chapter title",
    "summary": "2-sentence narrative summary preserving specific names, counts, items, and dates",
    "participants": ["Name"]
  }},
  "persona_deltas": [
    {{"speaker": "Name", "traits": [], "values": [], "boundaries_fears": [], "goals": [], "habits": [], "career_identity": []}}
  ]
}}"""


class _WindowExtractionCache:
    """Thread-safe content-addressable SQLite memoization for deterministic window extraction."""

    def __init__(self, path: str) -> None:
        self.path = path
        self._lock = threading.Lock()
        self._ready = False

    def _init_db(self) -> None:
        if self._ready or not self.path:
            return
        try:
            with sqlite3.connect(self.path, timeout=15.0) as conn:
                conn.execute("PRAGMA journal_mode=WAL")
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS window_extractions (
                        cache_key TEXT PRIMARY KEY,
                        payload_json TEXT NOT NULL,
                        created_at TEXT NOT NULL
                    )
                    """
                )
            self._ready = True
        except Exception:
            self._ready = False

    def get(self, key: str) -> dict[str, Any] | None:
        if os.environ.get("CML_WINDOW_CACHE_DISABLED", "").strip() in ("1", "true"):
            return None
        with self._lock:
            self._init_db()
            if not self._ready:
                return None
            try:
                with sqlite3.connect(self.path, timeout=15.0) as conn:
                    cur = conn.execute(
                        "SELECT payload_json FROM window_extractions WHERE cache_key = ?",
                        (key,),
                    )
                    row = cur.fetchone()
                return json.loads(row[0]) if row else None
            except Exception:
                return None

    def put(self, key: str, payload: dict[str, Any]) -> None:
        if os.environ.get("CML_WINDOW_CACHE_DISABLED", "").strip() in ("1", "true"):
            return
        with self._lock:
            self._init_db()
            if not self._ready:
                return
            try:
                with sqlite3.connect(self.path, timeout=15.0) as conn:
                    conn.execute(
                        """
                        INSERT OR REPLACE INTO window_extractions (cache_key, payload_json, created_at)
                        VALUES (?, ?, ?)
                        """,
                        (
                            key,
                            json.dumps(payload, ensure_ascii=False),
                            datetime.now(UTC).isoformat(),
                        ),
                    )
            except Exception:
                pass


_CACHE_SINGLETON = _WindowExtractionCache(
    os.environ.get("CML_WINDOW_CACHE_PATH", "/tmp/cml_window_consolidation_cache_v3.sqlite")
)


def _extract_json_dict(raw: str) -> dict[str, Any] | None:
    if not raw:
        return None
    txt = raw.strip()
    if "<think>" in txt and "</think>" in txt:
        txt = txt.split("</think>")[-1].strip()
    txt = re.sub(r"^```(?:json)?\s*", "", txt)
    txt = re.sub(r"\s*```$", "", txt)
    start = txt.find("{")
    end = txt.rfind("}")
    if start == -1 or end == -1 or end <= start:
        return None
    candidate = txt[start : end + 1]
    try:
        parsed = json.loads(candidate)
        return parsed if isinstance(parsed, dict) else None
    except json.JSONDecodeError:
        # Strip trailing commas before ] or }
        repaired = re.sub(r",\s*([}\]])", r"\1", candidate)
        try:
            parsed = json.loads(repaired)
            return parsed if isinstance(parsed, dict) else None
        except json.JSONDecodeError:
            return None


class WindowedMicroSleepConsolidator:
    """Runs multi-turn Micro-Sleep Consolidation across Hippocampal + Neocortical + DMN stores."""

    def __init__(
        self,
        episodic_store: Any,
        neocortical_store: Any,
        embedding_client: Any,
        llm_client: Any | None = None,
        dmn_store: PersistentDMNStore | None = None,
        window_size: int = 24,
        max_concurrency: int = 1,
    ) -> None:
        self.episodic_store = episodic_store
        self.neocortical_store = neocortical_store
        self.embedding_client = embedding_client
        self.llm_client = llm_client
        self.window_size = max(4, window_size)
        self.max_concurrency = max(1, max_concurrency)
        if dmn_store is not None:
            self.dmn_store = dmn_store
        else:
            pg_sf = getattr(episodic_store, "session_factory", None)
            sq_st = episodic_store if hasattr(episodic_store, "save_dmn_profile") else None
            self.dmn_store = PersistentDMNStore(pg_session_factory=pg_sf, sqlite_store=sq_st)

    @staticmethod
    def _is_verbatim_unconsolidated(rec: MemoryRecord) -> bool:
        meta = rec.metadata or {}
        if meta.get("window_consolidated"):
            return False
        if meta.get("proustian_cue_pointer") or meta.get("prospective_implication"):
            return False
        if meta.get("conway_tier") in ("chapter_gist", "dmn_self"):
            return False
        return rec.type in (
            MemoryType.EPISODIC_EVENT,
            MemoryType.CONVERSATION,
            MemoryType.MESSAGE,
            MemoryType.CONSTRAINT,
            MemoryType.OBSERVATION,
        )

    @staticmethod
    def _sort_episodes_chronologically(records: list[MemoryRecord]) -> list[MemoryRecord]:
        def _key(r: MemoryRecord) -> tuple[int, float, int, str]:
            meta = r.metadata or {}
            raw_tidx = meta.get("turn_idx", meta.get("turn_index", meta.get("dia_idx")))
            if raw_tidx is not None:
                s_idx = int(meta.get("session_idx", 0) or 0)
                return (0, float(s_idx), int(raw_tidx), (r.text or "")[:64])
            ts = r.timestamp.timestamp() if hasattr(r.timestamp, "timestamp") else 0.0
            return (1, ts, 0, (r.text or "")[:64])

        return sorted(records, key=_key)

    def _heuristic_window_extract(self, window: list[MemoryRecord]) -> dict[str, Any]:
        """Fast System-1 deterministic fallback when LLM extraction is disabled."""
        hyperedges: list[dict[str, Any]] = []
        pointers: list[dict[str, Any]] = []
        speakers: list[str] = []
        summary_bits: list[str] = []

        for idx, rec in enumerate(window):
            meta = rec.metadata or {}
            spk = str(meta.get("speaker") or "").strip()
            if spk and spk not in speakers:
                speakers.append(spk)
            ents = [
                m.group(0)
                for m in re.finditer(r"\b[A-Z][a-zA-Z0-9_-]{2,}\b", rec.text or "")
                if m.group(0) != spk
            ][:5]
            triggers = ProspectiveMemoryEngine.induce_triggers(rec.text or "", max_count=2)
            cues = [f"{t.implication} ({t.condition_cue})" for t in triggers]
            if ents or triggers:
                said_dt = (
                    rec.timestamp.strftime("%Y-%m-%d") if hasattr(rec.timestamp, "strftime") else ""
                )
                hyperedges.append(
                    {
                        "actors": [spk] if spk else [],
                        "entities": ents,
                        "event_summary": (rec.text or "")[:180],
                        "temporal_anchor": str(meta.get("event_date_display") or said_dt),
                        "location": "",
                        "causal_state": triggers[0].implication if triggers else "",
                        "prospective_cues": cues,
                        "turn_indices": [idx],
                    }
                )
            for t in triggers:
                pointers.append(
                    {
                        "speaker": spk,
                        "turn_index": idx,
                        "implicit_lesson_or_boundary": t.implication,
                        "future_cue_scenarios": [t.condition_cue],
                    }
                )
            if len(summary_bits) < 4 and len((rec.text or "").strip()) > 25:
                summary_bits.append(f"{spk + ': ' if spk else ''}{(rec.text or '').strip()[:100]}")

        return {
            "facts": [],
            "hyperedges": hyperedges[:8],
            "proustian_pointers": pointers[:6],
            "chapter_gist": {
                "title": f"Session dialogue ({', '.join(speakers[:3]) or 'conversation'})",
                "summary": " | ".join(summary_bits)[:360],
                "participants": speakers[:4],
            },
            "persona_deltas": [],
        }

    async def _extract_window(self, window: list[MemoryRecord]) -> dict[str, Any]:
        # First enrich verbatim records with deterministic Hippocampal Time Cell resolution
        lines: list[str] = []
        for i, rec in enumerate(window):
            meta = dict(rec.metadata or {})
            spk = str(meta.get("speaker") or "Speaker").strip()
            has_explicit_date = meta.get("date_str") not in (None, "") or "turn_idx" not in meta
            said = (
                rec.timestamp.strftime("%Y-%m-%d")
                if (has_explicit_date and hasattr(rec.timestamp, "strftime"))
                else "dialogue"
            )
            lines.append(f"[{i} | {said} | {spk}]: {(rec.text or '').strip()}")

        if self.llm_client is None:
            return self._heuristic_window_extract(window)

        window_blob = "\n".join(lines)
        model_id = str(getattr(self.llm_client, "model", "llm"))
        cache_key = hashlib.sha256(
            f"{_PROMPT_VERSION}:{model_id}:{window_blob}".encode()
        ).hexdigest()

        cached = _CACHE_SINGLETON.get(cache_key)
        if cached is not None:
            return cached

        prompt = _WINDOW_CONSOLIDATION_PROMPT.format(window_text=window_blob)
        try:
            raw = await self.llm_client.complete(
                prompt,
                temperature=0.0,
                max_tokens=1800,
            )
            parsed = _extract_json_dict(raw)
            if parsed is not None:
                _CACHE_SINGLETON.put(cache_key, parsed)
                return parsed
        except Exception as exc:
            logger.warning("window_llm_extract_fallback", extra={"error": str(exc)})

        return self._heuristic_window_extract(window)

    async def consolidate_tenant(
        self,
        tenant_id: str,
        *,
        session_id: str | None = None,
        max_episodes: int = 2000,
    ) -> dict[str, int]:
        """Consolidate unconsolidated verbatim turns for ``tenant_id`` in ordered windows."""
        filters: dict[str, Any] = {"status": "active"}
        if session_id:
            filters["source_session_id"] = session_id

        scanned = await self.episodic_store.scan(
            tenant_id=tenant_id,
            filters=filters,
            limit=max_episodes,
        )
        already_covered: set[str] = set()
        for r in scanned:
            rmeta = r.metadata or {}
            if rmeta.get("conway_tier") == "chapter_gist":
                for cid in rmeta.get("child_episode_ids") or []:
                    already_covered.add(str(cid))
        verbatim = [
            r
            for r in scanned
            if self._is_verbatim_unconsolidated(r) and str(r.id) not in already_covered
        ]
        if not verbatim:
            return {
                "windows_processed": 0,
                "episodes_consolidated": 0,
                "facts_stored": 0,
                "hyperedges_stored": 0,
                "proustian_pointers_stored": 0,
                "chapters_stored": 0,
                "dmn_profiles_updated": 0,
            }

        ordered = self._sort_episodes_chronologically(verbatim)

        # Content-Defined Chunking (CDC) + session boundaries:
        # Breaking at deterministic CRC32 anchor turns (min 14, avg ~22, max 30) means
        # inserting a localized dialogue cue inside a 600-turn conversation only alters
        # the 1 window containing that cue and immediately re-synchronizes on the next anchor!
        min_win = max(4, min(14, self.window_size // 2))
        max_win = max(self.window_size, 30)
        windows: list[list[MemoryRecord]] = []
        cur_win: list[MemoryRecord] = []
        cur_sess: str | None = None
        for rec in ordered:
            sess = rec.source_session_id
            norm_txt = (rec.text or "").strip().lower().encode()
            is_cdc_cut = len(cur_win) >= min_win and (zlib.crc32(norm_txt) & 7) == 0
            if cur_win and (
                len(cur_win) >= max_win or (sess and cur_sess and sess != cur_sess) or is_cdc_cut
            ):
                windows.append(cur_win)
                cur_win = []
            cur_win.append(rec)
            cur_sess = sess
        if cur_win:
            windows.append(cur_win)

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _run_extract(w: list[MemoryRecord]) -> tuple[list[MemoryRecord], dict[str, Any]]:
            async with sem:
                return w, await self._extract_window(w)

        extracted_pairs = await asyncio.gather(*[_run_extract(w) for w in windows])

        facts_count = 0
        pending_facts: list[tuple[str, str, list[str]]] = []
        engrams_to_store: list[HyperedgeEngram] = []
        all_dmn_deltas: list[dict[str, Any]] = []
        new_records_pending_embed: list[tuple[str, MemoryRecordCreate]] = []
        episode_patches: list[tuple[Any, dict[str, Any]]] = []

        hypergraph: CorticalHypergraphEngine | None = getattr(
            self.neocortical_store, "hypergraph", None
        )

        for win, payload in extracted_pairs:
            # 1. Enrich verbatim episodes only when Hippocampal Time Cell granularity adds display info
            for rec in win:
                meta = dict(rec.metadata or {})
                if hasattr(rec.timestamp, "year") and not meta.get("event_date_display"):
                    refs = resolve_temporal_references(rec.text or "", rec.timestamp)
                    non_vague = [r for r in refs if r.get("granularity") != "vague"]
                    if non_vague:
                        best_ref = non_vague[0]
                        changed = False
                        if best_ref.get("display"):
                            meta["event_date_display"] = best_ref["display"]
                            changed = True
                        if best_ref.get("resolved_date") and not meta.get("event_date"):
                            rd = best_ref["resolved_date"]
                            meta["event_date"] = (
                                rd.strftime("%Y-%m-%d") if hasattr(rd, "strftime") else str(rd)
                            )
                            changed = True
                        if best_ref.get("granularity"):
                            meta["temporal_granularity"] = best_ref["granularity"]
                        if changed:
                            meta["window_consolidated"] = True
                            episode_patches.append((rec.id, {"metadata": meta}))

            # 2. Facts -> Neocortical FactStore
            for fitem in payload.get("facts") or []:
                if not isinstance(fitem, dict):
                    continue
                subj = str(fitem.get("subject") or "").strip()
                pred = str(fitem.get("predicate") or "").strip()
                val = fitem.get("value")
                if not subj or not pred or not val:
                    continue
                cat_raw = str(fitem.get("category") or "attribute").strip().lower()
                t_indices = [
                    i
                    for i in (fitem.get("turn_indices") or [])
                    if isinstance(i, int) and 0 <= i < len(win)
                ]
                ev_ids = [str(win[i].id) for i in t_indices] or [str(win[0].id)]
                ev_disp = str(fitem.get("event_date_display") or "").strip()
                val_str = f"{val} ({ev_disp})" if ev_disp and ev_disp not in str(val) else str(val)
                key = f"{subj.lower()}:{cat_raw}:{pred.lower().replace(' ', '_')[:60]}"
                pending_facts.append((key[:190], val_str, ev_ids))

            # 3. Hyperedges -> CorticalHypergraphEngine
            for hitem in payload.get("hyperedges") or []:
                if not isinstance(hitem, dict):
                    continue
                summary = str(hitem.get("event_summary") or "").strip()
                if not summary:
                    continue
                t_indices = [
                    i
                    for i in (hitem.get("turn_indices") or [])
                    if isinstance(i, int) and 0 <= i < len(win)
                ]
                ep_ids = [str(win[i].id) for i in t_indices] or [str(win[0].id)]
                # Carry over any Perirhinal sensory descriptors from constituent episodes
                sensory: list[str] = []
                for idx_i in t_indices:
                    s_desc = (win[idx_i].metadata or {}).get("perirhinal_descriptors")
                    if isinstance(s_desc, list):
                        sensory.extend(str(x) for x in s_desc if x)
                eng_key = CorticalHypergraphEngine.make_engram_key(tenant_id, summary, ep_ids)
                engrams_to_store.append(
                    HyperedgeEngram(
                        engram_key=eng_key,
                        tenant_id=tenant_id,
                        event_summary=summary,
                        actors=[str(x) for x in (hitem.get("actors") or []) if x],
                        entities=[str(x) for x in (hitem.get("entities") or []) if x],
                        temporal_anchor=str(hitem.get("temporal_anchor") or "") or None,
                        location=str(hitem.get("location") or "") or None,
                        causal_state=str(hitem.get("causal_state") or "") or None,
                        prospective_cues=[
                            str(x) for x in (hitem.get("prospective_cues") or []) if x
                        ],
                        sensory_descriptors=sensory[:8],
                        episode_ids=ep_ids,
                        confidence=0.88,
                    )
                )

            # 4. Grounded Proustian Cue Pointers -> point directly to parent verbatim episode ID
            for pitem in payload.get("proustian_pointers") or []:
                if not isinstance(pitem, dict):
                    continue
                t_idx = pitem.get("turn_index")
                if not isinstance(t_idx, int) or not (0 <= t_idx < len(win)):
                    t_idx = 0
                parent_ep = win[t_idx]
                lesson = str(pitem.get("implicit_lesson_or_boundary") or "").strip()
                scenarios = [
                    str(s).strip()
                    for s in (pitem.get("future_cue_scenarios") or [])
                    if str(s).strip()
                ]
                spk = str(
                    pitem.get("speaker") or (parent_ep.metadata or {}).get("speaker") or ""
                ).strip()
                if not lesson and not scenarios:
                    continue
                cue_text = (
                    f"[Proustian Cue -> {spk or 'Speaker'}] {lesson}. "
                    f"Triggered by: {'; '.join(scenarios[:3])}"
                ).strip()
                ptr_key = (
                    f"proustian:{parent_ep.id}:{hashlib.md5(cue_text.encode()).hexdigest()[:10]}"
                )
                rec_create = MemoryRecordCreate(
                    tenant_id=tenant_id,
                    context_tags=list(parent_ep.context_tags or ["conversation"]),
                    source_session_id=parent_ep.source_session_id,
                    type=MemoryType.CONSTRAINT,
                    text=cue_text,
                    key=ptr_key[:190],
                    timestamp=parent_ep.timestamp,
                    confidence=0.85,
                    importance=0.82,
                    entities=[EntityMention(text=spk, normalized=spk, entity_type="PERSON")]
                    if spk
                    else [],
                    metadata={
                        "proustian_cue_pointer": True,
                        "is_prospective_index": True,
                        "parent_episode_id": str(parent_ep.id),
                        "speaker": spk,
                        "constraints": [
                            {
                                "constraint_type": "causal",
                                "provenance": [str(parent_ep.id)],
                            }
                        ],
                    },
                    provenance=Provenance(
                        source=MemorySource.AGENT_INFERRED,
                        evidence_refs=[str(parent_ep.id)],
                    ),
                )
                new_records_pending_embed.append((cue_text, rec_create))

            # 5. Conway Tier-2 Autobiographical Chapter Gist
            gist = payload.get("chapter_gist")
            if isinstance(gist, dict) and gist.get("summary"):
                title = str(gist.get("title") or "Session Chapter").strip()
                summary = str(gist.get("summary") or "").strip()
                parts = [str(p) for p in (gist.get("participants") or []) if p]
                d_start = (
                    win[0].timestamp.strftime("%Y-%m-%d")
                    if hasattr(win[0].timestamp, "strftime")
                    else ""
                )
                d_end = (
                    win[-1].timestamp.strftime("%Y-%m-%d")
                    if hasattr(win[-1].timestamp, "strftime")
                    else ""
                )
                span = d_start if d_start == d_end else f"{d_start} to {d_end}"
                gist_text = f"[Chapter Summary {span}] {title}: {summary}"
                chap_key = f"chapter:{win[0].id}"
                chap_create = MemoryRecordCreate(
                    tenant_id=tenant_id,
                    context_tags=list(win[0].context_tags or ["conversation"]),
                    source_session_id=win[0].source_session_id,
                    type=MemoryType.SEMANTIC_FACT,
                    text=gist_text,
                    key=chap_key[:190],
                    timestamp=win[-1].timestamp,
                    confidence=0.88,
                    importance=0.78,
                    entities=[
                        EntityMention(text=p, normalized=p, entity_type="PERSON") for p in parts[:4]
                    ],
                    metadata={
                        "conway_tier": "chapter_gist",
                        "child_episode_ids": [str(r.id) for r in win],
                        "participants": parts,
                        "date_span": span,
                    },
                    provenance=Provenance(
                        source=MemorySource.CONSOLIDATION,
                        evidence_refs=[str(r.id) for r in win[:6]],
                    ),
                )
                new_records_pending_embed.append((gist_text, chap_create))

            # 6. Conway Tier-3 DMN Conceptual-Self Deltas
            for pdelta in payload.get("persona_deltas") or []:
                if isinstance(pdelta, dict):
                    all_dmn_deltas.append(pdelta)

        # Batch-store Neocortical facts in micro-batches of 16
        if pending_facts:
            # Deduplicate by key within the tenant pass (last write wins)
            dedup_facts = {k: (k, v, ev) for k, v, ev in pending_facts}
            fact_items = list(dedup_facts.values())
            for f_start in range(0, len(fact_items), 16):
                chunk_f = fact_items[f_start : f_start + 16]
                f_res = await asyncio.gather(
                    *[
                        self.neocortical_store.store_fact(
                            tenant_id=tenant_id,
                            key=fk,
                            value=fv,
                            confidence=0.86,
                            evidence_ids=fev,
                        )
                        for fk, fv, fev in chunk_f
                    ],
                    return_exceptions=True,
                )
                facts_count += sum(1 for r in f_res if not isinstance(r, BaseException))

        # Batch-embed all Proustian Cue Pointers + Conway Chapter Gists in chunks of 64
        proustian_stored = 0
        chapters_stored = 0
        if new_records_pending_embed:
            texts = [t for t, _ in new_records_pending_embed]
            embeddings: list[list[float]] = []
            for start_i in range(0, len(texts), 64):
                batch_texts = texts[start_i : start_i + 64]
                if hasattr(self.embedding_client, "embed_batch"):
                    bres = await self.embedding_client.embed_batch(batch_texts)
                    embeddings.extend([r.embedding for r in bres])
                else:
                    for bt in batch_texts:
                        embeddings.append((await self.embedding_client.embed(bt)).embedding)

            # Parallelize Postgres/SQLite record upserts in micro-batches of 16
            to_upsert: list[MemoryRecordCreate] = []
            for (_, rec_c), emb in zip(new_records_pending_embed, embeddings, strict=False):
                rec_c.embedding = emb
                to_upsert.append(rec_c)
                if (rec_c.metadata or {}).get("proustian_cue_pointer"):
                    proustian_stored += 1
                else:
                    chapters_stored += 1
            for u_start in range(0, len(to_upsert), 16):
                await asyncio.gather(
                    *[self.episodic_store.upsert(rc) for rc in to_upsert[u_start : u_start + 16]]
                )

        # Persist Hyperedges & DMN profiles & mark episodes consolidated
        hyperedges_stored = 0
        if hypergraph is not None and engrams_to_store:
            hyperedges_stored = await hypergraph.upsert_engrams(tenant_id, engrams_to_store)

        dmn_updated = await self.dmn_store.apply_deltas(tenant_id, all_dmn_deltas)

        for ep_id, patch in episode_patches:
            try:
                await self.episodic_store.update(ep_id, patch, increment_version=False)
            except Exception:
                pass

        return {
            "windows_processed": len(windows),
            "episodes_consolidated": len(ordered),
            "facts_stored": facts_count,
            "hyperedges_stored": hyperedges_stored,
            "proustian_pointers_stored": proustian_stored,
            "chapters_stored": chapters_stored,
            "dmn_profiles_updated": dmn_updated,
        }
