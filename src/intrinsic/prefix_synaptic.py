"""Phases I-4 & I-7: Synaptic Interface (KV-Cache & Deterministic APC Prefix Hierarchy).

Provides two complementary mechanisms for synaptic KV-cache injection:
1. Universal HTTP / vLLM Automatic Prefix Caching (APC):
   Constructs a canonical, deterministically ordered memory prefix block
   (sorted by stable engram key/ID rather than query-dependent similarity score)
   so vLLM/SGLang/OpenAI prefix caches achieve O(1) KV-block reuse across turns.
2. Local PyTorch / LMCache-inspired Tiered KV Store:
   Maintains a bounded LRU cache of pre-computed KV representations (`hot` tier
   and `warm` serialized tier) keyed by engram hash.
"""

from __future__ import annotations

import hashlib
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any


@dataclass
class SynapticKVEntry:
    """Cached KV representation or canonical prefix segment for a memory engram."""

    engram_id: str
    content_hash: str
    canonical_text: str
    kv_tensor_handle: Any | None = None
    tier: str = "hot"  # "hot" | "warm"
    access_count: int = 1


class SynapticKVManager:
    """Manages deterministic KV-prefix synthesis and tiered synaptic KV cache."""

    def __init__(self, max_hot_entries: int = 256, max_warm_entries: int = 1024) -> None:
        self.max_hot_entries = max_hot_entries
        self.max_warm_entries = max_warm_entries
        self._hot_cache: OrderedDict[str, SynapticKVEntry] = OrderedDict()
        self._warm_cache: OrderedDict[str, SynapticKVEntry] = OrderedDict()

    def build_canonical_prefix(
        self,
        packet: Any,
        *,
        max_engrams: int = 12,
    ) -> tuple[str, str, list[str]]:
        """Build a deterministic KV-cache-friendly memory prefix block.

        Unlike standard prompt formatting—which orders memories by query-specific
        cosine similarity and invalidates the LLM's KV prefix cache on every query—
        this orders stable semantic facts, constraints, and core episodes by their
        immutable identity (key / UUID) so prefix blocks remain byte-identical
        across turns.

        Returns (canonical_prefix_str, prefix_sha256, engram_ids).
        """
        stable_items: list[tuple[str, str, str]] = []  # (sort_key, engram_id, canonical_line)

        all_mems = (
            list(getattr(packet, "constraints", [])[:4])
            + list(getattr(packet, "facts", [])[:5])
            + list(getattr(packet, "preferences", [])[:4])
            + list(getattr(packet, "recent_episodes", [])[:4])
        )
        seen_ids: set[str] = set()
        for mem in all_mems:
            rec = getattr(mem, "record", None)
            if rec is None:
                continue
            eid = str(getattr(rec, "id", "") or getattr(rec, "key", "") or "")
            if not eid or eid in seen_ids:
                continue
            seen_ids.add(eid)
            sort_key = str(getattr(rec, "key", "") or eid)
            mtype = getattr(getattr(rec, "type", None), "value", str(getattr(rec, "type", "mem")))
            text = str(getattr(rec, "text", "") or "").strip()
            if not text:
                continue
            line = f"[ENGRAM:{mtype}:{sort_key}] {text}"
            stable_items.append((sort_key, eid, line))

        if not stable_items:
            return "", "", []

        # Sort deterministically by sort_key so shared engram subsets share the exact
        # leading token prefix in vLLM's radix/block KV cache.
        stable_items.sort(key=lambda x: x[0])
        selected = stable_items[:max_engrams]

        engram_ids: list[str] = []
        lines = ["<synaptic_memory_kv_prefix>"]
        for sort_key, eid, line in selected:
            engram_ids.append(eid)
            lines.append(line)
            self._touch_cache(eid, line)
        lines.append("</synaptic_memory_kv_prefix>")

        prefix_str = "\n".join(lines)
        prefix_hash = hashlib.sha256(prefix_str.encode("utf-8")).hexdigest()[:16]
        return prefix_str, prefix_hash, engram_ids

    def _touch_cache(self, engram_id: str, canonical_text: str) -> SynapticKVEntry:
        chash = hashlib.sha256(canonical_text.encode("utf-8")).hexdigest()[:16]
        if engram_id in self._hot_cache:
            entry = self._hot_cache.pop(engram_id)
            entry.access_count += 1
            self._hot_cache[engram_id] = entry
            return entry

        if engram_id in self._warm_cache:
            entry = self._warm_cache.pop(engram_id)
            entry.tier = "hot"
            entry.access_count += 1
        else:
            entry = SynapticKVEntry(
                engram_id=engram_id,
                content_hash=chash,
                canonical_text=canonical_text,
                tier="hot",
            )

        self._hot_cache[engram_id] = entry
        if len(self._hot_cache) > self.max_hot_entries:
            evicted_id, evicted = self._hot_cache.popitem(last=False)
            evicted.tier = "warm"
            self._warm_cache[evicted_id] = evicted
            if len(self._warm_cache) > self.max_warm_entries:
                self._warm_cache.popitem(last=False)
        return entry

    def store_kv_tensors(self, engram_id: str, canonical_text: str, kv_tensors: Any) -> None:
        """Attach precomputed PyTorch past_key_values to a hot cache entry."""
        entry = self._touch_cache(engram_id, canonical_text)
        entry.kv_tensor_handle = kv_tensors

    def get_entry(self, engram_id: str) -> SynapticKVEntry | None:
        if engram_id in self._hot_cache:
            return self._hot_cache[engram_id]
        return self._warm_cache.get(engram_id)

    def stats(self) -> dict[str, int]:
        return {
            "hot_entries": len(self._hot_cache),
            "warm_entries": len(self._warm_cache),
        }
