# Phase 7: Memory Hierarchy & Cache Management

**Intrinsic Phase I-7** (planned; not yet implemented). See [ActiveCML/README.md](README.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

Intrinsic memory injection introduces a new resource management challenge: latent representations (steering vectors, KV pairs, encoded memories) must be stored, retrieved, and managed across a tiered memory hierarchy. This phase builds the **LMCache-inspired** tiered storage system that keeps hot memories on GPU, warm memories in CPU RAM, and cold memories on disk.

### Core Problem
- **Steering vectors** are small (~32KB each for 4096-dim fp16) but may number in the hundreds
- **KV pairs** are large (~2MB+ per memory for a 32-layer model with 64 virtual tokens)
- **Encoded memories** include all representations and metadata
- GPU HBM is limited (24GB on consumer, 80GB on A100)
- All must be available with **sub-millisecond latency** during the forward pass

### Design: Three-Tier Memory Hierarchy

| Tier | Storage | Latency | Capacity | Content |
|------|---------|---------|----------|---------|
| L1 | GPU HBM | < 0.1 ms | ~1GB | Active injection set (current turn) |
| L2 | CPU DRAM | ~1 ms | ~16GB | Warm memories (recent turns, pre-fetched) |
| L3 | NVMe SSD / Disk | ~10 ms | ~1TB | Cold memories (all encoded representations) |

### Biological Analogy
This mirrors the human memory hierarchy:
- L1 (GPU) = Working memory / attention span
- L2 (CPU) = Short-term memory / recent context
- L3 (Disk) = Long-term memory / hippocampal storage

### Dependencies
- Phase 1: Model Backend, Memory Bus
- Phase 4: KV pairs (largest memory objects)
- Phase 5: Controller (determines what's "hot")
- Phase 6: Encoded Memory Store (provides the data to cache)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Memory Hierarchy Manager                               │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  L1: GPU HBM (Hot)                                                     │  │
│  │  ─────────────────                                                     │  │
│  │  Budget: ~1GB configurable                                             │  │
│  │  Contents:                                                              │  │
│  │    - Active steering vectors (current forward pass)                    │  │
│  │    - Active KV pairs (injected into cache)                             │  │
│  │    - Pre-fetched next-turn candidates                                  │  │
│  │  Eviction: LRU + relevance-weighted                                    │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │ promote / evict                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  L2: CPU DRAM (Warm)                                                   │  │
│  │  ───────────────────                                                   │  │
│  │  Budget: ~16GB configurable                                            │  │
│  │  Contents:                                                              │  │
│  │    - Recent-turn memories (last N turns)                               │  │
│  │    - Session-scoped memories                                           │  │
│  │    - Pre-fetched from L3 based on prediction                           │  │
│  │  Eviction: Turn-based aging + access frequency                         │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                              │ promote / evict                               │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  L3: NVMe SSD / Disk (Cold)                                           │  │
│  │  ───────────────────────────                                           │  │
│  │  Budget: configurable (default unlimited)                              │  │
│  │  Contents:                                                              │  │
│  │    - All encoded memories (EncodedMemory objects)                      │  │
│  │    - Serialized steering vectors (.pt files)                           │  │
│  │    - Compressed KV pairs                                               │  │
│  │  Format: pickle + torch.save with compression                          │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Predictive Pre-fetcher                                                 │  │
│  │  ──────────────────────                                                 │  │
│  │  Analyzes conversation trajectory to predict which memories will be    │  │
│  │  needed next turn, and promotes them from L3 → L2 → L1 proactively.   │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
│                                                                              │
│  ┌────────────────────────────────────────────────────────────────────────┐  │
│  │  Budget Monitor & Eviction Engine                                       │  │
│  │  ────────────────────────────────                                       │  │
│  │  Monitors GPU/CPU memory usage and triggers evictions.                 │  │
│  │  Policies: LRU, LFU, Relevance-weighted, EvicPress (importance-aware) │  │
│  └────────────────────────────────────────────────────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Tiered Memory Store

### Sub-task 1.1: `TieredMemoryStore`

**File:** `src/intrinsic/cache/hierarchy.py`

```python
import logging
import os
import pickle
import time
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single entry in the cache with metadata."""
    memory_id: str
    data: Any                    # EncodedMemory, steering vector, KV pairs, etc.
    size_bytes: int = 0
    last_accessed: float = field(default_factory=time.time)
    access_count: int = 0
    created_at: float = field(default_factory=time.time)
    relevance_score: float = 0.0
    tier: str = "l1"             # Current tier: "l1", "l2", "l3"

    def touch(self):
        self.last_accessed = time.time()
        self.access_count += 1


class L1GPUCache:
    """
    GPU HBM cache for active injection data.
    
    Stores tensors directly on GPU for zero-copy access during forward pass.
    Budget-constrained with LRU eviction.
    """

    def __init__(self, budget_bytes: int = 1_073_741_824, device: str = "cuda"):
        self._budget = budget_bytes  # Default 1GB
        self._device = device
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._used_bytes = 0

    def get(self, memory_id: str) -> Optional[CacheEntry]:
        if memory_id in self._cache:
            entry = self._cache[memory_id]
            entry.touch()
            self._cache.move_to_end(memory_id)  # LRU: move to end
            return entry
        return None

    def put(self, memory_id: str, data: Any, relevance: float = 0.0) -> Optional[CacheEntry]:
        """Store data on GPU. Returns evicted entry if eviction was needed."""
        size = self._estimate_size(data)
        
        # Evict until we have room
        evicted = None
        while self._used_bytes + size > self._budget and self._cache:
            evicted = self._evict_one()
        
        if self._used_bytes + size > self._budget:
            logger.warning(f"L1 budget exceeded even after eviction; cannot store {memory_id}")
            return evicted

        # Move tensors to GPU
        gpu_data = self._to_device(data, self._device)
        
        entry = CacheEntry(
            memory_id=memory_id,
            data=gpu_data,
            size_bytes=size,
            relevance_score=relevance,
            tier="l1",
        )
        self._cache[memory_id] = entry
        self._used_bytes += size
        return evicted

    def remove(self, memory_id: str) -> Optional[CacheEntry]:
        if memory_id in self._cache:
            entry = self._cache.pop(memory_id)
            self._used_bytes -= entry.size_bytes
            return entry
        return None

    def _evict_one(self) -> Optional[CacheEntry]:
        """Evict the least-recently-used entry."""
        if not self._cache:
            return None
        # LRU: first item is least recently used
        memory_id, entry = self._cache.popitem(last=False)
        self._used_bytes -= entry.size_bytes
        logger.debug(f"L1 evicted: {memory_id} ({entry.size_bytes} bytes)")
        return entry

    def _estimate_size(self, data: Any) -> int:
        """Estimate byte size of data."""
        if isinstance(data, torch.Tensor):
            return data.nelement() * data.element_size()
        elif isinstance(data, dict):
            total = 0
            for v in data.values():
                if isinstance(v, torch.Tensor):
                    total += v.nelement() * v.element_size()
                elif isinstance(v, tuple):
                    for t in v:
                        if isinstance(t, torch.Tensor):
                            total += t.nelement() * t.element_size()
            return total
        return 1024  # Default estimate

    def _to_device(self, data: Any, device: str):
        """Move tensors to specified device."""
        if isinstance(data, torch.Tensor):
            return data.to(device)
        elif isinstance(data, dict):
            return {k: self._to_device(v, device) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(self._to_device(v, device) for v in data)
        return data

    @property
    def used_bytes(self) -> int:
        return self._used_bytes

    @property
    def budget_bytes(self) -> int:
        return self._budget

    @property
    def utilization(self) -> float:
        return self._used_bytes / self._budget if self._budget > 0 else 0


class L2CPUCache:
    """
    CPU DRAM cache for warm memories.
    Same interface as L1 but on CPU with larger budget.
    """

    def __init__(self, budget_bytes: int = 16_106_127_360):  # 15GB
        self._budget = budget_bytes
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._used_bytes = 0

    def get(self, memory_id: str) -> Optional[CacheEntry]:
        if memory_id in self._cache:
            entry = self._cache[memory_id]
            entry.touch()
            self._cache.move_to_end(memory_id)
            return entry
        return None

    def put(self, memory_id: str, data: Any, relevance: float = 0.0) -> Optional[CacheEntry]:
        size = self._estimate_size(data)
        evicted = None
        while self._used_bytes + size > self._budget and self._cache:
            evicted = self._evict_one()

        # Move to CPU
        cpu_data = self._to_cpu(data)
        entry = CacheEntry(
            memory_id=memory_id, data=cpu_data, size_bytes=size,
            relevance_score=relevance, tier="l2",
        )
        self._cache[memory_id] = entry
        self._used_bytes += size
        return evicted

    def remove(self, memory_id: str) -> Optional[CacheEntry]:
        if memory_id in self._cache:
            entry = self._cache.pop(memory_id)
            self._used_bytes -= entry.size_bytes
            return entry
        return None

    def _evict_one(self) -> Optional[CacheEntry]:
        if not self._cache:
            return None
        memory_id, entry = self._cache.popitem(last=False)
        self._used_bytes -= entry.size_bytes
        return entry

    def _estimate_size(self, data):
        if isinstance(data, torch.Tensor):
            return data.nelement() * data.element_size()
        elif isinstance(data, dict):
            return sum(self._estimate_size(v) for v in data.values())
        elif isinstance(data, tuple):
            return sum(self._estimate_size(v) for v in data)
        return 1024

    @staticmethod
    def _to_cpu(data):
        if isinstance(data, torch.Tensor):
            return data.cpu()
        elif isinstance(data, dict):
            return {k: L2CPUCache._to_cpu(v) for k, v in data.items()}
        elif isinstance(data, tuple):
            return tuple(L2CPUCache._to_cpu(v) for v in data)
        return data

    @property
    def utilization(self) -> float:
        return self._used_bytes / self._budget if self._budget > 0 else 0


class L3DiskCache:
    """
    Persistent disk cache for cold memories.
    Serializes EncodedMemory objects to disk with compression.
    """

    def __init__(self, cache_dir: str = ".cache/l3_memories"):
        self._cache_dir = Path(cache_dir)
        self._cache_dir.mkdir(parents=True, exist_ok=True)
        self._index: Dict[str, str] = {}  # memory_id → filename
        self._load_index()

    def _load_index(self):
        """Load index of cached files."""
        for f in self._cache_dir.glob("*.pt"):
            memory_id = f.stem
            self._index[memory_id] = str(f)

    def get(self, memory_id: str) -> Optional[Any]:
        if memory_id not in self._index:
            return None
        try:
            path = self._index[memory_id]
            return torch.load(path, map_location="cpu", weights_only=False)
        except Exception as e:
            logger.warning(f"L3 read error for {memory_id}: {e}")
            return None

    def put(self, memory_id: str, data: Any) -> None:
        path = self._cache_dir / f"{memory_id}.pt"
        try:
            # Move to CPU before saving
            cpu_data = L2CPUCache._to_cpu(data)
            torch.save(cpu_data, str(path))
            self._index[memory_id] = str(path)
        except Exception as e:
            logger.error(f"L3 write error for {memory_id}: {e}")

    def remove(self, memory_id: str) -> None:
        if memory_id in self._index:
            path = Path(self._index.pop(memory_id))
            path.unlink(missing_ok=True)

    @property
    def size(self) -> int:
        return len(self._index)


class TieredMemoryStore:
    """
    Unified three-tier memory hierarchy.
    
    Access pattern:
    1. Check L1 (GPU) → if hit, return immediately
    2. Check L2 (CPU) → if hit, promote to L1, return
    3. Check L3 (Disk) → if hit, promote to L2 → L1, return
    4. Miss → return None (caller must encode)
    
    Promotion/demotion is automatic based on access patterns.
    """

    def __init__(
        self,
        l1_budget_bytes: int = 1_073_741_824,    # 1GB GPU
        l2_budget_bytes: int = 16_106_127_360,    # 15GB CPU
        l3_cache_dir: str = ".cache/l3_memories",
        device: str = "cuda",
    ):
        self.l1 = L1GPUCache(l1_budget_bytes, device)
        self.l2 = L2CPUCache(l2_budget_bytes)
        self.l3 = L3DiskCache(l3_cache_dir)

    def get(self, memory_id: str, promote: bool = True) -> Optional[Any]:
        """
        Retrieve data from the hierarchy with automatic promotion.
        """
        # L1 hit
        entry = self.l1.get(memory_id)
        if entry:
            return entry.data

        # L2 hit → promote to L1
        entry = self.l2.get(memory_id)
        if entry:
            if promote:
                evicted = self.l1.put(memory_id, entry.data, entry.relevance_score)
                if evicted:
                    # Demote evicted L1 entry to L2
                    self.l2.put(evicted.memory_id, evicted.data, evicted.relevance_score)
            return entry.data

        # L3 hit → promote to L2 → L1
        data = self.l3.get(memory_id)
        if data is not None:
            if promote:
                evicted_l2 = self.l2.put(memory_id, data)
                if evicted_l2:
                    self.l3.put(evicted_l2.memory_id, evicted_l2.data)
                evicted_l1 = self.l1.put(memory_id, data)
                if evicted_l1:
                    self.l2.put(evicted_l1.memory_id, evicted_l1.data)
            return data

        return None

    def put(self, memory_id: str, data: Any, relevance: float = 0.0, tier: str = "l1") -> None:
        """Store data starting at the specified tier."""
        if tier == "l1":
            evicted = self.l1.put(memory_id, data, relevance)
            if evicted:
                self.l2.put(evicted.memory_id, evicted.data, evicted.relevance_score)
        elif tier == "l2":
            evicted = self.l2.put(memory_id, data, relevance)
            if evicted:
                self.l3.put(evicted.memory_id, evicted.data)
        elif tier == "l3":
            self.l3.put(memory_id, data)
        
        # Always persist to L3 for durability
        self.l3.put(memory_id, data)

    def remove(self, memory_id: str) -> None:
        """Remove from all tiers."""
        self.l1.remove(memory_id)
        self.l2.remove(memory_id)
        self.l3.remove(memory_id)

    def get_stats(self) -> Dict:
        return {
            "l1_utilization": f"{self.l1.utilization:.1%}",
            "l1_used_bytes": self.l1.used_bytes,
            "l2_utilization": f"{self.l2.utilization:.1%}",
            "l3_entries": self.l3.size,
        }
```

---

## Task 2: Predictive Pre-fetcher

### Sub-task 2.1: `PredictivePreFetcher`

```python
class PredictivePreFetcher:
    """
    Predicts which memories will be needed in the next turn
    and pre-promotes them from L3 → L2 → L1.
    
    Strategies:
    1. Topic continuity: if current turn is about "cooking",
       pre-fetch all cooking-related memories
    2. Session history: memories accessed in this session are likely
       to be accessed again
    3. Co-access patterns: if memory A is often accessed with B,
       pre-fetch B when A is accessed
    4. Recency: recently stored memories are more likely to be needed
    """

    def __init__(self, tiered_store: TieredMemoryStore, retriever=None):
        self._store = tiered_store
        self._retriever = retriever
        self._access_history: List[List[str]] = []  # Per-turn access lists
        self._co_access: Dict[str, Dict[str, int]] = {}  # memory_id → {co_memory_id: count}

    async def prefetch_for_next_turn(
        self,
        current_query: str,
        tenant_id: str,
        current_accessed: List[str],
    ) -> List[str]:
        """
        Pre-fetch memories predicted to be needed next turn.
        Returns list of pre-fetched memory IDs.
        """
        candidates = set()

        # Strategy 1: Co-access patterns
        for memory_id in current_accessed:
            co_accessed = self._co_access.get(memory_id, {})
            for co_id, count in co_accessed.items():
                if count >= 2:  # Accessed together at least twice
                    candidates.add(co_id)

        # Strategy 2: Session history (recent accesses likely again)
        if self._access_history:
            for recent_turn in self._access_history[-3:]:  # Last 3 turns
                candidates.update(recent_turn)

        # Update co-access tracking
        self._access_history.append(current_accessed)
        for i, mid_a in enumerate(current_accessed):
            for mid_b in current_accessed[i+1:]:
                self._co_access.setdefault(mid_a, {})
                self._co_access[mid_a][mid_b] = self._co_access[mid_a].get(mid_b, 0) + 1
                self._co_access.setdefault(mid_b, {})
                self._co_access[mid_b][mid_a] = self._co_access[mid_b].get(mid_a, 0) + 1

        # Pre-fetch: promote from L3/L2 to L1
        prefetched = []
        for memory_id in candidates:
            if memory_id not in current_accessed:
                data = self._store.get(memory_id, promote=True)
                if data is not None:
                    prefetched.append(memory_id)

        logger.debug(f"Pre-fetched {len(prefetched)} memories for next turn")
        return prefetched
```

---

## Task 3: Eviction Policies

### Sub-task 3.1: Advanced Eviction — EvicPress-Inspired

```python
class EvictionPolicy:
    """
    Advanced eviction policy for the memory hierarchy.
    
    EvicPress-inspired: eviction priority considers both recency
    and importance (relevance × confidence × type priority).
    
    Score = importance_weight × importance + recency_weight × recency
    Lowest score gets evicted first.
    """

    def __init__(
        self,
        importance_weight: float = 0.6,
        recency_weight: float = 0.4,
    ):
        self._imp_w = importance_weight
        self._rec_w = recency_weight

    def compute_eviction_score(self, entry: CacheEntry, current_time: float) -> float:
        """
        Compute eviction score. LOWER = evict first.
        """
        # Importance: relevance × access frequency
        importance = entry.relevance_score * (1 + 0.1 * entry.access_count)
        
        # Recency: exponential decay based on time since last access
        age_seconds = current_time - entry.last_accessed
        recency = 1.0 / (1.0 + age_seconds / 60.0)  # Decay over minutes
        
        score = self._imp_w * importance + self._rec_w * recency
        return score

    def select_eviction_candidate(self, entries: Dict[str, CacheEntry]) -> Optional[str]:
        """Select the best candidate for eviction (lowest score)."""
        if not entries:
            return None
        
        current_time = time.time()
        scores = {
            mid: self.compute_eviction_score(entry, current_time)
            for mid, entry in entries.items()
        }
        
        return min(scores, key=scores.get)
```

---

## Acceptance Criteria

1. `TieredMemoryStore` correctly promotes data L3 → L2 → L1 on access
2. L1 GPU cache respects budget and evicts LRU entries
3. L2 CPU cache handles overflow by demoting to L3
4. L3 disk cache persists and loads data correctly
5. Pre-fetcher predicts and pre-loads memories with > 50% hit rate after warmup
6. Eviction policy correctly prioritizes low-importance, stale entries
7. Memory utilization statistics are accurate and queryable
8. Zero data loss: evicted entries are always demoted (not deleted)
9. Thread-safe operations for concurrent access
10. Configuration allows tuning tier budgets

## Estimated Effort
- **Duration:** 2-3 weeks
- **Complexity:** Medium-High (GPU memory management, concurrent access)
- **Risk:** Medium (GPU OOM if budgets are misconfigured)

## Testing Strategy
1. Unit test promotion/demotion across tiers
2. Budget enforcement: fill L1 beyond capacity, verify eviction
3. Persistence test: store to L3, restart, verify data loads
4. Pre-fetcher accuracy test: simulate conversation, measure hit rate
5. Performance benchmark: L1 access latency < 0.1ms, L2 < 1ms, L3 < 10ms
6. Concurrent access test: multiple threads reading/writing simultaneously
