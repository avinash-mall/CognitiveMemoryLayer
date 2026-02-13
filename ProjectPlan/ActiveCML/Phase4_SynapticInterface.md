# Phase 4: Synaptic Interface — KV-Cache Memory Injection

**Intrinsic Phase I-4** (planned; not yet implemented). See [BaseCMLStatus.md](../BaseCML/BaseCMLStatus.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

The Synaptic Interface is the **deepest** non-weight-modifying injection mechanism. It operates on the attention mechanism's memory — the Key-Value (KV) cache — by injecting pre-computed KV pairs that the model treats as if it had "already read" the memory content. This effectively implants **virtual context** that the model attends to during generation, without those tokens ever appearing in the actual prompt.

### Core Principle: Virtual Context Injection
Standard context stuffing prepends text tokens to the prompt, consuming context window budget and incurring O(n^2) attention cost. KV-Cache injection bypasses tokenization entirely: we pre-compute the KV representations of memory content and inject them directly into the cache. The model "sees" the memory as if it were part of the conversation, but at a fraction of the computational cost.

### Security Advantage: "Shadow in the Cache"
Injected KV pairs are abstract vectors — not human-readable tokens. This means sensitive memory content (passwords, personal data) is transmitted as opaque tensors, providing a natural obfuscation layer. An attacker inspecting the KV cache cannot trivially reconstruct the original text.

### Theoretical Foundation
The KV-cache stores:
```
K_l = [k_1, k_2, ..., k_n]    (key vectors for each token at layer l)
V_l = [v_1, v_2, ..., v_n]    (value vectors for each token at layer l)
```
We inject virtual KV pairs:
```
K'_l = [k_virtual_1, ..., k_virtual_m, k_1, ..., k_n]
V'_l = [v_virtual_1, ..., v_virtual_m, v_1, ..., v_n]
```
The attention mechanism computes: `Attention(Q, K', V')`, naturally attending to virtual memories alongside real context.

### Dependencies
- Phase 1: Model Backend (KV-cache access), Hook Manager
- Phase 3: Activation Interface (steering vectors can inform KV encoding)
- Existing: Memory Retriever, Embedding Client

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                         Synaptic Interface                                    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                   KV Encoder Pipeline                                │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  memory_text → tokenize → forward pass (teacher forcing)            │    │
│  │              → extract KV pairs at each layer                       │    │
│  │              → compress (optional: attention pooling)               │    │
│  │              → store pre-computed KV tensors                         │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
│  │  KV Injector      │   │  Position Manager │   │  Budget Allocator       │  │
│  │  ──────────────── │   │  ────────────────│   │  ─────────────────────  │  │
│  │  prepend/append   │   │  rotary position  │   │  max_virtual_tokens     │  │
│  │  to active cache  │   │  remapping for    │   │  per-memory allocation  │  │
│  │  per-layer inject │   │  virtual tokens   │   │  priority-based pruning │  │
│  └──────────────────┘   └──────────────────┘   └─────────────────────────┘  │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                   Temporal Decay (SynapticRAG)                       │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  Injected KV pairs decay over turns:                                │    │
│  │  strength(t) = strength_0 * exp(-λ * t)                             │    │
│  │  When strength < threshold → evict from cache                       │    │
│  │  Recent memories: strong presence; old memories: fade               │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                   Attention Mask Manager                              │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  Manage causal mask for virtual tokens:                              │    │
│  │  - Virtual tokens attend to each other                               │    │
│  │  - Real tokens attend to virtual tokens                              │    │
│  │  - Virtual tokens do NOT attend to real tokens (pre-computed)        │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: KV Encoder Pipeline

### Description
Convert memory text into pre-computed KV pairs for each layer of the model. This is done by running the memory text through the model once (teacher forcing) and capturing the K and V projections at each layer.

### Sub-task 1.1: `KVEncoder` — Memory Text to KV Pairs

**File:** `src/intrinsic/encoding/kv_encoder.py`

```python
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import torch

logger = logging.getLogger(__name__)


@dataclass
class PrecomputedKV:
    """Pre-computed KV pairs for a memory record."""
    memory_id: str
    source_text: str
    num_virtual_tokens: int
    
    # KV pairs per layer: Dict[layer_idx, (K, V)]
    # K shape: [num_virtual_tokens, num_kv_heads, head_dim]
    # V shape: [num_virtual_tokens, num_kv_heads, head_dim]
    kv_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = field(default_factory=dict)
    
    # Metadata
    relevance_score: float = 0.0
    injection_strength: float = 1.0
    decay_factor: float = 1.0
    created_at_turn: int = 0


class KVEncoder:
    """
    Encodes memory text into pre-computed KV pairs.
    
    Pipeline:
    1. Tokenize the memory text
    2. Run through the model (forward pass with no gradient)
    3. At each layer, capture the K and V projections
    4. Optionally compress: reduce num_virtual_tokens via attention pooling
    5. Return PrecomputedKV ready for injection
    
    This is the "expensive" step that happens ONCE per memory.
    The resulting KV pairs can be reused across many generation calls.
    """

    def __init__(
        self,
        backend,       # ModelBackend
        tokenizer,
        max_virtual_tokens: int = 64,
        compress: bool = True,
        compression_ratio: float = 0.5,  # Compress to 50% of original tokens
        target_layers: Optional[List[int]] = None,
    ):
        self._backend = backend
        self._tokenizer = tokenizer
        self._max_virtual_tokens = max_virtual_tokens
        self._compress = compress
        self._compression_ratio = compression_ratio
        self._target_layers = target_layers  # None = all layers
        
        # For capturing KV during forward pass
        self._captured_kv: Dict[int, Tuple[torch.Tensor, torch.Tensor]] = {}

    async def encode(
        self,
        memory_id: str,
        memory_text: str,
        relevance: float = 1.0,
    ) -> PrecomputedKV:
        """
        Encode memory text into KV pairs.
        
        Steps:
        1. Tokenize memory text
        2. Register capture hooks on attention layers
        3. Run forward pass
        4. Collect KV pairs
        5. Optionally compress
        """
        # Step 1: Tokenize
        inputs = self._tokenizer(
            memory_text,
            return_tensors="pt",
            max_length=self._max_virtual_tokens * 2,  # Allow headroom for compression
            truncation=True,
        ).to(self._backend._device)
        
        num_tokens = inputs.input_ids.shape[1]
        self._captured_kv.clear()

        # Step 2: Register capture hooks
        hooks = []
        spec = self._backend.get_model_spec()
        layers = self._target_layers or list(range(spec.num_layers))
        
        for layer_idx in layers:
            hook = self._register_kv_capture_hook(layer_idx)
            hooks.append(hook)

        # Step 3: Forward pass
        try:
            with torch.no_grad():
                self._backend._model(**inputs)
        finally:
            # Clean up hooks
            for hook in hooks:
                hook.remove()

        # Step 4: Collect KV pairs
        kv_pairs = {}
        for layer_idx, (k, v) in self._captured_kv.items():
            # k, v shape: [batch, num_kv_heads, seq_len, head_dim]
            # Remove batch dim and transpose to [seq_len, num_kv_heads, head_dim]
            k = k.squeeze(0).transpose(0, 1)  # [seq_len, num_kv_heads, head_dim]
            v = v.squeeze(0).transpose(0, 1)
            kv_pairs[layer_idx] = (k, v)

        # Step 5: Compress if enabled
        if self._compress and num_tokens > self._max_virtual_tokens:
            kv_pairs = self._compress_kv(kv_pairs, num_tokens)
            num_tokens = min(
                int(num_tokens * self._compression_ratio),
                self._max_virtual_tokens,
            )

        return PrecomputedKV(
            memory_id=memory_id,
            source_text=memory_text,
            num_virtual_tokens=num_tokens,
            kv_pairs=kv_pairs,
            relevance_score=relevance,
        )

    def _register_kv_capture_hook(self, layer_idx: int):
        """
        Register a hook on the attention module to capture K and V AFTER
        the QKV projection but BEFORE the attention computation.
        """
        layer = self._backend._get_layer(layer_idx)
        
        # Find the attention module within the layer
        # Handle different architectures
        attn = None
        for attr in ["self_attn", "attention", "attn"]:
            attn = getattr(layer, attr, None)
            if attn is not None:
                break
        
        if attn is None:
            raise ValueError(f"Cannot find attention module in layer {layer_idx}")

        captured = self._captured_kv  # Reference for closure

        def _capture_hook(module, inputs, output):
            """
            Capture KV from the attention forward output.
            
            Most transformer implementations return:
            (attn_output, attn_weights, past_key_value)
            where past_key_value = (K, V) tensors
            """
            if isinstance(output, tuple) and len(output) >= 3:
                # (attn_output, attn_weights, (K, V))
                past_kv = output[2]
                if isinstance(past_kv, tuple) and len(past_kv) == 2:
                    captured[layer_idx] = (past_kv[0].detach(), past_kv[1].detach())
            elif hasattr(module, '_last_key_value'):
                # Some implementations store it as an attribute
                captured[layer_idx] = (
                    module._last_key_value[0].detach(),
                    module._last_key_value[1].detach(),
                )

        return attn.register_forward_hook(_capture_hook)

    def _compress_kv(
        self,
        kv_pairs: Dict[int, Tuple[torch.Tensor, torch.Tensor]],
        original_tokens: int,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Compress KV pairs by reducing the number of virtual tokens.
        
        Method: Weighted average pooling with attention scores as weights.
        Groups adjacent tokens and pools their KV representations.
        
        Alternative methods (future):
        - Learned compression (trainable pooling layer)
        - Selective retention (keep only high-attention tokens)
        - Quantization (reduce precision)
        """
        target_tokens = min(
            int(original_tokens * self._compression_ratio),
            self._max_virtual_tokens,
        )
        
        if target_tokens >= original_tokens:
            return kv_pairs

        compressed = {}
        group_size = max(1, original_tokens // target_tokens)

        for layer_idx, (k, v) in kv_pairs.items():
            # k, v shape: [seq_len, num_kv_heads, head_dim]
            seq_len = k.shape[0]
            
            # Group and average
            compressed_k_parts = []
            compressed_v_parts = []
            
            for i in range(0, seq_len, group_size):
                end = min(i + group_size, seq_len)
                compressed_k_parts.append(k[i:end].mean(dim=0, keepdim=True))
                compressed_v_parts.append(v[i:end].mean(dim=0, keepdim=True))
            
            compressed_k = torch.cat(compressed_k_parts, dim=0)[:target_tokens]
            compressed_v = torch.cat(compressed_v_parts, dim=0)[:target_tokens]
            
            compressed[layer_idx] = (compressed_k, compressed_v)

        return compressed
```

---

## Task 2: KV-Cache Injector

### Description
Inject pre-computed KV pairs into the model's active KV-cache during generation. Handles position encoding remapping, attention mask updates, and per-layer injection.

### Sub-task 2.1: `KVCacheInjector`

**File:** `src/intrinsic/interfaces/synaptic.py`

```python
import logging
from typing import Dict, List, Optional, Tuple
import torch

from ..encoding.kv_encoder import PrecomputedKV
from ..backends.base import ModelBackend

logger = logging.getLogger(__name__)


class KVCacheInjector:
    """
    Injects pre-computed KV pairs into the model's active KV-cache.
    
    The injector handles:
    1. Position encoding remapping (virtual tokens get position IDs)
    2. Attention mask updates (real tokens can attend to virtual ones)
    3. Multi-memory composition (multiple PrecomputedKV sets)
    4. Budget management (don't exceed max_virtual_tokens)
    
    Injection modes:
    - "prepend": Virtual KVs before real context (most natural)
    - "append": Virtual KVs after real context
    - "interleave": Mix virtual and real based on relevance
    """

    def __init__(
        self,
        backend: ModelBackend,
        max_virtual_tokens: int = 128,
        injection_mode: str = "prepend",
        scale_by_relevance: bool = True,
    ):
        self._backend = backend
        self._max_virtual_tokens = max_virtual_tokens
        self._injection_mode = injection_mode
        self._scale_by_relevance = scale_by_relevance
        self._injected_count = 0  # Total virtual tokens currently injected

    def inject(
        self,
        precomputed_kvs: List[PrecomputedKV],
        current_kv_cache: Optional[Dict[int, Tuple[torch.Tensor, torch.Tensor]]] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Inject pre-computed KV pairs into the cache.
        
        Returns the modified KV cache to be used in the next forward pass.
        
        precomputed_kvs: List of pre-computed KV sets from KVEncoder
        current_kv_cache: Existing KV cache (None for first generation step)
        """
        # Step 1: Budget allocation
        allocated = self._allocate_budget(precomputed_kvs)
        
        if not allocated:
            return current_kv_cache or {}

        # Step 2: Merge all allocated KV pairs per layer
        merged_per_layer: Dict[int, List[Tuple[torch.Tensor, torch.Tensor]]] = {}
        
        for pkv, token_budget in allocated:
            for layer_idx, (k, v) in pkv.kv_pairs.items():
                if layer_idx not in merged_per_layer:
                    merged_per_layer[layer_idx] = []
                
                # Truncate to allocated budget
                k_truncated = k[:token_budget]
                v_truncated = v[:token_budget]
                
                # Scale by relevance if enabled
                if self._scale_by_relevance:
                    scale = pkv.relevance_score * pkv.injection_strength * pkv.decay_factor
                    v_truncated = v_truncated * scale
                    # Note: We scale V but not K. Scaling V modulates the
                    # contribution of virtual tokens to the attention output.
                    # Scaling K would change attention patterns, which is riskier.
                
                merged_per_layer[layer_idx].append((k_truncated, v_truncated))

        # Step 3: Concatenate and inject
        result_cache = {}
        for layer_idx, kv_list in merged_per_layer.items():
            # Concatenate all virtual KV pairs for this layer
            all_k = torch.cat([k for k, v in kv_list], dim=0)  # [total_virtual, num_heads, head_dim]
            all_v = torch.cat([v for k, v in kv_list], dim=0)

            if current_kv_cache and layer_idx in current_kv_cache:
                existing_k, existing_v = current_kv_cache[layer_idx]
                
                if self._injection_mode == "prepend":
                    # Virtual tokens come first → model "reads" memory before context
                    new_k = torch.cat([all_k, existing_k], dim=0)
                    new_v = torch.cat([all_v, existing_v], dim=0)
                elif self._injection_mode == "append":
                    new_k = torch.cat([existing_k, all_k], dim=0)
                    new_v = torch.cat([existing_v, all_v], dim=0)
                else:
                    new_k = torch.cat([all_k, existing_k], dim=0)
                    new_v = torch.cat([all_v, existing_v], dim=0)
                
                result_cache[layer_idx] = (new_k, new_v)
            else:
                result_cache[layer_idx] = (all_k, all_v)

        total_injected = sum(k.shape[0] for k, v in result_cache.values()) // len(result_cache)
        self._injected_count = total_injected
        logger.info(f"Injected {total_injected} virtual tokens across {len(result_cache)} layers")

        return result_cache

    def _allocate_budget(
        self,
        precomputed_kvs: List[PrecomputedKV],
    ) -> List[Tuple[PrecomputedKV, int]]:
        """
        Allocate virtual token budget across memories.
        
        Strategy: proportional allocation based on relevance score.
        Higher relevance → more virtual tokens → stronger influence.
        """
        if not precomputed_kvs:
            return []

        # Sort by relevance (highest first)
        sorted_kvs = sorted(precomputed_kvs, key=lambda x: x.relevance_score, reverse=True)

        total_relevance = sum(pkv.relevance_score for pkv in sorted_kvs)
        if total_relevance < 1e-8:
            # Equal allocation
            per_memory = self._max_virtual_tokens // len(sorted_kvs)
            return [(pkv, min(per_memory, pkv.num_virtual_tokens)) for pkv in sorted_kvs]

        allocated = []
        remaining_budget = self._max_virtual_tokens

        for pkv in sorted_kvs:
            if remaining_budget <= 0:
                break
            
            # Proportional allocation
            share = int((pkv.relevance_score / total_relevance) * self._max_virtual_tokens)
            share = min(share, pkv.num_virtual_tokens, remaining_budget)
            share = max(share, 1)  # At least 1 token per memory
            
            allocated.append((pkv, share))
            remaining_budget -= share

        return allocated

    def get_position_offset(self) -> int:
        """Return the number of virtual tokens prepended (for position ID remapping)."""
        return self._injected_count if self._injection_mode == "prepend" else 0

    def create_extended_attention_mask(
        self,
        original_mask: torch.Tensor,  # [batch, seq_len]
        num_virtual_tokens: int,
    ) -> torch.Tensor:
        """
        Extend the attention mask to include virtual tokens.
        
        Virtual tokens are always attended to (mask = 1).
        The causal mask ensures:
        - Real tokens CAN attend to virtual tokens (read memories)
        - Real tokens CAN attend to other real tokens (normal attention)
        - Virtual tokens attend to each other (internal coherence)
        - Virtual tokens do NOT attend to real tokens (they are pre-computed)
        """
        batch_size = original_mask.shape[0]
        device = original_mask.device
        
        # Virtual tokens are always visible
        virtual_mask = torch.ones(batch_size, num_virtual_tokens, device=device)
        
        # Extend: [virtual_tokens, real_tokens]
        extended = torch.cat([virtual_mask, original_mask], dim=1)
        
        return extended
```

---

## Task 3: Temporal Decay (SynapticRAG)

### Description
Implement temporal decay for injected KV pairs, inspired by SynapticRAG. Memories fade over time (conversation turns), requiring periodic re-injection to maintain influence. This prevents stale memories from dominating.

### Sub-task 3.1: `TemporalDecayManager`

**File:** `src/intrinsic/interfaces/synaptic.py` (continued)

```python
import math
import time


class TemporalDecayManager:
    """
    Manages temporal decay of injected KV memories.
    
    Inspired by SynapticRAG: each injected memory has a "freshness" score
    that decays exponentially over time (conversation turns or wall clock).
    
    Decay models:
    1. Turn-based: strength *= exp(-λ * turns_since_injection)
    2. Time-based: strength *= exp(-λ * seconds_since_injection / half_life)
    3. Usage-based: strength increases when memory is relevant, decays otherwise
    
    When a memory's strength drops below threshold, it is evicted from
    the active injection set (but remains in long-term storage).
    """

    def __init__(
        self,
        decay_model: str = "turn_based",  # "turn_based" | "time_based" | "usage_based"
        half_life_turns: int = 5,
        half_life_seconds: float = 300.0,  # 5 minutes
        eviction_threshold: float = 0.1,
        reinforcement_boost: float = 0.5,
    ):
        self._decay_model = decay_model
        self._half_life_turns = half_life_turns
        self._half_life_seconds = half_life_seconds
        self._eviction_threshold = eviction_threshold
        self._reinforcement_boost = reinforcement_boost
        
        # Track injection state: memory_id → {strength, injected_at_turn, injected_at_time, relevance_history}
        self._injection_state: Dict[str, Dict] = {}
        self._current_turn = 0

    def register_injection(self, memory_id: str, initial_strength: float = 1.0) -> None:
        """Register a newly injected memory for decay tracking."""
        self._injection_state[memory_id] = {
            "strength": initial_strength,
            "injected_at_turn": self._current_turn,
            "injected_at_time": time.time(),
            "last_relevant_turn": self._current_turn,
            "relevance_history": [],
        }

    def advance_turn(self) -> None:
        """Advance the turn counter (called after each conversation turn)."""
        self._current_turn += 1

    def compute_decay(self, memory_id: str) -> float:
        """Compute current strength for a memory after decay."""
        state = self._injection_state.get(memory_id)
        if state is None:
            return 0.0

        if self._decay_model == "turn_based":
            turns_elapsed = self._current_turn - state["injected_at_turn"]
            # λ = ln(2) / half_life
            lam = math.log(2) / self._half_life_turns
            decay_factor = math.exp(-lam * turns_elapsed)

        elif self._decay_model == "time_based":
            seconds_elapsed = time.time() - state["injected_at_time"]
            lam = math.log(2) / self._half_life_seconds
            decay_factor = math.exp(-lam * seconds_elapsed)

        elif self._decay_model == "usage_based":
            turns_since_relevant = self._current_turn - state["last_relevant_turn"]
            lam = math.log(2) / self._half_life_turns
            decay_factor = math.exp(-lam * turns_since_relevant)

        else:
            decay_factor = 1.0

        strength = state["strength"] * decay_factor
        return strength

    def reinforce(self, memory_id: str, relevance_score: float) -> None:
        """
        Reinforce a memory that was relevant in the current turn.
        This slows or reverses decay — frequently relevant memories stay strong.
        """
        state = self._injection_state.get(memory_id)
        if state is None:
            return

        state["last_relevant_turn"] = self._current_turn
        state["relevance_history"].append(relevance_score)

        # Boost strength proportional to relevance
        boost = self._reinforcement_boost * relevance_score
        state["strength"] = min(1.0, state["strength"] + boost)

    def get_eviction_candidates(self) -> List[str]:
        """Return memory IDs whose strength has dropped below eviction threshold."""
        candidates = []
        for memory_id in list(self._injection_state.keys()):
            strength = self.compute_decay(memory_id)
            if strength < self._eviction_threshold:
                candidates.append(memory_id)
        return candidates

    def evict(self, memory_id: str) -> None:
        """Remove a memory from decay tracking (evicted from active injection)."""
        self._injection_state.pop(memory_id, None)

    def get_active_memories(self) -> Dict[str, float]:
        """Return all active memories with their current strengths."""
        return {
            mid: self.compute_decay(mid)
            for mid in self._injection_state
        }
```

---

## Task 4: Synaptic Interface — Complete Pipeline

### Description
Compose the KV Encoder, Injector, and Decay Manager into a complete Synaptic Interface that handles the full lifecycle: encode → inject → decay → evict.

### Sub-task 4.1: `SynapticInterface`

**File:** `src/intrinsic/interfaces/synaptic.py` (continued)

```python
class SynapticInterface:
    """
    Complete KV-Cache injection interface.
    
    Lifecycle:
    1. ENCODE: Convert memory text → PrecomputedKV (once per memory)
    2. INJECT: Insert KV pairs into active cache (each generation)
    3. DECAY: Reduce injection strength over time
    4. REINFORCE: Boost strength when memory is relevant
    5. EVICT: Remove faded memories from active set
    
    The interface maintains a pool of pre-computed KV sets,
    managing which are active and at what strength.
    """

    def __init__(
        self,
        backend: ModelBackend,
        tokenizer,
        max_virtual_tokens: int = 128,
        decay_half_life: int = 5,
    ):
        self._encoder = KVEncoder(
            backend=backend,
            tokenizer=tokenizer,
            max_virtual_tokens=max_virtual_tokens // 2,  # Reserve half for new memories
        )
        self._injector = KVCacheInjector(
            backend=backend,
            max_virtual_tokens=max_virtual_tokens,
        )
        self._decay = TemporalDecayManager(
            half_life_turns=decay_half_life,
        )
        
        # Pool of pre-computed KV sets
        self._kv_pool: Dict[str, PrecomputedKV] = {}  # memory_id → PrecomputedKV

    async def process_memories(self, memory_vectors: List) -> None:
        """
        Bus handler: encode and prepare memories for KV injection.
        
        For each memory:
        1. If not already encoded → encode (expensive, cached)
        2. Update relevance and decay
        3. Stage for injection
        """
        for mv in memory_vectors:
            memory_id = mv.memory_id
            
            # Encode if not in pool
            if memory_id not in self._kv_pool:
                try:
                    pkv = await self._encoder.encode(
                        memory_id=memory_id,
                        memory_text=mv.source_text,
                        relevance=mv.relevance_score,
                    )
                    self._kv_pool[memory_id] = pkv
                    self._decay.register_injection(memory_id, initial_strength=mv.injection_strength)
                    logger.info(f"Encoded memory {memory_id} → {pkv.num_virtual_tokens} virtual tokens")
                except Exception as e:
                    logger.error(f"Failed to encode memory {memory_id}: {e}")
                    continue
            
            # Update relevance
            pkv = self._kv_pool[memory_id]
            pkv.relevance_score = mv.relevance_score
            
            # Reinforce if relevant
            if mv.relevance_score > 0.5:
                self._decay.reinforce(memory_id, mv.relevance_score)

    def get_injection_cache(
        self,
        current_kv_cache: Optional[Dict] = None,
    ) -> Dict[int, Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get the KV cache with all active memories injected.
        Called before each generation step.
        
        Applies temporal decay to strength values before injection.
        """
        # Apply decay and collect active memories
        active_kvs = []
        for memory_id, pkv in list(self._kv_pool.items()):
            strength = self._decay.compute_decay(memory_id)
            if strength < self._decay._eviction_threshold:
                # Evict faded memory
                self._decay.evict(memory_id)
                del self._kv_pool[memory_id]
                logger.debug(f"Evicted faded memory: {memory_id}")
                continue
            
            # Update decay factor
            pkv.decay_factor = strength
            active_kvs.append(pkv)

        if not active_kvs:
            return current_kv_cache or {}

        # Inject all active memories
        return self._injector.inject(active_kvs, current_kv_cache)

    def advance_turn(self) -> None:
        """Call after each conversation turn to advance decay."""
        self._decay.advance_turn()
        
        # Evict spent memories
        for memory_id in self._decay.get_eviction_candidates():
            self._decay.evict(memory_id)
            self._kv_pool.pop(memory_id, None)

    def get_stats(self) -> Dict:
        return {
            "pool_size": len(self._kv_pool),
            "active_memories": self._decay.get_active_memories(),
            "total_virtual_tokens": sum(
                pkv.num_virtual_tokens for pkv in self._kv_pool.values()
            ),
        }

    def clear(self) -> None:
        """Clear all injected memories."""
        self._kv_pool.clear()
        self._decay._injection_state.clear()
```

---

## Task 5: Position Encoding Remapping

### Description
When prepending virtual KV tokens, the position IDs for real tokens must be shifted to account for the virtual prefix. This is critical for models using Rotary Position Embeddings (RoPE).

### Sub-task 5.1: Position Remapping Utility

```python
class PositionRemapper:
    """
    Handles position ID remapping when virtual tokens are injected.
    
    For RoPE-based models:
    - Virtual tokens at positions [0, 1, ..., m-1]
    - Real tokens at positions [m, m+1, ..., m+n-1]
    
    The KV pairs for virtual tokens are pre-computed with their own
    position embeddings. We need to ensure the real tokens' position
    IDs are offset by the number of virtual tokens.
    
    Alternative: "Position-free" virtual tokens
    - Pre-compute KV without position encoding
    - Let attention weights determine importance (position-agnostic)
    - Research shows this works well for factual memory injection
    """

    def __init__(self, use_position_free: bool = False):
        self._position_free = use_position_free

    def remap_position_ids(
        self,
        original_position_ids: torch.Tensor,  # [batch, seq_len]
        num_virtual_tokens: int,
    ) -> torch.Tensor:
        """Offset position IDs for real tokens."""
        if self._position_free:
            return original_position_ids  # No remapping needed
        
        return original_position_ids + num_virtual_tokens

    def compute_virtual_position_ids(
        self,
        num_virtual_tokens: int,
        batch_size: int = 1,
        device: str = "cuda",
    ) -> torch.Tensor:
        """Generate position IDs for virtual tokens."""
        if self._position_free:
            # All virtual tokens at position 0 (position-agnostic)
            return torch.zeros(batch_size, num_virtual_tokens, dtype=torch.long, device=device)
        
        # Sequential positions [0, 1, ..., m-1]
        ids = torch.arange(num_virtual_tokens, device=device).unsqueeze(0).expand(batch_size, -1)
        return ids
```

---

## Acceptance Criteria

1. `KVEncoder` produces valid KV pairs from memory text via forward pass
2. `KVCacheInjector` correctly prepends/appends virtual KV pairs to the cache
3. `TemporalDecayManager` exponentially decays injection strength over turns
4. Reinforcement boosts decay for frequently relevant memories
5. Budget allocator distributes virtual tokens proportional to relevance
6. Position remapping correctly offsets real token positions
7. Attention mask extended to include virtual tokens
8. Full lifecycle: encode → inject → decay → evict works end-to-end
9. Memory bus handler for `KV_CACHE_INJECTION` channel is functional
10. KV compression reduces token count while preserving quality

## Estimated Effort
- **Duration:** 3-4 weeks
- **Complexity:** Very High (KV-cache manipulation is model-architecture-sensitive)
- **Risk:** High (position encoding misalignment can cause severe quality degradation)

## Testing Strategy
1. Unit test KVEncoder with small model — verify KV shapes match model spec
2. Verify position remapping produces correct offsets
3. Integration test: inject "The capital of France is Paris" as KV → ask "What is the capital of France?"
4. Temporal decay test: verify memory strength decays and eviction triggers
5. Budget allocation test: verify proportional distribution across memories
6. Performance benchmark: compare KV injection latency vs. context stuffing latency
7. Quality benchmark: compare answer accuracy with KV injection vs. RAG context stuffing
