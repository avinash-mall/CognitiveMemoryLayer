# Phase 5: Controller & Gating Unit — Intelligent Orchestration

## Overview

The Controller is the **brain** of the intrinsic memory system. It decides **which memories** to inject, **through which interface** (logit, activation, KV-cache, or LoRA), and **at what strength**. Without it, the injection interfaces are just raw capabilities — the Controller provides the intelligence that ties them together.

### Core Problem
Given a user query and a set of retrieved memories, the Controller must answer:
1. **Relevance gating:** Which memories are relevant enough to inject?
2. **Interface routing:** Should this memory be a logit bias, a steering vector, KV-cache injection, or multiple?
3. **Strength calibration:** How strongly should each memory be injected?
4. **Budget allocation:** Given finite capacity (max tokens, max hooks, max LoRA adapters), how to allocate resources?
5. **Fallback chain:** If the preferred interface is unavailable, what's the fallback?

### Design Philosophy
The Controller operates as a **Sidecar Controller** — it runs alongside the LLM, observing its state and intervening as needed, but does not modify the LLM's core architecture. Think of it as a hippocampal circuit that monitors the neocortex and injects relevant memories when needed.

### Dependencies
- Phase 1: Model Backend, Memory Bus, Model Inspector
- Phase 2: Logit Interface
- Phase 3: Activation Interface
- Phase 4: Synaptic Interface
- Existing: Memory Retriever, Memory Orchestrator

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                          Controller & Gating Unit                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    Relevance Gate                                     │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  retrieved_memories → threshold filter → salience scoring            │    │
│  │  → temporal freshness → emotional valence → confidence weighting     │    │
│  │  → pass/reject decision per memory                                   │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    Interface Router                                   │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  For each gated memory, select optimal injection interface:          │    │
│  │                                                                      │    │
│  │  semantic_fact  → KV-Cache (deep) + Logit Bias (reinforcement)      │    │
│  │  preference     → Activation Steering (behavioral influence)         │    │
│  │  constraint     → KV-Cache (deep) + Activation (strong)             │    │
│  │  episodic_event → Activation Steering (subtle)                       │    │
│  │  procedure      → KV-Cache (full context) + Logit (key tokens)      │    │
│  │  style/tone     → Activation Steering (late layers)                  │    │
│  │                                                                      │    │
│  │  Respects backend capabilities (degrades gracefully)                 │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    Strength Calibrator                                │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  Per-memory injection strength based on:                             │    │
│  │  - relevance_score (from retriever)                                  │    │
│  │  - confidence (from memory record)                                   │    │
│  │  - memory_type weight (constraints > facts > hypotheses)             │    │
│  │  - temporal decay (from SynapticRAG)                                 │    │
│  │  - interference avoidance (conflicting memories → reduce both)       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    Budget Manager                                     │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  Enforce resource limits:                                            │    │
│  │  - max_virtual_tokens (KV-Cache budget)                              │    │
│  │  - max_steering_vectors (activation hooks budget)                    │    │
│  │  - max_logit_biased_tokens (API logit_bias budget)                  │    │
│  │  - max_concurrent_lora (adapter slots)                               │    │
│  │                                                                      │    │
│  │  Priority-based pruning when budget exceeded                         │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                              │                                               │
│                              ▼                                               │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                    Fallback Chain Manager                             │    │
│  │  ──────────────────────────────────────────────────────────────────  │    │
│  │  If preferred interface unavailable:                                  │    │
│  │  KV-Cache → Activation Steering → Logit Bias → Context Stuffing     │    │
│  │  (deepest)                                        (shallowest)       │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Relevance Gate

### Description
The first stage: filter retrieved memories based on relevance, confidence, and contextual appropriateness. Only memories that pass the gate are forwarded to the injection interfaces.

### Sub-task 1.1: `RelevanceGate`

**File:** `src/intrinsic/controller/gating.py`

```python
import logging
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum, auto

logger = logging.getLogger(__name__)


class GateDecision(Enum):
    INJECT = auto()     # Memory should be injected
    SKIP = auto()       # Memory is irrelevant
    DEFER = auto()      # Memory is marginally relevant — include if budget allows


@dataclass
class GatedMemory:
    """A memory that has passed (or been deferred by) the relevance gate."""
    memory_id: str
    source_text: str
    memory_type: str
    relevance_score: float
    confidence: float
    gate_decision: GateDecision
    gate_score: float             # Composite score used for priority ranking
    recommended_channel: Optional[str] = None  # Set by router
    injection_strength: float = 1.0


class RelevanceGate:
    """
    Multi-factor relevance gate for memory injection.
    
    Scoring formula:
        gate_score = w_r * relevance + w_c * confidence + w_t * type_weight + w_f * freshness
    
    Decision:
        gate_score >= inject_threshold → INJECT
        defer_threshold <= gate_score < inject_threshold → DEFER
        gate_score < defer_threshold → SKIP
    
    The gate prevents irrelevant memories from polluting the model's computation.
    Without it, low-relevance memories would add noise to hidden states.
    """

    def __init__(
        self,
        inject_threshold: float = 0.5,
        defer_threshold: float = 0.3,
        max_injections: int = 10,
        weights: Optional[Dict[str, float]] = None,
    ):
        self._inject_threshold = inject_threshold
        self._defer_threshold = defer_threshold
        self._max_injections = max_injections
        self._weights = weights or {
            "relevance": 0.4,
            "confidence": 0.25,
            "type_weight": 0.2,
            "freshness": 0.15,
        }

    def evaluate(
        self,
        memories: List[Dict],
        current_turn: int = 0,
    ) -> List[GatedMemory]:
        """
        Evaluate all retrieved memories and assign gate decisions.
        
        memories: List of dicts with keys:
            - memory_id, source_text, memory_type
            - relevance_score, confidence
            - created_at (turn number), last_accessed (turn number)
        """
        gated = []
        for mem in memories:
            score = self._compute_gate_score(mem, current_turn)
            
            if score >= self._inject_threshold:
                decision = GateDecision.INJECT
            elif score >= self._defer_threshold:
                decision = GateDecision.DEFER
            else:
                decision = GateDecision.SKIP

            gated.append(GatedMemory(
                memory_id=mem["memory_id"],
                source_text=mem["source_text"],
                memory_type=mem.get("memory_type", "semantic_fact"),
                relevance_score=mem.get("relevance_score", 0.0),
                confidence=mem.get("confidence", 1.0),
                gate_decision=decision,
                gate_score=score,
            ))

        # Sort by gate score (highest first)
        gated.sort(key=lambda x: x.gate_score, reverse=True)

        # Enforce max injections: promote top DEFERs to INJECT if budget allows
        inject_count = sum(1 for g in gated if g.gate_decision == GateDecision.INJECT)
        for g in gated:
            if g.gate_decision == GateDecision.DEFER and inject_count < self._max_injections:
                g.gate_decision = GateDecision.INJECT
                inject_count += 1

        return gated

    def _compute_gate_score(self, mem: Dict, current_turn: int) -> float:
        """Compute composite gate score."""
        w = self._weights
        
        relevance = mem.get("relevance_score", 0.0)
        confidence = mem.get("confidence", 1.0)
        type_weight = self._type_priority(mem.get("memory_type", "semantic_fact"))
        
        # Freshness: newer memories score higher
        created_at = mem.get("created_at_turn", 0)
        age = max(0, current_turn - created_at)
        freshness = 1.0 / (1.0 + age * 0.1)  # Decay factor
        
        score = (
            w["relevance"] * relevance +
            w["confidence"] * confidence +
            w["type_weight"] * type_weight +
            w["freshness"] * freshness
        )
        return score

    def _type_priority(self, memory_type: str) -> float:
        """Priority weight by memory type."""
        priorities = {
            "constraint": 1.0,       # Always respect constraints
            "semantic_fact": 0.9,     # Strong factual memories
            "knowledge": 0.85,        # Domain knowledge
            "preference": 0.8,        # User preferences
            "procedure": 0.75,        # How-to knowledge
            "episodic_event": 0.6,    # Recent events
            "hypothesis": 0.3,        # Unconfirmed — low priority
            "scratch": 0.1,           # Temporary — rarely inject
        }
        return priorities.get(memory_type, 0.5)
```

---

## Task 2: Interface Router

### Description
After the gate, the router decides which injection interface to use for each memory. The routing considers memory type, backend capabilities, current load, and injection budget.

### Sub-task 2.1: `InterfaceRouter`

**File:** `src/intrinsic/controller/router.py`

```python
import logging
from typing import Dict, List, Optional
from ..backends.base import InterfaceCapability, ModelBackend
from ..bus import InjectionChannel, MemoryVector
from .gating import GatedMemory, GateDecision

logger = logging.getLogger(__name__)


# Routing rules: memory_type → preferred channels (in priority order)
ROUTING_TABLE: Dict[str, List[InjectionChannel]] = {
    # Facts benefit most from deep injection (KV-cache gives full context)
    "semantic_fact": [
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.ACTIVATION_STEERING,
        InjectionChannel.LOGIT_BIAS,
    ],
    # Constraints must be strongly enforced — use multiple interfaces
    "constraint": [
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.ACTIVATION_STEERING,
        InjectionChannel.LOGIT_BIAS,
    ],
    # Preferences are behavioral — steering vectors work well
    "preference": [
        InjectionChannel.ACTIVATION_STEERING,
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.LOGIT_BIAS,
    ],
    # Episodic events — medium depth
    "episodic_event": [
        InjectionChannel.ACTIVATION_STEERING,
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.LOGIT_BIAS,
    ],
    # Procedures need full context — KV-cache is ideal
    "procedure": [
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.LOGIT_BIAS,
    ],
    # Knowledge — same as facts
    "knowledge": [
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.ACTIVATION_STEERING,
        InjectionChannel.LOGIT_BIAS,
    ],
    # Style/tone — late-layer activation steering
    "style": [
        InjectionChannel.ACTIVATION_STEERING,
    ],
    # Hypotheses — weak injection via logit bias only
    "hypothesis": [
        InjectionChannel.LOGIT_BIAS,
    ],
}


class InterfaceRouter:
    """
    Routes gated memories to the appropriate injection interface(s).
    
    The router:
    1. Looks up the routing table for the memory type
    2. Checks backend capabilities (which interfaces are available)
    3. Checks current budget (which interfaces have capacity)
    4. Assigns the best available interface
    5. For high-priority memories: may assign MULTIPLE interfaces
    """

    def __init__(
        self,
        backend: ModelBackend,
        multi_inject_threshold: float = 0.85,  # Score above which multiple interfaces are used
    ):
        self._backend = backend
        self._capabilities = backend.get_capabilities()
        self._multi_inject_threshold = multi_inject_threshold
        
        # Map interface capabilities to channels
        self._capability_map = {
            InjectionChannel.LOGIT_BIAS: InterfaceCapability.LOGIT_BIAS,
            InjectionChannel.KNN_LM: InterfaceCapability.LOGIT_DISTRIBUTION,
            InjectionChannel.ACTIVATION_STEERING: InterfaceCapability.ACTIVATION_HOOKS,
            InjectionChannel.KV_CACHE_INJECTION: InterfaceCapability.KV_CACHE_ACCESS,
            InjectionChannel.DYNAMIC_LORA: InterfaceCapability.WEIGHT_ACCESS,
        }

    def route(
        self,
        gated_memories: List[GatedMemory],
        budget: Optional[Dict[InjectionChannel, int]] = None,
    ) -> Dict[InjectionChannel, List[MemoryVector]]:
        """
        Route gated memories to injection channels.
        
        Returns: Dict mapping each channel to the MemoryVectors to inject through it.
        """
        routed: Dict[InjectionChannel, List[MemoryVector]] = {
            ch: [] for ch in InjectionChannel
        }
        
        budget = budget or self._default_budget()
        budget_used: Dict[InjectionChannel, int] = {ch: 0 for ch in InjectionChannel}

        for mem in gated_memories:
            if mem.gate_decision != GateDecision.INJECT:
                continue

            # Get preferred channels for this memory type
            preferred = ROUTING_TABLE.get(mem.memory_type, [InjectionChannel.LOGIT_BIAS])

            # Filter to available channels
            available = [
                ch for ch in preferred
                if self._is_channel_available(ch) and budget_used[ch] < budget.get(ch, 0)
            ]

            if not available:
                # Fallback: context stuffing (always available)
                available = [InjectionChannel.CONTEXT_STUFFING]

            # Create MemoryVector
            mv = MemoryVector(
                memory_id=mem.memory_id,
                source_text=mem.source_text,
                relevance_score=mem.relevance_score,
                injection_strength=mem.confidence * mem.gate_score,
            )

            # Assign primary channel
            primary_channel = available[0]
            routed[primary_channel].append(mv)
            budget_used[primary_channel] += 1

            # For high-priority memories: also assign secondary channel
            if mem.gate_score >= self._multi_inject_threshold and len(available) > 1:
                secondary = available[1]
                # Clone the vector for secondary channel (may need different encoding)
                mv_secondary = MemoryVector(
                    memory_id=mem.memory_id,
                    source_text=mem.source_text,
                    relevance_score=mem.relevance_score,
                    injection_strength=mem.confidence * mem.gate_score * 0.5,  # Reduced strength for secondary
                )
                routed[secondary].append(mv_secondary)
                budget_used[secondary] += 1

            mem.recommended_channel = primary_channel.value

        # Log routing decisions
        for ch, vectors in routed.items():
            if vectors:
                logger.info(f"Routed {len(vectors)} memories to {ch.value}")

        return routed

    def _is_channel_available(self, channel: InjectionChannel) -> bool:
        """Check if a channel is supported by the backend."""
        if channel == InjectionChannel.CONTEXT_STUFFING:
            return True  # Always available (existing RAG)
        
        required_cap = self._capability_map.get(channel)
        if required_cap is None:
            return False
        
        return bool(self._capabilities & required_cap)

    def _default_budget(self) -> Dict[InjectionChannel, int]:
        """Default per-channel injection budget (max memories per channel)."""
        return {
            InjectionChannel.KV_CACHE_INJECTION: 5,
            InjectionChannel.ACTIVATION_STEERING: 8,
            InjectionChannel.LOGIT_BIAS: 15,
            InjectionChannel.KNN_LM: 10,
            InjectionChannel.DYNAMIC_LORA: 2,
            InjectionChannel.CONTEXT_STUFFING: 20,
        }
```

---

## Task 3: Complete Controller Pipeline

### Description
The top-level controller that ties together gating, routing, strength calibration, and bus dispatch.

### Sub-task 3.1: `MemoryController`

**File:** `src/intrinsic/controller/controller.py`

```python
import logging
from typing import Any, Dict, List, Optional
from ..bus import IntrinsicMemoryBus, InjectionChannel, MemoryVector
from ..backends.base import ModelBackend
from .gating import RelevanceGate, GatedMemory, GateDecision
from .router import InterfaceRouter

logger = logging.getLogger(__name__)


class MemoryController:
    """
    The top-level controller for intrinsic memory injection.
    
    Pipeline per generation request:
    1. Receive retrieved memories from MemoryRetriever
    2. Pass through RelevanceGate → filter/score
    3. Pass through InterfaceRouter → assign channels
    4. Calibrate injection strengths
    5. Dispatch to IntrinsicMemoryBus
    6. Track what was injected for auditing
    
    The Controller is the SINGLE POINT OF CONTROL for all injection decisions.
    No interface injects memory independently — all go through the Controller.
    """

    def __init__(
        self,
        backend: ModelBackend,
        bus: IntrinsicMemoryBus,
        gate: Optional[RelevanceGate] = None,
        router: Optional[InterfaceRouter] = None,
        fallback_to_context: bool = True,
    ):
        self._backend = backend
        self._bus = bus
        self._gate = gate or RelevanceGate()
        self._router = router or InterfaceRouter(backend)
        self._fallback_to_context = fallback_to_context
        
        self._current_turn = 0
        self._injection_history: List[Dict] = []

    async def process_retrieval(
        self,
        retrieved_memories: List[Dict],
        query: str,
        session_id: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Main entry point: process retrieved memories and dispatch injections.
        
        retrieved_memories: List of dicts from MemoryRetriever with:
            - memory_id, source_text, memory_type
            - relevance_score, confidence
            - metadata (optional)
        
        Returns: injection summary for logging/debugging
        """
        # Step 1: Relevance Gate
        gated = self._gate.evaluate(retrieved_memories, self._current_turn)
        
        inject_count = sum(1 for g in gated if g.gate_decision == GateDecision.INJECT)
        skip_count = sum(1 for g in gated if g.gate_decision == GateDecision.SKIP)
        defer_count = sum(1 for g in gated if g.gate_decision == GateDecision.DEFER)
        
        logger.info(
            f"Gate results: {inject_count} inject, {defer_count} defer, {skip_count} skip "
            f"(from {len(retrieved_memories)} retrieved)"
        )

        # Step 2: Route to interfaces
        routed = self._router.route(gated)

        # Step 3: Calibrate strengths
        self._calibrate_strengths(routed)

        # Step 4: Dispatch to bus
        for channel, vectors in routed.items():
            if vectors and channel != InjectionChannel.CONTEXT_STUFFING:
                self._bus.enqueue(channel, vectors)

        # Flush all channels
        await self._bus.flush()

        # Step 5: Track injection
        summary = {
            "turn": self._current_turn,
            "query": query[:100],
            "retrieved_count": len(retrieved_memories),
            "gated_inject": inject_count,
            "gated_skip": skip_count,
            "routed": {ch.value: len(vecs) for ch, vecs in routed.items() if vecs},
            "context_stuffing_fallback": len(routed.get(InjectionChannel.CONTEXT_STUFFING, [])),
        }
        self._injection_history.append(summary)
        
        return summary

    def _calibrate_strengths(self, routed: Dict[InjectionChannel, List[MemoryVector]]) -> None:
        """
        Fine-tune injection strengths based on inter-memory relationships.
        
        Interference detection:
        If two memories conflict (e.g., "User lives in Paris" vs "User moved to London"),
        reduce both strengths to avoid confusing the model.
        
        Reinforcement detection:
        If two memories agree, slightly boost both.
        """
        all_vectors = []
        for vectors in routed.values():
            all_vectors.extend(vectors)

        if len(all_vectors) <= 1:
            return

        # Simple interference check based on cosine similarity of source texts
        # (Full implementation would use embedding similarity)
        for i in range(len(all_vectors)):
            for j in range(i + 1, len(all_vectors)):
                vi, vj = all_vectors[i], all_vectors[j]
                # Simple heuristic: if same memory type and low text overlap, potential conflict
                if vi.source_text == vj.source_text:
                    continue  # Same memory, skip
                
                # Check for conflicting information (simplified)
                # Full implementation: compute embedding similarity and check for contradiction
                # For now, just ensure total injection doesn't exceed safe levels
                pass

        # Global strength normalization: if too many injections, reduce all
        total_strength = sum(v.injection_strength for v in all_vectors)
        max_total_strength = 5.0  # Configurable cap

        if total_strength > max_total_strength:
            scale = max_total_strength / total_strength
            for v in all_vectors:
                v.injection_strength *= scale
            logger.debug(f"Normalized injection strengths by factor {scale:.2f}")

    def advance_turn(self) -> None:
        """Advance to next conversation turn."""
        self._current_turn += 1

    def get_injection_history(self) -> List[Dict]:
        return self._injection_history.copy()

    def get_stats(self) -> Dict:
        return {
            "current_turn": self._current_turn,
            "total_injections": len(self._injection_history),
            "bus_log": self._bus.get_injection_log()[-10:],  # Last 10
        }
```

---

## Task 4: Fallback Chain Manager

### Description
When the preferred injection interface is unavailable, the fallback chain ensures memories are still injected via a less deep but still functional interface.

### Sub-task 4.1: `FallbackChain`

```python
class FallbackChain:
    """
    Fallback chain for interface degradation.
    
    Order (deepest to shallowest):
    1. KV-Cache Injection (deepest — virtual context)
    2. Activation Steering (deep — hidden state influence)
    3. kNN-LM Interpolation (medium — distribution mixing)
    4. Logit Bias (shallow — token probability nudging)
    5. Context Stuffing (shallowest — existing RAG behavior)
    
    Each level provides less influence but wider compatibility.
    API-only backends start at level 4 (Logit Bias).
    """

    CHAIN = [
        InjectionChannel.KV_CACHE_INJECTION,
        InjectionChannel.ACTIVATION_STEERING,
        InjectionChannel.KNN_LM,
        InjectionChannel.LOGIT_BIAS,
        InjectionChannel.CONTEXT_STUFFING,
    ]

    def __init__(self, backend: ModelBackend):
        self._backend = backend
        self._capabilities = backend.get_capabilities()

    def get_best_available(self, preferred: InjectionChannel) -> InjectionChannel:
        """
        Get the best available channel at or below the preferred level.
        
        If the preferred channel is available, return it.
        Otherwise, walk down the chain until a supported channel is found.
        """
        preferred_idx = self.CHAIN.index(preferred) if preferred in self.CHAIN else 0
        
        for channel in self.CHAIN[preferred_idx:]:
            if channel == InjectionChannel.CONTEXT_STUFFING:
                return channel  # Always available
            
            cap_map = {
                InjectionChannel.KV_CACHE_INJECTION: InterfaceCapability.KV_CACHE_ACCESS,
                InjectionChannel.ACTIVATION_STEERING: InterfaceCapability.ACTIVATION_HOOKS,
                InjectionChannel.KNN_LM: InterfaceCapability.LOGIT_DISTRIBUTION,
                InjectionChannel.LOGIT_BIAS: InterfaceCapability.LOGIT_BIAS,
            }
            
            required = cap_map.get(channel)
            if required and (self._capabilities & required):
                return channel
        
        return InjectionChannel.CONTEXT_STUFFING  # Ultimate fallback

    def get_available_chain(self) -> List[InjectionChannel]:
        """Return the full fallback chain filtered to available interfaces."""
        available = []
        for channel in self.CHAIN:
            if channel == InjectionChannel.CONTEXT_STUFFING:
                available.append(channel)
                continue
            cap_map = {
                InjectionChannel.KV_CACHE_INJECTION: InterfaceCapability.KV_CACHE_ACCESS,
                InjectionChannel.ACTIVATION_STEERING: InterfaceCapability.ACTIVATION_HOOKS,
                InjectionChannel.KNN_LM: InterfaceCapability.LOGIT_DISTRIBUTION,
                InjectionChannel.LOGIT_BIAS: InterfaceCapability.LOGIT_BIAS,
            }
            required = cap_map.get(channel)
            if required and (self._capabilities & required):
                available.append(channel)
        return available
```

---

## Task 5: Integration with Existing Memory Pipeline

### Description
Wire the Controller into the existing `SeamlessMemoryProvider` and `MemoryRetriever` so that intrinsic injection happens transparently alongside existing RAG behavior.

### Sub-task 5.1: Enhanced `SeamlessMemoryProvider`

```python
# Additions to SeamlessMemoryProvider (src/memory/seamless_provider.py)

class SeamlessMemoryProvider:
    """Enhanced with intrinsic memory support."""
    
    def __init__(
        self,
        orchestrator,
        max_context_tokens=1500,
        auto_store=True,
        relevance_threshold=0.3,
        # NEW: intrinsic memory components
        memory_controller=None,  # MemoryController from Phase 5
    ):
        self.orchestrator = orchestrator
        self.max_context_tokens = max_context_tokens
        self.auto_store = auto_store
        self.relevance_threshold = relevance_threshold
        self._controller = memory_controller

    async def process_turn(self, tenant_id, user_message, ...):
        """Process turn with both RAG and intrinsic injection."""
        
        # Step 1: Retrieve memories (existing pipeline)
        memory_context, injected_memories = await self._retrieve_context(
            tenant_id, user_message
        )
        
        # Step 2: If controller is available, route through intrinsic pipeline
        injection_summary = None
        if self._controller:
            # Convert RetrievedMemory objects to dicts for the controller
            memory_dicts = [
                {
                    "memory_id": str(mem.record.id),
                    "source_text": mem.record.text,
                    "memory_type": getattr(mem.record.type, 'value', str(mem.record.type)),
                    "relevance_score": mem.relevance_score,
                    "confidence": mem.record.confidence,
                }
                for mem in injected_memories
            ]
            
            injection_summary = await self._controller.process_retrieval(
                retrieved_memories=memory_dicts,
                query=user_message,
            )
        
        # Step 3: Return results (both RAG context and intrinsic injection info)
        return SeamlessTurnResult(
            memory_context=memory_context,
            injected_memories=injected_memories,
            stored_count=stored_count,
            reconsolidation_applied=reconsolidation_applied,
            injection_summary=injection_summary,  # NEW
        )
```

---

## Acceptance Criteria

1. `RelevanceGate` correctly filters memories based on multi-factor scoring
2. Gate respects memory type priorities (constraints > facts > hypotheses)
3. `InterfaceRouter` routes memories to correct channels based on type and backend capabilities
4. High-priority memories get multi-interface injection
5. `MemoryController` orchestrates the full gate → route → calibrate → dispatch pipeline
6. Strength calibration prevents over-injection (total strength cap)
7. `FallbackChain` degrades gracefully: KV → Activation → Logit → Context
8. API-only backends route everything to Logit Bias and Context Stuffing
9. Integration with `SeamlessMemoryProvider` is transparent and backward-compatible
10. Injection history is tracked for debugging

## Estimated Effort
- **Duration:** 2-3 weeks
- **Complexity:** High (routing logic must handle many edge cases)
- **Risk:** Medium (routing decisions significantly affect quality — needs empirical tuning)

## Testing Strategy
1. Unit test `RelevanceGate` with various memory profiles (high/low relevance, different types)
2. Unit test `InterfaceRouter` with different backend capabilities
3. Integration test full pipeline: retrieve → gate → route → dispatch → verify bus state
4. Fallback test: disable interfaces one by one and verify graceful degradation
5. Multi-interface test: high-priority memory injected through 2 channels simultaneously
6. Budget enforcement test: exceed budget and verify priority-based pruning
