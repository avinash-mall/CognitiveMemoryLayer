# Phase 9: Integration, Migration & Backward Compatibility

**Intrinsic Phase I-9** (planned; not yet implemented). See [BaseCMLStatus.md](../BaseCML/BaseCMLStatus.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

Phase 9 brings all the intrinsic memory interfaces together into a cohesive system, integrated with the existing CognitiveMemoryLayer codebase. This phase ensures that the existing REST API, memory stores, and background workers continue to function while the new intrinsic memory capabilities are layered on top.

### Core Objectives
1. **Full end-to-end integration:** Wire all phases (1-8) into the existing application startup and request lifecycle
2. **Backward compatibility:** Existing API clients continue to work unchanged
3. **Hybrid mode:** Support running RAG and intrinsic memory simultaneously
4. **Migration path:** Gradually migrate from pure RAG to intrinsic memory
5. **Configuration-driven:** Feature flags control which interfaces are active

### Integration Points
- `src/api/app.py` — Application startup, component initialization
- `src/api/routes.py` — New intrinsic memory API endpoints
- `src/memory/seamless_provider.py` — Transparent integration point
- `src/memory/orchestrator.py` — Wire Controller alongside existing orchestration
- `.env` / `src/core/config.py` — Extended configuration

### Dependencies
- All prior phases (1-8)
- Existing codebase: API, storage, orchestrator, retriever

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                    Integrated System Architecture                             │
│                                                                              │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                      REST API Layer (FastAPI)                         │    │
│  │                                                                      │    │
│  │  Existing Endpoints            │    New Endpoints                    │    │
│  │  ─────────────────             │    ──────────────                   │    │
│  │  POST /memory/write            │    GET  /intrinsic/status          │    │
│  │  POST /memory/read             │    POST /intrinsic/configure       │    │
│  │  POST /memory/turn             │    GET  /intrinsic/diagnostics     │    │
│  │  POST /memory/forget           │    POST /intrinsic/encode          │    │
│  │  (unchanged — backward compat) │    POST /intrinsic/inject_test     │    │
│  │                                │    GET  /intrinsic/cache/stats     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│  ┌───────────────────────────────────▼──────────────────────────────────┐    │
│  │                   Seamless Memory Provider                            │    │
│  │  ─────────────────────────────────────────────                        │    │
│  │  process_turn():                                                      │    │
│  │    1. Retrieve memories (existing pipeline)          ← RAG path      │    │
│  │    2. Route through MemoryController                 ← Intrinsic     │    │
│  │    3. Return both context text AND injection state    ← Combined     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                      │                                       │
│       ┌──────────────────────────────┼──────────────────────────┐           │
│       ▼                              ▼                          ▼           │
│  ┌──────────┐   ┌───────────────────────────────────┐   ┌──────────────┐  │
│  │  Existing │   │      Intrinsic Memory Pipeline     │   │  Background  │  │
│  │  RAG Path │   │  ─────────────────────────────── │   │  Workers     │  │
│  │           │   │                                    │   │              │  │
│  │  MemoryOrc│   │  Controller → Gate → Router →     │   │  Consolid.   │  │
│  │  Retriever│   │  Bus → Interfaces → Model Backend │   │  Forgetting  │  │
│  │  PacketBld│   │                                    │   │  Maintenance │  │
│  └──────────┘   └───────────────────────────────────┘   └──────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Application Startup Integration

### Description
Wire the intrinsic memory system into the FastAPI application startup lifecycle. All components must be initialized in the correct order with proper dependency injection.

### Sub-task 1.1: Startup Initialization

**File:** `src/api/app.py` (modifications)

```python
# Additions to existing FastAPI app startup

async def initialize_intrinsic_memory(app):
    """
    Initialize the intrinsic memory system.
    Called during app startup, after existing components are ready.
    
    Order:
    1. Load intrinsic config
    2. Initialize Model Backend (based on config)
    3. Create Hook Manager
    4. Create Memory Bus
    5. Initialize interfaces (Logit, Activation, Synaptic, Weight)
    6. Create Controller
    7. Wire Controller into SeamlessMemoryProvider
    """
    from ..core.config import get_settings
    from ..intrinsic.config import IntrinsicMemoryConfig
    from ..intrinsic.backends.base import InterfaceCapability
    from ..intrinsic.bus import IntrinsicMemoryBus
    from ..intrinsic.hooks.manager import HookManager, SafetyGuard
    from ..intrinsic.inspector import ModelInspector
    from ..intrinsic.controller.controller import MemoryController
    from ..intrinsic.controller.gating import RelevanceGate
    from ..intrinsic.controller.router import InterfaceRouter
    
    settings = get_settings()
    intrinsic_config = IntrinsicMemoryConfig.from_yaml(settings.raw_config)
    
    if not intrinsic_config.enabled:
        logger.info("Intrinsic memory is disabled. Running in RAG-only mode.")
        app.state.intrinsic_enabled = False
        return
    
    logger.info("Initializing intrinsic memory system...")
    
    # Step 1: Backend
    backend = await _create_backend(intrinsic_config)
    app.state.model_backend = backend
    
    # Step 2: Hook Manager
    safety = SafetyGuard(
        max_norm_ratio=intrinsic_config.max_norm_ratio,
        nan_action=intrinsic_config.nan_action,
    )
    hook_manager = HookManager(backend, safety)
    app.state.hook_manager = hook_manager
    
    # Step 3: Memory Bus
    bus = IntrinsicMemoryBus()
    app.state.memory_bus = bus
    
    # Step 4: Inspector
    inspector = ModelInspector(backend)
    app.state.model_inspector = inspector
    logger.info(f"Model capabilities: {inspector.get_capability_report()}")
    
    # Step 5: Interfaces (conditional on capabilities)
    caps = backend.get_capabilities()
    
    if intrinsic_config.logit_bias_enabled and (caps & InterfaceCapability.LOGIT_BIAS):
        from ..intrinsic.interfaces.logit import setup_logit_interface
        logit_engine = await setup_logit_interface(
            bus, backend, app.state.tokenizer, app.state.llm_client
        )
        app.state.logit_engine = logit_engine
        logger.info("Logit interface initialized")
    
    if intrinsic_config.activation_steering_enabled and (caps & InterfaceCapability.ACTIVATION_HOOKS):
        from ..intrinsic.interfaces.activation import setup_activation_interface
        activation_engine = await setup_activation_interface(
            bus, backend, hook_manager, app.state.tokenizer
        )
        app.state.activation_engine = activation_engine
        logger.info("Activation interface initialized")
    
    if intrinsic_config.kv_cache_injection_enabled and (caps & InterfaceCapability.KV_CACHE_ACCESS):
        from ..intrinsic.interfaces.synaptic import SynapticInterface
        synaptic = SynapticInterface(backend, app.state.tokenizer)
        bus.register_handler(InjectionChannel.KV_CACHE_INJECTION, synaptic.process_memories)
        app.state.synaptic_interface = synaptic
        logger.info("Synaptic interface initialized")
    
    if intrinsic_config.dynamic_lora_enabled and (caps & InterfaceCapability.WEIGHT_ACCESS):
        from ..intrinsic.interfaces.weight import setup_weight_interface
        adapter_manager = await setup_weight_interface(
            bus, backend, app.state.adapter_registry, app.state.embedding_client
        )
        app.state.adapter_manager = adapter_manager
        logger.info("Weight adaptation interface initialized")
    
    # Step 6: Controller
    gate = RelevanceGate()
    router = InterfaceRouter(backend)
    controller = MemoryController(
        backend=backend,
        bus=bus,
        gate=gate,
        router=router,
    )
    app.state.memory_controller = controller
    
    # Step 7: Wire into SeamlessMemoryProvider
    if hasattr(app.state, 'seamless_provider'):
        app.state.seamless_provider._controller = controller
    
    app.state.intrinsic_enabled = True
    logger.info("Intrinsic memory system fully initialized")


async def _create_backend(config):
    """Create the appropriate backend based on configuration."""
    if config.backend_type == "local_transformer":
        from ..intrinsic.backends.local_transformer import LocalTransformerBackend
        import torch
        dtype_map = {"float16": torch.float16, "float32": torch.float32, "bfloat16": torch.bfloat16}
        return LocalTransformerBackend(
            model_name_or_path=config.model_path,
            device=config.device,
            dtype=dtype_map.get(config.dtype, torch.float16),
        )
    elif config.backend_type == "api_only":
        from ..intrinsic.backends.api_only import APIOnlyBackend
        from ..utils.llm import get_llm_client
        client = get_llm_client()
        return APIOnlyBackend(model_name=config.model_path, api_client=client)
    else:
        raise ValueError(f"Unknown backend type: {config.backend_type}")


# Add to existing lifespan or startup event:
# await initialize_intrinsic_memory(app)
```

---

## Task 2: New API Endpoints

### Description
Add API endpoints for managing and monitoring the intrinsic memory system.

### Sub-task 2.1: Intrinsic Memory API Routes

**File:** `src/api/intrinsic_routes.py` (new file)

```python
from fastapi import APIRouter, Depends, HTTPException, Request
from pydantic import BaseModel
from typing import Any, Dict, List, Optional

router = APIRouter(prefix="/intrinsic", tags=["intrinsic_memory"])


class IntrinsicStatusResponse(BaseModel):
    enabled: bool
    backend_type: str
    model_name: str
    capabilities: Dict[str, bool]
    active_interfaces: List[str]
    available_interfaces: List[str]


class DiagnosticsResponse(BaseModel):
    hook_stats: Dict[str, Any]
    bus_log: List[Dict]
    controller_stats: Dict[str, Any]
    cache_stats: Dict[str, Any]
    injection_history: List[Dict]


class EncodeRequest(BaseModel):
    memory_id: str
    memory_text: str
    memory_type: str = "semantic_fact"
    encoding_mode: str = "adaptive"


class EncodeResponse(BaseModel):
    memory_id: str
    encoding_mode: str
    has_logit: bool
    has_activation: bool
    has_kv: bool
    encoding_time_ms: float


class ConfigureRequest(BaseModel):
    logit_bias_enabled: Optional[bool] = None
    activation_steering_enabled: Optional[bool] = None
    kv_cache_enabled: Optional[bool] = None
    injection_strength: Optional[float] = None
    gate_threshold: Optional[float] = None


@router.get("/status", response_model=IntrinsicStatusResponse)
async def get_intrinsic_status(request: Request):
    """Get the status of the intrinsic memory system."""
    if not getattr(request.app.state, 'intrinsic_enabled', False):
        return IntrinsicStatusResponse(
            enabled=False,
            backend_type="none",
            model_name="",
            capabilities={},
            active_interfaces=[],
            available_interfaces=[],
        )
    
    inspector = request.app.state.model_inspector
    report = inspector.get_capability_report()
    
    return IntrinsicStatusResponse(
        enabled=True,
        backend_type=report["architecture"],
        model_name=report["model"],
        capabilities=report["capabilities"],
        active_interfaces=report["available_interfaces"],
        available_interfaces=report["available_interfaces"],
    )


@router.get("/diagnostics", response_model=DiagnosticsResponse)
async def get_diagnostics(request: Request):
    """Get detailed diagnostics for the intrinsic memory system."""
    if not getattr(request.app.state, 'intrinsic_enabled', False):
        raise HTTPException(status_code=404, detail="Intrinsic memory is not enabled")
    
    controller = request.app.state.memory_controller
    hook_manager = request.app.state.hook_manager
    bus = request.app.state.memory_bus
    
    return DiagnosticsResponse(
        hook_stats=hook_manager.get_diagnostics(),
        bus_log=bus.get_injection_log()[-20:],
        controller_stats=controller.get_stats(),
        cache_stats={},  # From Phase 7 TieredMemoryStore
        injection_history=controller.get_injection_history()[-10:],
    )


@router.post("/encode", response_model=EncodeResponse)
async def encode_memory(body: EncodeRequest, request: Request):
    """Manually trigger encoding of a memory for testing."""
    if not getattr(request.app.state, 'intrinsic_enabled', False):
        raise HTTPException(status_code=404, detail="Intrinsic memory is not enabled")
    
    from ..intrinsic.encoding.hippocampal_encoder import HippocampalEncoder
    encoder = HippocampalEncoder(
        backend=request.app.state.model_backend,
        tokenizer=request.app.state.tokenizer,
        llm_client=getattr(request.app.state, 'llm_client', None),
        embedding_client=getattr(request.app.state, 'embedding_client', None),
    )
    
    encoded = await encoder.encode(
        memory_id=body.memory_id,
        memory_text=body.memory_text,
        memory_type=body.memory_type,
        mode=body.encoding_mode,
    )
    
    return EncodeResponse(
        memory_id=encoded.memory_id,
        encoding_mode=encoded.encoding_method,
        has_logit=encoded.has_logit_representation(),
        has_activation=encoded.has_activation_representation(),
        has_kv=encoded.has_kv_representation(),
        encoding_time_ms=encoded.encoding_time_ms,
    )


@router.post("/configure")
async def configure_intrinsic(body: ConfigureRequest, request: Request):
    """Dynamically reconfigure intrinsic memory settings."""
    if not getattr(request.app.state, 'intrinsic_enabled', False):
        raise HTTPException(status_code=404, detail="Intrinsic memory is not enabled")
    
    # Apply configuration changes
    controller = request.app.state.memory_controller
    
    if body.gate_threshold is not None:
        controller._gate._inject_threshold = body.gate_threshold
    
    if body.injection_strength is not None:
        if hasattr(request.app.state, 'activation_engine'):
            request.app.state.activation_engine._default_strength = body.injection_strength
    
    return {"status": "configured", "applied": body.dict(exclude_none=True)}


@router.get("/cache/stats")
async def get_cache_stats(request: Request):
    """Get memory hierarchy cache statistics."""
    if not getattr(request.app.state, 'intrinsic_enabled', False):
        raise HTTPException(status_code=404, detail="Intrinsic memory is not enabled")
    
    tiered_store = getattr(request.app.state, 'tiered_store', None)
    if tiered_store:
        return tiered_store.get_stats()
    return {"status": "no tiered store configured"}
```

---

## Task 3: Enhanced `/memory/turn` Endpoint

### Description
Modify the existing turn processing endpoint to include intrinsic injection metadata in the response, while maintaining backward compatibility.

### Sub-task 3.1: Extended Turn Processing

```python
# Additions to existing ProcessTurnResponse (src/api/schemas.py)

class IntrinsicInjectionInfo(BaseModel):
    """Metadata about intrinsic memory injections performed during this turn."""
    enabled: bool = False
    gated_inject_count: int = 0
    gated_skip_count: int = 0
    channels_used: Dict[str, int] = {}  # channel_name → memory_count
    injection_strength_avg: float = 0.0
    fallback_to_context: int = 0


class ProcessTurnResponse(BaseModel):
    """Extended response with intrinsic injection info."""
    context: str                           # Existing: formatted memory context
    memories_used: int                     # Existing: count
    stored_count: int                      # Existing: memories stored
    reconsolidation_applied: bool          # Existing
    # NEW: Intrinsic injection metadata (optional for backward compat)
    intrinsic: Optional[IntrinsicInjectionInfo] = None
```

---

## Task 4: Migration Strategy

### Description
Define the migration path from pure RAG to intrinsic memory, supporting hybrid operation.

### Sub-task 4.1: Migration Modes

```python
class MigrationMode:
    """
    Migration modes for transitioning from RAG to intrinsic memory.
    
    Modes:
    1. RAG_ONLY: Existing behavior. Intrinsic system disabled.
    2. HYBRID_SHADOW: Intrinsic runs in shadow mode — computes injections
       but doesn't apply them. Logs what WOULD have been injected.
       Used for A/B testing and validation.
    3. HYBRID_ADDITIVE: Both RAG context AND intrinsic injection active.
       Memory appears in prompt AND is injected into model internals.
       Redundant but safe for transition period.
    4. INTRINSIC_PRIMARY: Intrinsic injection is primary. RAG context
       only as fallback for unsupported interfaces.
    5. INTRINSIC_ONLY: Pure intrinsic injection. No prompt-based context.
    
    Migration path: 1 → 2 → 3 → 4 → 5
    Each step can be validated before proceeding.
    """
    RAG_ONLY = "rag_only"
    HYBRID_SHADOW = "hybrid_shadow"
    HYBRID_ADDITIVE = "hybrid_additive"
    INTRINSIC_PRIMARY = "intrinsic_primary"
    INTRINSIC_ONLY = "intrinsic_only"


class MigrationManager:
    """Manages the RAG → Intrinsic migration."""

    def __init__(self, mode: str = MigrationMode.RAG_ONLY):
        self._mode = mode
        self._shadow_log: List[Dict] = []

    @property
    def mode(self) -> str:
        return self._mode

    def set_mode(self, mode: str) -> None:
        logger.info(f"Migration mode changed: {self._mode} → {mode}")
        self._mode = mode

    def should_run_rag(self) -> bool:
        """Should RAG context be included in prompt?"""
        return self._mode in (
            MigrationMode.RAG_ONLY,
            MigrationMode.HYBRID_SHADOW,
            MigrationMode.HYBRID_ADDITIVE,
            MigrationMode.INTRINSIC_PRIMARY,  # As fallback
        )

    def should_run_intrinsic(self) -> bool:
        """Should intrinsic injection be performed?"""
        return self._mode in (
            MigrationMode.HYBRID_SHADOW,
            MigrationMode.HYBRID_ADDITIVE,
            MigrationMode.INTRINSIC_PRIMARY,
            MigrationMode.INTRINSIC_ONLY,
        )

    def should_apply_intrinsic(self) -> bool:
        """Should intrinsic injections actually be applied (vs. shadow)?"""
        return self._mode in (
            MigrationMode.HYBRID_ADDITIVE,
            MigrationMode.INTRINSIC_PRIMARY,
            MigrationMode.INTRINSIC_ONLY,
        )

    def should_include_context_in_prompt(self) -> bool:
        """Should RAG context text be included in the LLM prompt?"""
        return self._mode in (
            MigrationMode.RAG_ONLY,
            MigrationMode.HYBRID_SHADOW,
            MigrationMode.HYBRID_ADDITIVE,
        )

    def log_shadow(self, injection_data: Dict) -> None:
        """Log what WOULD have been injected (shadow mode)."""
        if self._mode == MigrationMode.HYBRID_SHADOW:
            self._shadow_log.append(injection_data)

    def get_shadow_log(self) -> List[Dict]:
        return self._shadow_log.copy()
```

---

## Task 5: End-to-End Integration Tests

### Description
Comprehensive integration tests verifying the complete pipeline.

### Sub-task 5.1: Test Scenarios

```python
# tests/integration/test_intrinsic_integration.py

"""
Integration test scenarios for intrinsic memory.

Test Matrix:
┌──────────────────┬──────────┬────────────┬──────────┬──────────┐
│ Scenario         │ Backend  │ Interfaces │ Mode     │ Expected │
├──────────────────┼──────────┼────────────┼──────────┼──────────┤
│ API-only basic   │ api_only │ logit_bias │ hybrid   │ Pass     │
│ Local full       │ local    │ all        │ intrinsic│ Pass     │
│ Graceful degrade │ api_only │ all req'd  │ hybrid   │ Fallback │
│ Shadow mode      │ local    │ all        │ shadow   │ No inject│
│ Memory lifecycle  │ local    │ activation │ intrinsic│ Decay OK │
│ Multi-memory     │ local    │ all        │ intrinsic│ Compose  │
│ Backward compat  │ none     │ none       │ rag_only │ Unchanged│
│ Hot migration    │ local    │ all        │ 1→2→3→4 │ Smooth   │
└──────────────────┴──────────┴────────────┴──────────┴──────────┘
"""

import pytest

class TestIntrinsicIntegration:
    """End-to-end integration tests."""

    async def test_backward_compatibility_rag_only(self, app_client):
        """Existing API works unchanged when intrinsic is disabled."""
        # Write a memory
        resp = await app_client.post("/memory/write", json={
            "content": "User lives in Paris",
            "type": "semantic_fact",
        })
        assert resp.status_code == 200

        # Read it back via existing endpoint
        resp = await app_client.post("/memory/read", json={
            "query": "Where does the user live?",
        })
        assert resp.status_code == 200
        data = resp.json()
        assert "Paris" in data["context"]
        # intrinsic field should be None or not present
        assert data.get("intrinsic") is None or not data["intrinsic"]["enabled"]

    async def test_hybrid_additive_mode(self, intrinsic_app_client):
        """Both RAG context and intrinsic injection active."""
        # Write memory
        await intrinsic_app_client.post("/memory/write", json={
            "content": "User prefers vegetarian food",
            "type": "preference",
        })

        # Process turn
        resp = await intrinsic_app_client.post("/memory/turn", json={
            "user_message": "What should I eat for dinner?",
        })
        assert resp.status_code == 200
        data = resp.json()
        
        # RAG context should include the memory
        assert "vegetarian" in data["context"].lower()
        
        # Intrinsic injection should have been performed
        assert data["intrinsic"]["enabled"] is True
        assert data["intrinsic"]["gated_inject_count"] > 0

    async def test_fallback_chain(self, api_only_client):
        """API-only backend falls back to logit bias."""
        resp = await api_only_client.get("/intrinsic/status")
        data = resp.json()
        
        # Should only have logit_bias capability
        assert data["capabilities"]["logit_bias"] is True
        assert data["capabilities"]["activation_hooks"] is False

    async def test_temporal_decay(self, intrinsic_app_client):
        """Injected memories decay over turns."""
        # Inject a memory
        await intrinsic_app_client.post("/memory/write", json={
            "content": "Meeting at 3pm today",
            "type": "episodic_event",
        })

        # Process several turns
        for i in range(10):
            await intrinsic_app_client.post("/memory/turn", json={
                "user_message": f"Turn {i}: What's the weather like?",
            })

        # Check diagnostics — the memory should have decayed
        resp = await intrinsic_app_client.get("/intrinsic/diagnostics")
        data = resp.json()
        # Verify decay is happening (injection count should decrease)

    async def test_migration_shadow_to_active(self, intrinsic_app_client):
        """Migrate from shadow to active mode."""
        # Start in shadow mode
        await intrinsic_app_client.post("/intrinsic/configure", json={
            "migration_mode": "hybrid_shadow",
        })

        # Process a turn — injections logged but not applied
        resp = await intrinsic_app_client.post("/memory/turn", json={
            "user_message": "Tell me about Paris",
        })
        
        # Switch to active
        await intrinsic_app_client.post("/intrinsic/configure", json={
            "migration_mode": "hybrid_additive",
        })

        # Process another turn — injections now applied
        resp = await intrinsic_app_client.post("/memory/turn", json={
            "user_message": "Tell me about Paris",
        })
        data = resp.json()
        assert data["intrinsic"]["enabled"] is True
```

---

## Acceptance Criteria

1. Application starts cleanly with intrinsic memory enabled
2. Application starts cleanly with intrinsic memory disabled (RAG-only)
3. Existing API endpoints return identical responses when intrinsic is disabled
4. `/intrinsic/status` reports correct capabilities for each backend type
5. `/intrinsic/diagnostics` returns hook stats, bus logs, and injection history
6. `/memory/turn` response includes intrinsic injection metadata
7. Migration modes work: RAG_ONLY → HYBRID_SHADOW → HYBRID_ADDITIVE → INTRINSIC_PRIMARY
8. Shadow mode logs injections without applying them
9. Hot migration (changing mode at runtime) doesn't crash
10. All existing tests pass with intrinsic memory disabled
11. Integration tests pass for each backend type (local, API-only)

## Estimated Effort
- **Duration:** 3-4 weeks
- **Complexity:** High (many integration points across the codebase)
- **Risk:** Medium (backward compatibility must be preserved perfectly)

## Testing Strategy
1. Run full existing test suite with intrinsic disabled → all pass
2. Run full existing test suite with intrinsic enabled in shadow → all pass
3. Integration tests for each migration mode
4. API compatibility tests: existing client code works unchanged
5. Load test: verify intrinsic memory doesn't degrade throughput > 10%
6. Failure injection: kill backend, verify graceful degradation to RAG
