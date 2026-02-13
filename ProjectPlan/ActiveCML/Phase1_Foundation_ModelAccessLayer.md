# Phase 1: Foundation & Model Access Layer

**Intrinsic Phase I-1** (planned; not yet implemented). See [BaseCMLStatus.md](../BaseCML/BaseCMLStatus.md) for the mapping of I-1..I-10 to core CML phases.

## Overview

Phase 1 establishes the foundational infrastructure for intrinsic LLM memory integration. The current system operates as an external REST-based RAG tool — memories are retrieved as text chunks and stuffed into prompts. This phase builds the **Model Access Layer (MAL)** — the bridge between the CognitiveMemoryLayer and the LLM's internal computation graph (hidden states, attention KV-cache, logits, and optionally weights).

This phase does **not** implement any injection strategy; it creates the abstraction layer, hook system, and model introspection capabilities that all subsequent phases depend on.

### Goals
- Abstract away model-specific internals behind a unified interface
- Build a hook system for intercepting and modifying the forward pass
- Create model introspection utilities (layer shapes, attention heads, vocabulary mapping)
- Establish the `IntrinsicMemoryBus` — the central event/data channel connecting memory retrieval to injection points
- Define the memory vector format (latent representations vs. text)

### Dependencies
- PyTorch (local model support via vLLM, HuggingFace Transformers, or llama.cpp)
- Existing `src/utils/llm.py` OpenAI-compatible client (extended, not replaced)
- Existing `src/memory/orchestrator.py` (wired as upstream memory source)

---

## Architecture Diagram

```
┌──────────────────────────────────────────────────────────────────────────────┐
│                        Model Access Layer (MAL)                              │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
│  │  Model Registry   │   │ Hook Manager     │   │ Model Inspector         │  │
│  │  ─────────────    │   │ ──────────────   │   │ ───────────────         │  │
│  │  - model catalog  │   │ - register hooks │   │ - layer_count           │  │
│  │  - backend type   │   │ - lifecycle mgmt │   │ - hidden_dim            │  │
│  │  - capabilities   │   │ - hook ordering  │   │ - num_heads             │  │
│  │  - access mode    │   │ - safety guards  │   │ - vocab_size            │  │
│  └──────────────────┘   └──────────────────┘   │ - unembedding_matrix    │  │
│                                                  └─────────────────────────┘  │
│  ┌──────────────────────────────────────────────────────────────────────┐    │
│  │                     Intrinsic Memory Bus                             │    │
│  │  ─────────────────────────────────────────────────────────────────   │    │
│  │  Channels: logit_bias | activation_steering | kv_injection | lora   │    │
│  │  Events: pre_forward | layer_N_post | pre_logit | post_generate     │    │
│  └──────────────────────────────────────────────────────────────────────┘    │
│                                                                              │
│  ┌──────────────────┐   ┌──────────────────┐   ┌─────────────────────────┐  │
│  │  Backend:         │   │  Backend:         │   │  Backend:               │  │
│  │  LocalTransformer │   │  vLLM             │   │  API-Only (OpenAI)      │  │
│  │  (full access)    │   │  (hooks + logits) │   │  (logit bias only)      │  │
│  └──────────────────┘   └──────────────────┘   └─────────────────────────┘  │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Task 1: Model Backend Abstraction

### Description
Create a backend abstraction that unifies access to different LLM deployment modes. Each backend exposes different levels of access — from full PyTorch tensor access (local models) to API-only logit bias (OpenAI API). The system must gracefully degrade: if activation hooks aren't available, fall back to logit-level or prompt-level integration.

### Sub-task 1.1: Define `ModelBackend` Abstract Base Class

**File:** `src/intrinsic/backends/base.py`

```python
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Flag, auto
from typing import Any, Callable, Dict, List, Optional, Tuple
import torch


class InterfaceCapability(Flag):
    """What injection interfaces this backend supports."""
    NONE = 0
    LOGIT_BIAS = auto()          # Can apply token-level logit bias
    LOGIT_DISTRIBUTION = auto()  # Can access full logit distribution (kNN-LM)
    ACTIVATION_HOOKS = auto()    # Can hook into hidden states during forward pass
    KV_CACHE_ACCESS = auto()     # Can read/write KV-cache entries
    WEIGHT_ACCESS = auto()       # Can load/swap LoRA adapters at runtime
    EMBEDDING_ACCESS = auto()    # Can access embedding/unembedding matrices


@dataclass
class ModelSpec:
    """Introspection data about the model architecture."""
    model_name: str
    num_layers: int
    hidden_dim: int
    num_attention_heads: int
    num_kv_heads: int               # for GQA models
    head_dim: int
    vocab_size: int
    max_seq_length: int
    dtype: torch.dtype
    intermediate_dim: int           # MLP intermediate size
    rope_base: float = 10000.0
    has_bias: bool = False
    architecture: str = "unknown"   # "llama", "mistral", "qwen", etc.
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class HookHandle:
    """Reference to a registered hook for lifecycle management."""
    hook_id: str
    layer_idx: int
    hook_type: str   # "pre_forward", "post_forward", "pre_logit"
    _remove_fn: Optional[Callable] = None

    def remove(self):
        if self._remove_fn:
            self._remove_fn()


class ModelBackend(ABC):
    """
    Abstract interface to an LLM's internals.
    
    Implementations:
    - LocalTransformerBackend: full PyTorch access (HuggingFace, llama.cpp python bindings)
    - VLLMBackend: vLLM server with custom hook extensions
    - APIOnlyBackend: OpenAI/Anthropic API (logit_bias only)
    """

    @abstractmethod
    def get_capabilities(self) -> InterfaceCapability:
        """Return the set of interfaces this backend supports."""
        ...

    @abstractmethod
    def get_model_spec(self) -> ModelSpec:
        """Return architectural metadata about the model."""
        ...

    @abstractmethod
    def supports(self, capability: InterfaceCapability) -> bool:
        """Check if a specific capability is available."""
        ...

    # --- Hook Registration ---

    @abstractmethod
    def register_activation_hook(
        self,
        layer_idx: int,
        hook_fn: Callable[[torch.Tensor, int], Optional[torch.Tensor]],
        hook_type: str = "post_forward",
    ) -> HookHandle:
        """
        Register a hook on a transformer layer.
        
        hook_fn signature: (hidden_states: [batch, seq_len, hidden_dim], layer_idx: int) -> Optional[modified_states]
        If hook_fn returns None, original states are kept.
        
        hook_type: "pre_forward" | "post_forward" | "post_attention" | "post_mlp"
        """
        ...

    @abstractmethod
    def remove_all_hooks(self) -> None:
        """Remove all registered hooks (cleanup)."""
        ...

    # --- KV-Cache Access ---

    @abstractmethod
    def get_kv_cache(
        self, layer_idx: int
    ) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        """
        Get current KV-cache tensors for a layer.
        Returns: (key_states, value_states) each of shape [batch, num_kv_heads, seq_len, head_dim]
        Returns None if KV-cache access is not supported.
        """
        ...

    @abstractmethod
    def inject_kv_pairs(
        self,
        layer_idx: int,
        key_vectors: torch.Tensor,    # [num_virtual_tokens, num_kv_heads, head_dim]
        value_vectors: torch.Tensor,   # [num_virtual_tokens, num_kv_heads, head_dim]
        position: str = "prepend",     # "prepend" | "append" | "replace_range"
    ) -> None:
        """Inject pre-computed KV pairs into the active cache."""
        ...

    # --- Logit Access ---

    @abstractmethod
    def apply_logit_bias(
        self,
        bias_dict: Dict[int, float],
    ) -> None:
        """Apply token-level logit bias for next generation step."""
        ...

    @abstractmethod
    def get_logit_distribution(
        self,
    ) -> Optional[torch.Tensor]:
        """
        Get the full logit distribution from the last forward pass.
        Returns: [vocab_size] tensor, or None if not available.
        """
        ...

    # --- Weight / LoRA Access ---

    @abstractmethod
    def load_lora_adapter(
        self,
        adapter_path: str,
        adapter_name: str,
        scaling: float = 1.0,
    ) -> None:
        """Load a LoRA adapter and merge/unmerge with base weights."""
        ...

    @abstractmethod
    def unload_lora_adapter(self, adapter_name: str) -> None:
        """Unload a previously loaded LoRA adapter."""
        ...

    # --- Embedding Access ---

    @abstractmethod
    def get_unembedding_matrix(self) -> Optional[torch.Tensor]:
        """
        Get the unembedding (lm_head) weight matrix.
        Returns: [vocab_size, hidden_dim] tensor.
        Used for deriving "Identity V" semantic primes.
        """
        ...

    @abstractmethod
    def get_embedding_matrix(self) -> Optional[torch.Tensor]:
        """
        Get the input embedding weight matrix.
        Returns: [vocab_size, hidden_dim] tensor.
        """
        ...
```

### Sub-task 1.2: Implement `LocalTransformerBackend`

**File:** `src/intrinsic/backends/local_transformer.py`

This backend provides full access to a locally loaded HuggingFace Transformers model.

```python
import torch
from typing import Callable, Dict, List, Optional, Tuple
from transformers import AutoModelForCausalLM, AutoConfig
from .base import ModelBackend, ModelSpec, InterfaceCapability, HookHandle


class LocalTransformerBackend(ModelBackend):
    """Full-access backend for locally loaded HuggingFace models."""

    def __init__(self, model_name_or_path: str, device: str = "cuda", dtype=torch.float16):
        self._config = AutoConfig.from_pretrained(model_name_or_path)
        self._model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype=dtype, device_map=device
        )
        self._model.eval()
        self._hooks: Dict[str, torch.utils.hooks.RemovableHook] = {}
        self._hook_counter = 0
        self._last_logits: Optional[torch.Tensor] = None
        self._device = device

        # Cache model spec on init
        self._spec = self._introspect_model()

    def _introspect_model(self) -> ModelSpec:
        """Extract architectural metadata from the loaded model."""
        cfg = self._config
        # Handle different architecture naming conventions
        num_layers = getattr(cfg, 'num_hidden_layers', getattr(cfg, 'n_layer', 32))
        hidden_dim = getattr(cfg, 'hidden_size', getattr(cfg, 'n_embd', 4096))
        num_heads = getattr(cfg, 'num_attention_heads', getattr(cfg, 'n_head', 32))
        num_kv_heads = getattr(cfg, 'num_key_value_heads', num_heads)  # GQA
        head_dim = hidden_dim // num_heads
        vocab_size = getattr(cfg, 'vocab_size', 32000)
        max_seq = getattr(cfg, 'max_position_embeddings', 4096)
        intermediate = getattr(cfg, 'intermediate_size', hidden_dim * 4)

        # Detect architecture family
        arch_name = cfg.architectures[0] if hasattr(cfg, 'architectures') and cfg.architectures else "unknown"
        arch_map = {
            "LlamaForCausalLM": "llama",
            "MistralForCausalLM": "mistral",
            "Qwen2ForCausalLM": "qwen2",
            "GPTNeoXForCausalLM": "gpt-neox",
            "GemmaForCausalLM": "gemma",
        }
        architecture = arch_map.get(arch_name, arch_name.lower())

        return ModelSpec(
            model_name=str(cfg._name_or_path),
            num_layers=num_layers,
            hidden_dim=hidden_dim,
            num_attention_heads=num_heads,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            vocab_size=vocab_size,
            max_seq_length=max_seq,
            dtype=self._model.dtype,
            intermediate_dim=intermediate,
            architecture=architecture,
        )

    def get_capabilities(self) -> InterfaceCapability:
        return (
            InterfaceCapability.LOGIT_BIAS
            | InterfaceCapability.LOGIT_DISTRIBUTION
            | InterfaceCapability.ACTIVATION_HOOKS
            | InterfaceCapability.KV_CACHE_ACCESS
            | InterfaceCapability.WEIGHT_ACCESS
            | InterfaceCapability.EMBEDDING_ACCESS
        )

    def get_model_spec(self) -> ModelSpec:
        return self._spec

    def supports(self, capability: InterfaceCapability) -> bool:
        return bool(self.get_capabilities() & capability)

    def _get_layer(self, layer_idx: int):
        """Get the transformer layer module by index (handles different architectures)."""
        model = self._model
        # Try common attribute paths
        for attr_path in ["model.layers", "transformer.h", "gpt_neox.layers"]:
            obj = model
            for attr in attr_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj[layer_idx]
        raise ValueError(f"Cannot locate transformer layers for architecture: {self._spec.architecture}")

    def register_activation_hook(
        self,
        layer_idx: int,
        hook_fn: Callable[[torch.Tensor, int], Optional[torch.Tensor]],
        hook_type: str = "post_forward",
    ) -> HookHandle:
        self._hook_counter += 1
        hook_id = f"hook_{self._hook_counter}"
        layer = self._get_layer(layer_idx)

        if hook_type == "post_forward":
            def _wrapper(module, inputs, output):
                # output can be tuple (hidden_states, ...) or just hidden_states
                if isinstance(output, tuple):
                    hidden_states = output[0]
                    modified = hook_fn(hidden_states, layer_idx)
                    if modified is not None:
                        return (modified,) + output[1:]
                    return output
                else:
                    modified = hook_fn(output, layer_idx)
                    return modified if modified is not None else output

            handle = layer.register_forward_hook(_wrapper)

        elif hook_type == "pre_forward":
            def _wrapper(module, inputs):
                hidden_states = inputs[0]
                modified = hook_fn(hidden_states, layer_idx)
                if modified is not None:
                    return (modified,) + inputs[1:]
                return inputs

            handle = layer.register_forward_pre_hook(_wrapper)

        else:
            raise ValueError(f"Unsupported hook_type: {hook_type}")

        self._hooks[hook_id] = handle
        return HookHandle(
            hook_id=hook_id,
            layer_idx=layer_idx,
            hook_type=hook_type,
            _remove_fn=handle.remove,
        )

    def remove_all_hooks(self) -> None:
        for handle in self._hooks.values():
            handle.remove()
        self._hooks.clear()

    def get_kv_cache(self, layer_idx: int) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
        # KV cache is populated during generation; access via model's past_key_values
        # This requires intercepting during the forward pass
        # Implementation depends on generation loop integration (see Phase 4)
        return None  # Placeholder — wired in Phase 4

    def inject_kv_pairs(
        self,
        layer_idx: int,
        key_vectors: torch.Tensor,
        value_vectors: torch.Tensor,
        position: str = "prepend",
    ) -> None:
        # Detailed implementation in Phase 4 (Synaptic Interface)
        raise NotImplementedError("KV injection implemented in Phase 4")

    def apply_logit_bias(self, bias_dict: Dict[int, float]) -> None:
        # Store bias dict; applied in the logit processing hook
        self._pending_logit_bias = bias_dict

    def get_logit_distribution(self) -> Optional[torch.Tensor]:
        return self._last_logits

    def load_lora_adapter(self, adapter_path: str, adapter_name: str, scaling: float = 1.0) -> None:
        # Detailed implementation in Phase 8 (Weight Adaptation)
        raise NotImplementedError("LoRA loading implemented in Phase 8")

    def unload_lora_adapter(self, adapter_name: str) -> None:
        raise NotImplementedError("LoRA unloading implemented in Phase 8")

    def get_unembedding_matrix(self) -> Optional[torch.Tensor]:
        """Return lm_head weights: [vocab_size, hidden_dim]."""
        lm_head = getattr(self._model, 'lm_head', None)
        if lm_head is not None:
            return lm_head.weight.data
        return None

    def get_embedding_matrix(self) -> Optional[torch.Tensor]:
        """Return input embedding weights: [vocab_size, hidden_dim]."""
        for attr_path in ["model.embed_tokens", "transformer.wte", "gpt_neox.embed_in"]:
            obj = self._model
            for attr in attr_path.split("."):
                obj = getattr(obj, attr, None)
                if obj is None:
                    break
            if obj is not None:
                return obj.weight.data
        return None
```

### Sub-task 1.3: Implement `APIOnlyBackend`

**File:** `src/intrinsic/backends/api_only.py`

For OpenAI/Anthropic API where only logit bias is available. This is the graceful degradation path.

```python
import torch
from typing import Callable, Dict, Optional, Tuple
from .base import ModelBackend, ModelSpec, InterfaceCapability, HookHandle


class APIOnlyBackend(ModelBackend):
    """
    Backend for API-only LLM providers (OpenAI, Anthropic, etc.).
    Only supports logit_bias parameter — no activation or KV-cache access.
    Used as graceful degradation when local model is not available.
    """

    def __init__(self, model_name: str, api_client, tokenizer=None):
        self._model_name = model_name
        self._api_client = api_client  # existing OpenAICompatibleClient
        self._tokenizer = tokenizer
        self._pending_logit_bias: Dict[int, float] = {}

        # Approximate model specs (fetched from known model registry or API)
        self._spec = self._estimate_model_spec(model_name)

    def _estimate_model_spec(self, model_name: str) -> ModelSpec:
        """Estimate model specs from known model catalog."""
        # Known model dimensions (approximate — used for vector sizing)
        KNOWN_MODELS = {
            "gpt-4o": ModelSpec(model_name="gpt-4o", num_layers=0, hidden_dim=0,
                                num_attention_heads=0, num_kv_heads=0, head_dim=0,
                                vocab_size=200000, max_seq_length=128000, dtype=torch.float16,
                                intermediate_dim=0, architecture="gpt-4"),
            # ... other known models
        }
        return KNOWN_MODELS.get(model_name, ModelSpec(
            model_name=model_name, num_layers=0, hidden_dim=0, num_attention_heads=0,
            num_kv_heads=0, head_dim=0, vocab_size=100000, max_seq_length=8192,
            dtype=torch.float16, intermediate_dim=0, architecture="unknown"
        ))

    def get_capabilities(self) -> InterfaceCapability:
        return InterfaceCapability.LOGIT_BIAS

    def get_model_spec(self) -> ModelSpec:
        return self._spec

    def supports(self, capability: InterfaceCapability) -> bool:
        return bool(self.get_capabilities() & capability)

    # --- Logit bias is the ONLY supported interface ---

    def apply_logit_bias(self, bias_dict: Dict[int, float]) -> None:
        self._pending_logit_bias = bias_dict

    def get_pending_logit_bias(self) -> Dict[int, float]:
        """Called by the generation loop to fetch and clear pending bias."""
        bias = self._pending_logit_bias.copy()
        self._pending_logit_bias.clear()
        return bias

    # --- Everything else raises NotSupported or returns None ---

    def register_activation_hook(self, *args, **kwargs) -> HookHandle:
        raise NotImplementedError("Activation hooks not available for API-only backends")

    def remove_all_hooks(self) -> None:
        pass  # No hooks to remove

    def get_kv_cache(self, layer_idx: int) -> None:
        return None

    def inject_kv_pairs(self, *args, **kwargs) -> None:
        raise NotImplementedError("KV-cache not available for API-only backends")

    def get_logit_distribution(self) -> None:
        return None  # API doesn't expose raw logits

    def load_lora_adapter(self, *args, **kwargs) -> None:
        raise NotImplementedError("LoRA not available for API-only backends")

    def unload_lora_adapter(self, *args, **kwargs) -> None:
        raise NotImplementedError("LoRA not available for API-only backends")

    def get_unembedding_matrix(self) -> None:
        return None

    def get_embedding_matrix(self) -> None:
        return None
```

---

## Task 2: Hook Manager & Lifecycle

### Description
A centralized manager that handles hook registration, ordering, safety guards (norm explosion detection, NaN traps), and graceful cleanup. Multiple injection phases may register hooks on the same layer — the Hook Manager ensures deterministic execution order and isolation.

### Sub-task 2.1: Implement `HookManager`

**File:** `src/intrinsic/hooks/manager.py`

```python
import logging
from collections import defaultdict
from typing import Callable, Dict, List, Optional
import torch

from ..backends.base import ModelBackend, HookHandle

logger = logging.getLogger(__name__)


class SafetyGuard:
    """Monitors hook outputs for numerical instability."""

    def __init__(self, max_norm_ratio: float = 5.0, nan_action: str = "revert"):
        self.max_norm_ratio = max_norm_ratio
        self.nan_action = nan_action  # "revert" | "zero" | "raise"

    def check(
        self, original: torch.Tensor, modified: torch.Tensor, hook_id: str
    ) -> torch.Tensor:
        """
        Validate the modified tensor. If it's unstable, revert to original.
        
        Checks:
        1. NaN/Inf detection
        2. Norm explosion (modified norm >> original norm)
        3. Cosine divergence (optional, expensive)
        """
        if torch.isnan(modified).any() or torch.isinf(modified).any():
            logger.warning(f"[SafetyGuard] NaN/Inf detected in hook '{hook_id}', reverting")
            if self.nan_action == "revert":
                return original
            elif self.nan_action == "zero":
                return torch.where(torch.isfinite(modified), modified, torch.zeros_like(modified))
            else:
                raise RuntimeError(f"NaN detected in hook '{hook_id}'")

        orig_norm = torch.norm(original).item()
        mod_norm = torch.norm(modified).item()
        if orig_norm > 0 and (mod_norm / orig_norm) > self.max_norm_ratio:
            logger.warning(
                f"[SafetyGuard] Norm explosion in hook '{hook_id}': "
                f"{mod_norm:.2f} vs {orig_norm:.2f} (ratio={mod_norm/orig_norm:.2f})"
            )
            # Scale modified back to safe range
            safe_scale = self.max_norm_ratio * orig_norm / mod_norm
            return modified * safe_scale

        return modified


class HookManager:
    """
    Centralized hook lifecycle manager.
    
    Responsibilities:
    - Register/unregister hooks on model layers
    - Enforce execution order (priority-based)
    - Apply safety guards to hook outputs
    - Provide diagnostics (hook call counts, latencies)
    """

    def __init__(self, backend: ModelBackend, safety_guard: Optional[SafetyGuard] = None):
        self._backend = backend
        self._safety = safety_guard or SafetyGuard()
        self._hooks: Dict[str, HookHandle] = {}
        self._layer_hooks: Dict[int, List[str]] = defaultdict(list)  # layer_idx -> [hook_ids]
        self._priorities: Dict[str, int] = {}  # hook_id -> priority (lower = first)
        self._stats: Dict[str, Dict] = {}  # hook_id -> {calls: int, total_ms: float}
        self._enabled = True

    def register(
        self,
        layer_idx: int,
        hook_fn: Callable,
        hook_type: str = "post_forward",
        priority: int = 100,
        name: Optional[str] = None,
    ) -> str:
        """
        Register a hook on a specific layer.
        
        Lower priority values execute first.
        Returns the hook_id for later management.
        """
        def _guarded_hook(hidden_states: torch.Tensor, layer_idx: int):
            if not self._enabled:
                return None
            original = hidden_states.clone()
            result = hook_fn(hidden_states, layer_idx)
            if result is not None:
                result = self._safety.check(original, result, handle.hook_id)
            return result

        handle = self._backend.register_activation_hook(layer_idx, _guarded_hook, hook_type)
        if name:
            handle.hook_id = name
        self._hooks[handle.hook_id] = handle
        self._layer_hooks[layer_idx].append(handle.hook_id)
        self._priorities[handle.hook_id] = priority
        self._stats[handle.hook_id] = {"calls": 0, "total_ms": 0.0}

        logger.info(f"Registered hook '{handle.hook_id}' on layer {layer_idx} (priority={priority})")
        return handle.hook_id

    def unregister(self, hook_id: str) -> None:
        """Remove a specific hook."""
        if hook_id in self._hooks:
            self._hooks[hook_id].remove()
            layer_idx = self._hooks[hook_id].layer_idx
            self._layer_hooks[layer_idx].remove(hook_id)
            del self._hooks[hook_id]
            del self._priorities[hook_id]
            del self._stats[hook_id]

    def unregister_all(self) -> None:
        """Remove all hooks."""
        self._backend.remove_all_hooks()
        self._hooks.clear()
        self._layer_hooks.clear()
        self._priorities.clear()
        self._stats.clear()

    def enable(self) -> None:
        self._enabled = True

    def disable(self) -> None:
        """Temporarily disable all hooks without removing them."""
        self._enabled = False

    def get_diagnostics(self) -> Dict:
        return {
            "total_hooks": len(self._hooks),
            "hooks_per_layer": {k: len(v) for k, v in self._layer_hooks.items()},
            "enabled": self._enabled,
            "hook_stats": self._stats.copy(),
        }
```

---

## Task 3: Model Inspector & Vocabulary Mapper

### Description
Provide utilities to inspect the loaded model's architecture and build vocabulary mappings for the Logit Interface. The inspector answers questions like "how many layers?", "what's the hidden dim?", "what token ID maps to 'Paris'?" — information needed by every subsequent phase.

### Sub-task 3.1: Implement `ModelInspector`

**File:** `src/intrinsic/inspector.py`

```python
from typing import Dict, List, Optional, Tuple
import torch
from .backends.base import ModelBackend, ModelSpec, InterfaceCapability


class ModelInspector:
    """
    Introspection utilities for the loaded model.
    Provides layer info, dimension details, and vocabulary mapping.
    """

    def __init__(self, backend: ModelBackend):
        self._backend = backend
        self._spec = backend.get_model_spec()
        self._vocab_map: Optional[Dict[str, int]] = None  # lazy loaded

    @property
    def spec(self) -> ModelSpec:
        return self._spec

    @property
    def num_layers(self) -> int:
        return self._spec.num_layers

    @property
    def hidden_dim(self) -> int:
        return self._spec.hidden_dim

    @property
    def num_heads(self) -> int:
        return self._spec.num_attention_heads

    @property
    def head_dim(self) -> int:
        return self._spec.head_dim

    @property
    def vocab_size(self) -> int:
        return self._spec.vocab_size

    def recommended_injection_layers(self) -> List[int]:
        """
        Return recommended layer indices for activation steering.
        
        Research indicates:
        - Early layers (0-25%): token-level / syntactic
        - Middle layers (25-75%): semantic / conceptual — BEST for memory steering
        - Late layers (75-100%): output formatting / next-token prediction
        
        Returns indices in the middle third of the model.
        """
        n = self._spec.num_layers
        start = n // 3
        end = (2 * n) // 3
        return list(range(start, end))

    def get_unembedding_row(self, token_id: int) -> Optional[torch.Tensor]:
        """
        Get the unembedding vector for a specific token.
        This is the "Identity V" — the direction in hidden space that
        the model associates with producing this token.
        
        Useful for constructing semantic primes for injection.
        """
        unembed = self._backend.get_unembedding_matrix()
        if unembed is not None:
            return unembed[token_id]  # [hidden_dim]
        return None

    def compute_semantic_direction(
        self, token_ids_positive: List[int], token_ids_negative: List[int]
    ) -> Optional[torch.Tensor]:
        """
        Compute a semantic direction vector from contrasting token groups.
        
        E.g., positive=["Paris", "France"] vs negative=["London", "England"]
        yields a direction encoding "Paris/France-ness" vs "London/England-ness".
        
        Uses mean-centering on unembedding rows (Identity V approach).
        """
        unembed = self._backend.get_unembedding_matrix()
        if unembed is None:
            return None

        pos_vecs = torch.stack([unembed[tid] for tid in token_ids_positive])
        neg_vecs = torch.stack([unembed[tid] for tid in token_ids_negative])

        pos_mean = pos_vecs.mean(dim=0)
        neg_mean = neg_vecs.mean(dim=0)

        direction = pos_mean - neg_mean
        direction = direction / (torch.norm(direction) + 1e-8)  # normalize
        return direction

    def get_capability_report(self) -> Dict:
        """Human-readable capability report for this backend."""
        caps = self._backend.get_capabilities()
        return {
            "model": self._spec.model_name,
            "architecture": self._spec.architecture,
            "layers": self._spec.num_layers,
            "hidden_dim": self._spec.hidden_dim,
            "capabilities": {
                "logit_bias": bool(caps & InterfaceCapability.LOGIT_BIAS),
                "logit_distribution": bool(caps & InterfaceCapability.LOGIT_DISTRIBUTION),
                "activation_hooks": bool(caps & InterfaceCapability.ACTIVATION_HOOKS),
                "kv_cache_access": bool(caps & InterfaceCapability.KV_CACHE_ACCESS),
                "weight_access": bool(caps & InterfaceCapability.WEIGHT_ACCESS),
                "embedding_access": bool(caps & InterfaceCapability.EMBEDDING_ACCESS),
            },
            "recommended_injection_layers": self.recommended_injection_layers(),
            "available_interfaces": self._derive_available_interfaces(caps),
        }

    def _derive_available_interfaces(self, caps: InterfaceCapability) -> List[str]:
        """Determine which memory injection interfaces are available."""
        interfaces = []
        if caps & InterfaceCapability.LOGIT_BIAS:
            interfaces.append("logit_bias")
        if caps & InterfaceCapability.LOGIT_DISTRIBUTION:
            interfaces.append("knn_lm")
        if caps & InterfaceCapability.ACTIVATION_HOOKS:
            interfaces.append("activation_steering")
        if caps & InterfaceCapability.KV_CACHE_ACCESS:
            interfaces.append("kv_cache_injection")
        if caps & InterfaceCapability.WEIGHT_ACCESS:
            interfaces.append("dynamic_lora")
        return interfaces
```

---

## Task 4: Intrinsic Memory Bus

### Description
The Memory Bus is the central data channel connecting the existing memory retrieval pipeline (orchestrator, retriever) to the new injection interfaces. It receives retrieval results and routes them to the appropriate injection interface based on the Controller's decision (Controller is built in Phase 5; for now, the bus supports manual routing).

### Sub-task 4.1: Define Bus Channels and Memory Vector Format

**File:** `src/intrinsic/bus.py`

```python
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Callable, Dict, List, Optional
import torch
import asyncio
import logging

logger = logging.getLogger(__name__)


class InjectionChannel(Enum):
    """Available injection channels."""
    LOGIT_BIAS = "logit_bias"
    KNN_LM = "knn_lm"
    ACTIVATION_STEERING = "activation_steering"
    KV_CACHE_INJECTION = "kv_cache_injection"
    DYNAMIC_LORA = "dynamic_lora"
    CONTEXT_STUFFING = "context_stuffing"  # Fallback: existing RAG behavior


@dataclass
class MemoryVector:
    """
    A memory encoded in the model's latent space.
    This is the CORE data structure for intrinsic memory.
    
    A MemoryVector can be:
    - A steering vector (for activation injection)
    - A set of KV pairs (for KV-cache injection)
    - A logit bias map (for logit-level injection)
    - A LoRA delta (for weight adaptation)
    """
    memory_id: str                          # Links back to MemoryRecord
    source_text: str                        # Original text (for debugging)
    
    # Activation-space representation
    steering_vector: Optional[torch.Tensor] = None    # [hidden_dim]
    target_layers: List[int] = field(default_factory=list)  # Which layers to inject into
    
    # KV-cache representation
    key_vectors: Optional[torch.Tensor] = None        # [num_virtual_tokens, num_kv_heads, head_dim]
    value_vectors: Optional[torch.Tensor] = None      # [num_virtual_tokens, num_kv_heads, head_dim]
    
    # Logit representation
    logit_bias: Optional[Dict[int, float]] = None     # {token_id: bias_value}
    
    # Metadata
    relevance_score: float = 0.0
    injection_strength: float = 1.0
    decay_factor: float = 1.0              # SynapticRAG temporal decay
    channel: Optional[InjectionChannel] = None  # Assigned by Controller
    
    # Optional LoRA representation
    lora_adapter_path: Optional[str] = None


@dataclass
class InjectionRequest:
    """A request to inject memory into the forward pass."""
    channel: InjectionChannel
    memory_vectors: List[MemoryVector]
    generation_step: int = 0               # Which token generation step
    metadata: Dict[str, Any] = field(default_factory=dict)


class IntrinsicMemoryBus:
    """
    Central event/data bus connecting memory retrieval to injection interfaces.
    
    Flow:
    1. Memory retriever produces MemoryVectors (encoded memories)
    2. Controller routes them to appropriate channels
    3. Injection interfaces consume from their channels
    4. Bus tracks what was injected for logging/debugging
    """

    def __init__(self):
        self._channel_handlers: Dict[InjectionChannel, Callable] = {}
        self._pending_injections: Dict[InjectionChannel, List[MemoryVector]] = {
            ch: [] for ch in InjectionChannel
        }
        self._injection_log: List[Dict] = []
        self._listeners: Dict[str, List[Callable]] = {}

    def register_handler(self, channel: InjectionChannel, handler: Callable) -> None:
        """
        Register an injection handler for a channel.
        Handler signature: async (memory_vectors: List[MemoryVector]) -> None
        """
        self._channel_handlers[channel] = handler
        logger.info(f"Registered handler for channel: {channel.value}")

    def enqueue(self, channel: InjectionChannel, memory_vectors: List[MemoryVector]) -> None:
        """Enqueue memory vectors for injection on a specific channel."""
        self._pending_injections[channel].extend(memory_vectors)
        self._emit("enqueue", {"channel": channel, "count": len(memory_vectors)})

    async def flush(self, channel: Optional[InjectionChannel] = None) -> None:
        """
        Flush pending injections to their registered handlers.
        If channel is specified, only flush that channel.
        """
        channels = [channel] if channel else list(InjectionChannel)
        for ch in channels:
            vectors = self._pending_injections.get(ch, [])
            if not vectors:
                continue
            handler = self._channel_handlers.get(ch)
            if handler:
                await handler(vectors)
                self._injection_log.append({
                    "channel": ch.value,
                    "count": len(vectors),
                    "memory_ids": [v.memory_id for v in vectors],
                })
            else:
                logger.warning(f"No handler registered for channel {ch.value}, "
                               f"dropping {len(vectors)} memory vectors")
            self._pending_injections[ch] = []

    def clear(self) -> None:
        """Clear all pending injections."""
        for ch in self._pending_injections:
            self._pending_injections[ch] = []

    def get_injection_log(self) -> List[Dict]:
        """Return the log of all injections performed."""
        return self._injection_log.copy()

    def on(self, event: str, callback: Callable) -> None:
        """Subscribe to bus events (for observability)."""
        self._listeners.setdefault(event, []).append(callback)

    def _emit(self, event: str, data: Dict) -> None:
        for cb in self._listeners.get(event, []):
            try:
                cb(data)
            except Exception as e:
                logger.error(f"Bus listener error: {e}")
```

---

## Task 5: Configuration & Feature Flags

### Description
Extend the existing configuration system to support intrinsic memory settings. Feature flags control which interfaces are active, safety thresholds, and backend selection.

### Sub-task 5.1: Configuration Schema

**File:** `config/intrinsic.yaml` (new config section)

```yaml
intrinsic_memory:
  enabled: true
  
  backend:
    type: "local_transformer"   # "local_transformer" | "vllm" | "api_only"
    model_path: "meta-llama/Llama-3.1-8B-Instruct"
    device: "cuda"
    dtype: "float16"
    
  interfaces:
    logit_bias:
      enabled: true
      max_bias_value: 5.0           # Clamp bias values to [-5, 5]
      max_tokens_biased: 50         # Max tokens to bias per step
    
    activation_steering:
      enabled: true
      injection_layers: "auto"      # "auto" | [8, 9, 10, 11, 12]
      default_strength: 0.8         # α multiplier
      max_strength: 2.0             # Safety cap
      norm_preservation: true       # Scale vectors to match hidden state norms
    
    kv_cache_injection:
      enabled: true
      max_virtual_tokens: 128       # Budget: max KV pairs to inject
      temporal_decay: true          # SynapticRAG decay
      decay_half_life_turns: 5      # Halve injection strength every N turns
    
    dynamic_lora:
      enabled: false                # Advanced — disabled by default
      adapter_cache_dir: "./lora_adapters"
      max_loaded_adapters: 3
  
  safety:
    max_norm_ratio: 5.0             # Revert if modified/original norm ratio exceeds this
    nan_action: "revert"            # "revert" | "zero" | "raise"
    enable_cosine_divergence_check: false
    divergence_threshold: 0.5       # Max cosine distance from original
  
  fallback:
    strategy: "graceful_degrade"    # "graceful_degrade" | "strict" | "context_stuffing_only"
    # graceful_degrade: try activation → logit → context stuffing
    # strict: fail if requested interface unavailable
    # context_stuffing_only: always use existing RAG behavior
  
  bus:
    log_injections: true
    max_log_entries: 1000
```

### Sub-task 5.2: Configuration Loader Integration

Extend `src/core/config.py` to load intrinsic memory settings:

```python
# Addition to existing Settings class (src/core/config.py)

@dataclass
class IntrinsicMemoryConfig:
    """Configuration for intrinsic memory integration."""
    enabled: bool = False
    backend_type: str = "api_only"
    model_path: str = ""
    device: str = "cuda"
    dtype: str = "float16"
    
    # Interface toggles
    logit_bias_enabled: bool = True
    activation_steering_enabled: bool = False
    kv_cache_injection_enabled: bool = False
    dynamic_lora_enabled: bool = False
    
    # Safety
    max_norm_ratio: float = 5.0
    nan_action: str = "revert"
    
    # Fallback
    fallback_strategy: str = "graceful_degrade"

    @classmethod
    def from_yaml(cls, config_dict: dict) -> "IntrinsicMemoryConfig":
        """Parse from YAML config dict."""
        im = config_dict.get("intrinsic_memory", {})
        backend = im.get("backend", {})
        interfaces = im.get("interfaces", {})
        safety = im.get("safety", {})
        fallback = im.get("fallback", {})
        
        return cls(
            enabled=im.get("enabled", False),
            backend_type=backend.get("type", "api_only"),
            model_path=backend.get("model_path", ""),
            device=backend.get("device", "cuda"),
            dtype=backend.get("dtype", "float16"),
            logit_bias_enabled=interfaces.get("logit_bias", {}).get("enabled", True),
            activation_steering_enabled=interfaces.get("activation_steering", {}).get("enabled", False),
            kv_cache_injection_enabled=interfaces.get("kv_cache_injection", {}).get("enabled", False),
            dynamic_lora_enabled=interfaces.get("dynamic_lora", {}).get("enabled", False),
            max_norm_ratio=safety.get("max_norm_ratio", 5.0),
            nan_action=safety.get("nan_action", "revert"),
            fallback_strategy=fallback.get("strategy", "graceful_degrade"),
        )
```

---

## Task 6: Directory Structure & Module Scaffolding

### Description
Create the new `src/intrinsic/` package structure that all subsequent phases will build into.

### Sub-task 6.1: Directory Layout

```
src/intrinsic/
├── __init__.py
├── backends/
│   ├── __init__.py
│   ├── base.py              # ModelBackend ABC, ModelSpec, InterfaceCapability
│   ├── local_transformer.py  # Full-access HuggingFace backend
│   ├── vllm_backend.py       # vLLM server backend (Phase 3+)
│   └── api_only.py           # API-only logit bias backend
├── hooks/
│   ├── __init__.py
│   └── manager.py            # HookManager, SafetyGuard
├── interfaces/
│   ├── __init__.py
│   ├── logit.py              # Phase 2: Logit Interface
│   ├── activation.py         # Phase 3: Activation Interface
│   ├── synaptic.py           # Phase 4: KV-Cache Interface
│   └── weight.py             # Phase 8: Weight Adaptation Interface
├── encoding/
│   ├── __init__.py
│   ├── hippocampal_encoder.py # Phase 6: Memory → latent vector
│   ├── steering_vectors.py    # Phase 3/6: Steering vector derivation
│   └── kv_encoder.py          # Phase 4/6: Text → KV pairs
├── controller/
│   ├── __init__.py
│   ├── gating.py             # Phase 5: Strategy selection
│   └── router.py             # Phase 5: Multi-interface routing
├── cache/
│   ├── __init__.py
│   └── hierarchy.py          # Phase 7: L1/L2/L3 memory cache
├── inspector.py              # ModelInspector
├── bus.py                    # IntrinsicMemoryBus
└── config.py                 # IntrinsicMemoryConfig
```

---

## Acceptance Criteria

1. `ModelBackend` ABC is defined with all interface methods
2. `LocalTransformerBackend` can load a HuggingFace model and register forward hooks
3. `APIOnlyBackend` gracefully handles logit-bias-only scenarios
4. `HookManager` registers hooks with priority ordering and safety guards
5. `ModelInspector` reports model architecture and recommended injection layers
6. `IntrinsicMemoryBus` can enqueue and flush memory vectors to registered handlers
7. Configuration schema supports all intrinsic memory settings
8. All new modules import cleanly and existing tests pass
9. `src/intrinsic/` package structure is scaffolded

## Estimated Effort
- **Duration:** 2-3 weeks
- **Complexity:** Medium-High (requires understanding of transformer internals)
- **Risk:** Model-specific variations in layer naming and architecture
