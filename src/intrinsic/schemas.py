"""Schemas for ActiveCML Intrinsic Memory injection (Phases I-1..I-10)."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
from typing import Any


class IntrinsicDepth(StrEnum):
    """Four depths of intrinsic memory injection."""

    NONE = "none"
    LOGIT = "logit"  # I-2: Output distribution bias before sampling (universal HTTP/local)
    ACTIVATION = "activation"  # I-3: Residual stream steering vectors between layers (local hooks)
    SYNAPTIC_KV = "synaptic_kv"  # I-4: Attention KV-cache / deterministic APC prefix injection
    WEIGHT_LORA = "weight_lora"  # I-8: Dynamic LoRA adapter routing (vLLM multi-LoRA / local PEFT)


@dataclass
class IntrinsicInjectionSpec:
    """Complete intrinsic memory injection specification emitted by the Controller."""

    enabled: bool = True
    selected_depths: list[str] = field(default_factory=list)
    gate_strength: float = 0.0  # Calibrated alpha in [0.0, 1.0]
    routing_reason: str = ""

    # I-2: Logit bias map {token_id_str: bias_float} for OpenAI/vLLM chat completions
    logit_bias: dict[str, float] = field(default_factory=dict)
    boosted_terms: list[str] = field(default_factory=list)
    suppressed_terms: list[str] = field(default_factory=list)

    # I-3: Activation steering vector metadata & compressed basis vector
    steering_layer_indices: list[int] = field(default_factory=list)
    steering_vector: list[float] = field(default_factory=list)
    steering_scale: float = 0.0

    # I-4: Synaptic KV-cache / Deterministic Prefix Cache (vLLM APC & local KV cache)
    synaptic_prefix_hash: str | None = None
    canonical_kv_prefix: str | None = None
    kv_engram_ids: list[str] = field(default_factory=list)

    # I-8: Dynamic LoRA weight adaptation routing
    lora_adapter_id: str | None = None
    lora_domain: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a compact JSON-serializable representation."""
        data = asdict(self)
        # Keep steering_vector compact in wire metadata (first 16 dims preview + norm)
        if len(self.steering_vector) > 16:
            data["steering_vector_dim"] = len(self.steering_vector)
            data["steering_vector_preview"] = [round(x, 4) for x in self.steering_vector[:16]]
            del data["steering_vector"]
        return data

    def to_openai_extra_kwargs(self) -> dict[str, Any]:
        """Return kwargs directly consumable by OpenAI/vLLM chat.completions.create."""
        kwargs: dict[str, Any] = {}
        if self.logit_bias:
            kwargs["logit_bias"] = dict(self.logit_bias)
        if self.lora_adapter_id:
            kwargs["extra_body"] = {"lora_request": self.lora_adapter_id}
        return kwargs
