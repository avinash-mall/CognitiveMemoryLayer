"""Phases I-5, I-9 & I-10: ActiveCML Controller & Gating Unit.

Given a query and retrieved MemoryPacket, decides:
1. What calibrated strength alpha in [0, 1] to inject at (based on retrieval
   sufficiency, memory confidence, somatic arousal, and CA1 reality monitoring).
2. Which intrinsic interfaces to activate:
   - Logit Interface (I-2): token logit_bias boosting retrieved entities/dates and
     suppressing superseded beliefs & CA1 unverified presuppositions.
   - Activation Interface (I-3): residual-stream steering vector synthesis & hooks.
   - Synaptic KV Interface (I-4): canonical APC prefix block & tiered KV cache.
   - Weight Adaptation Interface (I-8): dynamic LoRA adapter routing.
"""

from __future__ import annotations

from typing import Any

from ..memory.dmn import DefaultModeNetwork
from ..retrieval.reality_monitor import CA1RealityMonitor
from .activation_hooks import ActivationSteeringEngine
from .logit_interface import LogitBiasSynthesizer
from .lora_router import DynamicLoRARouter
from .prefix_synaptic import SynapticKVManager
from .schemas import IntrinsicDepth, IntrinsicInjectionSpec


class IntrinsicController:
    """ActiveCML Controller & Gating Unit coordinating all four intrinsic memory depths."""

    def __init__(
        self,
        *,
        logit_synthesizer: LogitBiasSynthesizer | None = None,
        activation_engine: ActivationSteeringEngine | None = None,
        synaptic_kv: SynapticKVManager | None = None,
        lora_router: DynamicLoRARouter | None = None,
    ) -> None:
        self.logit_synthesizer = logit_synthesizer or LogitBiasSynthesizer()
        self.activation_engine = activation_engine or ActivationSteeringEngine()
        self.synaptic_kv = synaptic_kv or SynapticKVManager()
        self.lora_router = lora_router or DynamicLoRARouter()

    def compute_spec(
        self,
        packet: Any,
        *,
        tenant_id: str | None = None,
        intent: str | None = None,
        target_hidden_dim: int | None = None,
    ) -> IntrinsicInjectionSpec:
        """Evaluate the MemoryPacket and synthesize a full IntrinsicInjectionSpec."""
        all_mems = list(getattr(packet, "all_memories", []))
        if not all_mems:
            return IntrinsicInjectionSpec(
                enabled=True,
                selected_depths=[IntrinsicDepth.NONE.value],
                gate_strength=0.0,
                routing_reason="No retrieved memories to inject intrinsically.",
            )

        # 1. Calibrate gate strength alpha in [0.0, 1.0]
        suff = getattr(packet, "sufficiency", None) or {}
        top_strength = float(suff.get("evidence_strength", 0.0) or 0.0)
        if top_strength <= 0.0 and all_mems:
            top_strength = max(float(getattr(m, "relevance_score", 0.5) or 0.5) for m in all_mems)

        mean_conf = sum(
            float(getattr(getattr(m, "record", None), "confidence", 0.5) or 0.5)
            for m in all_mems[:8]
        ) / max(1, min(len(all_mems), 8))

        max_arousal = 0.0
        for m in all_mems[:8]:
            meta = getattr(getattr(m, "record", None), "metadata", None) or {}
            somatic = meta.get("somatic_marker")
            if isinstance(somatic, dict):
                max_arousal = max(max_arousal, float(somatic.get("arousal", 0.0) or 0.0))

        gate_strength = min(
            1.0,
            max(
                0.15,
                round(0.50 * min(1.0, top_strength) + 0.35 * mean_conf + 0.15 * max_arousal, 3),
            ),
        )

        # 2. Check CA1 Reality Monitoring for unverified presuppositions to suppress in logit_bias
        query = str(getattr(packet, "query", "") or "")
        verdict = CA1RealityMonitor.inspect(query, all_mems)

        # 3. Synthesize I-2 Logit Interface
        logit_bias, boosted, suppressed = self.logit_synthesizer.synthesize(
            packet,
            gate_strength=gate_strength,
            unverified_presuppositions=verdict.unverified_presuppositions,
        )

        # 4. Synthesize I-3 Activation Steering Interface
        steer_vec, steer_scale, steer_layers = self.activation_engine.synthesize_steering_vector(
            packet,
            gate_strength=gate_strength,
            target_dim=target_hidden_dim,
        )

        # 5. Synthesize I-4 Synaptic KV-Prefix Interface
        canonical_prefix, prefix_hash, engram_ids = self.synaptic_kv.build_canonical_prefix(packet)

        # 6. Synthesize I-8 Dynamic LoRA Routing
        is_mentalizing = DefaultModeNetwork.is_mentalizing_query(query)
        lora_id, lora_domain = self.lora_router.route(
            tenant_id=tenant_id,
            intent=intent,
            is_mentalizing=is_mentalizing,
        )

        selected_depths: list[str] = []
        if logit_bias:
            selected_depths.append(IntrinsicDepth.LOGIT.value)
        if steer_vec:
            selected_depths.append(IntrinsicDepth.ACTIVATION.value)
        if canonical_prefix:
            selected_depths.append(IntrinsicDepth.SYNAPTIC_KV.value)
        if lora_id:
            selected_depths.append(IntrinsicDepth.WEIGHT_LORA.value)

        return IntrinsicInjectionSpec(
            enabled=True,
            selected_depths=selected_depths or [IntrinsicDepth.NONE.value],
            gate_strength=gate_strength,
            routing_reason=(
                f"Calibrated gate={gate_strength:.2f} (strength={top_strength:.2f}, "
                f"conf={mean_conf:.2f}, arousal={max_arousal:.2f}, ca1_mismatch={verdict.has_mismatch})"
            ),
            logit_bias=logit_bias,
            boosted_terms=boosted,
            suppressed_terms=suppressed,
            steering_layer_indices=steer_layers,
            steering_vector=steer_vec,
            steering_scale=steer_scale,
            synaptic_prefix_hash=prefix_hash or None,
            canonical_kv_prefix=canonical_prefix or None,
            kv_engram_ids=engram_ids,
            lora_adapter_id=lora_id,
            lora_domain=lora_domain,
        )


_DEFAULT_INTRINSIC_CONTROLLER: IntrinsicController | None = None


def get_intrinsic_controller() -> IntrinsicController:
    """Return singleton IntrinsicController instance."""
    global _DEFAULT_INTRINSIC_CONTROLLER
    if _DEFAULT_INTRINSIC_CONTROLLER is None:
        _DEFAULT_INTRINSIC_CONTROLLER = IntrinsicController()
    return _DEFAULT_INTRINSIC_CONTROLLER
