"""Phase I-8: Weight Adaptation Interface (Dynamic LoRA Adapter Routing).

Routes queries to specialized runtime LoRA adapters (synaptic plasticity)
based on query intent, domain context tags, and tenant adapter mappings.
Works with vLLM multi-LoRA serving (`extra_body.lora_request`) and local PEFT
models (`model.set_adapter(adapter_id)`).
"""

from __future__ import annotations


class DynamicLoRARouter:
    """Selects runtime LoRA adapters for task- or tenant-specific synaptic plasticity."""

    def __init__(self, domain_adapters: dict[str, str] | None = None) -> None:
        self.domain_adapters: dict[str, str] = domain_adapters or {
            "temporal_query": "cml-lora-temporal-v1",
            "multi_hop": "cml-lora-multihop-v1",
            "constraint_check": "cml-lora-constraint-v1",
            "cognitive": "cml-lora-theory-of-mind-v1",
        }
        self._tenant_adapters: dict[str, str] = {}

    def register_tenant_adapter(self, tenant_id: str, adapter_id: str) -> None:
        """Register a dedicated plastic LoRA adapter for a tenant."""
        self._tenant_adapters[tenant_id] = adapter_id

    def route(
        self,
        *,
        tenant_id: str | None = None,
        intent: str | None = None,
        is_mentalizing: bool = False,
    ) -> tuple[str | None, str | None]:
        """Return (adapter_id, domain_name) for the given query context."""
        if tenant_id and tenant_id in self._tenant_adapters:
            return self._tenant_adapters[tenant_id], f"tenant:{tenant_id}"
        if is_mentalizing and "cognitive" in self.domain_adapters:
            return self.domain_adapters["cognitive"], "cognitive"
        norm_intent = (intent or "").lower().strip()
        if norm_intent in self.domain_adapters:
            return self.domain_adapters[norm_intent], norm_intent
        return None, None
