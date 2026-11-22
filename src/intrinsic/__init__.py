"""ActiveCML Intrinsic Memory Subsystem (Phases I-1..I-10).

Provides model-internal and HTTP-portable intrinsic memory injection:
- I-1: Model Access Layer & Forward-Hook Concurrency Isolation
- I-2: Logit Interface (Logit-Bias Synthesis & False-Premise/Superseded Suppression)
- I-3: Activation Interface (Residual-Stream Steering Vectors)
- I-4: Synaptic Interface (KV-Cache & Deterministic APC Prefix Framing)
- I-5: Controller & Gating Unit
- I-8: Weight Adaptation Interface (Dynamic LoRA Routing)
"""

from .activation_hooks import ActivationSteeringEngine, ActiveSteeringState
from .controller import IntrinsicController, get_intrinsic_controller
from .logit_interface import LogitBiasSynthesizer
from .lora_router import DynamicLoRARouter
from .prefix_synaptic import SynapticKVEntry, SynapticKVManager
from .schemas import IntrinsicDepth, IntrinsicInjectionSpec

__all__ = [
    "ActivationSteeringEngine",
    "ActiveSteeringState",
    "DynamicLoRARouter",
    "IntrinsicController",
    "IntrinsicDepth",
    "IntrinsicInjectionSpec",
    "LogitBiasSynthesizer",
    "SynapticKVEntry",
    "SynapticKVManager",
    "get_intrinsic_controller",
]
