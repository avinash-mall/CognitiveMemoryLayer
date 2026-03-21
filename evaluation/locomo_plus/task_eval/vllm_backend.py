"""Shared in-process vLLM backend for LoCoMo-Plus evaluation.

Holds a single LLM instance so the model is loaded once and reused across
Phase B (QA) and Phase C (judge). Requires: pip install vllm
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm import LLM

_llm_instance: LLM | None = None
_llm_model: str | None = None


def get_llm(model: str) -> LLM:
    """Return the vLLM LLM singleton, loading it on first call or on model change."""
    global _llm_instance, _llm_model
    if _llm_instance is not None and _llm_model == model:
        return _llm_instance

    try:
        from vllm import LLM as _LLM
    except ImportError:
        raise ImportError(
            "vllm is required for GPU inference. Install with:\n"
            "  pip install vllm\n"
            "or use the default --qa-backend openai_compatible / --backend call_llm"
        )

    try:
        import torch
        tensor_parallel_size = max(1, torch.cuda.device_count()) if torch.cuda.is_available() else 1
    except ImportError:
        tensor_parallel_size = 1

    print(
        f"[vLLM] Loading model '{model}' "
        f"(tensor_parallel_size={tensor_parallel_size})...",
        flush=True,
    )
    _llm_instance = _LLM(model=model, tensor_parallel_size=tensor_parallel_size)
    _llm_model = model
    print("[vLLM] Model loaded.", flush=True)
    return _llm_instance


def generate_batch(
    model: str,
    conversations: list[list[dict]],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> list[str]:
    """Run batched chat inference on GPU. Returns one response string per conversation."""
    from vllm import SamplingParams

    llm = get_llm(model)
    sampling_params = SamplingParams(temperature=temperature, max_tokens=max_tokens)
    outputs = llm.chat(conversations, sampling_params=sampling_params)
    results = []
    for out in outputs:
        if out.outputs:
            results.append((out.outputs[0].text or "").strip())
        else:
            results.append("")
    return results


def generate_single(
    model: str,
    messages: list[dict],
    temperature: float = 0.0,
    max_tokens: int = 512,
) -> str:
    """Run single-prompt chat inference on GPU."""
    results = generate_batch(model, [messages], temperature=temperature, max_tokens=max_tokens)
    return results[0] if results else ""
