"""
Evaluation framework utils.
- Load unified input samples (from JSON produced by data/unified_input.py).
- Predictions: evaluate uses call_llm / call_vllm; call_test is a placeholder.
"""

import json
import os
from pathlib import Path

CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. "
    "The conversation takes place over multiple days, "
    "and the date of each conversation is written at the beginning of the conversation.\n\n"
)

INSTRUCTION_QA = "Answer the following question based on the conversation above.\n\n"

INSTRUCTION_COGNITIVE = (
    "Your task: This is a memory-aware dialogue setting. "
    "You are continuing or reflecting on a prior conversation. "
    "Show that you are aware of the relevant memory or context from the evidence when you respond; "
    "your answer should naturally connect to or acknowledge that context.\n\n"
)


def _get_task_instruction(category: str) -> str:
    if (category or "").strip() == "Cognitive":
        return INSTRUCTION_COGNITIVE
    return INSTRUCTION_QA


def _build_model_input(
    input_prompt: str, category: str = "", name1: str = "A", name2: str = "B"
) -> str:
    conv = CONV_START_PROMPT.format(name1, name2)
    instruction = _get_task_instruction(category)
    body = (input_prompt or "").strip()
    return conv + instruction + (body if body else "")


def _prepend_conv_prefix(text: str, name1: str = "A", name2: str = "B") -> str:
    if not (text or "").strip():
        return (CONV_START_PROMPT.format(name1, name2)).strip()
    return CONV_START_PROMPT.format(name1, name2) + (text or "").strip()


def load_unified_samples(data_file: str):
    """Load samples from the unified input JSON."""
    path = Path(data_file)
    if not path.exists():
        raise FileNotFoundError(f"Unified input file not found: {data_file}")
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, list):
        raise ValueError("Unified input JSON must be a list of samples.")
    return data


def call_test(input_prompt: str, model: str, **kwargs) -> str:
    """Placeholder backend: same I/O as call_llm; returns fake prediction."""
    category = kwargs.get("category", "")
    content = (
        _build_model_input(input_prompt or "", category=category)
        if category
        else _prepend_conv_prefix(input_prompt or "")
    )
    if not content.strip():
        return "(empty)"
    return content[:-100] if len(content) > 100 else content


def _load_env_local_sh():
    """Load OPENAI_BASE_URL/OPENAI_API_KEY from scripts/env.local.sh when not already set."""
    env_local = Path(__file__).resolve().parent.parent / "scripts" / "env.local.sh"
    if not env_local.is_file():
        return
    with open(env_local, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or not line.startswith("export "):
                continue
            rest = line[7:].strip()
            if "=" not in rest:
                continue
            key, _, val = rest.partition("=")
            key = key.strip()
            val = val.strip().strip('"').strip("'").strip()
            if key in ("OPENAI_API_KEY", "OPENAI_BASE_URL") and val and not os.environ.get(key):
                os.environ[key] = val


def _load_project_dotenv():
    """Load project .env so LLM_EVAL__* / LLM_INTERNAL__* are available for judge (e.g. Ollama)."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    # evaluation/locomo_plus/task_eval/utils.py -> repo root is parent^4
    repo_root = Path(__file__).resolve().parent.parent.parent.parent
    env_file = repo_root / ".env"
    if env_file.is_file():
        load_dotenv(env_file)
    # Also allow env.local.sh to have set OPENAI_* already
    _load_env_local_sh()


def _get_openai_client():
    """Lazy init OpenAI client from OPENAI_API_KEY + OPENAI_BASE_URL, or from .env LLM_EVAL__* / LLM_INTERNAL__* (e.g. Ollama)."""
    _load_project_dotenv()
    api_key = (os.environ.get("OPENAI_API_KEY") or "").strip()
    base_url = (os.environ.get("OPENAI_BASE_URL") or "").strip()
    # If no OpenAI key but .env has LLM_EVAL__* or LLM_INTERNAL__* (e.g. Ollama), use that for judge
    if not api_key:
        llm_base = (
            os.environ.get("LLM_EVAL__BASE_URL") or os.environ.get("LLM_INTERNAL__BASE_URL") or ""
        ).strip()
        if llm_base:
            api_key = "ollama"
            base_url = llm_base
    if not api_key:
        raise ValueError(
            "OPENAI_API_KEY is not set and LLM_EVAL__BASE_URL/LLM_INTERNAL__BASE_URL is not set. "
            "Set OPENAI_API_KEY (or LLM_EVAL__* / LLM_INTERNAL__* in .env for Ollama), or use evaluation/locomo_plus/scripts/env.local.sh"
        )
    from openai import OpenAI

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


def call_llm(input_prompt: str, model: str, **kwargs) -> str:
    """Call OpenAI-compatible API for prediction."""
    client = _get_openai_client()
    temperature = kwargs.get("temperature", 0.3)
    max_tokens = kwargs.get("max_tokens", 2048)
    category = kwargs.get("category", "")
    content = (
        _build_model_input(input_prompt or "", category=category)
        if category
        else _prepend_conv_prefix(input_prompt or "")
    )
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": content}],
            temperature=float(temperature),
            max_tokens=int(max_tokens),
        )
        text = (response.choices[0].message.content or "").strip()
        return text if text else "(empty)"
    except Exception as e:
        return f"[API Error: {e}]"


def call_vllm(input_prompt: str, model: str, **kwargs) -> str:
    """Call local vLLM. Not implemented."""
    raise NotImplementedError("call_vllm not implemented yet; use --backend call_test for now.")


def call_model(input_prompt: str, model: str, backend: str = "call_test", **kwargs) -> str:
    """Dispatch by backend to call_test / call_llm / call_vllm."""
    if backend == "call_test":
        return call_test(input_prompt, model=model, **kwargs)
    if backend == "call_llm":
        return call_llm(input_prompt, model=model, **kwargs)
    if backend == "call_vllm":
        return call_vllm(input_prompt, model=model, **kwargs)
    raise ValueError(f"Unknown backend: {backend}. Use call_test, call_llm, or call_vllm.")


def extract_question_from_input_prompt(input_prompt: str) -> str:
    """Extract question from input_prompt. Locomo format: '...\\n\\nQuestion: ' -> text after 'Question:'."""
    if not (input_prompt or "").strip():
        return ""
    s = input_prompt.strip()
    if "Question:" in s:
        return s.split("Question:")[-1].strip()
    return ""


def build_output_record(
    sample: dict,
    prediction: str,
    model: str,
    question_input: str = "",
) -> dict:
    """Build one record for judge: question_input, evidence, category, ground_truth, prediction, model."""
    ground_truth = sample.get("answer")
    if ground_truth is None or (isinstance(ground_truth, str) and ground_truth.strip() == ""):
        ground_truth = ""
    record = {
        "question_input": question_input,
        "evidence": sample.get("evidence", ""),
        "category": sample.get("category"),
        "ground_truth": ground_truth,
        "prediction": prediction,
        "model": model,
    }
    if sample.get("time_gap") is not None and sample.get("time_gap") != "":
        record["time_gap"] = sample["time_gap"]
    return record
