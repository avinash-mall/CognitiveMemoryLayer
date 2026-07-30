"""LLM client abstraction for chunking and extraction."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from openai import AsyncOpenAI

from ..core.config import get_settings
from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class LLMResponse:
    """Structured LLM response."""

    content: str
    model: str
    tokens_used: int
    finish_reason: str


# Default base URLs per OpenAI-compatible provider (used when base_url not set in config)
_OPENAI_DEFAULT_BASE = "https://api.openai.com/v1"
_OPENAI_COMPATIBLE_DEFAULT_BASE = "http://localhost:8000/v1"
_OLLAMA_DEFAULT_BASE = "http://localhost:11434/v1"


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: str | None = None,
    ) -> str:
        """Return raw text completion."""
        ...

    @abstractmethod
    async def complete_json(
        self,
        prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        """Return parsed JSON from completion."""
        ...


def _content_or_warn(response: Any, *, where: str, max_tokens: int) -> str:
    """Return message content, logging loudly when it comes back empty.

    A reasoning model whose thinking is parsed into a separate field spends the token
    budget on `reasoning_content` and returns `content=""` once it hits the cap. Callers
    then see an empty string that is indistinguishable from a genuine empty answer, which
    silently degrades into a wrong result (a judge scoring 0, an extractor finding
    nothing). Surface it instead: either raise max_tokens or disable thinking via
    LLM_*__EXTRA_BODY={"chat_template_kwargs": {"enable_thinking": false}}.
    """
    choice = response.choices[0]
    content = choice.message.content or ""
    if not content.strip():
        logger.warning(
            "llm_empty_content",
            extra={
                "where": where,
                "model": getattr(response, "model", "?"),
                "finish_reason": getattr(choice, "finish_reason", None),
                "max_tokens": max_tokens,
                "reasoning_chars": len(
                    getattr(choice.message, "reasoning_content", None)
                    or getattr(choice.message, "reasoning", None)
                    or ""
                ),
            },
        )
    return content


def _parse_json_from_response(response: str) -> dict[str, Any]:
    """Extract and parse JSON from LLM response text."""
    try:
        return json.loads(response)
    except json.JSONDecodeError:
        match = re.search(r"\[.*\]|\{.*\}", response, re.DOTALL)
        if match:
            return json.loads(match.group())
        raise


class OpenAICompatibleClient(LLMClient):
    """Single client for any OpenAI-compatible API (OpenAI, local server, Ollama, proxies)."""

    def __init__(
        self,
        base_url: str | None = None,
        model: str | None = None,
        api_key: str | None = None,
        _provider: str | None = None,
        extra_body: dict | None = None,
    ) -> None:
        """Create client. Pass base_url/model/api_key, or leave None to use llm_internal config."""
        import os

        settings = get_settings()
        cfg = settings.llm_internal
        provider = _provider or cfg.provider
        self.model = model or cfg.model
        self._base_url = base_url
        self._api_key = api_key
        if base_url is None:
            self._base_url = cfg.base_url or (
                _OPENAI_DEFAULT_BASE
                if provider == "openai"
                else (
                    _OPENAI_COMPATIBLE_DEFAULT_BASE
                    if provider in ("openai_compatible", "vllm", "sglang")
                    else _OLLAMA_DEFAULT_BASE
                )
            )
        if self._api_key is None:
            if base_url is not None:
                self._api_key = "dummy"
            elif provider == "openai":
                self._api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
            else:
                self._api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
        self.extra_body = extra_body if extra_body is not None else cfg.extra_body
        self.client = AsyncOpenAI(base_url=self._base_url, api_key=self._api_key)

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: str | None = None,
    ) -> str:
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})

        response = await self.client.chat.completions.create(
            model=self.model,
            messages=messages,  # type: ignore[arg-type]
            temperature=temperature,
            max_tokens=max_tokens,
            extra_body=self.extra_body,
        )
        return _content_or_warn(response, where="complete", max_tokens=max_tokens)

    async def complete_json(
        self,
        prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
        max_tokens: int = 2000,
    ) -> dict[str, Any]:
        # max_tokens caps runaway generation: without it, a degenerate completion under
        # guided JSON decoding can run until the model's context fills (observed 90KB+ /
        # 4+ minutes on a local vLLM). Callers degrade gracefully on truncated JSON.
        messages = [
            {
                "role": "system",
                "content": "You are a JSON generator. Always respond with valid JSON only, no markdown.",
            },
            {"role": "user", "content": prompt},
        ]
        try:
            response = await self.client.chat.completions.create(  # type: ignore[call-overload]
                model=self.model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                response_format={"type": "json_object"},
                extra_body=self.extra_body,
            )
            text = _content_or_warn(response, where="complete_json", max_tokens=max_tokens) or "{}"
        except Exception:
            # Fallback for models/endpoints that don't support strict response_format
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,  # type: ignore[arg-type]
                temperature=temperature,
                max_tokens=max_tokens,
                extra_body=self.extra_body,
            )
            text = (
                _content_or_warn(response, where="complete_json_fallback", max_tokens=max_tokens)
                or "{}"
            )
        return _parse_json_from_response(text)


class MockLLMClient(LLMClient):
    """Mock LLM client for tests; returns fixed or programmable responses."""

    def __init__(
        self,
        fixed_response: str | None = None,
        fixed_json: dict[str, Any] | None = None,
    ) -> None:
        self.fixed_response = fixed_response or "[]"
        self.fixed_json = fixed_json

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: str | None = None,
    ) -> str:
        return self.fixed_response

    async def complete_json(
        self,
        prompt: str,
        schema: dict | None = None,
        temperature: float = 0.0,
    ) -> dict[str, Any]:
        if self.fixed_json is not None:
            return self.fixed_json
        return json.loads(self.fixed_response)


def _gemini_client(api_key: str, model: str) -> LLMClient:
    """Lazy import and return a Gemini-backed LLMClient adapter."""
    try:
        import asyncio

        import google.generativeai as genai
    except ImportError as e:
        raise ImportError(
            "gemini provider requires the google-generativeai package. "
            "Install with: pip install google-generativeai"
        ) from e

    class _GeminiClient(LLMClient):
        def __init__(self, api_key: str, model: str) -> None:
            genai.configure(api_key=api_key)
            self._model_name = model
            self._model = genai.GenerativeModel(model)

        async def complete(
            self,
            prompt: str,
            temperature: float = 0.0,
            max_tokens: int = 500,
            system_prompt: str | None = None,
        ) -> str:
            loop = asyncio.get_running_loop()
            contents = prompt
            if system_prompt:
                contents = f"{system_prompt}\n\n{prompt}"
            generation_config = genai.types.GenerationConfig(
                temperature=temperature,
                max_output_tokens=max_tokens,
            )
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(
                    contents,
                    generation_config=generation_config,
                ),
            )
            if not response or not response.text:
                return ""
            return response.text

        async def complete_json(
            self,
            prompt: str,
            schema: dict | None = None,
            temperature: float = 0.0,
        ) -> dict[str, Any]:
            response = await self.complete(
                prompt,
                temperature=temperature,
                system_prompt="You are a JSON generator. Always respond with valid JSON only, no markdown.",
            )
            return _parse_json_from_response(response)

    return _GeminiClient(api_key, model)


def _claude_client(api_key: str, model: str) -> LLMClient:
    """Lazy import and return a Claude-backed LLMClient adapter."""
    try:
        from anthropic import AsyncAnthropic
    except ImportError as e:
        raise ImportError(
            "claude provider requires the anthropic package. Install with: pip install anthropic"
        ) from e

    class _ClaudeClient(LLMClient):
        def __init__(self, api_key: str, model: str) -> None:
            self._client = AsyncAnthropic(api_key=api_key)
            self._model = model

        async def complete(
            self,
            prompt: str,
            temperature: float = 0.0,
            max_tokens: int = 500,
            system_prompt: str | None = None,
        ) -> str:
            kwargs: dict[str, Any] = dict(
                model=self._model,
                max_tokens=max_tokens,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature,
            )
            if system_prompt:
                kwargs["system"] = system_prompt
            response = await self._client.messages.create(**kwargs)
            if not response.content or not response.content[0].text:
                return ""
            return response.content[0].text

        async def complete_json(
            self,
            prompt: str,
            schema: dict | None = None,
            temperature: float = 0.0,
        ) -> dict[str, Any]:
            response = await self.complete(
                prompt,
                temperature=temperature,
                system_prompt="You are a JSON generator. Always respond with valid JSON only, no markdown.",
            )
            return _parse_json_from_response(response)

    return _ClaudeClient(api_key, model)


def _build_llm_client_from_config(
    provider: str,
    model: str,
    api_key: str | None,
    base_url: str | None,
    env_prefix: str = "LLM_INTERNAL",
    extra_body: dict | None = None,
) -> LLMClient:
    """Build LLM client from provider/model/api_key/base_url. Handles openai, ollama, anthropic, gemini, vllm, sglang, openai_compatible."""

    # anthropic is alias for claude
    if provider == "anthropic":
        provider = "claude"

    if provider in ("openai", "openai_compatible", "vllm", "sglang", "ollama"):
        if base_url is not None:
            url = base_url
        elif provider == "openai":
            url = _OPENAI_DEFAULT_BASE
        elif provider in ("openai_compatible", "vllm", "sglang"):
            url = _OPENAI_COMPATIBLE_DEFAULT_BASE
        else:
            url = _OLLAMA_DEFAULT_BASE
        key = api_key if provider == "openai" else (api_key or "dummy")
        return OpenAICompatibleClient(base_url=url, model=model, api_key=key, extra_body=extra_body)

    if provider == "gemini":
        if not api_key:
            raise ValueError(
                f"{env_prefix}__API_KEY (or OPENAI_API_KEY) is required for provider=gemini"
            )
        return _gemini_client(api_key, model)

    if provider == "claude":
        if not api_key:
            raise ValueError(
                f"{env_prefix}__API_KEY (or OPENAI_API_KEY) is required for provider=claude/anthropic"
            )
        return _claude_client(api_key, model)

    raise ValueError(
        f"Unknown LLM provider: {provider}. Supported: openai, ollama, anthropic, gemini, vllm, sglang, openai_compatible"
    )


def get_internal_llm_client() -> LLMClient:
    """Factory for LLM client used by internal tasks. Reads LLM_INTERNAL__* only."""
    import os

    settings = get_settings()
    cfg = settings.llm_internal
    api_key = cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
    return _build_llm_client_from_config(
        provider=cfg.provider,
        model=cfg.model,
        api_key=api_key,
        base_url=cfg.base_url,
        env_prefix="LLM_INTERNAL",
        extra_body=cfg.extra_body,
    )


def get_eval_llm_client() -> LLMClient:
    """Factory for LLM client used by evaluation (QA, judge). Reads LLM_EVAL__*; falls back to LLM_INTERNAL__*."""
    import os

    settings = get_settings()
    ev = settings.llm_eval
    cfg = settings.llm_internal
    provider = ev.provider if ev.provider is not None else cfg.provider
    model = ev.model if ev.model is not None else cfg.model
    base_url = ev.base_url if ev.base_url is not None else cfg.base_url
    api_key = (
        ev.api_key
        if ev.api_key is not None
        else cfg.api_key or os.environ.get("OPENAI_API_KEY", "")
    )
    return _build_llm_client_from_config(
        provider=provider,
        model=model,
        api_key=api_key,
        base_url=base_url,
        env_prefix="LLM_EVAL",
        extra_body=ev.extra_body if ev.extra_body is not None else cfg.extra_body,
    )


# Backward-compatibility alias
