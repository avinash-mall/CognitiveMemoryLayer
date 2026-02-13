"""LLM client abstraction for chunking and extraction."""

import json
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

from ..core.config import get_settings

try:
    from openai import AsyncOpenAI
except ImportError:
    AsyncOpenAI = None  # type: ignore


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
    ) -> None:
        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenAICompatibleClient")
        import os

        settings = get_settings()
        provider = settings.llm.provider
        self.model = model or settings.llm.model
        self._base_url = base_url
        self._api_key = api_key
        if base_url is None:
            self._base_url = settings.llm.base_url or (
                _OPENAI_DEFAULT_BASE
                if provider == "openai"
                else (
                    _OPENAI_COMPATIBLE_DEFAULT_BASE
                    if provider in ("openai_compatible", "vllm")
                    else _OLLAMA_DEFAULT_BASE
                )
            )
        if self._api_key is None:
            if base_url is not None:
                self._api_key = "dummy"
            elif provider == "openai":
                self._api_key = settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
            else:
                self._api_key = settings.llm.api_key or os.environ.get("OPENAI_API_KEY") or "dummy"
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
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

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
            generation_config = {"temperature": temperature, "max_output_tokens": max_tokens}
            response = await loop.run_in_executor(
                None,
                lambda: self._model.generate_content(contents, generation_config=generation_config),
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


def get_llm_client() -> LLMClient:
    """Factory function to get configured LLM client."""
    import os

    settings = get_settings()
    provider = settings.llm.provider
    model = settings.llm.model
    api_key = settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
    base_url = settings.llm.base_url

    if provider in ("openai", "openai_compatible", "vllm", "ollama"):
        if base_url:
            url = base_url
        elif provider == "openai":
            url = _OPENAI_DEFAULT_BASE
        elif provider in ("openai_compatible", "vllm"):
            url = _OPENAI_COMPATIBLE_DEFAULT_BASE
        else:
            url = _OLLAMA_DEFAULT_BASE
        key = api_key if provider == "openai" else (api_key or "dummy")
        return OpenAICompatibleClient(base_url=url, model=model, api_key=key)

    if provider == "gemini":
        if not api_key:
            raise ValueError("LLM__API_KEY (or OPENAI_API_KEY) is required for provider=gemini")
        return _gemini_client(api_key, model)

    if provider == "claude":
        if not api_key:
            raise ValueError("LLM__API_KEY (or OPENAI_API_KEY) is required for provider=claude")
        return _claude_client(api_key, model)

    raise ValueError(f"Unknown LLM provider: {provider}")


# Backward-compatibility alias
OpenAIClient = OpenAICompatibleClient
