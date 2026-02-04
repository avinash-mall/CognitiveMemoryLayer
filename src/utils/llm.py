"""LLM client abstraction for chunking and extraction."""
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

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


class LLMClient(ABC):
    """Abstract LLM client interface."""

    @abstractmethod
    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
    ) -> str:
        """Return raw text completion."""
        ...

    @abstractmethod
    async def complete_json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        """Return parsed JSON from completion."""
        ...


class OpenAIClient(LLMClient):
    """OpenAI API client."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        import os

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for OpenAIClient")
        settings = get_settings()
        self.client = AsyncOpenAI(
            api_key=api_key or settings.llm.api_key or os.environ.get("OPENAI_API_KEY", "")
        )
        self.model = model or settings.llm.model

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
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
        schema: Optional[Dict] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        import json
        import re

        response = await self.complete(
            prompt,
            temperature=temperature,
            system_prompt="You are a JSON generator. Always respond with valid JSON only, no markdown.",
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]|\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


class MockLLMClient(LLMClient):
    """Mock LLM client for tests; returns fixed or programmable responses."""

    def __init__(
        self,
        fixed_response: Optional[str] = None,
        fixed_json: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.fixed_response = fixed_response or "[]"
        self.fixed_json = fixed_json

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
    ) -> str:
        return self.fixed_response

    async def complete_json(
        self,
        prompt: str,
        schema: Optional[Dict] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        import json

        if self.fixed_json is not None:
            return self.fixed_json
        return json.loads(self.fixed_response)


class VLLMClient(LLMClient):
    """vLLM OpenAI-compatible API client (e.g. Llama 3.2 with vLLM in Docker)."""

    def __init__(
        self,
        base_url: Optional[str] = None,
        model: Optional[str] = None,
    ) -> None:
        import os

        if AsyncOpenAI is None:
            raise ImportError("openai package is required for VLLMClient")
        settings = get_settings()
        self.base_url = (
            base_url
            or settings.llm.vllm_base_url
            or os.environ.get("VLLM_BASE_URL", "http://localhost:8000/v1")
        )
        self.model = model or settings.llm.vllm_model
        self.client = AsyncOpenAI(
            base_url=self.base_url,
            api_key="dummy",  # vLLM often does not require a key
        )

    async def complete(
        self,
        prompt: str,
        temperature: float = 0.0,
        max_tokens: int = 500,
        system_prompt: Optional[str] = None,
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
        schema: Optional[Dict] = None,
        temperature: float = 0.0,
    ) -> Dict[str, Any]:
        import json
        import re

        response = await self.complete(
            prompt,
            temperature=temperature,
            system_prompt="You are a JSON generator. Always respond with valid JSON only, no markdown.",
        )
        try:
            return json.loads(response)
        except json.JSONDecodeError:
            match = re.search(r"\[.*\]|\{.*\}", response, re.DOTALL)
            if match:
                return json.loads(match.group())
            raise


def get_llm_client() -> LLMClient:
    """Factory function to get configured LLM client."""
    settings = get_settings()
    if settings.llm.provider == "openai":
        return OpenAIClient()
    if settings.llm.provider == "vllm":
        return VLLMClient()
    raise ValueError(f"Unknown LLM provider: {settings.llm.provider}")
