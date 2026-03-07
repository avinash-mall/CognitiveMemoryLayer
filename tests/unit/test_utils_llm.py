from __future__ import annotations

import os
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.utils import llm


def _settings(
    *,
    provider: str = "openai",
    model: str = "gpt-4o-mini",
    api_key: str | None = None,
    base_url: str | None = None,
    eval_provider: str | None = None,
    eval_model: str | None = None,
    eval_api_key: str | None = None,
    eval_base_url: str | None = None,
):
    return SimpleNamespace(
        llm_internal=SimpleNamespace(
            provider=provider,
            model=model,
            api_key=api_key,
            base_url=base_url,
        ),
        llm_eval=SimpleNamespace(
            provider=eval_provider,
            model=eval_model,
            api_key=eval_api_key,
            base_url=eval_base_url,
        ),
    )


def test_parse_json_from_response_handles_direct_and_embedded_json() -> None:
    assert llm._parse_json_from_response('{"ok": true}') == {"ok": True}
    assert llm._parse_json_from_response("prefix [1, 2, 3] suffix") == [1, 2, 3]


@pytest.mark.asyncio
async def test_mock_llm_client_returns_fixed_values() -> None:
    client = llm.MockLLMClient(fixed_response='{"answer": 1}')
    assert await client.complete("prompt") == '{"answer": 1}'
    assert await client.complete_json("prompt") == {"answer": 1}

    json_client = llm.MockLLMClient(fixed_json={"status": "ok"})
    assert await json_client.complete_json("prompt") == {"status": "ok"}


def test_openai_compatible_client_uses_provider_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    captures: list[tuple[str, str]] = []

    class _FakeAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            captures.append((base_url, api_key))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))

    monkeypatch.setattr(llm, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(provider="openai", api_key="real-key"))
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")
    openai_client = llm.OpenAICompatibleClient(_provider="openai")

    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: _settings(provider="openai_compatible", api_key=None, base_url=None),
    )
    compat_client = llm.OpenAICompatibleClient(_provider="openai_compatible")

    monkeypatch.setattr(llm, "get_settings", lambda: _settings(provider="ollama"))
    ollama_client = llm.OpenAICompatibleClient(_provider="ollama")

    assert captures[0] == (llm._OPENAI_DEFAULT_BASE, "real-key")
    assert captures[1] == (llm._OPENAI_COMPATIBLE_DEFAULT_BASE, "env-key")
    assert captures[2] == (llm._OLLAMA_DEFAULT_BASE, "env-key")
    assert openai_client.model == "gpt-4o-mini"
    assert compat_client.model == "gpt-4o-mini"
    assert ollama_client.model == "gpt-4o-mini"


@pytest.mark.asyncio
async def test_openai_compatible_client_complete_json_falls_back_without_response_format(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[dict[str, object]] = []

    async def _create(**kwargs):
        calls.append(kwargs)
        if "response_format" in kwargs:
            raise RuntimeError("unsupported")
        return SimpleNamespace(
            choices=[SimpleNamespace(message=SimpleNamespace(content='{"fallback": true}'))]
        )

    class _FakeAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=_create))

    monkeypatch.setattr(llm, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(llm, "get_settings", lambda: _settings(api_key="key"))
    client = llm.OpenAICompatibleClient()

    assert await client.complete_json("prompt") == {"fallback": True}
    assert len(calls) == 2
    assert "response_format" in calls[0]
    assert "response_format" not in calls[1]


def test_build_llm_client_from_config_routes_supported_providers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    openai_ctor = MagicMock(return_value="openai-client")
    gemini_ctor = MagicMock(return_value="gemini-client")
    claude_ctor = MagicMock(return_value="claude-client")
    monkeypatch.setattr(llm, "OpenAICompatibleClient", openai_ctor)
    monkeypatch.setattr(llm, "_gemini_client", gemini_ctor)
    monkeypatch.setattr(llm, "_claude_client", claude_ctor)

    assert llm._build_llm_client_from_config("openai", "m", "k", None) == "openai-client"
    assert (
        llm._build_llm_client_from_config("openai_compatible", "m", None, None)
        == "openai-client"
    )
    assert llm._build_llm_client_from_config("gemini", "m", "k", None) == "gemini-client"
    assert llm._build_llm_client_from_config("anthropic", "m", "k", None) == "claude-client"

    with pytest.raises(ValueError, match="Unknown LLM provider"):
        llm._build_llm_client_from_config("bad", "m", None, None)


def test_build_llm_client_from_config_validates_required_keys() -> None:
    with pytest.raises(ValueError, match="LLM_INTERNAL__API_KEY"):
        llm._build_llm_client_from_config("gemini", "m", None, None)
    with pytest.raises(ValueError, match="LLM_INTERNAL__API_KEY"):
        llm._build_llm_client_from_config("claude", "m", None, None)


def test_build_llm_client_from_config_propagates_helper_import_errors(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(llm, "_gemini_client", MagicMock(side_effect=ImportError("missing gemini")))
    monkeypatch.setattr(llm, "_claude_client", MagicMock(side_effect=ImportError("missing claude")))

    with pytest.raises(ImportError, match="missing gemini"):
        llm._build_llm_client_from_config("gemini", "m", "k", None)
    with pytest.raises(ImportError, match="missing claude"):
        llm._build_llm_client_from_config("claude", "m", "k", None)


def test_get_internal_llm_client_uses_internal_config_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = MagicMock(return_value="client")
    monkeypatch.setattr(llm, "_build_llm_client_from_config", builder)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: _settings(provider="openai", model="gpt-internal", api_key=None, base_url=None),
    )
    monkeypatch.setenv("OPENAI_API_KEY", "env-key")

    assert llm.get_internal_llm_client() == "client"
    builder.assert_called_once_with(
        provider="openai",
        model="gpt-internal",
        api_key="env-key",
        base_url=None,
        env_prefix="LLM_INTERNAL",
    )


def test_get_eval_llm_client_falls_back_to_internal_config(monkeypatch: pytest.MonkeyPatch) -> None:
    builder = MagicMock(return_value="eval-client")
    monkeypatch.setattr(llm, "_build_llm_client_from_config", builder)
    monkeypatch.setattr(
        llm,
        "get_settings",
        lambda: _settings(
            provider="openai_compatible",
            model="internal-model",
            api_key="internal-key",
            base_url="http://internal",
            eval_provider=None,
            eval_model="eval-model",
            eval_api_key=None,
            eval_base_url=None,
        ),
    )

    assert llm.get_eval_llm_client() == "eval-client"
    builder.assert_called_once_with(
        provider="openai_compatible",
        model="eval-model",
        api_key="internal-key",
        base_url="http://internal",
        env_prefix="LLM_EVAL",
    )


def test_openai_compatible_client_uses_dummy_key_for_explicit_base_url(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captures: list[tuple[str, str]] = []

    class _FakeAsyncOpenAI:
        def __init__(self, *, base_url: str, api_key: str) -> None:
            captures.append((base_url, api_key))
            self.chat = SimpleNamespace(completions=SimpleNamespace(create=AsyncMock()))

    monkeypatch.setattr(llm, "AsyncOpenAI", _FakeAsyncOpenAI)
    monkeypatch.setattr(llm, "get_settings", lambda: _settings())
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)

    client = llm.OpenAICompatibleClient(base_url="http://local/v1", model="x")

    assert captures == [("http://local/v1", "dummy")]
    assert client.model == "x"
    assert os.environ.get("OPENAI_API_KEY") is None
