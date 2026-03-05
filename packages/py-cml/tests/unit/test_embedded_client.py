"""Unit tests for embedded CML client (config and mocked orchestrator)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

from cml.embedded import EmbeddedCognitiveMemoryLayer
from cml.embedded_config import EmbeddedConfig


def test_embedded_config_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    """EmbeddedConfig has expected defaults."""
    for key in (
        "EMBEDDING_INTERNAL__PROVIDER",
        "EMBEDDING_INTERNAL__MODEL",
        "EMBEDDING_INTERNAL__DIMENSIONS",
        "EMBEDDING_INTERNAL__BASE_URL",
        "EMBEDDING_INTERNAL__LOCAL_MODEL",
        "LLM_INTERNAL__PROVIDER",
        "LLM_INTERNAL__MODEL",
        "LLM_INTERNAL__BASE_URL",
    ):
        monkeypatch.delenv(key, raising=False)
    config = EmbeddedConfig()
    assert config.storage_mode == "lite"
    assert config.tenant_id == "default"
    assert config.database.database_url.startswith("sqlite")
    assert config.embedding.provider == "local"
    assert config.llm.provider == "openai"


@pytest.mark.asyncio
async def test_embedded_write_returns_write_response_when_initialized() -> None:
    """When orchestrator is set, write() returns WriteResponse."""
    config = EmbeddedConfig(tenant_id="test-tenant")
    client = EmbeddedCognitiveMemoryLayer(config=config)
    mid = uuid4()
    client._orchestrator = MagicMock()
    client._orchestrator.write = AsyncMock(
        return_value={
            "memory_id": mid,
            "chunks_created": 1,
            "message": "ok",
        }
    )
    client._initialized = True
    result = await client.write("hello")
    assert result.success is True
    assert result.memory_id == mid
    assert result.chunks_created == 1
    client._orchestrator.write.assert_called_once()
    call_kw = client._orchestrator.write.call_args[1]
    assert call_kw["tenant_id"] == "test-tenant"
    assert call_kw["content"] == "hello"


@pytest.mark.asyncio
async def test_embedded_read_passes_filters_and_user_timezone() -> None:
    """Embedded read() passes memory_types, since, until, and user_timezone to orchestrator.read()."""
    from cml.models.enums import MemoryType

    config = EmbeddedConfig(tenant_id="t1")
    client = EmbeddedCognitiveMemoryLayer(config=config)
    client._orchestrator = MagicMock()
    client._orchestrator.read = AsyncMock(
        return_value=MagicMock(
            all_memories=[],
            facts=[],
            preferences=[],
            recent_episodes=[],
        )
    )
    client._initialized = True
    since = datetime(2025, 1, 1, tzinfo=UTC)
    until = datetime(2025, 1, 2, tzinfo=UTC)
    await client.read(
        "q",
        memory_types=[MemoryType.PREFERENCE],
        since=since,
        until=until,
        user_timezone="America/New_York",
    )
    client._orchestrator.read.assert_called_once()
    call_kw = client._orchestrator.read.call_args[1]
    assert call_kw["tenant_id"] == "t1"
    assert call_kw["query"] == "q"
    assert call_kw["memory_types"] == ["preference"]
    assert call_kw["since"] == since
    assert call_kw["until"] == until
    assert call_kw["user_timezone"] == "America/New_York"


def test_packet_to_read_response_includes_constraints_and_procedures() -> None:
    """Embedded packet mapping includes constraints/procedures in memories and constraints field."""
    from cml.embedded import _packet_to_read_response

    now = datetime.now(UTC)

    def _retrieved(text: str, mem_type: str) -> SimpleNamespace:
        record = SimpleNamespace(
            id=uuid4(),
            text=text,
            type=SimpleNamespace(value=mem_type),
            confidence=0.9,
            timestamp=now,
            metadata={},
        )
        return SimpleNamespace(record=record, relevance_score=0.8)

    fact = _retrieved("Fact memory", "semantic_fact")
    preference = _retrieved("Preference memory", "preference")
    episode = _retrieved("Episode memory", "episodic_event")
    procedure = _retrieved("Procedure memory", "procedure")
    constraint = _retrieved("Constraint memory", "constraint")
    packet = SimpleNamespace(
        facts=[fact],
        preferences=[preference],
        recent_episodes=[episode],
        procedures=[procedure],
        constraints=[constraint],
        all_memories=[fact, preference, episode, procedure, constraint],
    )

    result = _packet_to_read_response("q", packet, elapsed_ms=2.0)
    texts = [m.text for m in result.memories]
    assert "Procedure memory" in texts
    assert "Constraint memory" in texts
    assert len(result.constraints) == 1
    assert result.constraints[0].text == "Constraint memory"


@pytest.mark.asyncio
async def test_embedded_ensure_initialized_raises() -> None:
    """write() raises if not initialized."""
    client = EmbeddedCognitiveMemoryLayer()
    with pytest.raises(RuntimeError, match="not initialized"):
        await client.write("x")


@pytest.mark.asyncio
async def test_embedded_initialize_skips_llm_client_when_master_llm_disabled(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """initialize() should pass llm_client=None into create_lite when master LLM flag is off."""
    config = EmbeddedConfig(tenant_id="test-tenant")
    client = EmbeddedCognitiveMemoryLayer(config=config)

    fake_store = MagicMock()
    fake_store.initialize = AsyncMock(return_value=None)
    fake_store.close = AsyncMock(return_value=None)
    fake_orchestrator = MagicMock()
    mock_create_lite = AsyncMock(return_value=fake_orchestrator)

    class _Features:
        use_llm_enabled = False

    class _LLMInternal:
        base_url = None
        model = ""
        provider = "openai"

    class _Settings:
        features = _Features()
        llm_internal = _LLMInternal()

    monkeypatch.setattr("cml.embedded._check_embedded_deps", lambda: None)
    monkeypatch.setattr(
        "cml.storage.sqlite_store.SQLiteMemoryStore",
        lambda db_path: fake_store,
    )
    monkeypatch.setattr(
        "src.utils.embeddings.get_embedding_client",
        lambda: MagicMock(),
    )
    monkeypatch.setattr(
        "src.memory.orchestrator.MemoryOrchestrator.create_lite",
        mock_create_lite,
    )
    monkeypatch.setattr(
        "src.core.config.get_settings",
        lambda: _Settings(),
    )
    monkeypatch.setattr(
        "src.utils.llm.OpenAICompatibleClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("OpenAICompatibleClient must not be constructed in non-LLM mode")
        ),
    )

    await client.initialize()
    mock_create_lite.assert_awaited_once()
    assert mock_create_lite.await_args.args[2] is None
    await client.close()
