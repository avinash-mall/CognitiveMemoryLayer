"""Unit tests for embedded CML client (config and mocked orchestrator)."""

from datetime import UTC, datetime
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock
from uuid import uuid4

import pytest

try:
    from cml.storage.sqlite_store import SQLiteMemoryStore  # noqa: F401
except ImportError:
    pytest.skip(
        'Embedded engine not available. From repo root: pip install -e ".[embedded]".',
        allow_module_level=True,
    )

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
        "LLM_INTERNAL__API_KEY",
        "LLM_INTERNAL__EXTRA_BODY",
    ):
        monkeypatch.delenv(key, raising=False)
    config = EmbeddedConfig()
    assert config.storage_mode == "lite"
    assert config.tenant_id == "default"
    assert config.database.database_url.startswith("sqlite")
    assert config.embedding.provider == "local"
    assert config.embedding.dimensions == 768
    assert config.llm.provider == "openai"
    assert config.llm.extra_body is None


def test_embedded_llm_config_reads_api_key_and_extra_body(monkeypatch: pytest.MonkeyPatch) -> None:
    """EmbeddedLLMConfig reads LLM_INTERNAL__API_KEY and JSON LLM_INTERNAL__EXTRA_BODY."""
    monkeypatch.setenv("LLM_INTERNAL__PROVIDER", "openai_compatible")
    monkeypatch.setenv("LLM_INTERNAL__MODEL", "qwen3.8-27b")
    monkeypatch.setenv("LLM_INTERNAL__BASE_URL", "http://127.0.0.1:8000/v1")
    monkeypatch.setenv("LLM_INTERNAL__API_KEY", "sk-test-local")
    monkeypatch.setenv(
        "LLM_INTERNAL__EXTRA_BODY",
        '{"chat_template_kwargs": {"enable_thinking": false}}',
    )
    config = EmbeddedConfig()
    assert config.llm.provider == "openai_compatible"
    assert config.llm.model == "qwen3.8-27b"
    assert config.llm.base_url == "http://127.0.0.1:8000/v1"
    assert config.llm.api_key == "sk-test-local"
    assert config.llm.extra_body == {"chat_template_kwargs": {"enable_thinking": False}}


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


def test_packet_to_read_response_forwards_metadata_warnings_and_intrinsic() -> None:
    """_packet_to_read_response forwards warnings, open_questions, sufficiency, and retrieval_meta."""
    from cml.embedded import _packet_to_read_response

    packet = SimpleNamespace(
        facts=[],
        preferences=[],
        recent_episodes=[],
        procedures=[],
        constraints=[],
        all_memories=[],
        warnings=["CA1_SPEAKER_MISMATCH: Entity 'Alex' has no recorded link to 'surf'"],
        open_questions=["Did Alex mention surfing in another session?"],
        sufficiency={"sufficient": False, "top_relevance": 0.21},
        retrieval_meta={
            "sources_completed": 3,
            "intrinsic_injection": {
                "mode": "prefix_block",
                "prefix_block": "[ACTIVE_MEMORY] User is vegetarian.",
            },
        },
    )

    result = _packet_to_read_response("Did Alex surf?", packet, elapsed_ms=4.5)
    assert result.warnings == ["CA1_SPEAKER_MISMATCH: Entity 'Alex' has no recorded link to 'surf'"]
    assert result.open_questions == ["Did Alex mention surfing in another session?"]
    assert result.sufficiency == {"sufficient": False, "top_relevance": 0.21}
    assert result.retrieval_meta is not None
    assert result.retrieval_meta["sources_completed"] == 3
    assert result.intrinsic_injection == {
        "mode": "prefix_block",
        "prefix_block": "[ACTIVE_MEMORY] User is vegetarian.",
    }


@pytest.mark.asyncio
async def test_embedded_initialize_falls_back_when_cloud_provider_missing_key(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """initialize() falls back to llm_client=None when cloud provider has no API key."""
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__API_KEY", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__PROVIDER", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__MODEL", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__BASE_URL", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__EXTRA_BODY", raising=False)

    config = EmbeddedConfig(tenant_id="test-tenant")
    client = EmbeddedCognitiveMemoryLayer(config=config)

    fake_store = MagicMock()
    fake_store.initialize = AsyncMock(return_value=None)
    fake_store.close = AsyncMock(return_value=None)
    fake_orchestrator = MagicMock()
    mock_create_lite = AsyncMock(return_value=fake_orchestrator)

    class _Features:
        use_llm_enabled = True

    class _LLMInternal:
        base_url = None
        model = "gpt-4o-mini"
        provider = "openai"
        api_key = None
        extra_body = None

    class _Settings:
        features = _Features()
        llm_internal = _LLMInternal()

    monkeypatch.setattr("cml.embedded._check_embedded_deps", lambda: None)
    monkeypatch.setattr(
        "cml.storage.sqlite_store.SQLiteMemoryStore",
        lambda db_path: fake_store,
    )
    monkeypatch.setattr("src.utils.embeddings.get_embedding_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.memory.orchestrator.MemoryOrchestrator.create_lite",
        mock_create_lite,
    )
    monkeypatch.setattr("src.core.config.get_settings", lambda: _Settings())
    monkeypatch.setattr(
        "src.utils.llm.OpenAICompatibleClient",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("Cloud provider without API key must not construct LLM client")
        ),
    )

    await client.initialize()
    mock_create_lite.assert_awaited_once()
    assert mock_create_lite.await_args.args[2] is None
    await client.close()


@pytest.mark.asyncio
async def test_embedded_initialize_forwards_extra_body_for_local_provider(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """initialize() passes extra_body through to OpenAICompatibleClient for local providers."""
    monkeypatch.delenv("LLM_INTERNAL__PROVIDER", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__MODEL", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__BASE_URL", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__API_KEY", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__EXTRA_BODY", raising=False)

    from cml.embedded_config import EmbeddedLLMConfig

    config = EmbeddedConfig(
        tenant_id="test-tenant",
        llm=EmbeddedLLMConfig(
            provider="openai_compatible",
            model="qwen3.8-27b",
            base_url="http://127.0.0.1:8000/v1",
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        ),
    )
    client = EmbeddedCognitiveMemoryLayer(config=config)

    fake_store = MagicMock()
    fake_store.initialize = AsyncMock(return_value=None)
    fake_store.close = AsyncMock(return_value=None)
    fake_orchestrator = MagicMock()
    mock_create_lite = AsyncMock(return_value=fake_orchestrator)

    captured: dict[str, object] = {}

    def _fake_llm_client(**kwargs: object) -> MagicMock:
        captured.update(kwargs)
        return MagicMock(name="llm_client")

    class _Features:
        use_llm_enabled = True

    class _LLMInternal:
        base_url = None
        model = "gpt-4o-mini"
        provider = "openai"
        api_key = None
        extra_body = None

    class _Settings:
        features = _Features()
        llm_internal = _LLMInternal()

    monkeypatch.setattr("cml.embedded._check_embedded_deps", lambda: None)
    monkeypatch.setattr(
        "cml.storage.sqlite_store.SQLiteMemoryStore",
        lambda db_path: fake_store,
    )
    monkeypatch.setattr("src.utils.embeddings.get_embedding_client", lambda: MagicMock())
    monkeypatch.setattr(
        "src.memory.orchestrator.MemoryOrchestrator.create_lite",
        mock_create_lite,
    )
    monkeypatch.setattr("src.core.config.get_settings", lambda: _Settings())
    monkeypatch.setattr("src.utils.llm.OpenAICompatibleClient", _fake_llm_client)

    await client.initialize()
    mock_create_lite.assert_awaited_once()
    assert mock_create_lite.await_args.args[2] is not None
    assert captured["model"] == "qwen3.8-27b"
    assert captured["base_url"] == "http://127.0.0.1:8000/v1"
    assert captured["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    await client.close()


@pytest.mark.asyncio
async def test_embedded_consolidate_forgetting_and_batch_ops() -> None:
    """EmbeddedCognitiveMemoryLayer exposes consolidate, run_forgetting, batch_write, batch_read."""
    from types import SimpleNamespace

    client = EmbeddedCognitiveMemoryLayer(tenant_id="t1")
    client._initialized = True
    orch = MagicMock()
    orch.consolidation = MagicMock()
    orch.consolidation.consolidate = AsyncMock(
        return_value=SimpleNamespace(
            episodes_sampled=4,
            clusters_formed=1,
            clusters_skipped_no_recurrence=0,
            gists_extracted=1,
            details_recovered=0,
        )
    )
    orch.forgetting = MagicMock()
    orch.forgetting.run_forgetting = AsyncMock(
        return_value=SimpleNamespace(
            memories_scanned=5,
            result=SimpleNamespace(operations_applied=2),
        )
    )
    orch.write = AsyncMock(return_value={"memory_id": None, "chunks_created": 1, "message": "ok"})
    orch.read = AsyncMock(
        return_value=SimpleNamespace(
            facts=[],
            preferences=[],
            recent_episodes=[],
            procedures=[],
            constraints=[],
            all_memories=[],
        )
    )
    client._orchestrator = orch

    cons = await client.consolidate()
    assert cons["status"] == "consolidation_completed"
    assert cons["episodes_sampled"] == 4

    forg = await client.run_forgetting(dry_run=True)
    assert forg["status"] == "forgetting_completed"
    assert forg["memories_scanned"] == 5
    assert forg["dry_run"] is True

    writes = await client.batch_write([{"content": "a"}, {"content": "b"}])
    assert len(writes) == 2

    reads = await client.batch_read(["q1", "q2"])
    assert len(reads) == 2


def test_embedded_database_config_accepts_database_url_and_alias() -> None:
    """EmbeddedDatabaseConfig accepts both database_url and legacy postgres_url alias."""
    from cml.embedded_config import EmbeddedDatabaseConfig

    cfg_by_name = EmbeddedDatabaseConfig(database_url="sqlite+aiosqlite:///custom.db")
    assert cfg_by_name.database_url == "sqlite+aiosqlite:///custom.db"

    cfg_by_alias = EmbeddedDatabaseConfig(postgres_url="sqlite+aiosqlite:///legacy.db")
    assert cfg_by_alias.database_url == "sqlite+aiosqlite:///legacy.db"


def test_embedded_embedding_config_respects_explicit_provider_and_dimensions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit provider/dimensions on EmbeddedEmbeddingConfig must not be overwritten by env."""
    from cml.embedded_config import EmbeddedEmbeddingConfig

    monkeypatch.setenv("EMBEDDING_INTERNAL__PROVIDER", "local")
    monkeypatch.setenv("EMBEDDING_INTERNAL__DIMENSIONS", "768")

    cfg = EmbeddedEmbeddingConfig(provider="mock", dimensions=384)
    assert cfg.provider == "mock"
    assert cfg.dimensions == 384


@pytest.mark.asyncio
async def test_embedded_update_translates_memory_not_found_to_cml_not_found() -> None:
    """EmbeddedCognitiveMemoryLayer.update() translates engine MemoryNotFoundError to cml.NotFoundError."""
    from cml.exceptions import NotFoundError
    from src.core.exceptions import MemoryNotFoundError

    client = EmbeddedCognitiveMemoryLayer(tenant_id="t1")
    client._initialized = True
    orch = MagicMock()
    missing_id = uuid4()
    orch.update = AsyncMock(side_effect=MemoryNotFoundError(missing_id))
    client._orchestrator = orch

    with pytest.raises(NotFoundError):
        await client.update(missing_id, text="updated")
