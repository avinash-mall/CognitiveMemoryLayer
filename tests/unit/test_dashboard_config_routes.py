from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException

from src.api.dashboard import config_routes
from src.api.schemas import ConfigUpdateRequest

from .dashboard_support import ADMIN_AUTH


@pytest.mark.asyncio
async def test_dashboard_config_returns_sections_and_masks_secrets(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AUTH__API_KEY", "test-key")
    monkeypatch.setenv("AUTH__ADMIN_API_KEY", "admin-key")
    monkeypatch.setenv("LLM_INTERNAL__API_KEY", "llm-secret")
    monkeypatch.setenv("DATABASE__NEO4J_PASSWORD", "neo-secret")

    from src.core.config import get_settings

    get_settings.cache_clear()
    try:
        payload = await config_routes.dashboard_config(auth=ADMIN_AUTH, db=MagicMock())
    finally:
        get_settings.cache_clear()

    names = [section.name for section in payload.sections]
    assert "Application" in names
    assert "Database" in names
    assert "Embedding (Internal)" in names

    items = {
        item.key: item
        for section in payload.sections
        for item in section.items
    }
    assert items["auth.api_key"].value == "****"
    assert items["auth.api_key"].is_editable is False
    assert items["database.neo4j_password"].value == "****"
    assert items["embedding_internal.provider"].is_editable is True
    assert items["auth.api_key"].source == "env"


@pytest.mark.asyncio
async def test_dashboard_config_update_persists_editable_values(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_routes, "get_env_path", lambda: env_path)
    monkeypatch.setattr(config_routes, "update_env", lambda updates: captured.update(updates))

    body = ConfigUpdateRequest(
        updates={
            "embedding_internal.dimensions": 1536,
            "cors_origins": ["https://a.test", "https://b.test"],
            "debug": True,
        }
    )

    result = await config_routes.dashboard_config_update(body=body, auth=ADMIN_AUTH, db=MagicMock())

    assert result["success"] is True
    assert captured == {
        "EMBEDDING_INTERNAL__DIMENSIONS": 1536,
        "CORS_ORIGINS": "https://a.test,https://b.test",
        "DEBUG": True,
    }


@pytest.mark.asyncio
async def test_dashboard_config_update_handles_blank_cors_origins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_routes, "get_env_path", lambda: env_path)
    monkeypatch.setattr(config_routes, "update_env", lambda updates: captured.update(updates))

    await config_routes.dashboard_config_update(
        body=ConfigUpdateRequest(updates={"cors_origins": ""}),
        auth=ADMIN_AUTH,
        db=MagicMock(),
    )

    assert captured == {"CORS_ORIGINS": ""}


@pytest.mark.asyncio
async def test_dashboard_config_update_rejects_invalid_values() -> None:
    with pytest.raises(HTTPException, match="positive integer"):
        await config_routes.dashboard_config_update(
            body=ConfigUpdateRequest(updates={"embedding_internal.dimensions": 0}),
            auth=ADMIN_AUTH,
            db=MagicMock(),
        )


@pytest.mark.asyncio
async def test_dashboard_config_update_rejects_write_protected_fields() -> None:
    with pytest.raises(HTTPException, match="Cannot modify secret field"):
        await config_routes.dashboard_config_update(
            body=ConfigUpdateRequest(updates={"auth.api_key": "new-key"}),
            auth=ADMIN_AUTH,
            db=MagicMock(),
        )


@pytest.mark.asyncio
async def test_dashboard_config_update_rejects_unknown_fields() -> None:
    with pytest.raises(HTTPException, match="not editable"):
        await config_routes.dashboard_config_update(
            body=ConfigUpdateRequest(updates={"feature.unknown": True}),
            auth=ADMIN_AUTH,
            db=MagicMock(),
        )


@pytest.mark.asyncio
async def test_dashboard_config_update_returns_503_when_project_root_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / "missing" / ".env"
    monkeypatch.setattr(config_routes, "get_env_path", lambda: env_path)

    with pytest.raises(HTTPException, match="Project root not found"):
        await config_routes.dashboard_config_update(
            body=ConfigUpdateRequest(updates={"debug": False}),
            auth=ADMIN_AUTH,
            db=MagicMock(),
        )


@pytest.mark.asyncio
async def test_dashboard_config_update_returns_503_when_env_write_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("", encoding="utf-8")
    monkeypatch.setattr(config_routes, "get_env_path", lambda: env_path)
    monkeypatch.setattr(config_routes, "update_env", MagicMock(side_effect=OSError("read only")))

    with pytest.raises(HTTPException, match="Cannot write to .env"):
        await config_routes.dashboard_config_update(
            body=ConfigUpdateRequest(updates={"debug": False}),
            auth=ADMIN_AUTH,
            db=MagicMock(),
        )
