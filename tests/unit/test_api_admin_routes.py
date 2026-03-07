from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from fastapi import HTTPException

from src.api import admin_routes

AUTH = SimpleNamespace(tenant_id="tenant-a")


@pytest.mark.asyncio
async def test_trigger_consolidation_returns_summary() -> None:
    report = SimpleNamespace(episodes_sampled=7, clusters_formed=3, gists_extracted=2)
    orchestrator = SimpleNamespace(
        consolidation=SimpleNamespace(consolidate=AsyncMock(return_value=report))
    )

    result = await admin_routes.trigger_consolidation(
        user_id="user-1",
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result == {
        "status": "consolidation_completed",
        "user_id": "user-1",
        "episodes_sampled": 7,
        "clusters_formed": 3,
        "gists_extracted": 2,
    }


@pytest.mark.asyncio
async def test_trigger_consolidation_wraps_errors() -> None:
    orchestrator = SimpleNamespace(
        consolidation=SimpleNamespace(consolidate=AsyncMock(side_effect=RuntimeError("consolidate boom")))
    )

    with pytest.raises(HTTPException, match="Consolidation failed: consolidate boom"):
        await admin_routes.trigger_consolidation(
            user_id="user-1",
            auth=AUTH,
            orchestrator=orchestrator,
        )


@pytest.mark.asyncio
async def test_trigger_forgetting_returns_summary() -> None:
    report = SimpleNamespace(
        memories_scanned=11,
        result=SimpleNamespace(operations_applied=4),
    )
    orchestrator = SimpleNamespace(
        forgetting=SimpleNamespace(run_forgetting=AsyncMock(return_value=report))
    )

    result = await admin_routes.trigger_forgetting(
        user_id="user-1",
        dry_run=False,
        auth=AUTH,
        orchestrator=orchestrator,
    )

    assert result == {
        "status": "forgetting_completed",
        "user_id": "user-1",
        "dry_run": False,
        "memories_scanned": 11,
        "operations_applied": 4,
    }


@pytest.mark.asyncio
async def test_trigger_forgetting_wraps_errors() -> None:
    orchestrator = SimpleNamespace(
        forgetting=SimpleNamespace(run_forgetting=AsyncMock(side_effect=RuntimeError("forget boom")))
    )

    with pytest.raises(HTTPException, match="Forgetting failed: forget boom"):
        await admin_routes.trigger_forgetting(
            user_id="user-1",
            dry_run=True,
            auth=AUTH,
            orchestrator=orchestrator,
        )
