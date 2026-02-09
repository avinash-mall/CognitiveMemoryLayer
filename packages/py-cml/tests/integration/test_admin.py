"""Integration tests: admin endpoints (consolidate, run_forgetting, list_tenants, component_health)."""

import pytest


@pytest.mark.integration
@pytest.mark.asyncio
async def test_consolidate(live_client):
    """Consolidate (if server supports). Skip or expect 404 if no dashboard routes."""
    try:
        result = await live_client.consolidate()
        assert isinstance(result, dict)
    except Exception as e:
        if "404" in str(e) or "501" in str(e) or "403" in str(e):
            pytest.skip("Server may not implement consolidate or require admin key")
        raise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_run_forgetting_dry_run(live_client):
    """run_forgetting with dry_run=True."""
    try:
        result = await live_client.run_forgetting(dry_run=True)
        assert isinstance(result, dict)
    except Exception as e:
        if "404" in str(e) or "501" in str(e) or "403" in str(e):
            pytest.skip("Server may not implement run_forgetting or require admin key")
        raise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_list_tenants(live_client):
    """List tenants (admin). Skip if 404/403."""
    try:
        tenants = await live_client.list_tenants()
        assert isinstance(tenants, list)
    except Exception as e:
        if "404" in str(e) or "501" in str(e) or "403" in str(e):
            pytest.skip("Server may not implement list_tenants or require admin key")
        raise


@pytest.mark.integration
@pytest.mark.asyncio
async def test_component_health(live_client):
    """Component health (dashboard). Skip if 404."""
    try:
        health = await live_client.component_health()
        assert isinstance(health, dict)
    except Exception as e:
        if "404" in str(e) or "501" in str(e) or "403" in str(e):
            pytest.skip("Server may not implement component_health")
        raise
