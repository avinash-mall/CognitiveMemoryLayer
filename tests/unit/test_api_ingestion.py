import os
import uuid
from datetime import UTC, datetime

import pytest
import requests

# Skip entire module when evaluation package is not available (e.g. in Docker/CI)
pytest.importorskip("evaluation")
from evaluation.scripts.eval_locomo_plus import _cml_write


def _server_authed() -> bool:
    base_url = os.environ.get("CML_BASE_URL", "")
    api_key = os.environ.get("CML_API_KEY", "")
    if not api_key or not base_url:
        return False
    try:
        resp = requests.get(
            f"{base_url.rstrip('/')}/api/v1/memory/stats",
            headers={"X-API-Key": api_key},
            timeout=3,
        )
        return resp.status_code == 200
    except Exception:
        return False


@pytest.mark.e2e
@pytest.mark.skipif(not _server_authed(), reason="CML API server not running or auth not configured")
def test_api_ingestion():
    base_url = os.environ.get("CML_BASE_URL", "").rstrip("/")
    api_key = os.environ.get("CML_API_KEY", "")
    tenant_id = f"test-tenant-{uuid.uuid4()}"
    session_id = f"test-session-{uuid.uuid4()}"
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()

    print(f"Ingesting memory to API with timestamp: {historical_date}")

    _cml_write(
        base_url=base_url,
        api_key=api_key,
        tenant_id=tenant_id,
        content="I moved to New York in January 2023.",
        session_id=session_id,
        metadata={"test": True},
        turn_id="turn_0",
        timestamp=historical_date,
    )
    print("Ingestion complete.")
    print(f"Tenant ID to check: {tenant_id}")


if __name__ == "__main__":
    test_api_ingestion()
