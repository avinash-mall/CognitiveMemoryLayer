import uuid
from datetime import UTC, datetime

# Assuming API server is running locally
from evaluation.scripts.eval_locomo_plus import _cml_write


def test_api_ingestion():
    tenant_id = f"test-tenant-{uuid.uuid4()}"
    session_id = f"test-session-{uuid.uuid4()}"
    historical_date = datetime(2023, 1, 15, 12, 0, 0, tzinfo=UTC).isoformat()

    print(f"Ingesting memory to API with timestamp: {historical_date}")

    _cml_write(
        base_url="http://localhost:8000",
        api_key="test-key",
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
