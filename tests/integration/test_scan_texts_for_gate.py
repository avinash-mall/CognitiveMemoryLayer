"""The write gate's novelty window must see real memories and not machine paraphrases.

``scan_texts_for_gate`` excludes ``prospective:%`` rows, and that filter has a SQL trap:
most episodic rows have ``key IS NULL``, and ``NULL NOT LIKE 'x'`` evaluates to NULL,
which SQL treats as false. A bare ``NOT LIKE`` would therefore drop nearly every row and
blind the gate — silently, since the gate degrades to "everything is novel" rather than
erroring.
"""

from datetime import UTC, datetime
from uuid import uuid4

import pytest

from src.core.enums import MemorySource, MemoryType
from src.core.schemas import MemoryRecordCreate, Provenance
from src.storage.postgres import PostgresMemoryStore


def _record(tenant_id: str, text: str, key: str | None) -> MemoryRecordCreate:
    return MemoryRecordCreate(
        tenant_id=tenant_id,
        context_tags=[],
        type=MemoryType.EPISODIC_EVENT,
        text=text,
        key=key,
        embedding=None,
        entities=[],
        relations=[],
        metadata={},
        timestamp=datetime.now(UTC),
        provenance=Provenance(source=MemorySource.USER_EXPLICIT),
    )


@pytest.mark.asyncio
async def test_keyless_records_are_returned_and_prospective_rows_are_not(pg_session_factory):
    store = PostgresMemoryStore(pg_session_factory)
    tenant_id = f"t-{uuid4().hex[:8]}"

    await store.upsert(_record(tenant_id, "I walked the dog this morning.", None))
    await store.upsert(_record(tenant_id, "The dog is a greyhound.", None))
    await store.upsert(
        _record(tenant_id, "User may adopt another dog.", f"prospective:{uuid4()}:abc123")
    )
    await store.upsert(_record(tenant_id, "A keyed but ordinary memory.", "preference:xyz"))

    texts = await store.scan_texts_for_gate(tenant_id, limit=10)

    # The NULL trap: if `NOT LIKE` were used bare, both key-less rows would vanish.
    assert "I walked the dog this morning." in texts
    assert "The dog is a greyhound." in texts
    assert "A keyed but ordinary memory." in texts
    assert "User may adopt another dog." not in texts
