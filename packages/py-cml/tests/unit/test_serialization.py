"""Tests for serialization utilities (CMLJSONEncoder, serialize_for_api)."""

import json
from datetime import datetime
from uuid import uuid4

from cml.utils.serialization import CMLJSONEncoder, serialize_for_api


def test_serialize_for_api_uuid_and_datetime() -> None:
    """serialize_for_api converts UUID and datetime to JSON-safe values."""
    uid = uuid4()
    dt = datetime(2025, 1, 1, 12, 0, 0)
    data = {"id": uid, "created": dt, "nested": {"a": uid}}
    out = serialize_for_api(data)
    assert out["id"] == str(uid)
    assert out["created"] == dt.isoformat()
    assert out["nested"]["a"] == str(uid)


def test_cml_json_encoder_uuid_datetime() -> None:
    """CMLJSONEncoder serializes UUID and datetime in json.dumps."""
    uid = uuid4()
    dt = datetime(2025, 1, 1, 12, 0, 0)
    s = json.dumps({"id": uid, "ts": dt}, cls=CMLJSONEncoder)
    assert str(uid) in s
    assert dt.isoformat() in s
