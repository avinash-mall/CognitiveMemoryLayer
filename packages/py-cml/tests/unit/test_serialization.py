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


def test_serialize_for_api_strips_none() -> None:
    """serialize_for_api omits keys with None values."""
    data = {"a": 1, "b": None, "c": "x"}
    out = serialize_for_api(data)
    assert out["a"] == 1
    assert out["c"] == "x"
    assert "b" not in out


def test_serialize_for_api_nested_dict() -> None:
    """serialize_for_api recurses into nested dicts."""
    uid = uuid4()
    data = {"outer": {"inner": {"id": uid}}}
    out = serialize_for_api(data)
    assert out["outer"]["inner"]["id"] == str(uid)


def test_serialize_for_api_list_of_dicts() -> None:
    """serialize_for_api recurses into list elements that are dicts."""
    uid = uuid4()
    data = {"items": [{"id": uid}, {"id": "plain"}]}
    out = serialize_for_api(data)
    assert out["items"][0]["id"] == str(uid)
    assert out["items"][1]["id"] == "plain"


def test_serialize_for_api_list_non_dict_elements_unchanged() -> None:
    """List elements that are not dicts are left as-is."""
    data = {"tags": ["a", "b", 3]}
    out = serialize_for_api(data)
    assert out["tags"] == ["a", "b", 3]


def test_serialize_for_api_empty_dict() -> None:
    """Empty dict returns empty dict."""
    out = serialize_for_api({})
    assert out == {}
