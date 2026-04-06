"""Targeted tests for cml.eval.locomo helpers."""

from __future__ import annotations

from types import SimpleNamespace

from cml.eval.locomo import _dashboard_post


def test_dashboard_post_sends_csrf_header(monkeypatch) -> None:
    captured: dict[str, object] = {}

    class _Resp:
        status_code = 200

        def raise_for_status(self) -> None:
            return None

        def json(self) -> dict[str, bool]:
            return {"ok": True}

    def _fake_post(url, json, headers, timeout):
        captured["url"] = url
        captured["json"] = json
        captured["headers"] = headers
        captured["timeout"] = timeout
        return _Resp()

    def _stale_post(*args, **kwargs):
        raise AssertionError("should not use stale cached session")

    monkeypatch.setattr("cml.eval.locomo._SESSION", SimpleNamespace(post=_stale_post))
    monkeypatch.setattr("cml.eval.locomo.requests", SimpleNamespace(post=_fake_post))

    result = _dashboard_post(
        "http://localhost:8000", "test-key", "/consolidate", {"tenant_id": "t"}
    )

    assert result == {"ok": True}
    assert captured["headers"]["X-Requested-With"] == "XMLHttpRequest"
    assert captured["headers"]["X-API-Key"] == "test-key"
