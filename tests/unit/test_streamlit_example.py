"""Render tests for the Streamlit example app."""

from __future__ import annotations

from pathlib import Path

import pytest


def test_streamlit_app_renders_without_runtime_exceptions() -> None:
    streamlit_testing = pytest.importorskip("streamlit.testing.v1")

    app = streamlit_testing.AppTest.from_file(
        str(Path(__file__).resolve().parents[2] / "examples" / "streamlit_app.py")
    )
    app.run(timeout=30)

    assert not app.exception
    assert [title.value for title in app.title] == ["Cognitive Memory Layer Studio"]
