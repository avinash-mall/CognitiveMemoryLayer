from __future__ import annotations

from pathlib import Path

from hatch_build import get_version


def test_get_version_prefers_environment_variable_over_env_file(
    tmp_path: Path, monkeypatch
) -> None:
    (tmp_path / ".env").write_text("VERSION=1.2.3\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("1.2.2\n", encoding="utf-8")
    monkeypatch.setenv("VERSION", "1.2.4")

    assert get_version(tmp_path) == "1.2.4"


def test_get_version_falls_back_to_env_file_then_version_file(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.delenv("VERSION", raising=False)
    (tmp_path / ".env").write_text("VERSION=2.0.0\n", encoding="utf-8")
    (tmp_path / "VERSION").write_text("1.9.9\n", encoding="utf-8")

    assert get_version(tmp_path) == "2.0.0"

    (tmp_path / ".env").unlink()
    assert get_version(tmp_path) == "1.9.9"
