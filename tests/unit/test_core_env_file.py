from __future__ import annotations

from pathlib import Path

import pytest

from src.core import env_file


def test_format_env_value_handles_scalars_lists_and_quoted_strings() -> None:
    assert env_file._format_env_value(True) == "true"
    assert env_file._format_env_value(False) == "false"
    assert env_file._format_env_value(12) == "12"
    assert env_file._format_env_value(1.5) == "1.5"
    assert env_file._format_env_value(["a", "b", 3]) == "a,b,3"
    assert env_file._format_env_value("plain") == "plain"
    assert env_file._format_env_value("needs spaces") == '"needs spaces"'
    assert env_file._format_env_value('quote"and\\slash') == '"quote\\"and\\\\slash"'


def test_parse_env_line_handles_comments_quotes_and_booleans() -> None:
    assert env_file._parse_env_line("") == (None, None)
    assert env_file._parse_env_line("# comment") == (None, None)
    assert env_file._parse_env_line("NOT A KEY") == (None, None)
    assert env_file._parse_env_line("FEATURE_FLAG=TRUE") == ("FEATURE_FLAG", "true")
    assert env_file._parse_env_line("ENABLED=FALSE") == ("ENABLED", "false")
    assert env_file._parse_env_line('APP_NAME="My App"') == ("APP_NAME", "My App")
    assert env_file._parse_env_line('ESCAPED="a\\"b\\\\c"') == ("ESCAPED", 'a"b\\c')


def test_get_env_path_uses_project_root(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(env_file, "_project_root", lambda: tmp_path)
    assert env_file.get_env_path() == tmp_path / ".env"


def test_update_env_replaces_existing_and_appends_new(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text(
        "DEBUG=false\n"
        'APP_NAME="Old Name"\n'
        "UNCHANGED=value\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(env_file, "get_env_path", lambda: env_path)

    env_file.update_env(
        {
            "DEBUG": True,
            "APP_NAME": 'New "Name"',
            "CORS_ORIGINS": ["http://a.test", "http://b.test"],
        }
    )

    content = env_path.read_text(encoding="utf-8")
    assert "DEBUG=true\n" in content
    assert 'APP_NAME="New \\"Name\\""\n' in content
    assert "UNCHANGED=value\n" in content
    assert "CORS_ORIGINS=http://a.test,http://b.test\n" in content


def test_update_env_creates_new_file_when_missing(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    monkeypatch.setattr(env_file, "get_env_path", lambda: env_path)

    env_file.update_env({"AUTH__API_KEY": "test-key", "DEBUG": False})

    assert set(env_path.read_text(encoding="utf-8").splitlines()) == {
        "AUTH__API_KEY=test-key",
        "DEBUG=false",
    }


def test_update_env_cleans_temp_file_on_replace_error(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    env_path = tmp_path / ".env"
    env_path.write_text("DEBUG=false\n", encoding="utf-8")
    monkeypatch.setattr(env_file, "get_env_path", lambda: env_path)

    original_replace = Path.replace

    def _boom(self: Path, target: Path) -> Path:
        if self.parent == tmp_path and self.suffix == ".tmp":
            raise OSError("replace failed")
        return original_replace(self, target)

    monkeypatch.setattr(Path, "replace", _boom)

    with pytest.raises(OSError, match="replace failed"):
        env_file.update_env({"DEBUG": True})

    assert env_path.read_text(encoding="utf-8") == "DEBUG=false\n"
    assert list(tmp_path.glob(".env.*.tmp")) == []
