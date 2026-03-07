"""Tests for scripts/run_examples.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

MODULE_PATH = Path(__file__).resolve().parents[2] / "scripts" / "run_examples.py"
MODULE_NAME = "run_examples_test_module"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
assert SPEC and SPEC.loader
RUN_EXAMPLES = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = RUN_EXAMPLES
SPEC.loader.exec_module(RUN_EXAMPLES)


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def test_discover_examples_ignores_helper_docs_and_parses_metadata(tmp_path: Path) -> None:
    root_examples = tmp_path / "examples"
    sdk_examples = tmp_path / "packages" / "py-cml" / "examples"
    _write(
        root_examples / "alpha.py",
        'EXAMPLE_META = {"name": "alpha", "kind": "python", "summary": "alpha"}\n',
    )
    _write(
        root_examples / "beta.sh",
        '# EXAMPLE_META: {"name":"beta","kind":"shell","summary":"beta"}\n',
    )
    _write(root_examples / "_shared.py", "IGNORED = True\n")
    _write(root_examples / "README.md", "# ignored\n")
    _write(root_examples / "openclaw_skill" / "SKILL.md", "# ignored\n")
    _write(
        sdk_examples / "gamma.py",
        'EXAMPLE_META = {"name": "gamma", "kind": "python", "summary": "gamma"}\n',
    )

    discovered = RUN_EXAMPLES.discover_examples(roots=(root_examples, sdk_examples))

    assert [example.name for example in discovered] == ["alpha", "beta", "gamma"]
    assert discovered[0].kind == "python"
    assert discovered[1].kind == "shell"


def test_discover_examples_falls_back_when_metadata_is_missing(tmp_path: Path) -> None:
    root_examples = tmp_path / "examples"
    _write(root_examples / "streamlit_demo.py", "print('hello')\n")
    _write(root_examples / "plain.py", "print('hello')\n")

    discovered = RUN_EXAMPLES.discover_examples(roots=(root_examples,))

    by_name = {example.name: example for example in discovered}
    assert by_name["streamlit_demo"].kind == "streamlit"
    assert by_name["plain"].kind == "python"


def test_discover_examples_rejects_duplicate_names(tmp_path: Path) -> None:
    root_one = tmp_path / "examples"
    root_two = tmp_path / "packages" / "py-cml" / "examples"
    _write(root_one / "alpha.py", 'EXAMPLE_META = {"name": "dup", "kind": "python"}\n')
    _write(root_two / "beta.py", 'EXAMPLE_META = {"name": "dup", "kind": "python"}\n')

    with pytest.raises(ValueError, match="Duplicate example name"):
        RUN_EXAMPLES.discover_examples(roots=(root_one, root_two))


def test_select_examples_filters_kind_and_enables_explicit_example() -> None:
    examples = [
        RUN_EXAMPLES.ExampleSpec(
            name="quickstart",
            path=Path("quickstart.py"),
            kind="python",
            summary="",
        ),
        RUN_EXAMPLES.ExampleSpec(
            name="chat_with_memory",
            path=Path("chat_with_memory.py"),
            kind="python",
            summary="",
            requires_openai=True,
        ),
        RUN_EXAMPLES.ExampleSpec(
            name="api_curl_examples",
            path=Path("api_curl_examples.sh"),
            kind="shell",
            summary="",
        ),
    ]

    selected, include_llm = RUN_EXAMPLES.select_examples(
        examples,
        all_examples=True,
        example_name=None,
        kind="python",
        include_llm=False,
    )
    assert [example.name for example in selected] == ["quickstart"]
    assert include_llm is False

    selected, include_llm = RUN_EXAMPLES.select_examples(
        examples,
        all_examples=False,
        example_name="chat_with_memory",
        kind=None,
        include_llm=False,
    )
    assert [example.name for example in selected] == ["chat_with_memory"]
    assert include_llm is True


def test_skip_reason_covers_missing_env_and_tools(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CML_API_KEY", raising=False)
    monkeypatch.delenv("CML_BASE_URL", raising=False)
    monkeypatch.delenv("CML_ADMIN_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
    monkeypatch.delenv("ANTHROPIC_MODEL", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__MODEL", raising=False)
    monkeypatch.delenv("LLM_INTERNAL__BASE_URL", raising=False)

    api_spec = RUN_EXAMPLES.ExampleSpec(
        name="quickstart",
        path=Path("quickstart.py"),
        kind="python",
        summary="",
        requires_api_key=True,
        requires_base_url=True,
    )
    assert RUN_EXAMPLES.skip_reason(api_spec, include_llm=False, no_skip=False) == "missing CML_API_KEY"

    admin_spec = RUN_EXAMPLES.ExampleSpec(
        name="admin_dashboard",
        path=Path("admin_dashboard.py"),
        kind="python",
        summary="",
        requires_api_key=True,
        requires_base_url=True,
        requires_admin_key=True,
    )
    monkeypatch.setenv("CML_API_KEY", "test-key")
    monkeypatch.setenv("CML_BASE_URL", "http://localhost:8000")
    assert (
        RUN_EXAMPLES.skip_reason(admin_spec, include_llm=False, no_skip=False)
        == "missing CML_ADMIN_API_KEY"
    )

    llm_spec = RUN_EXAMPLES.ExampleSpec(
        name="chat_with_memory",
        path=Path("chat_with_memory.py"),
        kind="python",
        summary="",
        requires_openai=True,
    )
    assert RUN_EXAMPLES.skip_reason(llm_spec, include_llm=False, no_skip=False) == "LLM example (use --include-llm)"
    assert "missing OPENAI_API_KEY" in RUN_EXAMPLES.skip_reason(
        llm_spec, include_llm=True, no_skip=False
    )

    shell_spec = RUN_EXAMPLES.ExampleSpec(
        name="api_curl_examples",
        path=Path("api_curl_examples.sh"),
        kind="shell",
        summary="",
    )
    monkeypatch.setattr(RUN_EXAMPLES, "find_shell_executable", lambda: None)
    assert RUN_EXAMPLES.skip_reason(shell_spec, include_llm=False, no_skip=False) == "missing bash/sh"

    streamlit_spec = RUN_EXAMPLES.ExampleSpec(
        name="streamlit_app",
        path=Path("streamlit_app.py"),
        kind="streamlit",
        summary="",
    )
    monkeypatch.setattr(RUN_EXAMPLES, "find_streamlit_executable", lambda: None)
    assert (
        RUN_EXAMPLES.skip_reason(streamlit_spec, include_llm=False, no_skip=False)
        == "missing streamlit executable"
    )


def test_build_command_for_each_kind(monkeypatch: pytest.MonkeyPatch) -> None:
    python_spec = RUN_EXAMPLES.ExampleSpec(
        name="quickstart",
        path=Path("examples/quickstart.py"),
        kind="python",
        summary="",
    )
    shell_spec = RUN_EXAMPLES.ExampleSpec(
        name="api_curl_examples",
        path=Path("examples/api_curl_examples.sh"),
        kind="shell",
        summary="",
    )
    streamlit_spec = RUN_EXAMPLES.ExampleSpec(
        name="streamlit_app",
        path=Path("examples/streamlit_app.py"),
        kind="streamlit",
        summary="",
    )

    monkeypatch.setattr(RUN_EXAMPLES, "find_shell_executable", lambda: "/usr/bin/bash")
    monkeypatch.setattr(RUN_EXAMPLES, "find_streamlit_executable", lambda: "/usr/bin/streamlit")

    assert RUN_EXAMPLES.build_command(python_spec)[:2] == [sys.executable, "-u"]
    assert RUN_EXAMPLES.build_command(shell_spec) == [
        "/usr/bin/bash",
        Path("examples/api_curl_examples.sh").as_posix(),
    ]
    assert RUN_EXAMPLES.build_command(streamlit_spec, port=8765) == [
        "/usr/bin/streamlit",
        "run",
        str(Path("examples/streamlit_app.py")),
        "--server.headless",
        "true",
        "--server.address",
        "127.0.0.1",
        "--server.port",
        "8765",
        "--browser.gatherUsageStats",
        "false",
    ]


def test_repo_examples_have_metadata_and_unique_names() -> None:
    discovered = RUN_EXAMPLES.discover_examples()
    assert discovered
    assert len({example.name for example in discovered}) == len(discovered)
    assert {"session_scope", "admin_dashboard", "temporal_fidelity"} <= {
        example.name for example in discovered
    }
    assert all(example.summary is not None for example in discovered)
