"""Discover and run Cognitive Memory Layer examples."""

from __future__ import annotations

import argparse
import ast
import json
import os
import shutil
import socket
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, cast

REPO_ROOT = Path(__file__).resolve().parent.parent
DISCOVERY_ROOTS = (
    REPO_ROOT / "examples",
    REPO_ROOT / "packages" / "py-cml" / "examples",
)
IGNORE_FILES = {"__init__.py", "README.md", "requirements.txt"}
IGNORE_DIRS = {"__pycache__", "openclaw_skill"}
SUPPORTED_EXTENSIONS = {".py", ".sh"}
PYTHON_ENCODING_ENV = "PYTHONIOENCODING"
DEFAULT_STREAMLIT_STARTUP_TIMEOUT = 45.0


def load_repo_env() -> None:
    env_file = REPO_ROOT / ".env"
    if not env_file.is_file():
        return
    try:
        from dotenv import load_dotenv

        load_dotenv(env_file)
    except ImportError:
        return


@dataclass(frozen=True)
class ExampleSpec:
    name: str
    path: Path
    kind: Literal["python", "shell", "streamlit"]
    summary: str
    requires_api: bool = False
    requires_api_key: bool = False
    requires_base_url: bool = False
    requires_admin_key: bool = False
    requires_embedded: bool = False
    requires_openai: bool = False
    requires_anthropic: bool = False
    interactive: bool = False
    timeout_sec: int = 60

    @property
    def relative_path(self) -> str:
        if self.path.is_absolute():
            return str(self.path.relative_to(REPO_ROOT))
        return str(self.path)


@dataclass(frozen=True)
class ExampleResult:
    status: Literal["ok", "fail", "skip"]
    detail: str
    elapsed: float


def _coerce_metadata(path: Path, raw: dict[str, Any]) -> ExampleSpec:
    raw_kind = raw.get("kind") or ("streamlit" if path.name.startswith("streamlit") else "python")
    if raw_kind not in {"python", "shell", "streamlit"}:
        raise ValueError(f"Unsupported example kind '{raw_kind}' in {path}")
    kind = cast("Literal['python', 'shell', 'streamlit']", raw_kind)
    name = str(raw.get("name") or path.stem)
    return ExampleSpec(
        name=name,
        path=path,
        kind=kind,
        summary=str(raw.get("summary") or ""),
        requires_api=bool(raw.get("requires_api", False)),
        requires_api_key=bool(raw.get("requires_api_key", False)),
        requires_base_url=bool(raw.get("requires_base_url", False)),
        requires_admin_key=bool(raw.get("requires_admin_key", False)),
        requires_embedded=bool(raw.get("requires_embedded", False)),
        requires_openai=bool(raw.get("requires_openai", False)),
        requires_anthropic=bool(raw.get("requires_anthropic", False)),
        interactive=bool(raw.get("interactive", False)),
        timeout_sec=int(raw.get("timeout_sec", 60)),
    )


def parse_python_metadata(path: Path) -> dict[str, Any]:
    module = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    for node in module.body:
        if not isinstance(node, ast.Assign):
            continue
        if len(node.targets) != 1:
            continue
        target = node.targets[0]
        if isinstance(target, ast.Name) and target.id == "EXAMPLE_META":
            value = ast.literal_eval(node.value)
            if not isinstance(value, dict):
                raise ValueError(f"EXAMPLE_META must be a dict literal in {path}")
            return value
    return {}


def parse_shell_metadata(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        for _ in range(10):
            line = handle.readline()
            if not line:
                break
            stripped = line.strip()
            if stripped.startswith("# EXAMPLE_META:"):
                return json.loads(stripped.split(":", 1)[1].strip())
    return {}


def fallback_metadata(path: Path) -> dict[str, Any]:
    kind: Literal["python", "shell", "streamlit"]
    if path.suffix == ".sh":
        kind = "shell"
    elif path.name.startswith("streamlit"):
        kind = "streamlit"
    else:
        kind = "python"
    return {
        "name": path.stem,
        "kind": kind,
        "summary": "",
        "timeout_sec": 60,
    }


def should_ignore(path: Path) -> bool:
    if any(part in IGNORE_DIRS for part in path.parts):
        return True
    if path.name in IGNORE_FILES:
        return True
    return bool(path.name.startswith("_"))


def discover_examples(*, roots: tuple[Path, ...] = DISCOVERY_ROOTS) -> list[ExampleSpec]:
    seen_names: dict[str, Path] = {}
    discovered: list[ExampleSpec] = []
    for root in roots:
        if not root.exists():
            continue
        for path in sorted(root.rglob("*")):
            if not path.is_file() or path.suffix not in SUPPORTED_EXTENSIONS:
                continue
            if should_ignore(path.relative_to(root)):
                continue
            if path.suffix == ".py":
                metadata = parse_python_metadata(path)
            else:
                metadata = parse_shell_metadata(path)
            spec = _coerce_metadata(path, metadata or fallback_metadata(path))
            if spec.name in seen_names:
                raise ValueError(
                    f"Duplicate example name '{spec.name}' in {path} and {seen_names[spec.name]}"
                )
            seen_names[spec.name] = path
            discovered.append(spec)
    return discovered


def _has_embedded_support() -> bool:
    try:
        from cml import EmbeddedCognitiveMemoryLayer  # noqa: F401

        return True
    except Exception:
        return False


def _has_openai_env() -> bool:
    if os.environ.get("LLM_INTERNAL__MODEL") and os.environ.get("LLM_INTERNAL__BASE_URL"):
        return True
    return bool(os.environ.get("OPENAI_API_KEY") and os.environ.get("OPENAI_MODEL"))


def _has_anthropic_env() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY") and os.environ.get("ANTHROPIC_MODEL"))


def find_shell_executable() -> str | None:
    return shutil.which("bash") or shutil.which("sh")


def find_streamlit_executable() -> str | None:
    return shutil.which("streamlit")


def skip_reason(spec: ExampleSpec, *, include_llm: bool, no_skip: bool) -> str | None:
    if no_skip:
        return None
    if spec.requires_api_key and not os.environ.get("CML_API_KEY"):
        return "missing CML_API_KEY"
    if spec.requires_base_url and not os.environ.get("CML_BASE_URL"):
        return "missing CML_BASE_URL"
    if spec.requires_admin_key and not os.environ.get("CML_ADMIN_API_KEY"):
        return "missing CML_ADMIN_API_KEY"
    if spec.requires_embedded and not _has_embedded_support():
        return "embedded extras are not installed"
    if spec.requires_openai:
        if not include_llm:
            return "LLM example (use --include-llm)"
        if not _has_openai_env():
            return (
                "missing OPENAI_API_KEY+OPENAI_MODEL or LLM_INTERNAL__MODEL+LLM_INTERNAL__BASE_URL"
            )
    if spec.requires_anthropic:
        if not include_llm:
            return "LLM example (use --include-llm)"
        if not _has_anthropic_env():
            return "missing ANTHROPIC_API_KEY+ANTHROPIC_MODEL"
    if spec.kind == "shell":
        if not find_shell_executable():
            return "missing bash/sh"
        if not shutil.which("curl"):
            return "missing curl"
    if spec.kind == "streamlit" and not find_streamlit_executable():
        return "missing streamlit executable"
    return None


def select_examples(
    examples: list[ExampleSpec],
    *,
    all_examples: bool,
    example_name: str | None,
    kind: str | None,
    include_llm: bool,
) -> tuple[list[ExampleSpec], bool]:
    if example_name:
        selected = [example for example in examples if example.name == example_name]
        if not selected:
            raise ValueError(f"Unknown example: {example_name}")
        return _filter_by_kind(selected, kind), True

    if not all_examples:
        return [], include_llm

    selected = [
        example
        for example in examples
        if include_llm or (not example.requires_openai and not example.requires_anthropic)
    ]
    return _filter_by_kind(selected, kind), include_llm


def _filter_by_kind(examples: list[ExampleSpec], kind: str | None) -> list[ExampleSpec]:
    if not kind:
        return examples
    return [example for example in examples if example.kind == kind]


def build_subprocess_env(*, non_interactive: bool) -> dict[str, str]:
    env = os.environ.copy()
    env[PYTHON_ENCODING_ENV] = "utf-8"
    env.setdefault("DEBUG", "false")
    if non_interactive:
        env.setdefault("CML_EXAMPLE_NON_INTERACTIVE", "1")
    return env


def build_command(spec: ExampleSpec, *, port: int | None = None) -> list[str]:
    if spec.kind == "python":
        return [sys.executable, "-u", str(spec.path)]
    if spec.kind == "shell":
        shell = find_shell_executable()
        if not shell:
            raise RuntimeError("No shell executable available")
        return [shell, spec.relative_path.replace("\\", "/")]
    if spec.kind == "streamlit":
        streamlit = find_streamlit_executable()
        if not streamlit:
            raise RuntimeError("No streamlit executable available")
        if port is None:
            raise ValueError("Streamlit examples require a port")
        return [
            streamlit,
            "run",
            str(spec.path),
            "--server.headless",
            "true",
            "--server.address",
            "127.0.0.1",
            "--server.port",
            str(port),
            "--browser.gatherUsageStats",
            "false",
        ]
    raise ValueError(f"Unsupported example kind: {spec.kind}")


def pick_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


def _tail(output: str, *, line_count: int = 8) -> str:
    lines = [line for line in output.strip().splitlines() if line.strip()]
    if not lines:
        return "no output"
    return "\n".join(lines[-line_count:])


def _port_is_open(port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(0.2)
        return sock.connect_ex(("127.0.0.1", port)) == 0


def run_streamlit(spec: ExampleSpec, *, env: dict[str, str]) -> ExampleResult:
    port = pick_free_port()
    command = build_command(spec, port=port)
    started = time.perf_counter()
    process = subprocess.Popen(
        command,
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    try:
        deadline = time.perf_counter() + min(spec.timeout_sec, DEFAULT_STREAMLIT_STARTUP_TIMEOUT)
        while time.perf_counter() < deadline:
            if _port_is_open(port):
                elapsed = time.perf_counter() - started
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                return ExampleResult("ok", f"started on port {port}", elapsed)

            if process.poll() is not None:
                output = process.stdout.read() if process.stdout else ""
                elapsed = time.perf_counter() - started
                return ExampleResult("fail", _tail(output), elapsed)
            time.sleep(0.5)

        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
        output = process.stdout.read() if process.stdout else ""
        elapsed = time.perf_counter() - started
        return ExampleResult("fail", f"timeout waiting for startup\n{_tail(output)}", elapsed)
    finally:
        if process.poll() is None:
            process.kill()


def run_subprocess(spec: ExampleSpec, *, env: dict[str, str]) -> ExampleResult:
    command = build_command(spec)
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=str(REPO_ROOT),
            env=env,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=spec.timeout_sec,
        )
        elapsed = time.perf_counter() - started
        if completed.returncode == 0:
            return ExampleResult("ok", "", elapsed)
        output = completed.stderr or completed.stdout
        return ExampleResult("fail", _tail(output), elapsed)
    except subprocess.TimeoutExpired:
        elapsed = time.perf_counter() - started
        return ExampleResult("fail", "timeout", elapsed)


def run_one(
    spec: ExampleSpec,
    *,
    include_llm: bool,
    no_skip: bool,
) -> ExampleResult:
    reason = skip_reason(spec, include_llm=include_llm, no_skip=no_skip)
    if reason:
        return ExampleResult("skip", reason, 0.0)

    env = build_subprocess_env(non_interactive=True)
    if spec.kind == "streamlit":
        return run_streamlit(spec, env=env)
    return run_subprocess(spec, env=env)


def print_example_list(examples: list[ExampleSpec]) -> None:
    print("\nDiscovered examples:\n")
    for example in examples:
        summary = f" - {example.summary}" if example.summary else ""
        print(f"  {example.name:<24} [{example.kind:<9}] {example.relative_path}{summary}")
    print()


def print_summary(results: list[tuple[ExampleSpec, ExampleResult]]) -> None:
    ok_count = sum(1 for _, result in results if result.status == "ok")
    fail_count = sum(1 for _, result in results if result.status == "fail")
    skip_count = sum(1 for _, result in results if result.status == "skip")

    print("\n" + "=" * 78)
    print("  Example run summary")
    print("=" * 78)
    for example, result in results:
        label = f"{example.name} [{example.kind}]"
        if result.status == "ok":
            print(f"  {label:<40} ok    ({result.elapsed:.1f}s)")
        elif result.status == "skip":
            print(f"  {label:<40} skip  ({result.detail})")
        else:
            print(f"  {label:<40} fail  ({result.detail})")
    print("=" * 78)
    print(f"  ok: {ok_count}, fail: {fail_count}, skip: {skip_count}")
    print()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Discover and run CML examples from the repo.")
    parser.add_argument("--list", action="store_true", help="List discovered examples and exit.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--all", action="store_true", help="Run all discovered examples.")
    group.add_argument("--example", metavar="NAME", help="Run a single example by name.")
    parser.add_argument(
        "--kind",
        choices=("python", "shell", "streamlit"),
        help="Restrict results to a specific example kind.",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Include OpenAI and Anthropic examples when running all examples.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Disable env/dependency skip checks and try to execute everything.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    load_repo_env()
    args = parse_args(argv)
    try:
        examples = discover_examples()
    except Exception as exc:
        print(f"Failed to discover examples: {exc}", file=sys.stderr)
        return 1

    if args.list:
        print_example_list(_filter_by_kind(examples, args.kind))
        return 0

    if not args.all and not args.example:
        print_example_list(_filter_by_kind(examples, args.kind))
        print("Use --all to run every example or --example NAME to run one.\n")
        return 0

    try:
        selected, include_llm = select_examples(
            examples,
            all_examples=args.all,
            example_name=args.example,
            kind=args.kind,
            include_llm=args.include_llm or bool(args.example),
        )
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        print(
            "Available examples:",
            ", ".join(example.name for example in examples),
            file=sys.stderr,
        )
        return 1

    if not selected:
        print("No examples matched the requested filters.")
        return 0

    print("\nStarting example runs...\n")
    results: list[tuple[ExampleSpec, ExampleResult]] = []
    for example in selected:
        print(f"Running {example.name} [{example.kind}]...")
        result = run_one(example, include_llm=include_llm, no_skip=args.no_skip)
        print(f"  -> {result.status} ({result.elapsed:.1f}s)")
        if result.detail and result.status != "ok":
            print(f"     {result.detail}")
        results.append((example, result))

    print_summary(results)
    return 0 if all(result.status != "fail" for _, result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
