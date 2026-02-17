"""
Run Cognitive Memory Layer examples one-by-one and report Pass / Fail / Skip.

Run from repo root. Prerequisites:
  - pip install -r examples/requirements.txt (and for embedded: pip install -e "packages/py-cml[embedded]")
  - CML API up for API-dependent examples: docker compose -f docker/docker-compose.yml up api
  - .env at repo root with AUTH__API_KEY, optional CML_BASE_URL; for LLM examples add OPENAI_* or ANTHROPIC_*

Usage:
  python scripts/run_examples.py --all
  python scripts/run_examples.py --example embedded_mode
  python scripts/run_examples.py --all --include-llm
  python scripts/run_examples.py --all --no-skip
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

# Load .env from repo root so skip logic and subprocess env see AUTH__API_KEY, etc.
_env_file = REPO_ROOT / ".env"
if _env_file.is_file():
    try:
        from dotenv import load_dotenv

        load_dotenv(_env_file)
    except ImportError:
        pass

EXAMPLES = [
    {
        "name": "embedded_mode",
        "path": "examples/embedded_mode.py",
        "needs_api": False,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": False,
        "timeout_sec": 90,
    },
    {
        "name": "quickstart",
        "path": "examples/quickstart.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": False,
        "timeout_sec": 60,
    },
    {
        "name": "basic_usage",
        "path": "examples/basic_usage.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": False,
        "timeout_sec": 60,
    },
    {
        "name": "async_example",
        "path": "examples/async_example.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": False,
        "timeout_sec": 60,
    },
    {
        "name": "agent_integration",
        "path": "examples/agent_integration.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": False,
        "timeout_sec": 60,
    },
    {
        "name": "temporal_fidelity",
        "path": "packages/py-cml/examples/temporal_fidelity.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": False,
        "timeout_sec": 60,
    },
    {
        "name": "standalone_demo",
        "path": "examples/standalone_demo.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": False,
        "interactive": True,
        "timeout_sec": 150,
        "non_interactive_env": "CML_STANDALONE_NON_INTERACTIVE",
    },
    {
        "name": "chat_with_memory",
        "path": "examples/chat_with_memory.py",
        "needs_api": True,
        "needs_llm_openai": True,
        "needs_llm_anthropic": False,
        "interactive": True,
        "timeout_sec": 90,
        "stdin_input": b"quit\n",
    },
    {
        "name": "openai_tool_calling",
        "path": "examples/openai_tool_calling.py",
        "needs_api": True,
        "needs_llm_openai": True,
        "needs_llm_anthropic": False,
        "interactive": True,
        "timeout_sec": 90,
        "stdin_input": b"quit\n",
    },
    {
        "name": "anthropic_tool_calling",
        "path": "examples/anthropic_tool_calling.py",
        "needs_api": True,
        "needs_llm_openai": False,
        "needs_llm_anthropic": True,
        "interactive": True,
        "timeout_sec": 90,
        "stdin_input": b"quit\n",
    },
    {
        "name": "langchain_integration",
        "path": "examples/langchain_integration.py",
        "needs_api": True,
        "needs_llm_openai": True,
        "needs_llm_anthropic": False,
        "interactive": True,
        "timeout_sec": 90,
        "stdin_input": b"quit\n",
    },
]


def _has_api_key() -> bool:
    return bool(os.environ.get("AUTH__API_KEY") or os.environ.get("CML_API_KEY"))


def _has_openai() -> bool:
    return bool(
        os.environ.get("OPENAI_API_KEY")
        and (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM__MODEL"))
    )


def _has_anthropic() -> bool:
    return bool(os.environ.get("ANTHROPIC_API_KEY"))


def _should_skip(ex: dict, include_llm: bool, no_skip: bool) -> str | None:
    if no_skip:
        return None
    if ex["needs_api"] and not _has_api_key():
        return "missing AUTH__API_KEY or CML_API_KEY"
    if ex.get("needs_llm_openai") and not _has_openai():
        if not include_llm:
            return "LLM example (use --include-llm)"
        return "missing OPENAI_API_KEY or OPENAI_MODEL/LLM__MODEL"
    if ex.get("needs_llm_anthropic") and not _has_anthropic():
        if not include_llm:
            return "LLM example (use --include-llm)"
        return "missing ANTHROPIC_API_KEY"
    if ex["name"] == "embedded_mode":
        try:
            __import__("cml")
            from cml import EmbeddedCognitiveMemoryLayer  # noqa: F401
        except ImportError:
            return "cognitive-memory-layer[embedded] not installed"
    return None


def _run_one(
    ex: dict,
    include_llm: bool,
    no_skip: bool,
) -> tuple[str, str, float, str | None]:
    path = REPO_ROOT / ex["path"]
    if not path.is_file():
        return "fail", f"file not found: {path}", 0.0, None

    skip_reason = _should_skip(ex, include_llm, no_skip)
    if skip_reason:
        return "skip", skip_reason, 0.0, None

    env = os.environ.copy()
    if ex.get("non_interactive_env"):
        env[ex["non_interactive_env"]] = "1"
    # Use UTF-8 for subprocess I/O so Unicode (e.g. ✓/✗) does not break on Windows cp1252
    env["PYTHONIOENCODING"] = "utf-8"

    stdin_input = ex.get("stdin_input")
    stdin_pipe = subprocess.PIPE if stdin_input else None

    cmd = [sys.executable, "-u", str(path)]
    start = time.time()
    try:
        proc = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            stdin=stdin_pipe,
            input=stdin_input,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=ex["timeout_sec"],
        )
        elapsed = time.time() - start
        if proc.returncode == 0:
            return "ok", "", elapsed, None
        err = (proc.stderr or proc.stdout or "").strip()
        last_lines = "\n".join(err.splitlines()[-5:]) if err else "non-zero exit"
        return "fail", last_lines, elapsed, None
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        return "fail", "timeout", elapsed, None
    except Exception as e:
        elapsed = time.time() - start
        return "fail", str(e), elapsed, None


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run CML examples one-by-one and report Pass / Fail / Skip."
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--all",
        action="store_true",
        help="Run all examples (non-LLM by default; add --include-llm for LLM examples).",
    )
    group.add_argument(
        "--example",
        metavar="NAME",
        help="Run only the example with this name (e.g. quickstart, embedded_mode).",
    )
    parser.add_argument(
        "--include-llm",
        action="store_true",
        help="Also run OpenAI/Anthropic/LangChain examples (requires keys and CML API).",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Run even when env vars are missing (will likely fail).",
    )
    args = parser.parse_args()

    if args.example:
        candidates = [e for e in EXAMPLES if e["name"] == args.example]
        if not candidates:
            print(f"Unknown example: {args.example}", file=sys.stderr)
            print(
                "Available:",
                ", ".join(str(e["name"]) for e in EXAMPLES),
                file=sys.stderr,
            )
            return 1
        to_run = candidates
        include_llm = True
    else:
        if not args.all:
            parser.print_help()
            return 0
        include_llm = args.include_llm
        to_run = [
            e
            for e in EXAMPLES
            if not e.get("needs_llm_openai") and not e.get("needs_llm_anthropic")
        ] + (
            [e for e in EXAMPLES if e.get("needs_llm_openai") or e.get("needs_llm_anthropic")]
            if include_llm
            else []
        )

    results = []
    for ex in to_run:
        status, detail, elapsed, _ = _run_one(ex, include_llm, args.no_skip)
        results.append((ex["name"], status, detail, elapsed))

    ok_count = sum(1 for _, s, _, _ in results if s == "ok")
    fail_count = sum(1 for _, s, _, _ in results if s == "fail")
    skip_count = sum(1 for _, s, _, _ in results if s == "skip")

    print("\n" + "=" * 70)
    print("  Example run summary")
    print("=" * 70)
    for name, status, detail, elapsed in results:
        if status == "ok":
            print(f"  {name:<25} ok    ({elapsed:.1f}s)")
        elif status == "skip":
            print(f"  {name:<25} skip  ({detail})")
        else:
            print(f"  {name:<25} fail  ({detail})")
    print("=" * 70)
    print(f"  ok: {ok_count}, fail: {fail_count}, skip: {skip_count}")
    print()
    return 0 if fail_count == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
