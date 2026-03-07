"""Refresh README badge counts for version and tests.

Updates the existing `Tests` and `Version` badge URLs in the repo README files.
Unlike the older placeholder-based variant, this script edits the current badge
markup in place and can optionally run in check-only mode.

Usage:
    python scripts/update_readme_badges.py
    python scripts/update_readme_badges.py --check
"""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import REPO_ROOT, load_repo_env, normalize_bool_env

TEST_BADGE_PATTERN = re.compile(
    r"(\[!\[Tests\]\(https://img\.shields\.io/badge/Tests-)([^-\)]*)(-[^)]+\)\]\([^)]+\))"
)
VERSION_BADGE_PATTERN = re.compile(
    r"(\[!\[Version\]\(https://img\.shields\.io/badge/version-)([^-\)]*)(-[^)]+\)\]\([^)]+\))"
)


@dataclass(frozen=True)
class BadgeTarget:
    path: Path
    test_paths: tuple[str, ...]


BADGE_TARGETS = (
    BadgeTarget(REPO_ROOT / "README.md", ("tests", "packages/py-cml/tests")),
    BadgeTarget(REPO_ROOT / "packages" / "py-cml" / "README.md", ("packages/py-cml/tests",)),
)


def _read_version_from_env_file(root: Path) -> str | None:
    env_file = root / ".env"
    if not env_file.is_file():
        return None
    for line in env_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        if key.strip() == "VERSION":
            return value.strip().strip('"').strip("'") or None
    return None


def _read_version_from_version_file(root: Path) -> str | None:
    version_file = root / "VERSION"
    if not version_file.is_file():
        return None
    return version_file.read_text(encoding="utf-8").strip() or None


def get_version(root: Path | None = None) -> str:
    """Match the repo's version-resolution order without depending on hatchling."""
    root = root or REPO_ROOT
    return (
        _read_version_from_env_file(root)
        or os.environ.get("VERSION")
        or _read_version_from_version_file(root)
        or "0.0.0"
    )


def _pytest_env() -> dict[str, str]:
    env = os.environ.copy()
    env["DEBUG"] = "false"
    return env


def _collect_count(path: str, *, root: Path) -> int:
    result = subprocess.run(
        [sys.executable, "-m", "pytest", path, "--collect-only", "-q"],
        cwd=root,
        capture_output=True,
        text=True,
        timeout=180,
        env=_pytest_env(),
    )
    out = (result.stdout or "") + (result.stderr or "")
    match = re.search(r"(\d+)\s+(?:tests?\s+)?collected", out, re.IGNORECASE)
    if match:
        return int(match.group(1))
    if result.returncode != 0:
        raise RuntimeError(f"pytest collection failed for {path}:\n{out.strip()}")
    return 0


def get_test_count(paths: tuple[str, ...], *, root: Path | None = None) -> int:
    """Run `pytest --collect-only -q` for the provided test roots and sum counts."""
    effective_root = root or REPO_ROOT
    return sum(_collect_count(path, root=effective_root) for path in paths)


def update_badge_text(text: str, *, version: str, tests: int) -> tuple[str, bool]:
    """Replace version/tests badge values in a README body."""
    updated = text
    changed = False

    tests_value = str(tests)
    tests_replaced, tests_count = TEST_BADGE_PATTERN.subn(
        lambda match: f"{match.group(1)}{tests_value}{match.group(3)}",
        updated,
        count=1,
    )
    if tests_count:
        updated = tests_replaced
        changed = changed or (updated != text)

    version_value = quote(version, safe="")
    version_replaced, version_count = VERSION_BADGE_PATTERN.subn(
        lambda match: f"{match.group(1)}{version_value}{match.group(3)}",
        updated,
        count=1,
    )
    if version_count:
        changed = changed or (version_replaced != updated)
        updated = version_replaced

    return updated, changed


def process_target(target: BadgeTarget, *, version: str, check: bool) -> bool:
    if not target.path.is_file():
        print(f"Skip (not found): {target.path}", file=sys.stderr)
        return False

    original = target.path.read_text(encoding="utf-8")
    tests = get_test_count(target.test_paths)
    updated, changed = update_badge_text(original, version=version, tests=tests)
    rel = target.path.relative_to(REPO_ROOT)

    if not changed:
        print(f"Up to date: {rel} (version={version}, tests={tests})")
        return False

    if check:
        print(f"Out of date: {rel} (version={version}, tests={tests})", file=sys.stderr)
        return True

    target.path.write_text(updated, encoding="utf-8")
    print(f"Updated {rel} (version={version}, tests={tests})")
    return True


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Refresh README badge counts in place.")
    parser.add_argument(
        "--check",
        action="store_true",
        help="Exit non-zero when the README badges are stale instead of rewriting files.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    load_repo_env()
    normalize_bool_env("DEBUG")
    args = build_parser().parse_args(argv)
    version = get_version()

    changed = False
    for target in BADGE_TARGETS:
        changed = process_target(target, version=version, check=args.check) or changed

    if args.check and changed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
