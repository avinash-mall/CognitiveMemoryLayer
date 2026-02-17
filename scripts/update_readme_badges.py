"""Update version and test-count placeholders in README files.

Replaces {{VERSION}} with version from VERSION file or .env, and {{TESTS}} with
the current pytest collection count. Run from repo root.

Usage:
    python scripts/update_readme_badges.py
"""

import os
import re
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent


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
    """Same logic as hatch_build.get_version (no hatchling dependency)."""
    root = root or REPO_ROOT
    return (
        _read_version_from_env_file(root)
        or os.environ.get("VERSION")
        or _read_version_from_version_file(root)
        or "0.0.0"
    )


def get_test_count(root: Path | None = None) -> str:
    """Run pytest --collect-only -q on server and SDK tests; return combined count."""
    root = root or REPO_ROOT
    total = 0
    for path in ["tests", "packages/py-cml/tests"]:
        result = subprocess.run(
            [sys.executable, "-m", "pytest", path, "--collect-only", "-q"],
            cwd=root,
            capture_output=True,
            text=True,
            timeout=120,
        )
        out = (result.stdout or "") + (result.stderr or "")
        match = re.search(r"(\d+)\s+(?:tests?\s+)?collected", out, re.IGNORECASE)
        if match:
            total += int(match.group(1))
    return str(total) if total else "0"


def main() -> None:
    version = get_version()
    tests = get_test_count()

    readmes = [
        REPO_ROOT / "README.md",
        REPO_ROOT / "packages" / "py-cml" / "README.md",
    ]
    for path in readmes:
        if not path.is_file():
            print(f"Skip (not found): {path}", file=sys.stderr)
            continue
        text = path.read_text(encoding="utf-8")
        updated = False
        if "{{VERSION}}" in text:
            text = text.replace("{{VERSION}}", version)
            updated = True
        if "{{TESTS}}" in text:
            text = text.replace("{{TESTS}}", tests)
            updated = True
        if updated:
            path.write_text(text, encoding="utf-8")
            rel = path.relative_to(REPO_ROOT)
            print(f"Updated {rel} (version={version}, tests={tests})")


if __name__ == "__main__":
    main()
