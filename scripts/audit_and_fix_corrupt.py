#!/usr/bin/env python3
"""
Implement plan: Fix corrupt files vs a7a54e5.
Step 1: Audit every file for readability and valid data.
Step 1b: Validate usage of deleted items.
Step 1c: Compare each modified file (local vs a7a54e5).
Outputs: failed list, deleted-referenced list, modified-keep list, modified-replace list.
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
BASELINE = "a7a54e5"

# Binary extensions to skip for content checks (only readability / can read bytes)
BINARY_EXTENSIONS = {
    ".png",
    ".ico",
    ".woff",
    ".woff2",
    ".ttf",
    ".eot",
    ".pyc",
    ".so",
    ".dll",
    ".db",
}


def run_git(*args: str, cwd: Path | None = None) -> str:
    cwd = cwd or REPO_ROOT
    r = subprocess.run(
        ["git", *list(args)],
        cwd=cwd,
        capture_output=True,
        text=True,
    )
    if r.returncode != 0:
        raise RuntimeError(f"git {' '.join(args)}: {r.stderr or r.stdout}")
    return (r.stdout or "").strip()


def get_tracked_files() -> list[str]:
    out = run_git("ls-files")
    files = [f for f in out.splitlines() if f and ".git.corrupt" not in f]
    return files


def read_file(path: Path) -> tuple[bytes | None, str | None, str | None]:
    """Read file as bytes. Return (raw_bytes, decoded_text, error)."""
    try:
        raw = path.read_bytes()
    except OSError as e:
        return None, None, str(e)
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as e:
        return raw, None, f"utf-8 decode: {e}"
    if "\0" in text:
        return raw, text, "contains null byte"
    if "\ufffd" in text:
        return raw, text, "contains replacement character"
    return raw, text, None


def check_py(path: Path, text: str) -> str | None:
    try:
        ast.parse(text)
        return None
    except SyntaxError as e:
        return f"Python syntax: {e}"


def check_toml(path: Path, text: str) -> str | None:
    try:
        try:
            import tomllib

            tomllib.loads(text)
        except ImportError:
            try:
                import tomli

                tomli.loads(text)
            except ImportError:
                import toml

                toml.loads(text)
    except Exception as e:
        return f"TOML: {e}"
    return None


def check_json(path: Path, text: str) -> str | None:
    try:
        json.loads(text)
        return None
    except json.JSONDecodeError as e:
        return f"JSON: {e}"


def check_yaml(path: Path, text: str) -> str | None:
    try:
        import yaml

        yaml.safe_load(text)
        return None
    except ImportError:
        return None  # skip if no PyYAML
    except Exception as e:
        return f"YAML: {e}"


def audit_file(rel_path: str) -> str | None:
    """Return None if file passes, else error string."""
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return "not a file"
    ext = path.suffix.lower()
    if ext in BINARY_EXTENSIONS:
        return None  # skip binary
    _raw, text, read_err = read_file(path)
    if read_err:
        return read_err
    if text is None:
        return "no decoded text"
    # Type-specific validation
    if ext == ".py":
        return check_py(path, text)
    if ext == ".toml":
        return check_toml(path, text)
    if ext == ".lock":
        return check_toml(path, text)
    if ext == ".json":
        return check_json(path, text)
    if ext in (".yaml", ".yml"):
        return check_yaml(path, text)
    return None


def main_audit() -> list[str]:
    """Step 1: Audit every file. Return list of failed paths."""
    failed = []
    for rel in get_tracked_files():
        err = audit_file(rel)
        if err is not None:
            failed.append(rel)
            print(f"FAIL {rel}: {err}", file=sys.stderr)
    return failed


def get_deleted_paths() -> list[str]:
    out = run_git(
        "diff", "--name-only", "--diff-filter=D", BASELINE, "HEAD", "--", ".", ":!.git.corrupt"
    )
    return [f for f in out.splitlines() if f]


def get_modified_paths() -> list[str]:
    out = run_git(
        "diff", "--name-only", "--diff-filter=M", BASELINE, "HEAD", "--", ".", ":!.git.corrupt"
    )
    return [f for f in out.splitlines() if f]


def search_references(needle: str, haystack_files: list[str]) -> bool:
    """Return True if needle appears in any tracked file (as substring)."""
    # Normalize: path to module-like or key tokens
    tokens = set()
    tokens.add(needle)
    # e.g. src/api/dashboard/config_routes.py -> config_routes, dashboard
    base = os.path.basename(needle).replace(".py", "")
    tokens.add(base)
    tokens.add("dashboard")
    for part in needle.replace("\\", "/").split("/"):
        if part and part != "src" and part != "api" and not part.startswith("."):
            tokens.add(part)
    for rel in haystack_files:
        path = REPO_ROOT / rel
        if not path.is_file() or path.suffix.lower() != ".py":
            continue
        try:
            content = path.read_text(encoding="utf-8", errors="replace")
        except Exception:
            continue
        for t in tokens:
            if t in content and len(t) > 2:
                return True
    return False


def main_deleted_referenced() -> list[str]:
    """Step 1b: Deleted paths that are still referenced."""
    deleted = get_deleted_paths()
    tracked = get_tracked_files()
    referenced = []
    for rel in deleted:
        if search_references(rel, tracked):
            referenced.append(rel)
            print(f"REFERENCED (restore): {rel}", file=sys.stderr)
    return referenced


def get_file_at_ref(rel_path: str, ref: str) -> str | None:
    try:
        out = run_git("show", f"{ref}:{rel_path}")
        return out
    except RuntimeError:
        return None


def get_local_content(rel_path: str) -> str | None:
    path = REPO_ROOT / rel_path
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return None


def content_valid(rel_path: str, content: str) -> bool:
    """Run same validity as audit_file on content string."""
    ext = Path(rel_path).suffix.lower()
    if "\0" in content or "\ufffd" in content:
        return False
    if ext == ".py":
        return check_py(Path(rel_path), content) is None
    if ext in (".toml", ".lock"):
        return check_toml(Path(rel_path), content) is None
    if ext == ".json":
        return check_json(Path(rel_path), content) is None
    if ext in (".yaml", ".yml"):
        return check_yaml(Path(rel_path), content) is None
    return True


def main_modified_compare() -> tuple[list[str], list[str]]:
    """Step 1c: For each modified file, keep local or replace with a7a54e5.
    Returns (keep_local_list, replace_with_a7a54e5_list).
    """
    modified = get_modified_paths()
    keep_local = []
    replace = []
    for rel in modified:
        local = get_local_content(rel)
        base = get_file_at_ref(rel, BASELINE)
        if base is None:
            keep_local.append(rel)
            continue
        local_ok = local is not None and content_valid(rel, local)
        base_ok = content_valid(rel, base)
        if local_ok and not base_ok:
            keep_local.append(rel)
            continue
        if not local_ok and base_ok:
            replace.append(rel)
            continue
        if not local_ok and not base_ok:
            replace.append(rel)
            continue
        # Both valid: prefer local if it's "more complete" (e.g. more lines / not truncated)
        local_lines = len(local.splitlines()) if local else 0
        base_lines = len(base.splitlines())
        if local_lines >= base_lines and local and local.strip().endswith(("\n", "}", ">")):
            keep_local.append(rel)
        else:
            replace.append(rel)
    return keep_local, replace


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("step", choices=["audit", "deleted", "modified", "all"])
    args = ap.parse_args()

    if args.step in ("audit", "all"):
        failed = main_audit()
        out_path = REPO_ROOT / "corrupt_candidates.txt"
        out_path.write_text("\n".join(failed) + ("\n" if failed else ""), encoding="utf-8")
        print(f"Audit: {len(failed)} failed. Wrote {out_path}")

    if args.step in ("deleted", "all"):
        refs = main_deleted_referenced()
        out_path = REPO_ROOT / "deleted_still_referenced.txt"
        out_path.write_text("\n".join(refs) + ("\n" if refs else ""), encoding="utf-8")
        print(f"Deleted still referenced: {len(refs)}. Wrote {out_path}")

    if args.step in ("modified", "all"):
        keep, replace = main_modified_compare()
        (REPO_ROOT / "modified_keep_local.txt").write_text(
            "\n".join(keep) + ("\n" if keep else ""), encoding="utf-8"
        )
        (REPO_ROOT / "modified_replace_with_a7a54e5.txt").write_text(
            "\n".join(replace) + ("\n" if replace else ""), encoding="utf-8"
        )
        print(f"Modified: keep_local={len(keep)}, replace={len(replace)}")
