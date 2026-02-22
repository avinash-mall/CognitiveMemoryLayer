"""Helpers for reading and writing .env configuration."""

import re
import tempfile
from pathlib import Path
from typing import Any


def _project_root() -> Path:
    """Project root (directory containing alembic.ini)."""
    return Path(__file__).resolve().parent.parent.parent


def get_env_path() -> Path:
    """Path to .env file in project root."""
    return _project_root() / ".env"


def _format_env_value(value) -> str:
    """Format a value for .env (booleans, numbers, strings, lists)."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, list):
        return ",".join(str(v) for v in value)
    s = str(value)
    # Quote if contains spaces, #, =, or newlines
    if any(c in s for c in (" ", "#", "=", "\n", "\r", '"')):
        s = s.replace("\\", "\\\\").replace('"', '\\"')
        return f'"{s}"'
    return s


def _parse_env_line(line: str) -> tuple[str | None, str | None]:
    """Parse a single .env line. Returns (key, value) or (None, None) if not a KEY= line."""
    line = line.rstrip("\r\n")
    if not line or line.startswith("#"):
        return None, None
    m = re.match(r"^([A-Za-z_][A-Za-z0-9_]*(?:__[A-Za-z0-9_]+)*)\s*=\s*(.*)$", line)
    if not m:
        return None, None
    key, raw_val = m.group(1), m.group(2)
    val = raw_val.strip()
    if val.startswith('"') and val.endswith('"'):
        val = val[1:-1].replace('\\"', '"').replace("\\\\", "\\")
    elif val.upper() == "TRUE":
        val = "true"
    elif val.upper() == "FALSE":
        val = "false"
    return key, val


def update_env(updates: dict[str, Any]) -> None:
    """Update .env with key-value pairs. Keys are env var names (e.g. APP_NAME, EMBEDDING__DIMENSIONS).

    Updates existing lines; appends new ones. Writes atomically via temp file + replace.
    Raises OSError if file is read-only or cannot be written.
    """
    env_path = get_env_path()
    # Normalize: uppercase keys for lookup, keep original key for output
    updates_norm: dict[str, tuple[str, Any]] = {k.upper(): (k, v) for k, v in updates.items()}
    keys_updated: set[str] = set(updates_norm.keys())
    lines: list[str] = []

    if env_path.exists():
        content = env_path.read_text(encoding="utf-8")
        for line in content.splitlines(keepends=True):
            key, _ = _parse_env_line(line.rstrip("\r\n"))
            if key:
                key_upper = key.upper()
                if key_upper in keys_updated:
                    orig_key, new_val = updates_norm[key_upper]
                    lines.append(f"{orig_key}={_format_env_value(new_val)}\n")
                    keys_updated.discard(key_upper)
                    continue
            lines.append(line if line.endswith("\n") else line + "\n")
    else:
        lines.append("")

    # Append any new keys not found in file
    for key_upper in keys_updated:
        orig_key, val = updates_norm[key_upper]
        lines.append(f"{orig_key}={_format_env_value(val)}\n")

    # Write atomically
    fd, tmp_path = tempfile.mkstemp(prefix=".env.", suffix=".tmp", dir=env_path.parent, text=True)
    try:
        with open(fd, "w", encoding="utf-8") as f:
            f.writelines(lines)
        Path(tmp_path).replace(env_path)
    except Exception:
        Path(tmp_path).unlink(missing_ok=True)
        raise
