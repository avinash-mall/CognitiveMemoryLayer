"""Legacy compatibility wrapper for cml.modeling.prepare."""

from __future__ import annotations

import sys
from importlib import import_module
from pathlib import Path
from typing import Any


def _load_module():
    try:
        return import_module("cml.modeling.prepare")
    except ImportError:
        repo_root = Path(__file__).resolve().parents[3]
        sdk_src = repo_root / "packages" / "py-cml" / "src"
        if sdk_src.exists():
            sys.path.insert(0, str(sdk_src))
        return import_module("cml.modeling.prepare")


_module = _load_module()
for _key, _value in vars(_module).items():
    if _key in {"__name__", "__package__", "__loader__", "__spec__", "__file__", "__cached__"}:
        continue
    globals()[_key] = _value


def main(*args, **kwargs):
    """Delegate to cml.modeling.prepare.main."""
    return _module.main(*args, **kwargs)


def __getattr__(name: str) -> Any:
    """Fallback for dynamically re-exported attributes."""
    return getattr(_module, name)


if __name__ == "__main__":
    raise SystemExit(main())
