"""Legacy compatibility wrapper for cml.modeling.prepare.

Requires the cognitive-memory-layer package to be installed.
Run: pip install -e ".[modeling]" from the repository root.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any


def _load_module():
    try:
        return import_module("cml.modeling.prepare")
    except ImportError as exc:
        raise ImportError(
            "cml.modeling.prepare is not importable. "
            'Install the package first: pip install -e ".[modeling]"'
        ) from exc


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
