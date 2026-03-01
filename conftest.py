"""Global pytest hooks shared across test trees."""

from __future__ import annotations

import gc
import sys
import warnings

_SWIG_MODULE_PREFIXES = (
    "sentencepiece",
    "tokenizers.implementations.sentencepiece",
)


def pytest_sessionfinish(session, exitstatus):
    """Unload sentencepiece SWIG modules before interpreter teardown.

    This avoids a known third-party shutdown deprecation warning:
    "builtin type swigvarlink has no __module__ attribute".
    """
    removed_any = False
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        for module_name in list(sys.modules):
            lower_name = module_name.lower()
            if lower_name == "swig_runtime_data4" or any(
                lower_name.startswith(prefix) for prefix in _SWIG_MODULE_PREFIXES
            ):
                sys.modules.pop(module_name, None)
                removed_any = True
    if removed_any:
        gc.collect()
