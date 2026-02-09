"""Deprecation utilities for API evolution."""

from __future__ import annotations

import asyncio
import warnings
from functools import wraps
from typing import Any, Callable, TypeVar, cast

F = TypeVar("F", bound=Callable[..., Any])


def deprecated(alternative: str, removal_version: str) -> Callable[[F], F]:
    """Mark a function or method as deprecated.

    Args:
        alternative: What to use instead (e.g. "write()").
        removal_version: Version when the deprecated API will be removed (e.g. "2.0.0").

    Returns:
        Decorator that emits DeprecationWarning and calls the original.
    """

    def decorator(func: F) -> F:
        @wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in "
                f"v{removal_version}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return func(*args, **kwargs)

        @wraps(func)
        async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
            warnings.warn(
                f"{func.__name__} is deprecated and will be removed in "
                f"v{removal_version}. Use {alternative} instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            return await func(*args, **kwargs)

        if asyncio.iscoroutinefunction(func):
            return cast("F", async_wrapper)
        return cast("F", wrapper)

    return decorator
