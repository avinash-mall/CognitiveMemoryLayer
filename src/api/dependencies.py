"""Shared FastAPI dependencies for API routes.

Re-exports commonly used auth dependencies so route modules can import
from a single place::

    from .dependencies import get_auth_context, require_write_permission, AuthContext
"""

from .auth import (
    AuthContext,
    get_auth_context,
    require_admin_permission,
    require_write_permission,
)

__all__ = [
    "AuthContext",
    "get_auth_context",
    "require_write_permission",
    "require_admin_permission",
]
