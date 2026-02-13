"""API layer: FastAPI app, routes, auth, and dependencies."""

from .app import create_app
from .dependencies import (
    AuthContext,
    get_auth_context,
    require_admin_permission,
    require_write_permission,
)

__all__ = [
    "AuthContext",
    "create_app",
    "get_auth_context",
    "require_admin_permission",
    "require_write_permission",
]
