"""API layer: FastAPI app, routes, auth, and dependencies."""

from .app import create_app
from .dependencies import AuthContext, get_auth_context, require_admin_permission, require_write_permission

__all__ = [
    "create_app",
    "AuthContext",
    "get_auth_context",
    "require_write_permission",
    "require_admin_permission",
]
