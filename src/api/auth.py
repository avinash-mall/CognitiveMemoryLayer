"""API authentication and authorization."""
from dataclasses import dataclass
from typing import Optional

from fastapi import Depends, Header, HTTPException, Security
from fastapi.security import APIKeyHeader

from ..core.config import get_settings

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


@dataclass
class AuthContext:
    """Authentication context for a request."""

    tenant_id: str
    user_id: Optional[str] = None
    api_key: str = ""
    can_read: bool = True
    can_write: bool = True
    can_admin: bool = False


class AuthService:
    """
    Handles authentication and authorization.
    In production, integrate with proper auth system.
    """

    def __init__(self):
        self._api_keys: dict = {}
        self._load_keys()

    def _load_keys(self):
        """Load API keys from config/database."""
        self._api_keys = {
            "demo-key-123": AuthContext(
                tenant_id="demo",
                can_read=True,
                can_write=True,
                can_admin=False,
                api_key="demo-key-123",
            ),
            "admin-key-456": AuthContext(
                tenant_id="admin",
                can_read=True,
                can_write=True,
                can_admin=True,
                api_key="admin-key-456",
            ),
        }

    def validate_key(self, api_key: str) -> Optional[AuthContext]:
        """Validate API key and return context."""
        return self._api_keys.get(api_key)

    def check_permission(self, context: AuthContext, permission: str) -> bool:
        """Check if context has permission."""
        if permission == "read":
            return context.can_read
        if permission == "write":
            return context.can_write
        if permission == "admin":
            return context.can_admin
        return False


_auth_service = AuthService()


async def get_auth_context(
    api_key: Optional[str] = Security(api_key_header),
    x_tenant_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
) -> AuthContext:
    """Dependency to get auth context from request."""
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    context = _auth_service.validate_key(api_key)
    if not context:
        raise HTTPException(status_code=401, detail="Invalid API key")

    if x_user_id:
        context = AuthContext(
            tenant_id=context.tenant_id,
            user_id=x_user_id,
            api_key=context.api_key,
            can_read=context.can_read,
            can_write=context.can_write,
            can_admin=context.can_admin,
        )
    return context


async def require_write_permission(
    context: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    """Require write permission."""
    if not context.can_write:
        raise HTTPException(status_code=403, detail="Write permission required")
    return context


async def require_admin_permission(
    context: AuthContext = Depends(get_auth_context),
) -> AuthContext:
    """Require admin permission."""
    if not context.can_admin:
        raise HTTPException(status_code=403, detail="Admin permission required")
    return context
