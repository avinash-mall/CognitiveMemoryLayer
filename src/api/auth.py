"""API authentication and authorization (config-based, no hardcoded keys)."""

import hmac
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


def _build_api_keys() -> dict:
    """Build API key map from settings (env: AUTH__API_KEY, AUTH__ADMIN_API_KEY, AUTH__DEFAULT_TENANT_ID)."""
    settings = get_settings()
    auth = settings.auth
    keys: dict = {}
    if auth.api_key:
        keys[auth.api_key] = AuthContext(
            tenant_id=auth.default_tenant_id,
            can_read=True,
            can_write=True,
            can_admin=(auth.api_key == auth.admin_api_key and bool(auth.admin_api_key)),
            api_key=auth.api_key,
        )
    if auth.admin_api_key and auth.admin_api_key != auth.api_key:
        keys[auth.admin_api_key] = AuthContext(
            tenant_id=auth.default_tenant_id,
            can_read=True,
            can_write=True,
            can_admin=True,
            api_key=auth.admin_api_key,
        )
    return keys


async def get_auth_context(
    api_key: Optional[str] = Security(api_key_header),
    x_tenant_id: Optional[str] = Header(None),
    x_user_id: Optional[str] = Header(None),
) -> AuthContext:
    """Dependency to get auth context from request.

    The X-Tenant-Id header can override the default tenant derived from the
    API key. The X-User-Id header can set the user identity.
    """
    if not api_key:
        raise HTTPException(status_code=401, detail="API key required")

    api_keys = _build_api_keys()
    context = None
    for known_key, ctx in api_keys.items():
        if hmac.compare_digest(api_key, known_key):
            context = ctx
            break
    if not context:
        raise HTTPException(status_code=401, detail="Invalid API key")

    # Apply header overrides: X-Tenant-Id only for admin keys (SEC-03)
    tenant_id = (x_tenant_id if (x_tenant_id and context.can_admin) else None) or context.tenant_id
    user_id = x_user_id or context.user_id
    if tenant_id != context.tenant_id or user_id != context.user_id:
        context = AuthContext(
            tenant_id=tenant_id,
            user_id=user_id,
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
