"""Per-scope sensory buffer management.

Uses :class:`BoundedStateMap` to prevent unbounded buffer growth in
long-running servers while keeping per-scope isolation.
"""

from ...utils.bounded_state import BoundedStateMap
from .buffer import SensoryBuffer, SensoryBufferConfig


class SensoryBufferManager:
    """
    Manages per-scope sensory buffers.
    Each scope gets its own isolated buffer.

    Bounded by LRU eviction (``max_scopes``) and TTL
    (``scope_ttl_seconds``) to prevent memory leaks.
    """

    def __init__(
        self,
        config: SensoryBufferConfig | None = None,
        max_scopes: int = 1000,
        scope_ttl_seconds: float = 600.0,
    ):
        self.config = config or SensoryBufferConfig()
        self._buffers: BoundedStateMap[SensoryBuffer] = BoundedStateMap(
            max_size=max_scopes,
            ttl_seconds=scope_ttl_seconds,
        )

    def _get_key(self, tenant_id: str, scope_id: str) -> str:
        return f"{tenant_id}:{scope_id}"

    async def get_buffer(self, tenant_id: str, scope_id: str) -> SensoryBuffer:
        """Get or create buffer for scope."""
        key = self._get_key(tenant_id, scope_id)
        return await self._buffers.get_or_create(
            key,
            factory=lambda: SensoryBuffer(self.config),
        )

    async def ingest(
        self,
        tenant_id: str,
        scope_id: str,
        text: str,
        turn_id: str | None = None,
        role: str | None = None,
    ) -> int:
        """Ingest text into scope's buffer."""
        buffer = await self.get_buffer(tenant_id, scope_id)
        return await buffer.ingest(text, turn_id, role)

    async def get_recent_text(
        self,
        tenant_id: str,
        scope_id: str,
        max_tokens: int | None = None,
    ) -> str:
        """Get recent text from scope's buffer."""
        buffer = await self.get_buffer(tenant_id, scope_id)
        return await buffer.get_text(max_tokens=max_tokens)

    async def clear_user(self, tenant_id: str, scope_id: str) -> None:
        """Clear a specific scope's buffer."""
        buffer = await self._buffers.get(self._get_key(tenant_id, scope_id))
        if buffer:
            await buffer.clear()

    async def cleanup_inactive(self, inactive_seconds: float = 300) -> None:
        """Remove expired buffers via the bounded state map's TTL."""
        await self._buffers.cleanup_expired()
