"""Per-scope sensory buffer management."""

from typing import Dict, Optional

import asyncio

from .buffer import SensoryBuffer, SensoryBufferConfig


class SensoryBufferManager:
    """
    Manages per-scope sensory buffers.
    Each scope gets its own isolated buffer.
    """

    def __init__(self, config: Optional[SensoryBufferConfig] = None):
        self.config = config or SensoryBufferConfig()
        self._buffers: Dict[str, SensoryBuffer] = {}
        self._lock = asyncio.Lock()

    def _get_key(self, tenant_id: str, scope_id: str) -> str:
        return f"{tenant_id}:{scope_id}"

    async def get_buffer(self, tenant_id: str, scope_id: str) -> SensoryBuffer:
        """Get or create buffer for scope."""
        key = self._get_key(tenant_id, scope_id)
        async with self._lock:
            if key not in self._buffers:
                self._buffers[key] = SensoryBuffer(self.config)
            return self._buffers[key]

    async def ingest(
        self,
        tenant_id: str,
        scope_id: str,
        text: str,
        turn_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> int:
        """Ingest text into scope's buffer."""
        buffer = await self.get_buffer(tenant_id, scope_id)
        return buffer.ingest(text, turn_id, role)

    async def get_recent_text(
        self,
        tenant_id: str,
        scope_id: str,
        max_tokens: Optional[int] = None,
    ) -> str:
        """Get recent text from scope's buffer."""
        buffer = await self.get_buffer(tenant_id, scope_id)
        return buffer.get_text(max_tokens=max_tokens)

    async def clear_user(self, tenant_id: str, scope_id: str) -> None:
        """Clear a specific scope's buffer."""
        key = self._get_key(tenant_id, scope_id)
        async with self._lock:
            if key in self._buffers:
                self._buffers[key].clear()

    async def cleanup_inactive(self, inactive_seconds: float = 300) -> None:
        """Remove buffers that are empty (inactive)."""
        async with self._lock:
            to_remove = [key for key, buffer in self._buffers.items() if buffer.is_empty]
            for key in to_remove:
                del self._buffers[key]
