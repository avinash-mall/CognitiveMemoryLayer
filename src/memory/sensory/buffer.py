"""Sensory buffer: high-fidelity short-term token storage with decay."""

from collections import deque
from dataclasses import dataclass
from typing import Deque, List, Optional

import asyncio
import time


@dataclass
class BufferedToken:
    """A token with its ingestion timestamp."""

    token: str
    timestamp: float  # Unix timestamp for fast comparison
    turn_id: Optional[str] = None
    role: Optional[str] = None  # "user", "assistant", "system"


@dataclass
class SensoryBufferConfig:
    """Configuration for the sensory buffer."""

    max_tokens: int = 500
    decay_seconds: float = 30.0
    cleanup_interval_seconds: float = 5.0


class SensoryBuffer:
    """
    High-fidelity short-term buffer for raw input tokens.

    Mimics sensory memory: stores everything briefly, then decays.
    Uses deque for O(1) append and popleft operations.
    """

    def __init__(self, config: Optional[SensoryBufferConfig] = None):
        self.config = config or SensoryBufferConfig()
        self._tokens: Deque[BufferedToken] = deque()
        self._lock = asyncio.Lock()
        self._cleanup_task: Optional[asyncio.Task] = None
        self._last_activity: float = time.time()

    async def ingest(
        self,
        text: str,
        turn_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> int:
        """
        Ingest new text into the buffer.

        Args:
            text: Raw text to buffer
            turn_id: Optional identifier for the conversation turn
            role: "user", "assistant", or "system"

        Returns:
            Number of tokens ingested
        """
        now = time.time()
        tokens = self._tokenize(text)

        async with self._lock:
            for token in tokens:
                self._tokens.append(
                    BufferedToken(
                        token=token,
                        timestamp=now,
                        turn_id=turn_id,
                        role=role,
                    )
                )
            self._cleanup(now)
            self._last_activity = now

        return len(tokens)

    async def get_recent(
        self,
        max_tokens: Optional[int] = None,
        since_seconds: Optional[float] = None,
        role_filter: Optional[str] = None,
    ) -> List[BufferedToken]:
        """
        Get recent tokens from buffer.

        Args:
            max_tokens: Maximum tokens to return
            since_seconds: Only tokens from last N seconds
            role_filter: Filter by role ("user", "assistant")

        Returns:
            List of buffered tokens (oldest first)
        """
        now = time.time()
        cutoff = now - (since_seconds or self.config.decay_seconds)

        async with self._lock:
            result = []
            for bt in self._tokens:
                if bt.timestamp < cutoff:
                    continue
                if role_filter and bt.role != role_filter:
                    continue
                result.append(bt)
                if max_tokens and len(result) >= max_tokens:
                    break
            return result

    async def get_text(
        self,
        max_tokens: Optional[int] = None,
        role_filter: Optional[str] = None,
    ) -> str:
        """Get buffered content as joined text."""
        tokens = await self.get_recent(max_tokens=max_tokens, role_filter=role_filter)
        return " ".join(bt.token for bt in tokens)

    async def clear(self) -> None:
        """Clear all buffered tokens."""
        async with self._lock:
            self._tokens.clear()

    def _cleanup(self, now: float) -> None:
        """Remove expired tokens and enforce capacity."""
        cutoff = now - self.config.decay_seconds
        while self._tokens and self._tokens[0].timestamp < cutoff:
            self._tokens.popleft()
        while len(self._tokens) > self.config.max_tokens:
            self._tokens.popleft()

    def _tokenize(self, text: str) -> List[str]:
        """Simple whitespace tokenization. For production, use tiktoken."""
        return text.split()

    @property
    def size(self) -> int:
        """Current number of tokens in buffer."""
        return len(self._tokens)

    @property
    def is_empty(self) -> bool:
        return len(self._tokens) == 0

    @property
    def last_activity(self) -> float:
        """Unix timestamp of last ingest activity."""
        return self._last_activity

    async def start_cleanup_loop(self) -> None:
        """Start background cleanup task."""

        async def cleanup_loop() -> None:
            while True:
                await asyncio.sleep(self.config.cleanup_interval_seconds)
                async with self._lock:
                    self._cleanup(time.time())

        self._cleanup_task = asyncio.create_task(cleanup_loop())

    async def stop_cleanup_loop(self) -> None:
        """Stop background cleanup task."""
        if self._cleanup_task:
            self._cleanup_task.cancel()
            try:
                await self._cleanup_task
            except asyncio.CancelledError:
                pass
            self._cleanup_task = None
