"""Sensory buffer: high-fidelity short-term token storage with decay.

Phase 5.1 optimisation: stores **token IDs** (ints) instead of decoded
strings to reduce CPU, GC pressure, and per-token object count.
The tiktoken encoder is lazily initialised and reused across calls.
"""

import asyncio
import contextlib
import time
from collections import deque
from dataclasses import dataclass
from typing import Any


@dataclass
class BufferedToken:
    """A token with its ingestion timestamp.

    ``token_id`` is the integer token ID (from tiktoken or a hash-based
    fallback).  The original text is recovered via :meth:`SensoryBuffer.get_text`.
    """

    token_id: int
    timestamp: float  # Unix timestamp for fast comparison
    turn_id: str | None = None
    role: str | None = None  # "user", "assistant", "system"

    # Backward-compat: expose a .token property returning an empty string
    # (callers should use SensoryBuffer.get_text instead).
    @property
    def token(self) -> str:  # pragma: no cover - compat shim
        return ""


@dataclass
class SensoryBufferConfig:
    """Configuration for the sensory buffer."""

    max_tokens: int = 500
    decay_seconds: float = 30.0
    cleanup_interval_seconds: float = 5.0


# Module-level tiktoken encoder (shared, thread-safe, lazy)
_tiktoken_encoder: Any | None = None
_tiktoken_available: bool | None = None


def _get_tiktoken_encoder() -> Any | None:
    """Return a cached tiktoken encoder, or *None* if unavailable."""
    global _tiktoken_encoder, _tiktoken_available
    if _tiktoken_available is None:
        try:
            import tiktoken

            _tiktoken_encoder = tiktoken.get_encoding("cl100k_base")
            _tiktoken_available = True
        except ImportError:
            _tiktoken_available = False
    return _tiktoken_encoder


class SensoryBuffer:
    """
    High-fidelity short-term buffer for raw input tokens.

    Mimics sensory memory: stores everything briefly, then decays.
    Uses deque for O(1) append and popleft operations.

    Tokens are stored as integer IDs; text is decoded only when
    :meth:`get_text` is called (single batch decode).
    """

    def __init__(self, config: SensoryBufferConfig | None = None):
        self.config = config or SensoryBufferConfig()
        self._tokens: deque[BufferedToken] = deque()
        self._lock = asyncio.Lock()
        self._cleanup_task: asyncio.Task | None = None
        self._last_activity: float = time.time()

    async def ingest(
        self,
        text: str,
        turn_id: str | None = None,
        role: str | None = None,
    ) -> int:
        """Ingest new text into the buffer.

        Returns the number of tokens ingested.
        """
        now = time.time()
        token_ids = self._tokenize(text)

        async with self._lock:
            for tid in token_ids:
                self._tokens.append(
                    BufferedToken(
                        token_id=tid,
                        timestamp=now,
                        turn_id=turn_id,
                        role=role,
                    )
                )
            self._cleanup(now)
            self._last_activity = now

        return len(token_ids)

    async def get_recent(
        self,
        max_tokens: int | None = None,
        since_seconds: float | None = None,
        role_filter: str | None = None,
    ) -> list[BufferedToken]:
        """Get recent tokens from buffer (oldest first)."""
        now = time.time()
        cutoff = now - (since_seconds or self.config.decay_seconds)

        async with self._lock:
            result: list[BufferedToken] = []
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
        max_tokens: int | None = None,
        role_filter: str | None = None,
    ) -> str:
        """Decode buffered token IDs into text in a single batch."""
        tokens = await self.get_recent(max_tokens=max_tokens, role_filter=role_filter)
        if not tokens:
            return ""

        enc = _get_tiktoken_encoder()
        if enc is not None:
            token_ids = [bt.token_id for bt in tokens]
            return enc.decode(token_ids)

        # Fallback (no tiktoken): token_ids are hash-based; can't decode
        return ""

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

    @staticmethod
    def _tokenize(text: str) -> list[int]:
        """Tokenize text, returning token IDs (not decoded strings).

        This avoids the previous per-token ``enc.decode([t])`` loop that
        created N string objects and N decode calls.
        """
        enc = _get_tiktoken_encoder()
        if enc is not None:
            return enc.encode(text)
        # Fallback: use hash of whitespace tokens as pseudo-IDs
        return [hash(w) & 0xFFFFFFFF for w in text.split()]

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
            with contextlib.suppress(asyncio.CancelledError):
                await self._cleanup_task
            self._cleanup_task = None
