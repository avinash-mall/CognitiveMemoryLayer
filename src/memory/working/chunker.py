"""Semantic chunking via semchunk (Hugging Face tokenizer)."""

import hashlib
from datetime import UTC, datetime

import semchunk

from .models import ChunkType, SemanticChunk


def _generate_chunk_id(text: str, index: int) -> str:
    """Generate deterministic chunk ID."""
    content = f"{text}:{index}"
    return hashlib.sha256(content.encode()).hexdigest()[:16]


class SemchunkChunker:
    """
    Uses semchunk with a Hugging Face tokenizer for semantic chunking.

    Configure via CHUNKER__TOKENIZER, CHUNKER__CHUNK_SIZE, CHUNKER__OVERLAP_PERCENT.
    Lazy-loads tokenizer and chunker on first use.
    """

    def __init__(self, tokenizer_id: str, chunk_size: int, overlap_percent: float) -> None:
        self._tokenizer_id = tokenizer_id
        self._chunk_size = chunk_size
        self._overlap_percent = overlap_percent
        self._chunker = None

    def _get_chunker(self):
        """Lazy-load tokenizer and build chunker."""
        if self._chunker is None:
            from transformers import AutoTokenizer

            tokenizer = AutoTokenizer.from_pretrained(self._tokenizer_id)
            self._chunker = semchunk.chunkerify(tokenizer, self._chunk_size)
        return self._chunker

    def chunk(
        self,
        text: str,
        turn_id: str | None = None,
        role: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[SemanticChunk]:
        """Chunk text with semchunk and map to SemanticChunk list (synchronous)."""
        if not text.strip():
            return []

        chunker = self._get_chunker()
        raw_chunks = chunker(text, overlap=self._overlap_percent)

        chunk_timestamp = timestamp or datetime.now(UTC)
        chunks = []
        for i, segment in enumerate(raw_chunks):
            if not segment.strip():
                continue
            chunk_id = _generate_chunk_id(segment, i)
            chunks.append(
                SemanticChunk(
                    id=chunk_id,
                    text=segment,
                    chunk_type=ChunkType.STATEMENT,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=0.5,
                    confidence=0.8,
                    timestamp=chunk_timestamp,
                )
            )

        if not chunks:
            chunks.append(
                SemanticChunk(
                    id=_generate_chunk_id(text, 0),
                    text=text,
                    chunk_type=ChunkType.STATEMENT,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=0.5,
                    confidence=0.8,
                    timestamp=chunk_timestamp,
                )
            )
        return chunks

    async def chunk_async(
        self,
        text: str,
        turn_id: str | None = None,
        role: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[SemanticChunk]:
        """Async wrapper: runs the synchronous HuggingFace tokenizer in a thread
        pool executor so it does not block the asyncio event loop during the
        cold-start load (~30 s) or for CPU-intensive chunking of large texts.
        """
        import functools

        import anyio

        return await anyio.to_thread.run_sync(
            functools.partial(self.chunk, text, turn_id=turn_id, role=role, timestamp=timestamp)
        )

    async def preload_async(self) -> None:
        """Warm the tokenizer in a background thread without blocking the event loop.
        Call once at server startup so the first real write is not penalised.
        """
        import anyio

        await anyio.to_thread.run_sync(self._get_chunker)
