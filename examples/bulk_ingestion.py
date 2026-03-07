"""Concurrent bulk ingestion example using semaphore-limited async writes."""

from __future__ import annotations

import asyncio
import time

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import AsyncCognitiveMemoryLayer
from cml.models import MemoryType

EXAMPLE_META = {
    "name": "bulk_ingestion",
    "kind": "python",
    "summary": "Semaphore-limited concurrent ingestion for large documents.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 120,
}

CONCURRENCY_LIMIT = 4


def chunk_document(text: str, chunk_size: int = 420) -> list[str]:
    words = text.split()
    chunks: list[str] = []
    current: list[str] = []
    current_length = 0
    for word in words:
        current.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current))
            current = []
            current_length = 0
    if current:
        chunks.append(" ".join(current))
    return chunks


async def ingest_chunk(
    memory: AsyncCognitiveMemoryLayer,
    semaphore: asyncio.Semaphore,
    chunk: str,
    *,
    index: int,
    total: int,
    session_id: str,
) -> bool:
    async with semaphore:
        response = await memory.write(
            chunk,
            session_id=session_id,
            memory_type=MemoryType.SEMANTIC_FACT,
            context_tags=["bulk-ingestion", "document"],
        )
        print(f"Chunk {index}/{total}: success={response.success}")
        return response.success


async def main_async() -> int:
    print_header("CML Bulk Ingestion")
    session_id = "examples-bulk-ingestion"
    source_text = (
        "Artificial intelligence systems can ingest, summarize, retrieve, and revise facts over time. "
        "Memory layers help applications preserve user preferences, long-term constraints, and prior events. "
    ) * 24
    chunks = chunk_document(source_text)
    started_at = time.perf_counter()

    try:
        async with AsyncCognitiveMemoryLayer(config=build_cml_config(timeout=60.0)) as memory:
            semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
            results = await asyncio.gather(
                *[
                    ingest_chunk(
                        memory,
                        semaphore,
                        chunk,
                        index=index,
                        total=len(chunks),
                        session_id=session_id,
                    )
                    for index, chunk in enumerate(chunks, start=1)
                ]
            )
        elapsed = time.perf_counter() - started_at
        print(f"Stored {sum(results)}/{len(results)} chunks in {elapsed:.2f}s")
        if elapsed > 0:
            print(f"Throughput: {len(results) / elapsed:.2f} chunks/sec")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
