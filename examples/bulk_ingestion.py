"""
Bulk Optimized Document Ingestion Example - Cognitive Memory Layer.

This script demonstrates how to efficiently ingest large documents or multiple
documents into the Cognitive Memory Layer using asynchronous concurrent writes
with rate limiting (Semaphore) to avoid overwhelming the server.

Required .env variables (place in .env in the root or parent folder):
  CML_API_KEY - Your API key for authentication.
  CML_BASE_URL - The base URL of the Cognitive Memory Layer API (e.g., http://localhost:8000).

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
       docker compose -f docker/docker-compose.yml up api
    2. pip install cognitive-memory-layer
"""

import asyncio
import os
import time
from pathlib import Path
from typing import List

try:
    from dotenv import load_dotenv

    # Load environment variables from .env file
    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from cml import AsyncCognitiveMemoryLayer

# Maximum number of concurrent write requests to the server
# This prevents overwhelming the API and database with too many simultaneous connections
CONCURRENCY_LIMIT = 5

def _memory_config():
    """Retrieve configuration from environment variables."""
    base_url = (os.environ.get("CML_BASE_URL") or "").strip() or "http://localhost:8000"
    api_key = os.environ.get("CML_API_KEY")
    
    if not api_key:
        print("Warning: CML_API_KEY is not set. If the server requires authentication, requests might fail.")
        
    return {
        "api_key": api_key,
        "base_url": base_url,
    }

def chunk_document(text: str, chunk_size: int = 500) -> List[str]:
    """
    Very simple chunking logic for splitting a large document into smaller pieces.
    In a real scenario, consider using a proper text splitter (e.g., from LangChain)
    which can split intelligently by sentence or paragraph.
    """
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        current_chunk.append(word)
        current_length += len(word) + 1
        if current_length >= chunk_size:
            chunks.append(" ".join(current_chunk))
            current_chunk = []
            current_length = 0
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
        
    return chunks

async def ingest_chunk(
    memory: AsyncCognitiveMemoryLayer, 
    chunk: str, 
    session_id: str, 
    semaphore: asyncio.Semaphore,
    index: int,
    total: int
):
    """Ingest a single chunk with concurrency limit."""
    async with semaphore:
        try:
            print(f"[{index}/{total}] Ingesting chunk...")
            result = await memory.write(
                chunk,
                session_id=session_id,
                memory_type="semantic_fact",
                context_tags=["bulk-ingestion", "document"]
            )
            return result
        except Exception as e:
            print(f"[{index}/{total}] Failed to ingest chunk: {e}")
            return None

async def bulk_ingest(document_text: str, session_id: str):
    """
    Split the document into chunks and concurrently ingest them into CML 
    while respecting the concurrency limit.
    """
    chunks = chunk_document(document_text)
    print(f"Document split into {len(chunks)} chunks.")
    
    config = _memory_config()
    semaphore = asyncio.Semaphore(CONCURRENCY_LIMIT)
    
    start_time = time.time()
    
    async with AsyncCognitiveMemoryLayer(**config) as memory:
        tasks = []
        for i, chunk in enumerate(chunks, 1):
            tasks.append(
                ingest_chunk(memory, chunk, session_id, semaphore, i, len(chunks))
            )
            
        print(f"Starting optimized bulk ingestion with concurrency limit of {CONCURRENCY_LIMIT}...")
        
        # asyncio.gather runs all tasks concurrently
        results = await asyncio.gather(*tasks)
        
        # Count successful writes (assuming WriteResponse has a 'success' attribute or we infer from non-None)
        success_count = sum(1 for r in results if getattr(r, 'success', r is not None))
            
    elapsed = time.time() - start_time
    print("-" * 40)
    print("Bulk Ingestion Summary:")
    print(f"Total chunks processed: {len(chunks)}")
    print(f"Successful writes:      {success_count}")
    print(f"Time taken:             {elapsed:.2f} seconds")
    print(f"Throughput:             {len(chunks) / elapsed:.2f} chunks/sec")
    print("-" * 40)

def main():
    # Example large document text
    sample_document = """
    Artificial intelligence (AI) is intelligence demonstrated by machines, as opposed to the natural 
    intelligence displayed by animals including humans. Leading AI textbooks define the field as the study 
    of "intelligent agents": any system that perceives its environment and takes actions that maximize its 
    chance of achieving its goals. Some popular accounts use the term "artificial intelligence" to describe 
    machines that mimic "cognitive" functions that humans associate with the human mind, such as "learning" 
    and "problem solving", however this definition is rejected by major AI researchers.
    
    The various sub-fields of AI research are centered around particular goals and the use of particular tools. 
    The traditional goals of AI research include reasoning, knowledge representation, planning, learning, natural 
    language processing, perception and the ability to move and manipulate objects. General intelligence 
    (the ability to solve an arbitrary problem) is among the field's long-term goals. To solve these problems, 
    AI researchers have adapted and integrated a wide range of problem-solving techniques—including search and 
    mathematical optimization, formal logic, artificial neural networks, and methods based on statistics, 
    probability and economics. AI also draws upon computer science, psychology, linguistics, philosophy, and 
    many other fields.
    """ * 10  # Multiply to simulate a larger document

    session_id = "bulk-ingestion-demo-session"
    
    print("=" * 60)
    print("Cognitive Memory Layer - Bulk Optimized Ingestion")
    print("=" * 60)
    
    try:
        asyncio.run(bulk_ingest(sample_document, session_id))
    except Exception as e:
        if "Connect" in str(type(e).__name__) or "Connection" in str(e):
            print(
                "\n✗ Could not connect. Start API: docker compose -f docker/docker-compose.yml up api"
            )
        else:
            raise

if __name__ == "__main__":
    main()
