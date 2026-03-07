"""Embedded mode example using the in-process CML engine."""

from __future__ import annotations

import asyncio
from pathlib import Path

from _shared import print_header

from cml import EmbeddedCognitiveMemoryLayer

EXAMPLE_META = {
    "name": "embedded_mode",
    "kind": "python",
    "summary": "In-process embedded mode with in-memory and file-backed storage.",
    "requires_api": False,
    "requires_api_key": False,
    "requires_base_url": False,
    "requires_admin_key": False,
    "requires_embedded": True,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 120,
}


async def main_async() -> int:
    print_header("CML Embedded Mode")
    db_path = Path("embedded_example.db")
    try:
        async with EmbeddedCognitiveMemoryLayer() as memory:
            await memory.write("User prefers Python over JavaScript.")
            await memory.write("User uses VS Code as their primary editor.")
            result = await memory.read("development preferences")
            print(f"In-memory read returned {result.total_count} memories")

        async with EmbeddedCognitiveMemoryLayer(db_path=str(db_path)) as memory:
            await memory.write("This memory persists between restarts.")

        async with EmbeddedCognitiveMemoryLayer(db_path=str(db_path)) as memory:
            result = await memory.read("persists")
            print(f"Persistent read returned {result.total_count} memories")
        return 0
    finally:
        if db_path.exists():
            db_path.unlink()


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
