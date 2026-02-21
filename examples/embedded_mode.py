"""Use py-cml without a server — embedded mode with serverless SQLite.

Uses aiosqlite (in-memory or file-based). No HTTP or CML API required.

Monorepo setup: pip install -e . && pip install -e "packages/py-cml[embedded]"
Released package: pip install cognitive-memory-layer[embedded]
"""

import asyncio

from cml import EmbeddedCognitiveMemoryLayer


async def main():
    async with EmbeddedCognitiveMemoryLayer() as memory:
        await memory.write("User prefers Python over JavaScript")
        await memory.write("User uses VS Code as their primary editor")
        await memory.write("User follows TDD methodology")
        result = await memory.read("development tools")
        print(f"Found {result.total_count} relevant memories:")
        for mem in result.memories:
            print(f"  - {mem.text}")
        stats = await memory.stats()
        print(f"\nTotal memories: {stats.total_memories}")
    async with EmbeddedCognitiveMemoryLayer(db_path="./my_app.db") as memory:
        await memory.write("This memory persists between restarts")
    async with EmbeddedCognitiveMemoryLayer(db_path="./my_app.db") as memory:
        result = await memory.read("persists")
        if result.memories:
            print(f"\nPersistent memory found: {result.memories[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
