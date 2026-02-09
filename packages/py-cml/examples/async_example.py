"""Async usage of py-cml with asyncio."""

import asyncio
from cml import AsyncCognitiveMemoryLayer


async def main():
    async with AsyncCognitiveMemoryLayer(api_key="your-key", base_url="http://localhost:8000") as memory:
        await asyncio.gather(
            memory.write("User likes hiking in the mountains"),
            memory.write("User prefers morning workouts"),
            memory.write("User is training for a marathon"),
        )
        result = await memory.read("exercise habits")
        print(f"Found {result.total_count} memories about exercise")
        for mem in result.memories:
            print(f"  - {mem.text}")
        results = await memory.batch_read(["exercise habits", "outdoor activities", "fitness goals"])
        for r in results:
            print(f"\nQuery: {r.query} -> {r.total_count} results")


if __name__ == "__main__":
    asyncio.run(main())
