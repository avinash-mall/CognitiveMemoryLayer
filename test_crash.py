import asyncio

from src.utils.embeddings import LocalEmbeddings

embeddings = LocalEmbeddings("nomic-ai/nomic-embed-text-v2-moe")


async def worker(i):
    # This will now use the lock we added inside embed
    await embeddings.embed(f"Test string {i}")


async def main():
    await asyncio.gather(*(worker(i) for i in range(20)))
    print("Success! No crashes.")


if __name__ == "__main__":
    asyncio.run(main())
