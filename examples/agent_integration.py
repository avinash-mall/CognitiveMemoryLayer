"""Integrate py-cml with an AI agent framework.

Set AUTH__API_KEY (or CML_API_KEY) and CML_BASE_URL in .env.
"""

import asyncio
import os
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from cml import AsyncCognitiveMemoryLayer


class MemoryAgent:
    """Simple agent with persistent memory."""

    def __init__(self, memory: AsyncCognitiveMemoryLayer, agent_id: str):
        self.memory = memory
        self.agent_id = agent_id

    async def observe(self, observation: str) -> None:
        """Store an observation."""
        await self.memory.write(
            observation,
            context_tags=["observation"],
            agent_id=self.agent_id,
        )

    async def plan(self, goal: str) -> str:
        """Create a plan using memory context."""
        context = await self.memory.get_context(goal)
        # In a real agent, you'd call an LLM here with the context
        return f"Plan for '{goal}' with context:\n{context}"

    async def reflect(self, topic: str) -> None:
        """Reflect on past observations."""
        result = await self.memory.read(
            topic,
            context_filter=["observation"],
            max_results=20,
        )
        print(f"Reflections on '{topic}':")
        for mem in result.memories:
            print(f"  [{mem.timestamp}] {mem.text}")


async def main():
    async with AsyncCognitiveMemoryLayer(
        api_key=os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY"),
        base_url=os.environ.get("CML_BASE_URL")
        or os.environ.get("MEMORY_API_URL")
        or "http://localhost:8000",
    ) as memory:
        agent = MemoryAgent(memory, agent_id="agent-001")

        # Agent observes things
        await agent.observe("The deployment pipeline took 15 minutes today")
        await agent.observe("Three tests failed due to timeout issues")
        await agent.observe("The database migration completed successfully")

        # Agent plans based on observations
        plan = await agent.plan("improve deployment speed")
        print(plan)

        # Agent reflects on past events
        await agent.reflect("deployment")


if __name__ == "__main__":
    asyncio.run(main())
