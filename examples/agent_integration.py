"""Minimal agent loop that uses graceful memory retrieval."""

from __future__ import annotations

import asyncio

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import AsyncCognitiveMemoryLayer

EXAMPLE_META = {
    "name": "agent_integration",
    "kind": "python",
    "summary": "Agent observations with read_safe-based planning context.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 60,
}


class MemoryAgent:
    def __init__(self, memory: AsyncCognitiveMemoryLayer, agent_id: str) -> None:
        self.memory = memory
        self.agent_id = agent_id

    async def observe(self, observation: str) -> None:
        await self.memory.write(
            observation,
            context_tags=["observation"],
            agent_id=self.agent_id,
            session_id="examples-agent",
        )

    async def plan(self, goal: str) -> str:
        result = await self.memory.read_safe(goal, context_filter=["observation"], max_results=5)
        if not result.memories:
            return f"Plan for '{goal}': proceed without prior context."
        return f"Plan for '{goal}':\n{result.context}"

    async def reflect(self, topic: str) -> None:
        result = await self.memory.read_safe(topic, context_filter=["observation"], max_results=5)
        print(f"Reflection on '{topic}': {result.total_count} memories")
        for item in result.memories:
            print(f"  - {item.text}")


async def main_async() -> int:
    print_header("CML Agent Integration")
    try:
        async with AsyncCognitiveMemoryLayer(config=build_cml_config()) as memory:
            agent = MemoryAgent(memory, agent_id="agent-001")
            await agent.observe("Deployment pipeline took 15 minutes today.")
            await agent.observe("Three tests failed due to timeout issues.")
            await agent.observe("Database migration completed successfully.")
            print(await agent.plan("improve deployment speed"))
            await agent.reflect("deployment")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


def main() -> int:
    return asyncio.run(main_async())


if __name__ == "__main__":
    raise SystemExit(main())
