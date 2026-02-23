"""OpenAI integration helper for py-cml. Use OPENAI_MODEL or LLM_INTERNAL__MODEL in .env for model."""

from __future__ import annotations

import os
from typing import Any, Protocol, runtime_checkable

from cml.client import CognitiveMemoryLayer


@runtime_checkable
class MemoryProvider(Protocol):
    """Protocol for memory-enhanced LLM providers."""

    def get_context(self, query: str) -> str:
        """Get memory context for a query."""
        ...

    def store_exchange(
        self,
        user_message: str,
        assistant_response: str,
        session_id: str,
    ) -> None:
        """Store a conversation exchange."""
        ...

    def clear_session(self, session_id: str) -> None:
        """Clear a session's memories."""
        ...


class CMLOpenAIHelper:
    """Helper for integrating CML memory with OpenAI chat completions.

    Example:
        from openai import OpenAI
        from cml import CognitiveMemoryLayer
        from cml.integrations import CMLOpenAIHelper

        memory = CognitiveMemoryLayer(api_key="...")
        openai_client = OpenAI()
        helper = CMLOpenAIHelper(memory, openai_client)

        response = helper.chat("What should I eat tonight?", session_id="s1")
    """

    def __init__(
        self,
        memory_client: CognitiveMemoryLayer,
        openai_client: Any,
        *,
        model: str | None = None,
    ) -> None:
        self.memory = memory_client
        self.openai = openai_client
        self.model = (
            model or os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_INTERNAL__MODEL") or ""
        )

    def chat(
        self,
        user_message: str,
        *,
        session_id: str | None = None,
        system_prompt: str = "You are a helpful assistant.",
        extra_messages: list[dict[str, str]] | None = None,
    ) -> str:
        """Send a message with automatic memory context.

        1. Retrieves relevant memories via turn().
        2. Injects them into the system prompt.
        3. Calls OpenAI chat completion.
        4. Stores the exchange for future recall.

        Args:
            user_message: The user's message.
            session_id: Optional session ID for conversation continuity.
            system_prompt: Base system prompt (memories are appended).
            extra_messages: Optional list of {role, content} messages before the user message.

        Returns:
            The assistant's reply text.
        """
        # Retrieve context only (no store) to avoid duplicating the user message
        read_result = self.memory.read(
            query=user_message,
            max_results=10,
            response_format="llm_context",
        )
        memory_context = read_result.llm_context or read_result.context or ""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": f"{system_prompt}\n\n## Relevant Memories\n{memory_context}",
            }
        ]
        if extra_messages:
            messages.extend(extra_messages)
        messages.append({"role": "user", "content": user_message})

        if not self.model:
            raise ValueError(
                "Model not set; pass model= to CMLOpenAIHelper() or set OPENAI_MODEL or LLM_INTERNAL__MODEL in .env"
            )
        response = self.openai.chat.completions.create(
            model=self.model,
            messages=messages,
        )
        assistant_message = response.choices[0].message.content or ""

        # Store the full exchange once
        self.memory.turn(
            user_message=user_message,
            assistant_response=assistant_message,
            session_id=session_id,
        )

        return assistant_message
