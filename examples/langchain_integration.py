"""LangChain conversation example backed by Cognitive Memory Layer."""

from __future__ import annotations

from typing import Any, Literal, cast

from _shared import (
    build_cml_config,
    explain_connection_failure,
    iter_user_inputs,
    openai_settings,
    print_header,
)
from langchain.chains import ConversationChain
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict, SecretStr

from cml import CognitiveMemoryLayer
from cml.models import MemoryType

EXAMPLE_META = {
    "name": "langchain_integration",
    "kind": "python",
    "summary": "LangChain conversation example with CML-backed memory injection.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": True,
    "requires_anthropic": False,
    "interactive": True,
    "timeout_sec": 120,
}


class CognitiveMemory(BaseMemory):
    """LangChain memory wrapper around py-cml."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = "default"
    memory_client: CognitiveMemoryLayer | None = None
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "response"
    return_messages: bool = False

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        if self.memory_client is None:
            return {self.memory_key: "" if not self.return_messages else []}
        query = str(inputs.get(self.input_key, "")).strip()
        if not query:
            return {self.memory_key: "" if not self.return_messages else []}

        result = self.memory_client.read_safe(
            query,
            max_results=5,
            response_format=cast("Literal['packet', 'list', 'llm_context']", "llm_context"),
        )
        if self.return_messages:
            return {
                self.memory_key: [AIMessage(content=f"[Memory]\n{result.context}")]
                if result.context
                else []
            }
        return {self.memory_key: result.context}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        if self.memory_client is None:
            return
        user_text = str(inputs.get(self.input_key, "")).strip()
        if user_text:
            self.memory_client.write(
                f"User said: {user_text}",
                session_id=self.session_id,
                context_tags=["conversation"],
                memory_type=MemoryType.EPISODIC_EVENT,
            )
        assistant_text = str(outputs.get(self.output_key, "")).strip()
        if assistant_text:
            self.memory_client.write(
                f"Assistant replied: {assistant_text[:240]}",
                session_id=self.session_id,
                context_tags=["conversation"],
                memory_type=MemoryType.EPISODIC_EVENT,
            )

    def clear(self) -> None:
        return None


PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template=(
        "You are a helpful assistant with persistent long-term memory.\n\n"
        "{history}\n\n"
        "Human: {input}\n"
        "Assistant:"
    ),
)


def create_chain() -> ConversationChain:
    settings = openai_settings()
    llm_kwargs: dict[str, Any] = {
        "model": str(settings["model"]),
        "api_key": SecretStr(str(settings["api_key"])),
        "temperature": 0.2,
    }
    if settings["base_url"] is not None:
        llm_kwargs["base_url"] = settings["base_url"]
    llm = ChatOpenAI(**llm_kwargs)
    memory = CognitiveMemory(
        session_id="examples-langchain",
        memory_client=CognitiveMemoryLayer(config=build_cml_config(timeout=60.0)),
    )
    return ConversationChain(llm=llm, memory=memory, prompt=PROMPT, verbose=False)


def main() -> int:
    print_header("CML LangChain Integration")
    chain = create_chain()
    try:
        for user_input in iter_user_inputs(
            "You: ",
            default_inputs=[
                "Remember that I prefer short status updates.",
                "How should you deliver status updates?",
                "quit",
            ],
        ):
            if user_input.lower() in {"quit", "exit"}:
                break
            output = chain.invoke({"input": user_input})
            response = output.get("response", output.get("output", str(output)))
            if hasattr(response, "content"):
                response = response.content
            print(f"Bot: {response}\n")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1
    finally:
        memory = cast("CognitiveMemory", chain.memory)
        if memory.memory_client is not None:
            memory.memory_client.close()


if __name__ == "__main__":
    raise SystemExit(main())
