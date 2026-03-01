"""LangChain integration with Cognitive Memory Layer.

Set CML_API_KEY, CML_BASE_URL, OPENAI_API_KEY, OPENAI_MODEL in .env.
"""

import os
from pathlib import Path
from typing import Any, Literal, cast

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

from langchain.chains import ConversationChain
from langchain_core.memory import BaseMemory
from langchain_core.messages import AIMessage
from langchain_core.prompts import PromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import ConfigDict, SecretStr

from cml import CognitiveMemoryLayer


class CognitiveMemory(BaseMemory):
    """LangChain memory backed by Cognitive Memory Layer (py-cml)."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    session_id: str = "default"
    memory_client: CognitiveMemoryLayer | None = None
    api_url: str = ""
    api_key: str = ""
    auto_store: bool = True
    store_human: bool = True
    store_ai: bool = False
    max_retrieval_results: int = 5
    retrieval_format: str = "llm_context"
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "response"
    return_messages: bool = False

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        if self.memory_client is None:
            base_url = (
                self.api_url or os.environ.get("CML_BASE_URL") or "http://localhost:8000"
            ).strip()
            key = self.api_key or os.environ.get("CML_API_KEY")
            self.memory_client = CognitiveMemoryLayer(api_key=key, base_url=base_url)

    @property
    def memory_variables(self) -> list[str]:
        return [self.memory_key]

    def load_memory_variables(self, inputs: dict[str, Any]) -> dict[str, Any]:
        query = inputs.get(self.input_key, "")
        if not query:
            return {self.memory_key: "" if not self.return_messages else []}
        if self.memory_client is None:
            return {self.memory_key: "" if not self.return_messages else []}
        try:
            fmt = cast(
                "Literal['packet', 'list', 'llm_context']",
                self.retrieval_format,
            )
            r = self.memory_client.read(
                query, max_results=self.max_retrieval_results, response_format=fmt
            )
            ctx = r.context
            if self.return_messages:
                return {self.memory_key: [AIMessage(content=f"[Memory]\n{ctx}")] if ctx else []}
            return {self.memory_key: ctx}
        except Exception as e:
            print(f"Warning: Could not load memories: {e}")
            return {self.memory_key: "" if not self.return_messages else []}

    def save_context(self, inputs: dict[str, Any], outputs: dict[str, str]) -> None:
        if not self.auto_store:
            return
        if self.memory_client is None:
            return
        try:
            from cml.models.enums import MemoryType

            if self.store_human:
                human = inputs.get(self.input_key, "")
                if human:
                    self.memory_client.write(
                        f"User said: {human}",
                        session_id=self.session_id,
                        context_tags=["conversation"],
                        memory_type=MemoryType.EPISODIC_EVENT,
                    )
            if self.store_ai:
                ai = outputs.get(self.output_key, "")
                if ai:
                    self.memory_client.write(
                        f"Assistant: {ai[:200]}",
                        session_id=self.session_id,
                        context_tags=["conversation"],
                        memory_type=MemoryType.EPISODIC_EVENT,
                    )
        except Exception as e:
            print(f"Warning: Could not save: {e}")

    def clear(self) -> None:
        """Clear memories. Uses semantic query - clears tenant memories matching
        'conversation session', not just this session. CML has no session-scoped forget."""
        if self.memory_client is None:
            return
        try:
            self.memory_client.forget(query="conversation session memories", action="delete")
        except Exception:
            pass


PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful assistant with long-term memory.

{history}

Human: {input}
Assistant:""",
)


def create_memory_chain(
    session_id: str = "default",
    llm_model: str | None = None,
    memory_api_url: str | None = None,
    memory_api_key: str | None = None,
) -> ConversationChain:
    base_url = (
        memory_api_url or os.environ.get("CML_BASE_URL") or ""
    ).strip() or "http://localhost:8000"
    key = memory_api_key or os.environ.get("CML_API_KEY")
    # Prefer LLM_INTERNAL__* when set (e.g. .env lines 20-22)
    model = (
        llm_model or os.environ.get("LLM_INTERNAL__MODEL") or os.environ.get("OPENAI_MODEL") or ""
    ).strip()
    if not model:
        raise ValueError("Set OPENAI_MODEL or LLM_INTERNAL__MODEL")
    memory = CognitiveMemory(
        session_id=session_id,
        api_url=base_url,
        api_key=key,
        auto_store=True,
        store_human=True,
        store_ai=False,
    )
    llm_base = (os.environ.get("LLM_INTERNAL__BASE_URL") or "").strip()
    if llm_base:
        api_key = (
            os.environ.get("LLM_INTERNAL__API_KEY") or os.environ.get("OPENAI_API_KEY") or "dummy"
        )
        llm = ChatOpenAI(
            model=model, temperature=0.7, base_url=llm_base, api_key=SecretStr(api_key)
        )
    else:
        llm = ChatOpenAI(model=model, temperature=0.7)
    return ConversationChain(llm=llm, memory=memory, prompt=PROMPT, verbose=False)


def main():
    print("=" * 60)
    print("LangChain + Cognitive Memory Layer")
    print("=" * 60)
    has_llm_internal = (
        os.environ.get("LLM_INTERNAL__PROVIDER")
        and os.environ.get("LLM_INTERNAL__MODEL")
        and os.environ.get("LLM_INTERNAL__BASE_URL")
    )
    has_openai = os.getenv("OPENAI_API_KEY") and (
        os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_INTERNAL__MODEL")
    )
    if not has_llm_internal and not has_openai:
        print(
            "Set OPENAI_API_KEY and OPENAI_MODEL (or LLM_INTERNAL__PROVIDER/MODEL/BASE_URL) in .env"
        )
        return
    if not (os.environ.get("OPENAI_MODEL") or os.environ.get("LLM_INTERNAL__MODEL")):
        print("Set OPENAI_MODEL or LLM_INTERNAL__MODEL in .env")
        return
    chain = create_memory_chain(session_id="langchain-demo")
    print("\nType 'quit' to exit.\n")
    try:
        while True:
            user_input = input("You: ").strip()
            if user_input.lower() in ("quit", "exit", "bye"):
                print("\nBot: Goodbye!")
                break
            if not user_input:
                continue
            out = chain.invoke({"input": user_input})
            response = out.get("response", out.get("output", str(out)))
            if hasattr(response, "content"):
                response = response.content
            print(f"\nBot: {response}\n")
    except KeyboardInterrupt:
        print("\nGoodbye!")


if __name__ == "__main__":
    main()
