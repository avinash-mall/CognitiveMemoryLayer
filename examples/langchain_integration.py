"""
LangChain Integration - Cognitive Memory Layer

This example demonstrates how to integrate the Cognitive Memory Layer
with LangChain as a custom memory class.

The CognitiveMemory class can be used with any LangChain chain or agent,
providing persistent long-term memory across sessions.

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api
    
    2. Install dependencies:
       pip install langchain langchain-openai httpx
    
    3. Set your OpenAI API key:
       export OPENAI_API_KEY=sk-...
"""

import os
from typing import Any, Dict, List, Optional
from langchain.memory.chat_memory import BaseChatMemory
from langchain.schema import BaseMessage, HumanMessage, AIMessage
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate

from memory_client import CognitiveMemoryClient


class CognitiveMemory(BaseChatMemory):
    """
    LangChain-compatible memory class that uses the Cognitive Memory Layer.
    
    This provides persistent, semantic memory that:
    - Retrieves relevant context before each LLM call
    - Stores important information from conversations
    - Persists across sessions and restarts
    
    Example:
        from langchain_openai import ChatOpenAI
        from langchain.chains import ConversationChain
        
        memory = CognitiveMemory(scope="session", scope_id="session-123")
        llm = ChatOpenAI()
        chain = ConversationChain(llm=llm, memory=memory)
        
        response = chain.predict(input="My name is Alice")
    """
    
    scope: str = "session"
    scope_id: str
    memory_client: Optional[CognitiveMemoryClient] = None
    api_url: str = "http://localhost:8000"
    api_key: str = "demo-key-123"
    
    # Control what gets stored
    auto_store: bool = True
    store_human: bool = True
    store_ai: bool = False
    
    # Memory retrieval settings
    max_retrieval_results: int = 5
    retrieval_format: str = "llm_context"
    
    # LangChain memory keys
    memory_key: str = "history"
    input_key: str = "input"
    output_key: str = "response"
    return_messages: bool = False
    
    class Config:
        arbitrary_types_allowed = True
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if self.memory_client is None:
            self.memory_client = CognitiveMemoryClient(
                base_url=self.api_url,
                api_key=self.api_key
            )
    
    @property
    def memory_variables(self) -> List[str]:
        """Return the memory variables this class provides."""
        return [self.memory_key]
    
    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """
        Load relevant memories based on the current input.
        
        This is called by LangChain before each LLM invocation.
        """
        # Get the current input to use as query
        query = inputs.get(self.input_key, "")
        
        if not query:
            return {self.memory_key: "" if not self.return_messages else []}
        
        try:
            # Retrieve relevant memories
            result = self.memory_client.read(
                scope=self.scope,
                scope_id=self.scope_id,
                query=query,
                max_results=self.max_retrieval_results,
                format=self.retrieval_format
            )
            
            if self.return_messages:
                # Return as LangChain messages
                messages = []
                if result.llm_context:
                    messages.append(AIMessage(content=f"[Memory Context]\n{result.llm_context}"))
                return {self.memory_key: messages}
            else:
                # Return as string
                return {self.memory_key: result.llm_context or ""}
                
        except Exception as e:
            print(f"Warning: Could not load memories: {e}")
            return {self.memory_key: "" if not self.return_messages else []}
    
    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:
        """
        Save the conversation context to memory.
        
        This is called by LangChain after each LLM invocation.
        """
        if not self.auto_store:
            return
        
        try:
            # Store human input
            if self.store_human:
                human_input = inputs.get(self.input_key, "")
                if human_input:
                    self.memory_client.write(
                        scope=self.scope,
                        scope_id=self.scope_id,
                        content=f"User said: {human_input}",
                        memory_type="episodic_event"
                    )
            
            # Store AI response (usually not needed as it's derived)
            if self.store_ai:
                ai_output = outputs.get(self.output_key, "")
                if ai_output:
                    self.memory_client.write(
                        scope=self.scope,
                        scope_id=self.scope_id,
                        content=f"Assistant responded about: {ai_output[:200]}",
                        memory_type="episodic_event"
                    )
                    
        except Exception as e:
            print(f"Warning: Could not save to memory: {e}")
    
    def clear(self) -> None:
        """Clear all memories for this scope."""
        try:
            self.memory_client.forget(
                scope=self.scope,
                scope_id=self.scope_id,
                query="*",
                action="delete"
            )
        except Exception as e:
            print(f"Warning: Could not clear memory: {e}")
    
    @property
    def chat_memory(self) -> Any:
        """Required by BaseChatMemory but not used in our implementation."""
        return None
    
    def add_message(self, message: BaseMessage) -> None:
        """Add a message directly to memory."""
        content = message.content
        memory_type = "episodic_event"
        
        if isinstance(message, HumanMessage):
            content = f"User said: {content}"
        elif isinstance(message, AIMessage):
            content = f"Assistant said: {content}"
        
        try:
            self.memory_client.write(
                scope=self.scope,
                scope_id=self.scope_id,
                content=content,
                memory_type=memory_type
            )
        except Exception as e:
            print(f"Warning: Could not add message to memory: {e}")


# Custom prompt that includes memory context
COGNITIVE_MEMORY_PROMPT = PromptTemplate(
    input_variables=["history", "input"],
    template="""You are a helpful assistant with access to long-term memory about the user.

{history}

Current conversation:
Human: {input}
Assistant:"""
)


def create_memory_chain(
    scope_id: str,
    scope: str = "session",
    llm_model: str = "gpt-4o-mini",
    memory_api_url: str = "http://localhost:8000",
    memory_api_key: str = "demo-key-123"
) -> ConversationChain:
    """
    Create a LangChain conversation chain with cognitive memory.
    
    Args:
        scope_id: Unique identifier for the memory scope (e.g., session ID)
        scope: Memory scope type (session, agent, namespace, global)
        llm_model: OpenAI model to use
        memory_api_url: Cognitive Memory Layer API URL
        memory_api_key: API key for memory service
        
    Returns:
        A ConversationChain with persistent memory
    """
    # Create memory
    memory = CognitiveMemory(
        scope=scope,
        scope_id=scope_id,
        api_url=memory_api_url,
        api_key=memory_api_key,
        auto_store=True,
        store_human=True,
        store_ai=False
    )
    
    # Create LLM
    llm = ChatOpenAI(model=llm_model, temperature=0.7)
    
    # Create chain
    chain = ConversationChain(
        llm=llm,
        memory=memory,
        prompt=COGNITIVE_MEMORY_PROMPT,
        verbose=False
    )
    
    return chain


def main():
    """Demo the LangChain integration."""
    
    print("=" * 60)
    print("LangChain Integration with Cognitive Memory Layer")
    print("=" * 60)
    
    if not os.getenv("OPENAI_API_KEY"):
        print("\nError: OPENAI_API_KEY not set")
        print("  export OPENAI_API_KEY=sk-...")
        return
    
    # Create a chain with memory
    print("\nCreating conversation chain with persistent memory...")
    chain = create_memory_chain(
        scope_id="langchain-demo-session",
        scope="session",
        llm_model="gpt-4o-mini"
    )
    
    print("\nType 'quit' to exit.\n")
    print("Bot: Hello! I'm a LangChain-powered assistant with long-term memory. Tell me about yourself!")
    print()
    
    try:
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nBot: Goodbye! I'll remember our conversation.")
                break
            
            if not user_input:
                continue
            
            # Use the chain
            response = chain.predict(input=user_input)
            print(f"\nBot: {response}\n")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")


# Example of using the memory class directly
def example_direct_usage():
    """Show direct usage of CognitiveMemory class."""
    
    print("\n--- Direct Usage Example ---\n")
    
    memory = CognitiveMemory(
        scope="session",
        scope_id="direct-usage-demo-session",
        auto_store=True
    )
    
    # Simulate storing conversation
    memory.save_context(
        inputs={"input": "My favorite color is blue and I love hiking."},
        outputs={"response": "That's great! Blue is a calming color."}
    )
    
    # Retrieve relevant memories
    memories = memory.load_memory_variables({"input": "What are my hobbies?"})
    print(f"Retrieved memories:\n{memories['history']}")
    
    # Clean up
    memory.memory_client.close()


if __name__ == "__main__":
    main()
    # Uncomment to run direct usage example:
    # example_direct_usage()
