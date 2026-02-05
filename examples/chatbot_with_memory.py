"""
Complete Chatbot with Memory - Cognitive Memory Layer

A full-featured chatbot that demonstrates all memory capabilities:
- Automatic context injection before every response
- Intelligent memory storage based on conversation
- Memory management commands
- Works with any LLM provider

This example can be used as a template for production chatbots.

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api
    
    2. Install dependencies:
       pip install openai httpx
    
    3. Set your OpenAI API key:
       export OPENAI_API_KEY=sk-...
"""

import os
import re
from typing import Optional, List, Tuple
from dataclasses import dataclass
from datetime import datetime
from openai import OpenAI
from memory_client import CognitiveMemoryClient


@dataclass
class ConversationTurn:
    """A single turn in the conversation."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime


class MemoryPoweredChatbot:
    """
    A chatbot that intelligently uses memory to provide personalized responses.
    
    Features:
    - Automatic memory retrieval before each response
    - Intelligent extraction of memorable information
    - Support for memory commands (!remember, !forget, !stats)
    - Configurable memory injection strategies
    """
    
    def __init__(
        self,
        session_id: str,
        llm_api_key: Optional[str] = None,
        memory_api_url: str = "http://localhost:8000",
        memory_api_key: Optional[str] = None,
        llm_model: str = "gpt-4o-mini",
        auto_remember: bool = True,
        memory_context_tokens: int = 1500,
        scope: str = "session"
    ):
        """
        Initialize the chatbot.
        
        Args:
            session_id: Unique identifier for the session (used as scope_id)
            llm_api_key: OpenAI API key (or set OPENAI_API_KEY env var)
            memory_api_url: URL of the Cognitive Memory Layer API
            memory_api_key: API key for memory service (default: AUTH__API_KEY from env)
            llm_model: LLM model to use
            auto_remember: Automatically extract and store memorable info
            memory_context_tokens: Max tokens for memory context
            scope: Memory scope to use (session, agent, namespace, global)
        """
        self.session_id = session_id
        self.scope = scope
        self.scope_id = session_id
        self.llm_model = llm_model
        self.auto_remember = auto_remember
        self.memory_context_tokens = memory_context_tokens
        
        # Initialize clients
        self.llm = OpenAI(api_key=llm_api_key or os.getenv("OPENAI_API_KEY"))
        self.memory = CognitiveMemoryClient(
            base_url=memory_api_url,
            api_key=memory_api_key or os.environ.get("AUTH__API_KEY", "")
        )
        
        # Conversation history (current session only)
        self.history: List[ConversationTurn] = []
        
        # Base system prompt
        self.base_system_prompt = """You are a helpful, friendly assistant with access to memories about the user.

Use the provided memory context to personalize your responses. If you know something about the user, use it naturally in conversation without explicitly mentioning that you "remember" it unless relevant.

Be conversational and helpful. If you don't know something about the user, it's okay to ask."""
    
    def _handle_command(self, message: str) -> Optional[str]:
        """
        Handle special memory commands.
        
        Commands:
            !remember <info> - Explicitly store information
            !forget <query> - Forget matching memories
            !stats - Show memory statistics
            !search <query> - Search memories
            !clear - Clear session history
        """
        message = message.strip()
        
        if message.startswith("!remember "):
            info = message[10:].strip()
            if info:
                result = self.memory.write(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    content=info,
                    memory_type="semantic_fact"
                )
                return f"✓ Stored: {info}" if result.success else f"✗ Failed: {result.message}"
            return "Usage: !remember <information to store>"
        
        elif message.startswith("!forget "):
            query = message[8:].strip()
            if query:
                result = self.memory.forget(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    query=query,
                    action="delete"
                )
                count = result.get("affected_count", 0)
                return f"✓ Forgot {count} memories matching '{query}'"
            return "Usage: !forget <query>"
        
        elif message == "!stats":
            stats = self.memory.stats(self.scope, self.scope_id)
            return f"""📊 Memory Statistics for {self.scope}/{self.scope_id}:
- Total memories: {stats.total_memories}
- Active memories: {stats.active_memories}
- Average confidence: {stats.avg_confidence:.0%}
- By type: {stats.by_type}"""
        
        elif message.startswith("!search "):
            query = message[8:].strip()
            if query:
                result = self.memory.read(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    query=query,
                    max_results=5
                )
                if result.memories:
                    lines = [f"🔍 Found {result.total_count} memories:"]
                    for mem in result.memories[:5]:
                        lines.append(f"  [{mem.type}] {mem.text[:80]}...")
                    return "\n".join(lines)
                return "No memories found matching that query."
            return "Usage: !search <query>"
        
        elif message == "!clear":
            self.history.clear()
            return "✓ Session history cleared (long-term memory preserved)"
        
        elif message == "!help":
            return """📚 Memory Commands:
- !remember <info> - Store information explicitly
- !forget <query> - Forget matching memories  
- !stats - Show memory statistics
- !search <query> - Search your memories
- !clear - Clear session history
- !help - Show this help"""
        
        return None
    
    def _get_memory_context(self, message: str) -> str:
        """Retrieve relevant memory context for the current message."""
        try:
            result = self.memory.read(
                scope=self.scope,
                scope_id=self.scope_id,
                query=message,
                max_results=10,
                format="llm_context"
            )
            return result.llm_context or ""
        except Exception as e:
            print(f"Warning: Could not retrieve memories: {e}")
            return ""
    
    def _extract_memorable_info(self, message: str, response: str) -> List[Tuple[str, str]]:
        """
        Use LLM to extract memorable information from the conversation.
        
        Returns list of (content, memory_type) tuples.
        """
        extraction_prompt = f"""Analyze this conversation turn and extract any important information worth remembering about the user.

User message: {message}
Assistant response: {response}

Extract information that would be useful to remember for future conversations, such as:
- Personal facts (name, occupation, location)
- Preferences (likes, dislikes, favorites)
- Constraints (allergies, restrictions, requirements)
- Goals or plans

Respond in JSON format with an array of objects:
[{{"content": "extracted fact", "type": "semantic_fact|preference|constraint|hypothesis"}}]

If nothing worth remembering, respond with: []

Only extract clear, factual information. Don't make assumptions."""

        try:
            response = self.llm.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": extraction_prompt}],
                response_format={"type": "json_object"},
                max_tokens=500
            )
            
            import json
            data = json.loads(response.choices[0].message.content)
            
            # Handle both array and object with array
            if isinstance(data, list):
                return [(item["content"], item["type"]) for item in data]
            elif isinstance(data, dict) and "items" in data:
                return [(item["content"], item["type"]) for item in data["items"]]
            elif isinstance(data, dict) and len(data) == 0:
                return []
            
        except Exception as e:
            print(f"Warning: Could not extract memories: {e}")
        
        return []
    
    def chat(self, message: str) -> str:
        """
        Process a user message and return a response.
        
        Args:
            message: The user's message
            
        Returns:
            The assistant's response
        """
        # Check for commands
        command_result = self._handle_command(message)
        if command_result is not None:
            return command_result
        
        # Get relevant memory context
        memory_context = self._get_memory_context(message)
        
        # Build system prompt with memory
        system_prompt = self.base_system_prompt
        if memory_context:
            system_prompt += f"\n\n--- MEMORY CONTEXT ---\n{memory_context}"
        
        # Build messages for LLM
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add recent conversation history (last 10 turns)
        for turn in self.history[-10:]:
            messages.append({"role": turn.role, "content": turn.content})
        
        # Add current message
        messages.append({"role": "user", "content": message})
        
        # Get response from LLM
        response = self.llm.chat.completions.create(
            model=self.llm_model,
            messages=messages,
            max_tokens=1000
        )
        
        assistant_response = response.choices[0].message.content
        
        # Store conversation in history
        self.history.append(ConversationTurn("user", message, datetime.now()))
        self.history.append(ConversationTurn("assistant", assistant_response, datetime.now()))
        
        # Auto-extract and store memorable information
        if self.auto_remember:
            memories_to_store = self._extract_memorable_info(message, assistant_response)
            for content, memory_type in memories_to_store:
                try:
                    self.memory.write(
                        scope=self.scope,
                        scope_id=self.scope_id,
                        content=content,
                        memory_type=memory_type
                    )
                    print(f"  [Auto-stored: {memory_type}] {content[:50]}...")
                except Exception as e:
                    print(f"  Warning: Could not store memory: {e}")
        
        return assistant_response
    
    def close(self):
        """Clean up resources."""
        self.memory.close()


def main():
    """Run the chatbot in interactive mode."""
    
    print("=" * 60)
    print("Memory-Powered Chatbot")
    print("=" * 60)
    print("""
This chatbot automatically:
- Retrieves relevant memories before each response
- Extracts and stores important information from conversations
- Supports memory commands (type !help for list)

Type 'quit' to exit.
""")
    
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        print("  export OPENAI_API_KEY=sk-...")
        return
    
    # Create chatbot with session scope
    chatbot = MemoryPoweredChatbot(
        session_id="chatbot-demo-session",
        auto_remember=True,
        scope="session"
    )
    
    try:
        print("Bot: Hello! I'm your personal assistant. I remember things about you across our conversations. How can I help you today?")
        print("     (Type !help for memory commands)\n")
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nBot: Goodbye! I'll remember our conversation.")
                break
            
            if not user_input:
                continue
            
            response = chatbot.chat(user_input)
            print(f"\nBot: {response}\n")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        chatbot.close()


if __name__ == "__main__":
    main()
