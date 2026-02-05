"""
Anthropic Claude Tool Use Integration - Cognitive Memory Layer

This example demonstrates how to use the Cognitive Memory Layer
as tools with Anthropic Claude's tool use feature.

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api
    
    2. Install dependencies:
       pip install anthropic httpx
    
    3. Set your Anthropic API key:
       export ANTHROPIC_API_KEY=sk-ant-...
"""

import json
import os
from typing import Optional, List, Dict, Any
from anthropic import Anthropic
from memory_client import CognitiveMemoryClient


# ==============================================
# Tool Definitions for Anthropic Claude
# ==============================================

MEMORY_TOOLS = [
    {
        "name": "memory_write",
        "description": "Store new information in the session's long-term memory. Use this when the user shares important personal information, preferences, facts about themselves, or when you learn something significant about them.",
        "input_schema": {
            "type": "object",
            "properties": {
                "content": {
                    "type": "string",
                    "description": "The information to store. Be specific and factual."
                },
                "memory_type": {
                    "type": "string",
                    "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis"],
                    "description": "Type of memory. Use 'semantic_fact' for facts, 'preference' for preferences, 'constraint' for rules that must be followed."
                }
            },
            "required": ["content"]
        }
    },
    {
        "name": "memory_read",
        "description": "Retrieve relevant memories to inform your response. Use this before answering questions about the user's preferences, history, or personal information.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Natural language query describing what information you need"
                },
                "memory_types": {
                    "type": "array",
                    "items": {
                        "type": "string",
                        "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis"]
                    },
                    "description": "Optional: filter by specific memory types"
                }
            },
            "required": ["query"]
        }
    },
    {
        "name": "memory_update",
        "description": "Update or provide feedback on an existing memory. Use when the user corrects information or confirms a fact.",
        "input_schema": {
            "type": "object",
            "properties": {
                "memory_id": {
                    "type": "string",
                    "description": "UUID of the memory to update"
                },
                "feedback": {
                    "type": "string",
                    "enum": ["correct", "incorrect", "outdated"],
                    "description": "Feedback type: 'correct' reinforces, 'incorrect' marks invalid, 'outdated' marks as historical"
                }
            },
            "required": ["memory_id", "feedback"]
        }
    },
    {
        "name": "memory_forget",
        "description": "Forget specific information when the user explicitly requests deletion.",
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Description of what to forget"
                },
                "action": {
                    "type": "string",
                    "enum": ["delete", "archive"],
                    "description": "Action: 'delete' removes permanently, 'archive' keeps but hides"
                }
            },
            "required": ["query"]
        }
    }
]


class ClaudeMemoryAssistant:
    """
    An Anthropic Claude-powered assistant with persistent memory.
    
    Uses Claude's tool use feature to interact with the Cognitive Memory Layer.
    """
    
    def __init__(
        self,
        session_id: str,
        anthropic_api_key: Optional[str] = None,
        memory_api_url: str = "http://localhost:8000",
        memory_api_key: str = "demo-key-123",
        model: str = "claude-sonnet-4-20250514",
        scope: str = "session"
    ):
        self.session_id = session_id
        self.scope = scope
        self.scope_id = session_id
        self.model = model
        
        # Initialize Anthropic client
        self.anthropic = Anthropic(api_key=anthropic_api_key or os.getenv("ANTHROPIC_API_KEY"))
        
        # Initialize memory client
        self.memory = CognitiveMemoryClient(
            base_url=memory_api_url,
            api_key=memory_api_key
        )
        
        # Conversation history
        self.messages: List[Dict[str, Any]] = []
        
        # System prompt
        self.system_prompt = """You are a helpful assistant with long-term memory capabilities.

You have access to memory tools that allow you to remember information about the user across conversations.

Guidelines:
1. Use memory_read BEFORE answering questions about the user's preferences, history, or personal information.
2. Use memory_write when the user shares important information (name, preferences, facts about themselves).
3. Use memory_update when the user corrects or confirms information.
4. Use memory_forget only when the user explicitly asks you to forget something.
5. For constraints (allergies, restrictions), ALWAYS respect them.

Be natural and conversational. Don't mention the memory system explicitly unless asked."""
    
    def _execute_tool(self, tool_name: str, tool_input: dict) -> str:
        """Execute a memory tool and return the result."""
        try:
            if tool_name == "memory_write":
                result = self.memory.write(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    content=tool_input["content"],
                    memory_type=tool_input.get("memory_type")
                )
                return json.dumps({
                    "success": result.success,
                    "message": result.message
                })
            
            elif tool_name == "memory_read":
                result = self.memory.read(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    query=tool_input["query"],
                    memory_types=tool_input.get("memory_types"),
                    format="llm_context"
                )
                return result.llm_context or "No relevant memories found."
            
            elif tool_name == "memory_update":
                result = self.memory.update(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    memory_id=tool_input["memory_id"],
                    feedback=tool_input["feedback"]
                )
                return json.dumps(result)
            
            elif tool_name == "memory_forget":
                result = self.memory.forget(
                    scope=self.scope,
                    scope_id=self.scope_id,
                    query=tool_input["query"],
                    action=tool_input.get("action", "archive")
                )
                return json.dumps(result)
            
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        Claude may use memory tools automatically.
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Keep trying until we get a final response (handle tool calls)
        while True:
            response = self.anthropic.messages.create(
                model=self.model,
                max_tokens=4096,
                system=self.system_prompt,
                tools=MEMORY_TOOLS,
                messages=self.messages
            )
            
            # Check if Claude wants to use tools
            if response.stop_reason == "tool_use":
                # Add assistant response with tool use
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                # Process each tool use block
                tool_results = []
                for block in response.content:
                    if block.type == "tool_use":
                        tool_name = block.name
                        tool_input = block.input
                        
                        print(f"  [Tool: {tool_name}] {tool_input}")
                        
                        result = self._execute_tool(tool_name, tool_input)
                        
                        tool_results.append({
                            "type": "tool_result",
                            "tool_use_id": block.id,
                            "content": result
                        })
                
                # Add tool results
                self.messages.append({
                    "role": "user",
                    "content": tool_results
                })
            
            else:
                # No more tool calls, extract text response
                text_response = ""
                for block in response.content:
                    if hasattr(block, "text"):
                        text_response += block.text
                
                self.messages.append({
                    "role": "assistant",
                    "content": response.content
                })
                
                return text_response
    
    def close(self):
        """Clean up resources."""
        self.memory.close()


def main():
    """Interactive chat session with Claude memory assistant."""
    
    print("=" * 60)
    print("Anthropic Claude Tool Use with Cognitive Memory Layer")
    print("=" * 60)
    print("\nThis assistant remembers information across conversations.")
    print("Try telling it about yourself, then ask questions later!")
    print("Type 'quit' to exit.\n")
    
    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("Warning: ANTHROPIC_API_KEY not set. Please set it:")
        print("  export ANTHROPIC_API_KEY=sk-ant-...")
        return
    
    # Create assistant with session scope
    assistant = ClaudeMemoryAssistant(
        session_id="claude-demo-session",
        model="claude-sonnet-4-20250514",  # or claude-3-opus-20240229
        scope="session"
    )
    
    try:
        print("Claude: Hello! I'm your memory-enabled assistant powered by Claude. I can remember things about you across our conversations. What would you like to share or ask?")
        print()
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nClaude: Goodbye! I'll remember our conversation for next time.")
                break
            
            if not user_input:
                continue
            
            response = assistant.chat(user_input)
            print(f"\nClaude: {response}\n")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        assistant.close()


# Example of programmatic usage
def example_conversation():
    """Example showing a complete conversation flow."""
    
    print("\n--- Example Conversation with Claude ---\n")
    
    assistant = ClaudeMemoryAssistant(
        session_id="claude-example-session",
        model="claude-sonnet-4-20250514",
        scope="session"
    )
    
    conversations = [
        "Hello! I'm Marcus and I work as a chef in New York.",
        "I specialize in Italian cuisine, especially pasta dishes.",
        "I'm vegetarian and allergic to tree nuts.",
        "What can you tell me about my profession?",
        "What dietary restrictions should you keep in mind for me?",
    ]
    
    for message in conversations:
        print(f"User: {message}")
        response = assistant.chat(message)
        print(f"Claude: {response}\n")
    
    assistant.close()


if __name__ == "__main__":
    main()
    # Uncomment to run the example conversation:
    # example_conversation()
