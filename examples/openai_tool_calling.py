"""
OpenAI Tool Calling Integration - Cognitive Memory Layer

This example demonstrates how to use the Cognitive Memory Layer
as tools with OpenAI's function calling (tool use) feature.

The LLM can autonomously decide when to:
- Store new information in memory
- Retrieve relevant context before answering
- Update memories based on user corrections
- Forget information when requested

Prerequisites:
    1. Start the API server:
       docker compose -f docker/docker-compose.yml up api
    
    2. Install dependencies:
       pip install openai httpx
    
    3. Set your OpenAI API key:
       export OPENAI_API_KEY=sk-...
"""

import json
import os
from typing import Optional
from openai import OpenAI
from memory_client import CognitiveMemoryClient


# ==============================================
# Tool Definitions for OpenAI Function Calling
# ==============================================

MEMORY_TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "memory_write",
            "description": "Store important information in long-term memory. The system automatically manages context. Use when the user shares personal information, preferences, or significant facts.",
            "parameters": {
                "type": "object",
                "properties": {
                    "content": {
                        "type": "string",
                        "description": "The information to store. Be specific and factual."
                    },
                    "memory_type": {
                        "type": "string",
                        "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis", "conversation", "message", "tool_result", "reasoning_step", "scratch", "knowledge", "observation", "plan"],
                        "description": "Type of memory. Use 'semantic_fact' for facts, 'preference' for preferences, 'constraint' for rules that must be followed."
                    }
                },
                "required": ["content"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_read",
            "description": "Retrieve relevant memories. Usually automatic, but call explicitly for specific queries (e.g. user preferences, history, or personal information).",
            "parameters": {
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
                            "enum": ["episodic_event", "semantic_fact", "preference", "task_state", "procedure", "constraint", "hypothesis", "conversation", "message", "tool_result", "reasoning_step", "scratch", "knowledge", "observation", "plan"]
                        },
                        "description": "Optional: filter by specific memory types"
                    }
                },
                "required": ["query"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_update",
            "description": "Update or provide feedback on an existing memory. Use when the user corrects information or confirms a fact.",
            "parameters": {
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
        }
    },
    {
        "type": "function",
        "function": {
            "name": "memory_forget",
            "description": "Forget specific information when the user explicitly requests deletion.",
            "parameters": {
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
    }
]


class MemoryEnabledAssistant:
    """
    An OpenAI-powered assistant with persistent memory.
    
    The assistant automatically uses memory tools to:
    - Remember user information across conversations
    - Retrieve relevant context before answering
    - Update its knowledge based on corrections
    """
    
    def __init__(
        self,
        session_id: str,
        openai_api_key: Optional[str] = None,
        memory_api_url: str = "http://localhost:8000",
        memory_api_key: Optional[str] = None,
        model: str = "gpt-4o-mini",
    ):
        self.session_id = session_id
        self.model = model
        
        # Initialize OpenAI client
        self.openai = OpenAI(api_key=openai_api_key or os.getenv("OPENAI_API_KEY"))
        
        # Initialize memory client
        self.memory = CognitiveMemoryClient(
            base_url=memory_api_url,
            api_key=memory_api_key or os.environ.get("AUTH__API_KEY", "")
        )
        
        # Conversation history
        self.messages = [
            {
                "role": "system",
                "content": """You are a helpful assistant with long-term memory capabilities.

You have access to memory tools that allow you to remember information about the user across conversations.

Guidelines:
1. Use memory_read BEFORE answering questions about the user's preferences, history, or personal information.
2. Use memory_write when the user shares important information (name, preferences, facts about themselves).
3. Use memory_update when the user corrects or confirms information.
4. Use memory_forget only when the user explicitly asks you to forget something.
5. For constraints (allergies, restrictions), ALWAYS respect them.

Be natural and conversational. Don't mention the memory system explicitly unless asked."""
            }
        ]
    
    def _execute_tool(self, tool_name: str, arguments: dict) -> str:
        """Execute a memory tool and return the result."""
        try:
            if tool_name == "memory_write":
                result = self.memory.write(
                    arguments["content"],
                    session_id=self.session_id,
                    context_tags=["conversation"],
                    memory_type=arguments.get("memory_type"),
                )
                return json.dumps({
                    "success": result.success,
                    "message": result.message
                })
            elif tool_name == "memory_read":
                result = self.memory.read(
                    arguments["query"],
                    memory_types=arguments.get("memory_types"),
                    format="llm_context"
                )
                return result.llm_context or "No relevant memories found."
            elif tool_name == "memory_update":
                result = self.memory.update(
                    memory_id=arguments["memory_id"],
                    feedback=arguments["feedback"]
                )
                return json.dumps(result)
            elif tool_name == "memory_forget":
                result = self.memory.forget(
                    query=arguments["query"],
                    action=arguments.get("action", "archive")
                )
                return json.dumps(result)
            
            else:
                return json.dumps({"error": f"Unknown tool: {tool_name}"})
                
        except Exception as e:
            return json.dumps({"error": str(e)})
    
    def chat(self, user_message: str) -> str:
        """
        Send a message and get a response.
        The assistant may use memory tools automatically.
        """
        # Add user message
        self.messages.append({"role": "user", "content": user_message})
        
        # Keep trying until we get a final response (handle tool calls)
        while True:
            response = self.openai.chat.completions.create(
                model=self.model,
                messages=self.messages,
                tools=MEMORY_TOOLS,
                tool_choice="auto"
            )
            
            assistant_message = response.choices[0].message
            
            # Check if the model wants to use tools
            if assistant_message.tool_calls:
                # Add assistant message with tool calls
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content,
                    "tool_calls": [
                        {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments
                            }
                        }
                        for tc in assistant_message.tool_calls
                    ]
                })
                
                # Execute each tool call
                for tool_call in assistant_message.tool_calls:
                    tool_name = tool_call.function.name
                    arguments = json.loads(tool_call.function.arguments)
                    
                    print(f"  [Tool: {tool_name}] {arguments}")
                    
                    result = self._execute_tool(tool_name, arguments)
                    
                    # Add tool result
                    self.messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "content": result
                    })
            
            else:
                # No more tool calls, we have the final response
                self.messages.append({
                    "role": "assistant",
                    "content": assistant_message.content
                })
                return assistant_message.content
    
    def close(self):
        """Clean up resources."""
        self.memory.close()


def main():
    """Interactive chat session with memory-enabled assistant."""
    
    print("=" * 60)
    print("OpenAI Tool Calling with Cognitive Memory Layer")
    print("=" * 60)
    print("\nThis assistant remembers information across conversations.")
    print("Try telling it about yourself, then ask questions later!")
    print("Type 'quit' to exit.\n")
    
    # Check for OpenAI API key
    if not os.getenv("OPENAI_API_KEY"):
        print("Warning: OPENAI_API_KEY not set. Please set it:")
        print("  export OPENAI_API_KEY=sk-...")
        return
    
    # Create assistant with session scope
    assistant = MemoryEnabledAssistant(
        session_id="openai-demo-session",
        model="gpt-4o-mini",  # or "gpt-4o" for better reasoning
    )
    
    try:
        # Example conversation
        print("Assistant: Hello! I'm your memory-enabled assistant. I can remember things about you across our conversations. What would you like to tell me or ask?")
        print()
        
        while True:
            user_input = input("You: ").strip()
            
            if user_input.lower() in ["quit", "exit", "bye"]:
                print("\nAssistant: Goodbye! I'll remember our conversation for next time.")
                break
            
            if not user_input:
                continue
            
            response = assistant.chat(user_input)
            print(f"\nAssistant: {response}\n")
            
    except KeyboardInterrupt:
        print("\n\nGoodbye!")
    finally:
        assistant.close()


# Example of programmatic usage
def example_programmatic():
    """Example showing programmatic usage of the assistant."""
    
    assistant = MemoryEnabledAssistant(
        session_id="demo-programmatic-session",
        model="gpt-4o-mini",
    )
    
    # Simulate a conversation
    conversations = [
        "Hi! My name is Sarah and I'm a data scientist.",
        "I prefer working with Python and I love hiking on weekends.",
        "I'm allergic to peanuts, please remember that.",
        "What do you know about me?",
        "Can you recommend a good programming language for me?",
    ]
    
    print("\n--- Programmatic Example ---\n")
    
    for message in conversations:
        print(f"User: {message}")
        response = assistant.chat(message)
        print(f"Assistant: {response}\n")
    
    assistant.close()


if __name__ == "__main__":
    main()
    # Uncomment to run programmatic example:
    # example_programmatic()
