# Cognitive Memory Layer - Examples

This folder contains working code examples demonstrating how to use the Cognitive Memory Layer with LLMs and as a standalone service.

## Prerequisites

Before running any examples:

1. **Start the API Server**:
   ```bash
   # From the project root
   docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
   docker compose -f docker/docker-compose.yml up api
   ```

2. **Install Python dependencies**:
   ```bash
   pip install httpx openai anthropic langchain langchain-openai
   ```

3. **Set API keys** (for LLM examples):
   ```bash
   export OPENAI_API_KEY=sk-...
   export ANTHROPIC_API_KEY=sk-ant-...
   ```

## Examples Overview

### 1. Memory Client (`memory_client.py`)

A reusable Python client for the Cognitive Memory Layer API. Used by other examples.

```python
from memory_client import CognitiveMemoryClient

client = CognitiveMemoryClient(api_key="demo-key-123")

# Store a memory with explicit scope
client.write(
    scope="session",
    scope_id="session-123",
    content="User prefers vegetarian food",
    memory_type="preference"
)

# Retrieve memories
result = client.read(
    scope="session",
    scope_id="session-123",
    query="dietary preferences",
    format="llm_context"
)
print(result.llm_context)
```

### 2. Basic Usage (`basic_usage.py`)

Simple example showing core operations without LLM dependencies:
- Writing different memory types
- Reading and querying memories
- Updating memories with feedback
- Forgetting memories
- Getting statistics

```bash
python examples/basic_usage.py
```

### 3. Standalone Demo (`standalone_demo.py`)

Interactive demo that doesn't require any LLM API keys. Great for:
- Understanding the API
- Testing the memory system
- Development and debugging

```bash
python examples/standalone_demo.py
```

### 4. OpenAI Tool Calling (`openai_tool_calling.py`)

Integration with OpenAI's function calling (tool use) feature. The LLM autonomously decides when to:
- Store new information
- Retrieve relevant context
- Update memories
- Forget information

```bash
python examples/openai_tool_calling.py
```

**Interactive chat example:**
```
You: My name is Alice and I work as a data scientist.
  [Tool: memory_write] {"content": "User's name is Alice...", "memory_type": "semantic_fact"}
Assistant: Nice to meet you, Alice! Data science is a fascinating field...

You: What do you know about me?
  [Tool: memory_read] {"query": "user information"}
Assistant: Based on what you've told me, your name is Alice and you work as a data scientist...
```

### 5. Anthropic Claude Tool Use (`anthropic_tool_calling.py`)

Same concept as OpenAI but using Anthropic Claude's tool use feature.

```bash
python examples/anthropic_tool_calling.py
```

### 6. Complete Chatbot (`chatbot_with_memory.py`)

A full-featured chatbot demonstrating:
- Automatic context injection before responses
- Intelligent extraction of memorable information
- Memory management commands (!remember, !forget, !stats)
- Works with any LLM

```bash
python examples/chatbot_with_memory.py
```

**Commands:**
- `!remember <info>` - Explicitly store information
- `!forget <query>` - Forget matching memories
- `!stats` - Show memory statistics
- `!search <query>` - Search memories
- `!clear` - Clear session history
- `!help` - Show help

### 7. LangChain Integration (`langchain_integration.py`)

Custom LangChain memory class that uses the Cognitive Memory Layer:

```python
from langchain_openai import ChatOpenAI
from langchain.chains import ConversationChain
from langchain_integration import CognitiveMemory

memory = CognitiveMemory(scope="session", scope_id="session-123")
llm = ChatOpenAI()
chain = ConversationChain(llm=llm, memory=memory)

response = chain.predict(input="My favorite color is blue")
```

```bash
python examples/langchain_integration.py
```

### 8. Async Usage (`async_usage.py`)

Asynchronous patterns for high-performance applications:
- Concurrent reads and writes
- Batch processing
- Pipeline patterns
- Timeout handling

```bash
python examples/async_usage.py
```

## Memory Scopes

The API uses a scope-based system for organizing memories:

| Scope | Use Case | Example |
|-------|----------|---------|
| `session` | Single conversation session | `session-abc123` |
| `agent` | Agent-specific context | `code-reviewer-agent` |
| `namespace` | Project or team level | `project-acme` |
| `global` | Shared across all contexts | `global-knowledge` |
| `user` | User-specific memories | `user-12345` |

## Memory Types

When storing memories, use the appropriate type:

| Type | Use Case | Example |
|------|----------|---------|
| `semantic_fact` | Permanent facts | "User's name is Alice" |
| `preference` | User preferences | "User prefers dark mode" |
| `constraint` | Must-follow rules | "User is allergic to peanuts" |
| `episodic_event` | Specific events | "User mentioned trip to Paris on Jan 15" |
| `hypothesis` | Uncertain info | "User might be interested in cooking" |
| `task_state` | Current task progress | "Step 3 of 5 complete" |
| `procedure` | How to do things | "To export data: click File > Export" |

## API Quick Reference

### Write Memory
```python
client.write(
    scope="session",
    scope_id="session-123",
    content="User prefers morning meetings",
    memory_type="preference",  # optional
    metadata={"source": "calendar"}  # optional
)
```

### Read Memory
```python
result = client.read(
    scope="session",
    scope_id="session-123",
    query="meeting preferences",
    max_results=10,
    format="llm_context"  # or "packet"
)
print(result.llm_context)  # Ready for LLM system prompt
```

### Update Memory
```python
client.update(
    scope="session",
    scope_id="session-123",
    memory_id="uuid-here",
    feedback="correct"  # or "incorrect", "outdated"
)
```

### Forget Memory
```python
client.forget(
    scope="session",
    scope_id="session-123",
    query="old address",
    action="delete"  # or "archive"
)
```

### Get Stats
```python
stats = client.stats("session", "session-123")
print(f"Total: {stats.total_memories}")
```

## Tool Definitions for LLMs

Copy these tool definitions for your LLM integration:

### OpenAI Format
```json
{
  "type": "function",
  "function": {
    "name": "memory_write",
    "description": "Store information in long-term memory",
    "parameters": {
      "type": "object",
      "properties": {
        "content": {"type": "string"},
        "memory_type": {"type": "string", "enum": ["semantic_fact", "preference", "constraint"]}
      },
      "required": ["content"]
    }
  }
}
```

### Anthropic Format
```json
{
  "name": "memory_write",
  "description": "Store information in long-term memory",
  "input_schema": {
    "type": "object",
    "properties": {
      "content": {"type": "string"},
      "memory_type": {"type": "string", "enum": ["semantic_fact", "preference", "constraint"]}
    },
    "required": ["content"]
  }
}
```

## Best Practices

1. **Read before responding**: Always retrieve relevant memories before generating responses about the user.

2. **Use appropriate memory types**: Constraints are never auto-forgotten, hypotheses need confirmation.

3. **Respect constraints**: Always check for and respect constraint memories (allergies, restrictions).

4. **Confirm uncertain info**: Mark inferences as hypotheses and confirm with the user.

5. **Use LLM context format**: Request `format="llm_context"` to get pre-formatted markdown ready for system prompts.

6. **Handle errors gracefully**: Memory operations may fail; always have fallback behavior.

## Troubleshooting

### "Could not connect to API"
```bash
# Check if API is running
curl http://localhost:8000/api/v1/health

# Start the API
docker compose -f docker/docker-compose.yml up -d postgres neo4j redis
docker compose -f docker/docker-compose.yml up api
```

### "API key required"
Make sure to include the `X-API-Key` header. Default key is `demo-key-123`.

### "No memories found"
- Check the scope and scope_id are correct
- Verify memories were stored successfully
- Try a broader query

### Memory not being stored
- Check the write gate threshold (default 0.3)
- Very short or trivial content may be filtered
- Use explicit memory_type to ensure storage

## Further Reading

- [Usage Documentation](../ProjectPlan/UsageDocumentation.md) - Complete API reference
- [README](../README.md) - Architecture and research foundations
- [API Docs](http://localhost:8000/docs) - Interactive OpenAPI documentation (when server is running)
