# CML Memory -- Cognitive Memory Layer for OpenClaw

> Structured long-term memory powered by the Cognitive Memory Layer (CML) API.
> Store facts, preferences, constraints, and episodes across conversations.

## When to Use This Skill

Activate this skill whenever you need to:

- **Remember** something the user told you (preferences, facts, personal details).
- **Recall** previously stored information before answering a question.
- **Forget** information the user asks you to remove.
- **Track** important events, decisions, or conversations over time.

This skill gives you persistent, structured memory that survives across sessions and conversations. Use it proactively -- store important information as soon as you learn it, and retrieve context before answering questions about the user.

## Configuration

The following environment variables must be set (in your shell or OpenClaw config):

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `CML_BASE_URL` | Yes | `http://localhost:8000` | CML API server URL |
| `CML_API_KEY` | Yes | -- | API key for authentication |
| `CML_TENANT_ID` | No | `openclaw` | Tenant partition for memories |

## Memory Types

Choose the correct type when storing memories. This affects how memories are consolidated and retrieved.

| Type | When to Use | Examples |
|------|-------------|---------|
| `semantic_fact` | Permanent, factual information about the user or world | "User's name is Alice", "User works at Acme Corp" |
| `preference` | User likes, dislikes, and preferred ways of doing things | "Prefers dark mode", "Likes concise answers" |
| `constraint` | Hard rules, restrictions, allergies, must-follow policies | "Allergic to peanuts", "Never schedule before 9am" |
| `episodic_event` | Specific events, meetings, decisions with a time component | "Had a job interview on Feb 10", "Deployed v2.0 today" |
| `hypothesis` | Uncertain information that needs confirmation | "Might be moving to NYC", "Seems interested in Python" |

## API Operations

### 1. Process Turn (Recommended for Conversations)

The simplest way to use CML. Sends the user's message, retrieves relevant context, and stores the message -- all in one call. **Use this as your default approach.**

```bash
curl -s -X POST "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/turn" \
  -H "Authorization: Bearer ${CML_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "user_message": "I just got promoted to senior engineer at Acme Corp!",
    "session_id": "openclaw-main",
    "max_context_tokens": 1500
  }'
```

Response contains `memory_context` (inject into your prompt), `memories_retrieved`, and `memories_stored`.

### 2. Store a Memory

Use when you learn something specific and want to store it with a particular type.

```bash
curl -s -X POST "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/write" \
  -H "Authorization: Bearer ${CML_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "content": "User is allergic to shellfish",
    "memory_type": "constraint",
    "session_id": "openclaw-main",
    "context_tags": ["health", "dietary"],
    "namespace": "openclaw"
  }'
```

### 3. Retrieve Memories

Query for relevant memories before answering questions about the user.

```bash
curl -s -X POST "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/read" \
  -H "Authorization: Bearer ${CML_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "dietary restrictions and food preferences",
    "max_results": 10,
    "format": "llm_context"
  }'
```

Use `"format": "llm_context"` to get a pre-formatted string ready to inject into your prompt. Use `"format": "packet"` when you need structured data with individual memory items.

You can filter by memory type:

```bash
curl -s -X POST "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/read" \
  -H "Authorization: Bearer ${CML_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "user restrictions",
    "memory_types": ["constraint", "preference"],
    "format": "llm_context"
  }'
```

### 4. Update a Memory

Provide feedback when a memory is confirmed correct, found incorrect, or outdated.

```bash
curl -s -X POST "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/update" \
  -H "Authorization: Bearer ${CML_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "memory_id": "<uuid-from-read-response>",
    "feedback": "correct"
  }'
```

Valid feedback values: `correct`, `incorrect`, `outdated`.

### 5. Forget Memories

When the user asks you to forget something.

```bash
curl -s -X POST "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/forget" \
  -H "Authorization: Bearer ${CML_API_KEY}" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "old home address",
    "action": "delete"
  }'
```

Actions: `delete` (permanent), `archive` (soft-delete, recoverable), `silence` (suppress from results).

### 6. Memory Statistics

Check what you have stored.

```bash
curl -s "${CML_BASE_URL:-http://localhost:8000}/api/v1/memory/stats" \
  -H "Authorization: Bearer ${CML_API_KEY}"
```

## Behavioral Guidelines

### When to Store Memories

Store a memory immediately when the user shares:

- Personal information (name, location, job, family)
- Preferences or opinions ("I prefer...", "I like...", "I hate...")
- Constraints or restrictions ("I'm allergic to...", "Don't ever...", "I can't...")
- Important events ("I just...", "Yesterday I...", "Next week I...")
- Goals or plans ("I want to...", "I'm planning to...")

### When to Retrieve Memories

Retrieve context before responding when:

- The user asks a question about themselves or past conversations
- You need to personalize a recommendation or suggestion
- The conversation topic relates to previously stored information
- The user references something from a prior conversation ("remember when...")

### Best Practices

1. **Use `/memory/turn` as your default** -- it handles both retrieval and storage in one call.
2. **Use `/memory/write` with explicit types** when you identify a specific fact, preference, or constraint worth storing separately.
3. **Always include `session_id`** to track conversation lineage.
4. **Use `context_tags`** to categorize memories (e.g., `["work", "project-alpha"]`).
5. **Inject retrieved context** at the start of your reasoning when making personalized responses.
6. **Confirm before forgetting** -- ask the user to confirm deletion of important memories.
7. **Mark incorrect memories** with `feedback: "outdated"` instead of deleting, so CML can learn patterns.

## Example Conversation Flows

### Flow 1: Learning and Recalling a Preference

```
User: "I'm vegetarian and I really dislike spicy food"

Agent thinking:
  1. Store preference via /memory/write with type "preference"
  2. Store constraint via /memory/write with type "constraint"

[bash] curl -s -X POST "$CML_BASE_URL/api/v1/memory/write" \
  -H "Authorization: Bearer $CML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"User is vegetarian","memory_type":"preference","session_id":"openclaw-main","context_tags":["dietary"]}'

[bash] curl -s -X POST "$CML_BASE_URL/api/v1/memory/write" \
  -H "Authorization: Bearer $CML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"content":"User dislikes spicy food","memory_type":"constraint","session_id":"openclaw-main","context_tags":["dietary"]}'

Agent: "Got it! I'll remember that you're vegetarian and prefer non-spicy food."
```

Later:

```
User: "Can you suggest a restaurant for dinner tonight?"

Agent thinking:
  1. Retrieve dietary preferences and constraints from CML

[bash] curl -s -X POST "$CML_BASE_URL/api/v1/memory/read" \
  -H "Authorization: Bearer $CML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"query":"food preferences dietary restrictions","format":"llm_context"}'

  -> Returns: "User is vegetarian. User dislikes spicy food."

Agent: "Based on your preferences, here are some vegetarian-friendly restaurants
        with mild cuisine options..."
```

### Flow 2: Using Process Turn for Seamless Memory

```
User: "I just accepted the offer at Google! Starting March 1st."

Agent thinking:
  1. Use /memory/turn to store this and retrieve any related context

[bash] curl -s -X POST "$CML_BASE_URL/api/v1/memory/turn" \
  -H "Authorization: Bearer $CML_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"user_message":"I just accepted the offer at Google! Starting March 1st.","session_id":"openclaw-main"}'

  -> Returns memory_context with any prior job-search memories
  -> Automatically stores this as a new memory

Agent: "Congratulations on the Google offer! That's a great next step after
        your interviews last month. March 1st start date noted!"
```
