# OpenClaw + Cognitive Memory Layer

An [OpenClaw](https://openclaw.ai/) skill that gives your personal AI assistant persistent, structured long-term memory powered by the Cognitive Memory Layer (CML).

## Why CML for OpenClaw?

OpenClaw has built-in memory, but CML adds a richer layer:

| Feature | OpenClaw Built-in | CML |
|---------|------------------|-----|
| Memory types | General notes | Typed: facts, preferences, constraints, episodes, hypotheses |
| Knowledge graph | No | Yes -- entities and relationships in Neo4j |
| Consolidation | No | Automatic: merges, deduplicates, and strengthens memories |
| Forgetting | Manual only | Biologically-inspired decay with configurable policies |
| Multi-agent | Per-agent | Shared across agents via tenant partitioning |
| Search | Keyword | Semantic (embedding-based) + knowledge graph |

CML acts as a structured memory backend that any OpenClaw agent can use alongside or instead of the default memory system.

## Prerequisites

1. **OpenClaw** installed and running ([install guide](https://openclaw.ai/))
2. **CML server** running (Docker recommended):
   ```bash
   # Clone the CognitiveMemoryLayer repo (if you haven't already)
   git clone https://github.com/avinash-mall/CognitiveMemoryLayer.git
   cd CognitiveMemoryLayer

   # Copy and configure environment
   cp .env.example .env
   # Edit .env -- set AUTH__API_KEY, LLM_INTERNAL__MODEL, EMBEDDING_INTERNAL__MODEL, etc.

   # Start all services
   docker compose -f docker/docker-compose.yml up -d
   ```
3. **CML API key** -- the `AUTH__API_KEY` value from your `.env` file

## Installation

Copy the skill into your OpenClaw workspace:

### macOS / Linux

```bash
mkdir -p ~/.openclaw/workspace/skills/cml-memory
cp SKILL.md ~/.openclaw/workspace/skills/cml-memory/SKILL.md
```

### Windows (PowerShell)

```powershell
New-Item -ItemType Directory -Force -Path "$env:USERPROFILE\.openclaw\workspace\skills\cml-memory"
Copy-Item SKILL.md "$env:USERPROFILE\.openclaw\workspace\skills\cml-memory\SKILL.md"
```

## Configuration

Set these environment variables for the OpenClaw process (in your shell profile, OpenClaw config, or `.env`):

```bash
export CML_BASE_URL="http://localhost:8000"
export CML_API_KEY="your-api-key-here"
```

Or add them to your OpenClaw configuration file (`~/.openclaw/openclaw.json`):

```jsonc
{
  "env": {
    "CML_BASE_URL": "http://localhost:8000",
    "CML_API_KEY": "your-api-key-here"
  }
}
```

## Usage

Once installed, your OpenClaw agent will automatically use CML for memory operations. Try these interactions:

### Store information

> "My name is Alice and I work as a data scientist at Acme Corp."

The agent will store this as a `semantic_fact` in CML.

### Recall information

> "What do you know about me?"

The agent will query CML and respond with everything it has stored.

### Store preferences

> "I prefer Python over JavaScript, and I like concise code reviews."

The agent will store these as `preference` type memories.

### Store constraints

> "I'm allergic to latex. Never suggest latex gloves in lab protocols."

The agent will store this as a `constraint` -- the highest-priority memory type.

### Forget something

> "Please forget my old address."

The agent will call the forget endpoint to remove matching memories.

### Check memory stats

> "How many things do you remember about me?"

The agent will query the stats endpoint and report counts by type.

## Verification

Confirm the skill is loaded by asking your OpenClaw agent:

> "What skills do you have?"

You should see `cml-memory` in the list. Then verify CML connectivity:

> "Can you check if the memory system is working?"

The agent should call the CML health endpoint and report the status.

## Architecture

```
User (WhatsApp / Telegram / Discord / ...)
  |
  v
OpenClaw Agent
  |
  |-- [cml-memory skill]
  |     |
  |     v
  |   CML API (http://localhost:8000)
  |     |
  |     +-- PostgreSQL (memories, facts)
  |     +-- Neo4j (knowledge graph)
  |     +-- Redis (sessions, cache)
  |
  +-- OpenClaw built-in memory (optional, can coexist)
```

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Connection refused" errors | Verify CML server is running: `curl http://localhost:8000/api/v1/health` |
| "401 Unauthorized" | Check `CML_API_KEY` matches your server `AUTH__API_KEY` in CML `.env` |
| Skill not appearing | Verify `SKILL.md` has the required YAML frontmatter and is at `~/.openclaw/workspace/skills/cml-memory/SKILL.md` |
| No memories returned | Ensure you have written memories first; try a broader query |
| Slow responses | CML needs embedding generation on write; first write may be slower |

## Further Reading

- [CML Documentation](../../ProjectPlan/UsageDocumentation.md)
- [py-cml Python SDK](../../packages/py-cml/README.md)
- [CML API Reference](http://localhost:8000/docs) (when server is running)
- [OpenClaw Skills Guide](https://github.com/openclaw/openclaw)


