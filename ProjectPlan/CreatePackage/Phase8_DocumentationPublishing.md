# Phase 8: Documentation & Publishing

## Objective

Create comprehensive documentation, usage examples, and establish the GitHub release and PyPI publishing pipeline for py-cml. Ensure developers can discover, install, and integrate the package within minutes.

---

## Task 8.1: Package README

### Sub-Task 8.1.1: README.md

**Architecture**: The README is the package's landing page on both GitHub and PyPI. It should convey value in 10 seconds and get developers to a working example in 60 seconds.

**Implementation** (`packages/py-cml/README.md`):
```markdown
# py-cml

**Python SDK for [CognitiveMemoryLayer](https://github.com/<org>/CognitiveMemoryLayer)** — neuro-inspired memory for AI applications.

[![PyPI](https://img.shields.io/pypi/v/py-cml)](https://pypi.org/project/py-cml/)
[![Python](https://img.shields.io/pypi/pyversions/py-cml)](https://pypi.org/project/py-cml/)
[![License](https://img.shields.io/github/license/<org>/CognitiveMemoryLayer)](LICENSE)
[![Tests](https://img.shields.io/github/actions/workflow/status/<org>/CognitiveMemoryLayer/py-cml-test.yml)](https://github.com/<org>/CognitiveMemoryLayer/actions)

Give your AI applications human-like memory — store, retrieve, consolidate, and forget information just like the brain does.

## Installation

```bash
pip install py-cml
```

For embedded mode (no server required):
```bash
pip install py-cml[embedded]
```

## Quick Start

```python
from cml import CognitiveMemoryLayer

# Connect to a CML server
memory = CognitiveMemoryLayer(
    api_key="your-api-key",
    base_url="http://localhost:8000",
)

# Store a memory
memory.write("User prefers vegetarian food and lives in Paris.")

# Retrieve relevant memories
result = memory.read("What does the user eat?")
for item in result.memories:
    print(f"  {item.text} (relevance: {item.relevance:.2f})")

# Seamless chat integration
turn = memory.turn(
    user_message="What should I eat tonight?",
    session_id="chat-001",
)
# Inject turn.memory_context into your LLM prompt
```

## Async Support

```python
from cml import AsyncCognitiveMemoryLayer

async with AsyncCognitiveMemoryLayer(api_key="...") as memory:
    await memory.write("User is a Python developer.")
    result = await memory.read("programming skills")
```

## Embedded Mode (No Server)

```python
from cml import EmbeddedCognitiveMemoryLayer

# Zero-config: SQLite + local embeddings
async with EmbeddedCognitiveMemoryLayer() as memory:
    await memory.write("User prefers dark mode.")
    result = await memory.read("UI preferences")
```

## Features

- **Write** — Store information with automatic chunking, embedding, and entity extraction
- **Read** — Hybrid retrieval (vector + graph + lexical) with relevance ranking
- **Turn** — Seamless chat integration: auto-retrieve context, auto-store new info
- **Update** — Modify memories with feedback (correct/incorrect/outdated)
- **Forget** — Remove memories by ID, query, or time (with soft/hard delete)
- **Consolidate** — Migrate episodic memories to semantic knowledge
- **Active Forgetting** — Automatically prune low-relevance memories
- **Multi-tenant** — Tenant isolation with namespace support
- **Sync + Async** — Both synchronous and asynchronous clients
- **Embedded Mode** — Run in-process without a server (SQLite or PostgreSQL)

## Configuration

```python
# Environment variables
# CML_API_KEY=your-api-key
# CML_BASE_URL=http://localhost:8000
# CML_TENANT_ID=my-tenant

# Or direct initialization
memory = CognitiveMemoryLayer(
    api_key="...",
    base_url="http://localhost:8000",
    tenant_id="my-tenant",
    timeout=30.0,
    max_retries=3,
)
```

## Documentation

- [Getting Started](docs/getting-started.md)
- [API Reference](docs/api-reference.md)
- [Configuration](docs/configuration.md)
- [Examples](docs/examples.md)

## License

GPL-3.0 — See [LICENSE](LICENSE) for details.
```

**Pseudo-code for README sections**:
```
SECTIONS:
1. Title + badges (PyPI, Python versions, license, CI)
2. One-line description
3. Installation (pip install py-cml / py-cml[embedded])
4. Quick Start (sync, 5 lines)
5. Async Support (async context manager, 4 lines)
6. Embedded Mode (zero config, 4 lines)
7. Features (bullet list, 12 items)
8. Configuration (env vars + direct params)
9. Documentation links
10. License
```

---

## Task 8.2: API Reference Documentation

### Sub-Task 8.2.1: Getting Started Guide

**Implementation** (`docs/getting-started.md`):

**Pseudo-code for content**:
```
# Getting Started

## Prerequisites
- Python 3.11+
- A running CML server (or use embedded mode)

## Installation
pip install py-cml

## Connect to a Server
1. Start CML server (link to parent project docs)
2. Get your API key
3. Initialize client
4. Write first memory
5. Read it back

## Next Steps
- Read API Reference for all operations
- See Examples for common patterns
- Try Embedded Mode for serverless usage
```

### Sub-Task 8.2.2: API Reference

**Implementation** (`docs/api-reference.md`):

**Pseudo-code for content**:
```
# API Reference

## CognitiveMemoryLayer (Sync Client)

### Constructor
    CognitiveMemoryLayer(
        api_key: str | None = None,
        base_url: str = "http://localhost:8000",
        tenant_id: str = "default",
        config: CMLConfig | None = None,
        timeout: float = 30.0,
        max_retries: int = 3,
    )

### Methods

#### write(content, **kwargs) -> WriteResponse
    Store new information in memory.
    Parameters: content, context_tags, session_id, memory_type, namespace, metadata
    Returns: WriteResponse(success, memory_id, chunks_created, message)
    Raises: AuthenticationError, ValidationError

#### read(query, **kwargs) -> ReadResponse
    Retrieve relevant memories.
    Parameters: query, max_results, context_filter, memory_types, since, until, format
    Returns: ReadResponse(query, memories, facts, preferences, episodes, llm_context)

#### turn(user_message, **kwargs) -> TurnResponse
    Process a conversation turn with seamless memory.
    Parameters: user_message, assistant_response, session_id, max_context_tokens
    Returns: TurnResponse(memory_context, memories_retrieved, memories_stored)

#### update(memory_id, **kwargs) -> UpdateResponse
    Update an existing memory.
    Parameters: memory_id, text, confidence, importance, metadata, feedback
    Returns: UpdateResponse(success, memory_id, version)

#### forget(**kwargs) -> ForgetResponse
    Forget memories.
    Parameters: memory_ids, query, before, action
    Returns: ForgetResponse(success, affected_count)

#### stats() -> StatsResponse
    Get memory statistics.

#### health() -> HealthResponse
    Check server health.

#### get_context(query, max_results) -> str
    Convenience: get formatted LLM context string.

#### delete_all(confirm=False) -> int
    Delete all memories (GDPR). Requires confirm=True.

## AsyncCognitiveMemoryLayer (Async Client)
    Same methods as above, but all are async.

## EmbeddedCognitiveMemoryLayer
    Same methods as async client.
    Additional: consolidate(), run_forgetting()

## Models

### MemoryType (enum)
    EPISODIC_EVENT, SEMANTIC_FACT, PREFERENCE, ...

### MemoryStatus (enum)
    ACTIVE, SILENT, COMPRESSED, ARCHIVED, DELETED

### MemoryItem
    id, text, type, confidence, relevance, timestamp, metadata

### WriteResponse, ReadResponse, TurnResponse, UpdateResponse, ForgetResponse, StatsResponse

## Exceptions

### CMLError (base)
### AuthenticationError (401)
### AuthorizationError (403)
### NotFoundError (404)
### ValidationError (422)
### RateLimitError (429)
### ServerError (5xx)
### ConnectionError (network)
### TimeoutError (timeout)

## Configuration

### CMLConfig
    api_key, base_url, tenant_id, timeout, max_retries, retry_delay, admin_api_key

### Environment Variables
    CML_API_KEY, CML_BASE_URL, CML_TENANT_ID, ...
```

### Sub-Task 8.2.3: Configuration Guide

**Implementation** (`docs/configuration.md`):

**Pseudo-code for content**:
```
# Configuration

## Environment Variables
Table of all CML_ env vars with descriptions and defaults.

## Direct Initialization
Code example with all params.

## Config Object
CMLConfig usage with all fields.

## Priority Order
1. Direct params → 2. Env vars → 3. .env file → 4. Defaults

## Embedded Configuration
EmbeddedConfig with storage_mode, database, embedding, llm settings.

## Storage Modes (Embedded)
- Lite: SQLite + local embeddings
- Standard: PostgreSQL + pgvector
- Full: PostgreSQL + Neo4j + Redis
```

---

## Task 8.3: Usage Examples

### Sub-Task 8.3.1: Quickstart Example

**Implementation** (`examples/quickstart.py`):
```python
"""py-cml Quickstart — store and retrieve memories in 30 seconds."""

from cml import CognitiveMemoryLayer

def main():
    # Initialize client
    memory = CognitiveMemoryLayer(
        api_key="your-api-key",
        base_url="http://localhost:8000",
    )

    # Store some information
    memory.write("User prefers vegetarian food and lives in Paris.")
    memory.write("User works at a tech startup as a backend engineer.")
    memory.write("User has a meeting with the design team every Tuesday.")

    # Retrieve relevant memories
    result = memory.read("What does the user do for work?")
    print(f"Found {result.total_count} relevant memories:")
    for item in result.memories:
        print(f"  [{item.type}] {item.text}")
        print(f"    Relevance: {item.relevance:.2f}, Confidence: {item.confidence:.2f}")

    # Get formatted context for LLM
    context = memory.get_context("dietary restrictions")
    print(f"\nLLM Context:\n{context}")

    # Check stats
    stats = memory.stats()
    print(f"\nMemory Stats: {stats.total_memories} memories stored")

    memory.close()

if __name__ == "__main__":
    main()
```

### Sub-Task 8.3.2: Chat with Memory Example

**Implementation** (`examples/chat_with_memory.py`):
```python
"""Build a chatbot with persistent memory using py-cml and OpenAI."""

from openai import OpenAI
from cml import CognitiveMemoryLayer


def chat_with_memory():
    # Initialize clients
    memory = CognitiveMemoryLayer(api_key="cml-key")
    openai = OpenAI()

    session_id = "chat-demo-001"
    print("Chat with Memory (type 'quit' to exit)\n")

    while True:
        user_input = input("You: ").strip()
        if user_input.lower() == "quit":
            break

        # Get memory context
        turn = memory.turn(
            user_message=user_input,
            session_id=session_id,
        )

        # Build prompt with memory
        messages = [
            {
                "role": "system",
                "content": (
                    "You are a helpful assistant with persistent memory. "
                    "Use the following memories to personalize your responses:\n\n"
                    f"{turn.memory_context}"
                ),
            },
            {"role": "user", "content": user_input},
        ]

        # Call OpenAI
        response = openai.chat.completions.create(
            model="gpt-4o-mini",
            messages=messages,
        )
        assistant_msg = response.choices[0].message.content

        # Store the exchange
        memory.turn(
            user_message=user_input,
            assistant_response=assistant_msg,
            session_id=session_id,
        )

        print(f"Assistant: {assistant_msg}\n")
        print(f"  [memories retrieved: {turn.memories_retrieved}]")

    memory.close()


if __name__ == "__main__":
    chat_with_memory()
```

### Sub-Task 8.3.3: Async Example

**Implementation** (`examples/async_example.py`):
```python
"""Async usage of py-cml with asyncio."""

import asyncio
from cml import AsyncCognitiveMemoryLayer


async def main():
    async with AsyncCognitiveMemoryLayer(api_key="your-key") as memory:
        # Store multiple memories concurrently
        import asyncio
        await asyncio.gather(
            memory.write("User likes hiking in the mountains"),
            memory.write("User prefers morning workouts"),
            memory.write("User is training for a marathon"),
        )

        # Read memories
        result = await memory.read("exercise habits")
        print(f"Found {result.total_count} memories about exercise")
        for mem in result.memories:
            print(f"  - {mem.text}")

        # Batch read
        results = await memory.batch_read([
            "exercise habits",
            "outdoor activities",
            "fitness goals",
        ])
        for r in results:
            print(f"\nQuery: {r.query} -> {r.total_count} results")


if __name__ == "__main__":
    asyncio.run(main())
```

### Sub-Task 8.3.4: Embedded Mode Example

**Implementation** (`examples/embedded_mode.py`):
```python
"""Use py-cml without a server — embedded mode with SQLite."""

import asyncio
from cml import EmbeddedCognitiveMemoryLayer


async def main():
    # Zero-config: uses SQLite in-memory + local embeddings
    async with EmbeddedCognitiveMemoryLayer() as memory:
        # Store memories
        await memory.write("User prefers Python over JavaScript")
        await memory.write("User uses VS Code as their primary editor")
        await memory.write("User follows TDD methodology")

        # Retrieve
        result = await memory.read("development tools")
        print(f"Found {result.total_count} relevant memories:")
        for mem in result.memories:
            print(f"  - {mem.text}")

        # Get stats
        stats = await memory.stats()
        print(f"\nTotal memories: {stats.total_memories}")

    # With persistent storage:
    async with EmbeddedCognitiveMemoryLayer(db_path="./my_app.db") as memory:
        await memory.write("This memory persists between restarts")

    # Read it back in a new instance:
    async with EmbeddedCognitiveMemoryLayer(db_path="./my_app.db") as memory:
        result = await memory.read("persists")
        print(f"\nPersistent memory found: {result.memories[0].text}")


if __name__ == "__main__":
    asyncio.run(main())
```

### Sub-Task 8.3.5: Agent Framework Example

**Implementation** (`examples/agent_integration.py`):
```python
"""Integrate py-cml with an AI agent framework."""

import asyncio
from cml import AsyncCognitiveMemoryLayer


class MemoryAgent:
    """Simple agent with persistent memory."""

    def __init__(self, memory: AsyncCognitiveMemoryLayer, agent_id: str):
        self.memory = memory
        self.agent_id = agent_id

    async def observe(self, observation: str):
        """Store an observation."""
        await self.memory.write(
            observation,
            context_tags=["observation"],
            agent_id=self.agent_id,
        )

    async def plan(self, goal: str) -> str:
        """Create a plan using memory context."""
        context = await self.memory.get_context(goal)
        # In a real agent, you'd call an LLM here with the context
        return f"Plan for '{goal}' with context:\n{context}"

    async def reflect(self, topic: str):
        """Reflect on past observations."""
        result = await self.memory.read(
            topic,
            context_filter=["observation"],
            max_results=20,
        )
        print(f"Reflections on '{topic}':")
        for mem in result.memories:
            print(f"  [{mem.timestamp}] {mem.text}")


async def main():
    async with AsyncCognitiveMemoryLayer(api_key="...") as memory:
        agent = MemoryAgent(memory, agent_id="agent-001")

        # Agent observes things
        await agent.observe("The deployment pipeline took 15 minutes today")
        await agent.observe("Three tests failed due to timeout issues")
        await agent.observe("The database migration completed successfully")

        # Agent plans based on observations
        plan = await agent.plan("improve deployment speed")
        print(plan)

        # Agent reflects on past events
        await agent.reflect("deployment")


if __name__ == "__main__":
    asyncio.run(main())
```

---

## Task 8.4: GitHub Repository Setup

> **Note**: All templates and workflows are added to the existing CognitiveMemoryLayer repo.
> Issue templates are shared across the whole repo. CI workflows use path filters to only
> trigger on `packages/py-cml/**` changes.

### Sub-Task 8.4.1: Repository Templates

**Implementation**:

**Issue Templates** (`.github/ISSUE_TEMPLATE/bug_report.md`):
```markdown
---
name: Bug Report
about: Report a bug in py-cml
title: "[BUG] "
labels: bug
---

**Describe the bug**
A clear description of what the bug is.

**To Reproduce**
```python
from cml import CognitiveMemoryLayer
# Steps to reproduce...
```

**Expected behavior**
What you expected to happen.

**Environment**
- py-cml version:
- Python version:
- CML server version:
- OS:
```

**Feature Request Template** (`.github/ISSUE_TEMPLATE/feature_request.md`):
```markdown
---
name: Feature Request
about: Suggest a feature for py-cml
title: "[FEATURE] "
labels: enhancement
---

**Use Case**
What problem does this solve?

**Proposed API**
```python
# How should this look for developers?
```

**Alternatives**
Other approaches considered.
```

### Sub-Task 8.4.2: Pull Request Template

**Implementation** (`.github/pull_request_template.md`):
```markdown
## Summary
<!-- Brief description of changes -->

## Changes
- [ ] Change 1
- [ ] Change 2

## Testing
- [ ] Unit tests added/updated
- [ ] Integration tests pass
- [ ] mypy passes
- [ ] ruff passes

## Documentation
- [ ] Docstrings updated
- [ ] README updated (if applicable)
- [ ] CHANGELOG updated
```

---

## Task 8.5: Publishing Pipeline

### Sub-Task 8.5.1: GitHub Release Process

**Architecture**: Use prefixed git tags (`py-cml-v*`) to trigger PyPI publishing from the monorepo. GitHub Releases are created from these tags.

**Pseudo-code**:
```
RELEASE PROCESS (from existing CognitiveMemoryLayer repo):

1. Bump version in packages/py-cml/pyproject.toml and _version.py
2. Update packages/py-cml/CHANGELOG.md with new version section
3. Commit: "chore(py-cml): prepare release v0.1.0"
4. Create prefixed git tag: git tag py-cml-v0.1.0
5. Push: git push origin main --tags
6. py-cml-publish.yml workflow triggers (matches py-cml-v* tag):
   a. Checks out repo
   b. Builds wheel and sdist from packages/py-cml/
   c. Publishes to PyPI via trusted publishing (OIDC)
7. Create GitHub Release from tag (optional):
   gh release create py-cml-v0.1.0 \
     --title "py-cml v0.1.0 — Initial Release" \
     --notes-file packages/py-cml/CHANGELOG.md
8. Verify: pip install py-cml==0.1.0
```

### Sub-Task 8.5.2: PyPI Configuration

**Architecture**: Use PyPI trusted publishing (OIDC) — no API tokens needed. The publisher is configured for the existing CognitiveMemoryLayer repo.

**Pseudo-code for setup**:
```
1. Create PyPI account (if not exists)
2. Reserve package name "py-cml" on PyPI:
   a. Go to pypi.org/manage/account/publishing/
   b. Add new pending publisher:
      - PyPI project name: py-cml
      - Owner: <github-org>
      - Repository: CognitiveMemoryLayer    ← existing repo, NOT a new one
      - Workflow: py-cml-publish.yml         ← SDK-specific workflow
      - Environment: pypi
3. Create GitHub environment "pypi" in CognitiveMemoryLayer repo settings
4. Push py-cml-publish.yml workflow
5. Create first py-cml-v0.1.0 tag to trigger publish
```

### Sub-Task 8.5.3: TestPyPI First

**Pseudo-code**:
```
BEFORE first real release:

1. Configure TestPyPI trusted publisher:
   - test.pypi.org/manage/account/publishing/
   - Same settings as PyPI

2. Create test publish workflow:
   - Trigger on push to "release/*" branches
   - Build and publish to TestPyPI

3. Verify installation from TestPyPI:
   pip install --index-url https://test.pypi.org/simple/ py-cml

4. Once verified, create real release to publish to PyPI
```

---

## Task 8.6: Versioning Strategy

### Sub-Task 8.6.1: Semantic Versioning

**Architecture**: Follow SemVer 2.0.0.

```
MAJOR.MINOR.PATCH

MAJOR: Breaking API changes
  - Removing a public method
  - Changing method signature (required params)
  - Changing return types

MINOR: New features (backwards compatible)
  - Adding new methods
  - Adding optional parameters
  - New embedded storage backends

PATCH: Bug fixes
  - Fixing incorrect behavior
  - Performance improvements
  - Documentation updates
```

### Sub-Task 8.6.2: Version History Plan

| Version | Milestone | Key Features |
|:--------|:----------|:-------------|
| 0.1.0 | Alpha | Async/sync clients, write/read/turn, basic models |
| 0.2.0 | Beta | Full API surface, embedded mode, batch ops |
| 0.3.0 | RC | Admin ops, namespace, session management |
| 0.9.0 | Pre-GA | Complete test suite, documentation, CI/CD |
| 1.0.0 | GA | Stable API, production ready, full docs |
| 1.1.0 | Post-GA | Framework integrations (LangChain, CrewAI) |
| 1.2.0 | Post-GA | Streaming support, webhook events |

---

## Task 8.7: Community & Support

### Sub-Task 8.7.1: Contributing Guide

**Implementation** (`CONTRIBUTING.md`):

**Pseudo-code for content**:
```
# Contributing to py-cml

## Development Setup
1. Fork and clone the repo
2. Create a virtual environment
3. pip install -e ".[dev]"
4. pre-commit install

## Running Tests
pytest tests/unit/       # Fast unit tests
pytest tests/ -m "not integration"  # All non-integration tests

## Code Style
- Ruff for linting: ruff check src/
- Ruff for formatting: ruff format src/
- mypy for types: mypy src/cml/

## PR Process
1. Create a branch from main
2. Make changes
3. Add tests
4. Update CHANGELOG.md
5. Submit PR
6. Pass CI checks
7. Get review approval

## Commit Messages
Use conventional commits:
- feat: New feature
- fix: Bug fix
- docs: Documentation
- test: Tests
- chore: Maintenance
```

### Sub-Task 8.7.2: Security Policy

**Implementation** (`SECURITY.md`):
```markdown
# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 1.x     | Yes       |
| 0.x     | Best effort |

## Reporting Vulnerabilities

Please report security vulnerabilities via GitHub Security Advisories:
https://github.com/<org>/CognitiveMemoryLayer/security/advisories/new

Do NOT open a public issue for security vulnerabilities.
```

---

## Acceptance Criteria

- [ ] README.md on GitHub/PyPI conveys value in < 10 seconds
- [ ] Quickstart works in < 60 seconds (pip install + 5 lines of code)
- [ ] API reference documents all public methods, parameters, return types
- [ ] Getting started guide covers install → first memory in 5 steps
- [ ] Configuration guide covers all options (env vars, direct, config object)
- [ ] 5+ usage examples covering common patterns
- [ ] GitHub issue templates for bugs and features
- [ ] PR template with checklist
- [ ] CONTRIBUTING.md with dev setup and PR process
- [ ] SECURITY.md with reporting instructions
- [ ] CHANGELOG.md follows Keep a Changelog format
- [ ] CI publishes to PyPI on GitHub Release
- [ ] TestPyPI verified before first real release
- [ ] Package installable: `pip install py-cml`
- [ ] Package importable: `from cml import CognitiveMemoryLayer`
- [ ] All SDK files live under `packages/py-cml/` in the existing repo
- [ ] CI workflows scoped to `packages/py-cml/**` path changes
- [ ] Publish workflow triggers on `py-cml-v*` tag pattern
- [ ] PyPI trusted publisher configured for the CognitiveMemoryLayer repo
