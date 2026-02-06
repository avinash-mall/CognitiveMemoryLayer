# Credentials, Tokens & Keys Inventory

> **Purpose:** Comprehensive inventory of all credentials, tokens, and keys used in the codebase for future centralized secrets management implementation.

---

## Summary

**Canonical template:** `.env.example` is the canonical template for local/dev credentials. Keep it in sync with this inventory.

| Category | Count | Source |
|----------|-------|--------|
| Database Credentials | 4 | Environment/Config |
| API Authentication | 2 | Environment/Config |
| Identity Headers | 2 | HTTP Headers |
| LLM Provider Keys | 5 | Environment/Config |
| Embedding Keys | 1 | Environment/Config |
| Environment Fallback | 1 | `OPENAI_API_KEY` |

---

## 1. Database Credentials

### 1.1 PostgreSQL

| Setting | Environment Variable | Default | Used In |
|---------|---------------------|---------|---------|
| Connection URL | `DATABASE__POSTGRES_URL` | `postgresql+asyncpg://memory:memory@localhost/memory` | `config.py:26`, `connection.py:28` |

**Note:** URL may contain embedded username/password (e.g., `postgresql+asyncpg://user:pass@host/db`). The default in config and `.env.example` includes `memory:memory@` to match Docker (`POSTGRES_USER`/`POSTGRES_PASSWORD`) and avoid OS-username auth failures when no `.env` is loaded.

### 1.2 Neo4j

| Setting | Environment Variable | Default | Used In |
|---------|---------------------|---------|---------|
| URL | `DATABASE__NEO4J_URL` | `bolt://localhost:7687` | `config.py:27` |
| Username | `DATABASE__NEO4J_USER` | `neo4j` | `config.py:28`, `connection.py:46` |
| Password | `DATABASE__NEO4J_PASSWORD` | `""` (empty) | `config.py:29`, `connection.py:47` |

**Validation:** Password required for non-localhost connections (`connection.py:41-43`)

### 1.3 Redis

| Setting | Environment Variable | Default | Used In |
|---------|---------------------|---------|---------|
| Connection URL | `DATABASE__REDIS_URL` | `redis://localhost:6379` | `config.py:30`, `connection.py:51` |

---

## 2. API Authentication Keys

### 2.1 Application API Keys

| Setting | Environment Variable | Default | Used In |
|---------|---------------------|---------|---------|
| User API Key | `AUTH__API_KEY` | `None` | `config.py:71`, `auth.py:32-38` |
| Admin API Key | `AUTH__ADMIN_API_KEY` | `None` | `config.py:72`, `auth.py:40-46` |
| Default Tenant | `AUTH__DEFAULT_TENANT_ID` | `default` | `config.py:73` |

**Header:** `X-API-Key` (defined in `auth.py:12`)

**Security Feature:** Uses `hmac.compare_digest()` for timing-safe comparison (`auth.py:67`)

### 2.2 Identity Headers (Non-Secret)

| Header | Purpose | Source |
|--------|---------|--------|
| `X-Tenant-ID` | Override tenant context | `auth.py:53`, `middleware.py:58` |
| `X-User-Id` | Set user identity | `auth.py:54` |

**Note:** These are identity tokens, not secrets, but should be validated in production.

---

## 3. LLM Provider API Keys

### 3.1 Primary LLM Configuration

| Setting | Environment Variable | Default | Used In |
|---------|---------------------|---------|---------|
| Provider | `LLM__PROVIDER` | `openai` | `config.py:47` |
| API Key | `LLM__API_KEY` | `None` | `config.py:50`, `llm.py:277` |
| Base URL | `LLM__BASE_URL` | `None` | `config.py:51` |

**Supported Providers:** `openai`, `vllm`, `ollama`, `gemini`, `claude`

### 3.2 Provider-Specific Keys

| Provider | Required Key | Environment Fallback | Notes |
|----------|--------------|----------------------|-------|
| OpenAI | `LLM__API_KEY` | `OPENAI_API_KEY` | Via `os.environ.get()` |
| vLLM | Optional | Falls back to `"dummy"` | Local deployment |
| Ollama | Optional | Falls back to `"dummy"` | Local deployment |
| Gemini | `LLM__API_KEY` | `OPENAI_API_KEY` | Required, validates at factory |
| Claude | `LLM__API_KEY` | `OPENAI_API_KEY` | Required, validates at factory |

**Files:** `llm.py:96-98`, `llm.py:277`, `llm.py:293-300`

---

## 4. Embedding Provider Keys

| Setting | Environment Variable | Default | Used In |
|---------|---------------------|---------|---------|
| Provider | `EMBEDDING__PROVIDER` | `openai` | `config.py:36` |
| API Key | `EMBEDDING__API_KEY` | `None` | `config.py:40`, `embeddings.py:54` |
| Base URL | `EMBEDDING__BASE_URL` | `None` | `config.py:41` |

**Fallback:** `OPENAI_API_KEY` environment variable (`embeddings.py:54`, `embeddings.py:273`)

---

## 5. Docker Compose Credentials

Location: `docker/docker-compose.yml`. App/env credentials in the compose file are aligned with `.env.example` (Postgres URL with `memory:memory`, Neo4j password `password`).

| Service | Credential | Value | Line |
|---------|------------|-------|------|
| PostgreSQL | `POSTGRES_USER` | `memory` | 7 |
| PostgreSQL | `POSTGRES_PASSWORD` | `memory` | 8 |
| Neo4j | `NEO4J_AUTH` | `neo4j/password` | 21 |
| App/API | `DATABASE__NEO4J_PASSWORD` | `password` | 80, 101 |

---

## 6. Secret Detection (WriteGate)

The `WriteGate` class (`write_gate.py:47-53`) detects and blocks storage of secrets:

```python
secret_patterns = [
    r"password\s*[:=]\s*\S+",
    r"api[_-]?key\s*[:=]\s*\S+",
    r"secret\s*[:=]\s*\S+",
    r"token\s*[:=]\s*\S+",
]
```

**Protected PII patterns:** SSN, credit card, email, phone (`write_gate.py:41-45`)

---

## 7. Files Containing Credential Logic

| File | Credential Types |
|------|------------------|
| `src/core/config.py` | All settings definitions |
| `src/api/auth.py` | API key authentication |
| `src/storage/connection.py` | Database credential usage |
| `src/utils/llm.py` | LLM API keys |
| `src/utils/embeddings.py` | Embedding API keys |
| `src/memory/hippocampal/write_gate.py` | Secret detection |
| `docker/docker-compose.yml` | Docker environment credentials |
| `.env.example` | Template for all credentials |

---

## 8. Optional Provider Dependencies

Some providers require additional packages (defined in `pyproject.toml:28-30`):

| Provider | Extra | Install Command |
|----------|-------|-----------------|
| Gemini | `gemini` | `poetry install -E gemini` |
| Claude | `claude` | `poetry install -E claude` |

---

## 9. Recommendations for Centralized Secrets Management

1. **Create unified secrets interface** - Single `SecretsProvider` abstraction
2. **Support vault backends** - HashiCorp Vault, AWS Secrets Manager, Azure Key Vault
3. **Lazy loading** - Fetch secrets on-demand, not at startup
4. **Secret rotation** - API for rotating credentials without restart
5. **Audit logging** - Track secret access for compliance
6. **Environment isolation** - Separate secrets per environment (dev/staging/prod)

---

*Generated: 2026-02-06*
