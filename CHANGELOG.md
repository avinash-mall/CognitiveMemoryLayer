# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added

- **Documentation**
  - `CONTRIBUTING.md` — Guidelines for contributors and development setup.
  - `SECURITY.md` — Security policy and vulnerability reporting.
  - `.env.example` — Example environment configuration for database, API, LLM, and embedding settings.

- **Public API**
  - Meaningful exports in package `__init__.py` files:
    - `src.core`: `get_settings`, `Settings`, `MemoryContext`, `MemoryType`, `MemoryStatus`, `MemorySource`, `MemoryRecord`, `MemoryRecordCreate`, `MemoryPacket`.
    - `src.api`: `create_app`, `AuthContext`, `get_auth_context`, `require_write_permission`, `require_admin_permission`.
    - `src.storage`: `DatabaseManager`, `Base`, `EventLogModel`, `MemoryRecordModel`, `SemanticFactModel`, `PostgresMemoryStore`.
    - `src.utils`: `EmbeddingClient`, `OpenAIEmbeddings`, `LLMClient`, `get_llm_client`.
    - `src.memory`: `MemoryOrchestrator`, `ShortTermMemory`, `HippocampalStore`, `NeocorticalStore`.
    - `src.retrieval`: `MemoryRetriever`.
    - `src.extraction`: `EntityExtractor`, `FactExtractor`, `LLMFactExtractor`.
    - `src.consolidation`: `ConsolidationReport`, `ConsolidationWorker`.
    - `src.forgetting`: `ForgettingReport`, `ForgettingWorker`, `ForgettingScheduler`.
    - `src.reconsolidation`: `LabileMemory`, `LabileSession`, `LabileStateTracker`, `ReconsolidationResult`, `ReconsolidationService`.

- **Sensory buffer**
  - `SensoryBuffer._tokenize()` now uses tiktoken (cl100k_base) when available for accurate token counting, with a whitespace-split fallback when tiktoken is not installed.

### Changed

- **CORS** — Default origins no longer use a placeholder. Tiered behavior: use `cors_origins` if set; in debug mode use `["*"]`; otherwise use `["http://localhost:3000", "http://localhost:8080"]`.
- **DatabaseManager.close()** — Safe shutdown: only disposes/closes connections that are not `None`, avoiding `AttributeError` on partial initialization.
- **Orchestrator read API** — Removed unused `memory_types` and `time_filter` parameters from `MemoryOrchestrator.read()` and from API route call sites; filtering remains at the retrieval layer where supported.

### Fixed

- Empty `src/api/dependencies.py` — Implemented with re-exports of auth dependencies.
- Redundant empty `src/memory/hippocampal/encoder.py` — Removed; encoding is handled by `HippocampalStore` in `store.py`.

---

*For earlier history and detailed project plan, see [ProjectPlan_Complete.md](ProjectPlan/ProjectPlan_Complete.md) and [CurrentIssues.md](ProjectPlan/CurrentIssues.md).*
