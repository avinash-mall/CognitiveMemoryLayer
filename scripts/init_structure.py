"""Create project directory structure for Cognitive Memory Layer."""

from pathlib import Path

# Reflects actual project layout. Run from repo root to create missing dirs/files (touch).
STRUCTURE = {
    ".github": {"workflows": ["ci.yml"]},
    "docker": ["Dockerfile", "docker-compose.yml"],
    "examples": [
        "__init__.py",
        "anthropic_tool_calling.py",
        "async_usage.py",
        "basic_usage.py",
        "chatbot_with_memory.py",
        "langchain_integration.py",
        "memory_client.py",
        "openai_tool_calling.py",
        "README.md",
        "requirements.txt",
        "standalone_demo.py",
    ],
    "migrations": {
        "versions": ["001_initial_schema.py"],
        "__init__.py": None,
        "env.py": None,
    },
    "src": {
        "__init__.py": None,
        "main.py": None,
        "celery_app.py": None,
        "api": [
            "__init__.py",
            "admin_routes.py",
            "app.py",
            "auth.py",
            "dependencies.py",
            "middleware.py",
            "routes.py",
            "schemas.py",
        ],
        "core": [
            "__init__.py",
            "config.py",
            "enums.py",
            "exceptions.py",
            "schemas.py",
        ],
        "consolidation": [
            "__init__.py",
            "clusterer.py",
            "migrator.py",
            "sampler.py",
            "schema_aligner.py",
            "summarizer.py",
            "triggers.py",
            "worker.py",
        ],
        "extraction": [
            "__init__.py",
            "entity_extractor.py",
            "fact_extractor.py",
            "relation_extractor.py",
        ],
        "forgetting": [
            "__init__.py",
            "actions.py",
            "compression.py",
            "executor.py",
            "interference.py",
            "scorer.py",
            "worker.py",
        ],
        "memory": {
            "__init__.py": None,
            "conversation.py": None,
            "knowledge_base.py": None,
            "orchestrator.py": None,
            "scratch_pad.py": None,
            "seamless_provider.py": None,
            "short_term.py": None,
            "tool_memory.py": None,
            "hippocampal": [
                "__init__.py",
                "encoder.py",
                "redactor.py",
                "store.py",
                "write_gate.py",
            ],
            "neocortical": [
                "__init__.py",
                "fact_store.py",
                "schema_manager.py",
                "schemas.py",
                "store.py",
            ],
            "sensory": ["__init__.py", "buffer.py", "manager.py"],
            "working": ["__init__.py", "chunker.py", "manager.py", "models.py"],
        },
        "reconsolidation": [
            "__init__.py",
            "belief_revision.py",
            "conflict_detector.py",
            "labile_tracker.py",
            "service.py",
        ],
        "retrieval": [
            "__init__.py",
            "classifier.py",
            "memory_retriever.py",
            "packet_builder.py",
            "planner.py",
            "query_types.py",
            "reranker.py",
            "retriever.py",
        ],
        "storage": [
            "__init__.py",
            "base.py",
            "connection.py",
            "event_log.py",
            "models.py",
            "neo4j.py",
            "postgres.py",
            "redis.py",
        ],
        "utils": [
            "__init__.py",
            "embeddings.py",
            "llm.py",
            "logging_config.py",
            "metrics.py",
            "timing.py",
        ],
    },
    "tests": {
        "unit": [
            "__init__.py",
            "test_core_enums_schemas_config.py",
            "test_sensory_buffer_working_memory.py",
            "test_embeddings_mock_client.py",
            "test_hippocampal_write_gate_redactor.py",
            "test_neocortical_schemas.py",
            "test_retrieval_classifier_planner_reranker.py",
            "test_reconsolidation_labile_conflict_belief.py",
            "test_consolidation_triggers_clusterer_sampler.py",
            "test_celery_forgetting_task.py",
            "test_forgetting_scorer_policy_interference.py",
            "test_api_auth_schemas.py",
        ],
        "integration": [
            "__init__.py",
            "conftest.py",
            "test_storage_event_log_repository.py",
            "test_short_term_memory_flow.py",
            "test_hippocampal_encode_flow.py",
            "test_fact_store_integration.py",
            "test_neocortical_store_flow.py",
            "test_retrieval_flow.py",
            "test_reconsolidation_flow.py",
            "test_consolidation_flow.py",
            "test_forgetting_flow.py",
            "test_forgetting_llm_compression.py",
            "test_api_flow.py",
        ],
        "e2e": ["__init__.py", "test_api_flows.py"],
        "conftest.py": None,
    },
}


def create_structure(base_path: Path, structure: dict) -> None:
    for name, content in structure.items():
        path = base_path / name
        if isinstance(content, dict):
            path.mkdir(parents=True, exist_ok=True)
            create_structure(path, content)
        elif isinstance(content, list):
            path.mkdir(parents=True, exist_ok=True)
            for file in content:
                (path / file).touch()
        elif content is None:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.touch()


if __name__ == "__main__":
    create_structure(Path(__file__).resolve().parent.parent, STRUCTURE)
    print("Directory structure created.")
