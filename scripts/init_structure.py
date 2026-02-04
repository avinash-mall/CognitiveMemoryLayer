"""Create project directory structure for Cognitive Memory Layer."""
from pathlib import Path

STRUCTURE = {
    "src": {
        "api": ["__init__.py", "routes.py", "dependencies.py", "middleware.py"],
        "core": ["__init__.py", "models.py", "schemas.py", "enums.py", "exceptions.py", "config.py"],
        "memory": {
            "sensory": ["__init__.py", "buffer.py"],
            "working": ["__init__.py", "manager.py", "chunker.py"],
            "hippocampal": ["__init__.py", "store.py", "encoder.py"],
            "neocortical": ["__init__.py", "store.py", "schema_manager.py"],
            "__init__.py": None,
            "orchestrator.py": None,
        },
        "retrieval": ["__init__.py", "planner.py", "retriever.py", "reranker.py"],
        "consolidation": ["__init__.py", "worker.py", "clusterer.py", "summarizer.py"],
        "forgetting": ["__init__.py", "scorer.py", "worker.py"],
        "extraction": ["__init__.py", "entity_extractor.py", "fact_extractor.py"],
        "storage": [
            "__init__.py", "postgres.py", "neo4j.py", "redis.py", "base.py",
            "models.py", "event_log.py", "connection.py",
        ],
        "utils": ["__init__.py", "embeddings.py", "llm.py", "timing.py"],
        "__init__.py": None,
    },
    "tests": {
        "unit": ["__init__.py"],
        "integration": ["__init__.py"],
        "conftest.py": None,
    },
    "config": ["settings.yaml", "logging.yaml"],
    "migrations": ["__init__.py", "env.py"],
    "migrations/versions": [],
    "docker": ["Dockerfile", "docker-compose.yml"],
    "docs": [],
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
            path.touch()


if __name__ == "__main__":
    create_structure(Path(__file__).resolve().parent.parent, STRUCTURE)
    print("Directory structure created.")
