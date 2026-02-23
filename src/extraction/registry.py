"""Extractor plugin registry (A-01).

Allows third-party or project extensions to register custom extractors,
retrieval sources, or memory types without modifying core source files.

Usage:
    from src.extraction.registry import extractor_registry

    # Register a custom entity extractor
    @extractor_registry.register("entity", "custom_ner")
    class CustomNERExtractor(EntityExtractor):
        ...

    # Retrieve it
    cls = extractor_registry.get("entity", "custom_ner")
"""

from __future__ import annotations

import threading
from collections import defaultdict
from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class ExtractorRegistry:
    """Thread-safe registry for pluggable extractors.

    Categories:
        - "entity": EntityExtractor implementations
        - "relation": RelationExtractor implementations
        - "constraint": ConstraintExtractor implementations
        - "fact": FactExtractor implementations
        - "retrieval": Retrieval source implementations
        - "memory_type": Custom MemoryType handlers
    """

    def __init__(self) -> None:
        self._registry: dict[str, dict[str, type]] = defaultdict(dict)
        self._lock = threading.Lock()

    def register(self, category: str, name: str) -> Any:
        """Decorator to register an extractor class.

        Example::

            @extractor_registry.register("entity", "spacy_ner")
            class SpacyNERExtractor(EntityExtractor):
                ...
        """

        def decorator(cls: type) -> type:
            with self._lock:
                if name in self._registry[category]:
                    logger.warning(
                        "extractor_registry_overwrite",
                        category=category,
                        name=name,
                        old_cls=self._registry[category][name].__name__,
                        new_cls=cls.__name__,
                    )
                self._registry[category][name] = cls
            logger.info("extractor_registered", category=category, name=name, cls=cls.__name__)
            return cls

        return decorator

    def get(self, category: str, name: str) -> type | None:
        """Retrieve a registered extractor class by category and name."""
        with self._lock:
            return self._registry.get(category, {}).get(name)

    def list_registered(self, category: str | None = None) -> dict[str, list[str]]:
        """List all registered extractors, optionally filtered by category."""
        with self._lock:
            if category:
                return {category: list(self._registry.get(category, {}).keys())}
            return {cat: list(names.keys()) for cat, names in self._registry.items()}

    def create(self, category: str, name: str, **kwargs: Any) -> Any:
        """Instantiate a registered extractor by category and name."""
        cls = self.get(category, name)
        if cls is None:
            raise KeyError(
                f"No extractor registered for category={category!r}, name={name!r}. "
                f"Available: {self.list_registered(category)}"
            )
        return cls(**kwargs)


# Singleton instance for global registration
extractor_registry = ExtractorRegistry()
