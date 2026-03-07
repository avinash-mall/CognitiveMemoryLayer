from __future__ import annotations

import pytest

import src.consolidation as consolidation_pkg
import src.memory as memory_pkg
from src.consolidation.worker import ConsolidationReport, ConsolidationWorker
from src.memory.hippocampal.store import HippocampalStore
from src.memory.neocortical.store import NeocorticalStore
from src.memory.orchestrator import MemoryOrchestrator
from src.memory.short_term import ShortTermMemory


def test_memory_package_lazy_exports_resolve_expected_types() -> None:
    assert memory_pkg.__getattr__("HippocampalStore") is HippocampalStore
    assert memory_pkg.__getattr__("NeocorticalStore") is NeocorticalStore
    assert memory_pkg.__getattr__("MemoryOrchestrator") is MemoryOrchestrator
    assert memory_pkg.__getattr__("ShortTermMemory") is ShortTermMemory


def test_memory_package_lazy_exports_reject_unknown_names() -> None:
    with pytest.raises(AttributeError, match="UnknownMemory"):
        memory_pkg.__getattr__("UnknownMemory")


def test_consolidation_package_lazy_exports_resolve_expected_types() -> None:
    assert consolidation_pkg.__getattr__("ConsolidationReport") is ConsolidationReport
    assert consolidation_pkg.__getattr__("ConsolidationWorker") is ConsolidationWorker


def test_consolidation_package_lazy_exports_reject_unknown_names() -> None:
    with pytest.raises(AttributeError, match="UnknownConsolidation"):
        consolidation_pkg.__getattr__("UnknownConsolidation")
