"""Shadow mode comparison logger for heuristic vs model decisions.

Run both the heuristic and model paths in parallel, compare decisions,
and log deltas before switching defaults.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ShadowComparison:
    """Single comparison between heuristic and model decisions."""

    component: str
    task: str
    heuristic_result: Any
    model_result: Any
    agreed: bool
    heuristic_latency_ms: float
    model_latency_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)


class ShadowModeLogger:
    """Collects and reports shadow comparisons between heuristic and model paths."""

    def __init__(self, *, enabled: bool = False, sample_rate: float = 1.0):
        self.enabled = enabled
        self.sample_rate = max(0.0, min(1.0, sample_rate))
        self._comparisons: list[ShadowComparison] = []
        self._count = 0

    def compare(
        self,
        *,
        component: str,
        task: str,
        heuristic_fn: Callable[[], Any],
        model_fn: Callable[[], Any],
        agreement_fn: Callable[[Any, Any], bool] | None = None,
    ) -> Any:
        """Run both paths and log comparison. Returns the heuristic result (primary).

        Args:
            component: Source component name (e.g. "write_gate", "retriever")
            task: Task being compared (e.g. "novelty_pair", "importance")
            heuristic_fn: Zero-arg callable returning heuristic result
            model_fn: Zero-arg callable returning model result
            agreement_fn: Optional comparator; defaults to equality check
        """
        if not self.enabled:
            return heuristic_fn()

        self._count += 1
        if random.random() > self.sample_rate:
            return heuristic_fn()

        t0 = time.perf_counter()
        heuristic_result = heuristic_fn()
        h_ms = (time.perf_counter() - t0) * 1000

        model_result = None
        m_ms = 0.0
        try:
            t1 = time.perf_counter()
            model_result = model_fn()
            m_ms = (time.perf_counter() - t1) * 1000
        except Exception as exc:
            logger.debug(
                "shadow_model_error",
                extra={"component": component, "task": task, "error": str(exc)},
            )

        if agreement_fn is not None:
            agreed = agreement_fn(heuristic_result, model_result)
        else:
            agreed = heuristic_result == model_result

        comp = ShadowComparison(
            component=component,
            task=task,
            heuristic_result=_safe_serialize(heuristic_result),
            model_result=_safe_serialize(model_result),
            agreed=agreed,
            heuristic_latency_ms=round(h_ms, 3),
            model_latency_ms=round(m_ms, 3),
        )
        self._comparisons.append(comp)

        if not agreed:
            logger.info(
                "shadow_disagreement",
                extra={
                    "component": component,
                    "task": task,
                    "heuristic": _safe_serialize(heuristic_result),
                    "model": _safe_serialize(model_result),
                    "h_ms": round(h_ms, 1),
                    "m_ms": round(m_ms, 1),
                },
            )

        return heuristic_result

    def get_summary(self) -> dict[str, Any]:
        """Return aggregate summary of all shadow comparisons."""
        if not self._comparisons:
            return {"total": 0, "agreement_rate": 0.0, "by_component": {}}

        total = len(self._comparisons)
        agreed = sum(1 for c in self._comparisons if c.agreed)

        by_component: dict[str, dict[str, Any]] = {}
        for c in self._comparisons:
            key = f"{c.component}:{c.task}"
            if key not in by_component:
                by_component[key] = {
                    "total": 0,
                    "agreed": 0,
                    "h_latency_ms": [],
                    "m_latency_ms": [],
                }
            by_component[key]["total"] += 1
            if c.agreed:
                by_component[key]["agreed"] += 1
            by_component[key]["h_latency_ms"].append(c.heuristic_latency_ms)
            by_component[key]["m_latency_ms"].append(c.model_latency_ms)

        for stats in by_component.values():
            h_lats = stats.pop("h_latency_ms")
            m_lats = stats.pop("m_latency_ms")
            stats["agreement_rate"] = round(stats["agreed"] / max(1, stats["total"]), 4)
            stats["avg_heuristic_ms"] = round(sum(h_lats) / max(1, len(h_lats)), 3)
            stats["avg_model_ms"] = round(sum(m_lats) / max(1, len(m_lats)), 3)

        return {
            "total": total,
            "agreed": agreed,
            "agreement_rate": round(agreed / max(1, total), 4),
            "by_component": by_component,
        }

    def reset(self) -> None:
        """Clear collected comparisons."""
        self._comparisons.clear()
        self._count = 0


def _safe_serialize(value: Any) -> Any:
    """Convert value to a JSON-safe representation."""
    if value is None or isinstance(value, (bool, int, float, str)):
        return value
    if isinstance(value, (list, tuple)):
        return [_safe_serialize(v) for v in value]
    if isinstance(value, dict):
        return {str(k): _safe_serialize(v) for k, v in value.items()}
    return str(value)


_shadow_logger: ShadowModeLogger | None = None


def get_shadow_logger() -> ShadowModeLogger:
    """Return process-cached shadow mode logger."""
    global _shadow_logger
    if _shadow_logger is None:
        _shadow_logger = ShadowModeLogger(enabled=False)
    return _shadow_logger
