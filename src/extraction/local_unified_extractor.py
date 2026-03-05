"""Local unified write extraction composing model-based sub-extractors.

Replaces the LLM-based UnifiedWritePathExtractor when LLM is disabled,
using fact_extraction_structured, write_importance_regression, and
pii_span_detection models from modelpack.
"""

from __future__ import annotations

from ..utils.logging_config import get_logger
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime

logger = get_logger(__name__)


class LocalUnifiedWriteExtractor:
    """Compose local model extractors for the write path.

    Uses available task-level models for:
    - Fact extraction (fact_extraction_structured)
    - Importance scoring (write_importance_regression)
    - PII span detection (pii_span_detection)
    Falls back to default values when models are unavailable.
    """

    def __init__(self, modelpack: ModelPackRuntime | None = None):
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

    @property
    def available(self) -> bool:
        """True if at least one task model is loaded."""
        return (
            getattr(self.modelpack, "has_task_model", lambda _: False)("fact_extraction_structured")
            or getattr(self.modelpack, "has_task_model", lambda _: False)("write_importance_regression")
            or getattr(self.modelpack, "has_task_model", lambda _: False)("pii_span_detection")
        )

    async def extract(self, text: str, *, context: str = "") -> dict:
        """Run local extraction pipeline and return structured result.

        Returns a dict compatible with the UnifiedWritePathExtractor output:
        {
            "facts": [...],
            "importance": float,
            "pii_spans": [...],
            "memory_type": str | None,
            "constraints": [...],
            "source": "local_unified",
        }
        """
        result: dict = {
            "facts": [],
            "importance": 0.5,
            "pii_spans": [],
            "memory_type": None,
            "constraints": [],
            "source": "local_unified",
        }

        # Fact extraction
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)("fact_extraction_structured"):
                spans_pred = self.modelpack.predict_spans("fact_extraction_structured", text)
                if spans_pred is not None and spans_pred.spans:
                    result["facts"] = [
                        {"start": s[0], "end": s[1], "label": s[2], "text": text[s[0] : s[1]]}
                        for s in spans_pred.spans
                        if s[0] < len(text) and s[1] <= len(text)
                    ]
        except Exception as exc:
            logger.debug("local_fact_extraction_failed", extra={"error": str(exc)})

        # Importance regression
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)("write_importance_regression"):
                score_pred = self.modelpack.predict_score_single(
                    "write_importance_regression", text
                )
                if score_pred is not None:
                    result["importance"] = max(0.0, min(1.0, score_pred.score))
        except Exception as exc:
            logger.debug("local_importance_failed", extra={"error": str(exc)})

        # PII span detection
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)("pii_span_detection"):
                pii_pred = self.modelpack.predict_spans("pii_span_detection", text)
                if pii_pred is not None and pii_pred.spans:
                    result["pii_spans"] = [
                        {"start": s[0], "end": s[1], "label": s[2]} for s in pii_pred.spans
                    ]
        except Exception as exc:
            logger.debug("local_pii_detection_failed", extra={"error": str(exc)})

        # Memory type from router model (if available)
        try:
            mt_pred = self.modelpack.predict_single("memory_type", text)
            if mt_pred is not None:
                result["memory_type"] = mt_pred.label
        except Exception:
            pass

        return result
