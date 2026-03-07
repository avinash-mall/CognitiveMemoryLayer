"""Local unified write extraction composing model-based sub-extractors.

Replaces the LLM-based UnifiedWritePathExtractor when LLM is disabled,
using fact_extraction_structured, write_importance_regression,
pii_span_detection, and router tasks (context_tag, confidence_bin, decay_profile).
"""

from __future__ import annotations

from ..utils.logging_config import get_logger
from ..utils.modelpack import ModelPackRuntime, get_modelpack_runtime
from .fact_span_adapter import build_structured_fact_records, span_prediction_confidence
from .write_time_facts import ExtractedFact, _derive_predicate, _label_to_category

logger = get_logger(__name__)

_CONFIDENCE_BIN_MAP: dict[str, float] = {
    "low": 0.35,
    "medium": 0.65,
    "high": 0.9,
}

_DECAY_PROFILE_MAP: dict[str, float] = {
    "very_fast": 0.35,
    "fast": 0.2,
    "medium": 0.1,
    "slow": 0.05,
    "very_slow": 0.02,
}


class LocalUnifiedWriteExtractor:
    """Compose local model extractors for the write path.

    Uses available task-level models for:
    - Fact extraction (fact_extraction_structured)
    - Importance scoring (write_importance_regression)
    - PII span detection (pii_span_detection)
    - Router: context_tag, confidence_bin, decay_profile
    Falls back to default values when models are unavailable.
    """

    def __init__(self, modelpack: ModelPackRuntime | None = None):
        self.modelpack = modelpack if modelpack is not None else get_modelpack_runtime()

    @property
    def available(self) -> bool:
        """True if at least one task model is loaded (router or _task family)."""
        if getattr(self.modelpack, "available", False):
            return True
        return (
            getattr(self.modelpack, "has_task_model", lambda _: False)("fact_extraction_structured")
            or getattr(self.modelpack, "has_task_model", lambda _: False)(
                "write_importance_regression"
            )
            or getattr(self.modelpack, "has_task_model", lambda _: False)("pii_span_detection")
        )

    async def extract(self, text: str, *, context: str = "") -> dict:
        """Run local extraction pipeline and return structured result.

        Returns a dict compatible with the UnifiedWritePathExtractor output:
        {
            "facts": [ExtractedFact, ...],
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
            "context_tags": [],
            "confidence": 0.5,
            "decay_rate": None,
            "source": "local_unified",
        }

        # Fact extraction
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)(
                "fact_extraction_structured"
            ):
                spans_pred = self.modelpack.predict_spans("fact_extraction_structured", text)
                if spans_pred is not None and spans_pred.spans:
                    records = build_structured_fact_records(
                        text,
                        spans_pred,
                        derive_predicate=_derive_predicate,
                        label_to_category=_label_to_category,
                        confidence=span_prediction_confidence(spans_pred),
                    )
                    result["facts"] = [
                        ExtractedFact(
                            key=record.key,
                            category=record.category,
                            predicate=record.predicate,
                            value=record.value,
                            confidence=record.confidence,
                        )
                        for record in records
                    ]
        except Exception as exc:
            logger.debug("local_fact_extraction_failed", extra={"error": str(exc)})

        # Importance regression
        try:
            if getattr(self.modelpack, "has_task_model", lambda _: False)(
                "write_importance_regression"
            ):
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

        # Context tag from router (single label -> list)
        try:
            if getattr(self.modelpack, "available", False):
                ct_pred = self.modelpack.predict_single("context_tag", text)
                if ct_pred is not None and ct_pred.label and ct_pred.label.strip():
                    result["context_tags"] = [ct_pred.label.strip()]
        except Exception as exc:
            logger.debug("local_context_tag_failed", extra={"error": str(exc)})

        # Confidence from router confidence_bin
        try:
            if getattr(self.modelpack, "available", False):
                cb_pred = self.modelpack.predict_single("confidence_bin", text)
                if cb_pred is not None and cb_pred.label:
                    label = cb_pred.label.strip().lower()
                    if label in _CONFIDENCE_BIN_MAP:
                        result["confidence"] = _CONFIDENCE_BIN_MAP[label]
                    elif cb_pred.confidence is not None:
                        result["confidence"] = max(0.0, min(1.0, cb_pred.confidence))
        except Exception as exc:
            logger.debug("local_confidence_bin_failed", extra={"error": str(exc)})

        # Decay rate from router decay_profile
        try:
            if getattr(self.modelpack, "available", False):
                dp_pred = self.modelpack.predict_single("decay_profile", text)
                if dp_pred is not None and dp_pred.label:
                    label = dp_pred.label.strip().lower().replace("-", "_")
                    if label in _DECAY_PROFILE_MAP:
                        val = _DECAY_PROFILE_MAP[label]
                        if 0.01 <= val <= 0.5:
                            result["decay_rate"] = val
        except Exception as exc:
            logger.debug("local_decay_profile_failed", extra={"error": str(exc)})

        return result
