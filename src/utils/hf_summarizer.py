"""Hugging Face summarizer backend for non-LLM consolidation paths."""

from __future__ import annotations

import asyncio
from functools import lru_cache
from typing import Any

from .logging_config import get_logger

_logger = get_logger(__name__)


class HuggingFaceSummarizer:
    """Lazy-loaded transformers summarizer with resilient fallback behavior."""

    def __init__(
        self,
        *,
        model: str,
        task: str = "summarization",
        max_input_chars: int = 2400,
        max_output_chars: int = 320,
        min_length: int = 24,
        max_length: int = 96,
        device: int = -1,
    ) -> None:
        self.model = model
        self.task = task
        self.max_input_chars = max(200, max_input_chars)
        self.max_output_chars = max(80, max_output_chars)
        self.min_length = max(8, min_length)
        self.max_length = max(self.min_length, max_length)
        self.device = device
        self._pipeline: Any | None = None
        self._pipeline_load_attempted = False

    async def summarize(self, text: str, *, max_chars: int | None = None) -> str:
        """Summarize text asynchronously."""
        return await asyncio.to_thread(self._summarize_sync, text, max_chars)

    def _summarize_sync(self, text: str, max_chars: int | None = None) -> str:
        cleaned = " ".join((text or "").split())
        if not cleaned:
            return ""

        cap = max(60, min(max_chars or self.max_output_chars, self.max_output_chars))
        if len(cleaned) <= cap:
            return cleaned

        pipeline = self._get_pipeline()
        if pipeline is None:
            return self._truncate(cleaned, cap)

        try:
            output = pipeline(
                cleaned[: self.max_input_chars],
                min_length=self.min_length,
                max_length=self.max_length,
                do_sample=False,
                truncation=True,
            )
            summary = ""
            if isinstance(output, list) and output:
                row = output[0]
                if isinstance(row, dict):
                    summary = str(row.get("summary_text", "")).strip()
                else:
                    summary = str(row).strip()
            else:
                summary = str(output).strip()
            if not summary:
                return self._truncate(cleaned, cap)
            return self._truncate(summary, cap)
        except Exception as exc:
            _logger.warning(
                "hf_summarizer_inference_failed",
                extra={"model": self.model, "error": str(exc)},
            )
            return self._truncate(cleaned, cap)

    def _get_pipeline(self) -> Any | None:
        if self._pipeline is not None:
            return self._pipeline
        if self._pipeline_load_attempted:
            return None
        self._pipeline_load_attempted = True

        try:
            from transformers import pipeline

            pipeline_task: Any = self.task
            self._pipeline = pipeline(
                pipeline_task,
                model=self.model,
                tokenizer=self.model,
                device=self.device,
            )
            _logger.info(
                "hf_summarizer_ready",
                extra={"model": self.model, "task": self.task, "device": self.device},
            )
            return self._pipeline
        except Exception as exc:
            _logger.warning(
                "hf_summarizer_load_failed",
                extra={"model": self.model, "task": self.task, "error": str(exc)},
            )
            return None

    @staticmethod
    def _truncate(text: str, max_chars: int) -> str:
        if len(text) <= max_chars:
            return text
        return text[: max_chars - 3].rstrip() + "..."


@lru_cache(maxsize=8)
def get_hf_summarizer(
    *,
    model: str,
    task: str = "summarization",
    max_input_chars: int = 2400,
    max_output_chars: int = 320,
    min_length: int = 24,
    max_length: int = 96,
    device: int = -1,
) -> HuggingFaceSummarizer:
    """Return a cached HF summarizer instance by config tuple."""
    return HuggingFaceSummarizer(
        model=model,
        task=task,
        max_input_chars=max_input_chars,
        max_output_chars=max_output_chars,
        min_length=min_length,
        max_length=max_length,
        device=device,
    )
