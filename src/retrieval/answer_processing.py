"""Answer post-processing: adversarial verification and F1-friendly compression.

Two quick-win improvements from the Improvement Report:

1. **Adversarial verification** (Section 8.3): Binary check whether retrieved
   context actually answers the question, preventing hallucination on
   unanswerable questions.  Expected: +20-40 F1 on adversarial.

2. **Answer compression** (Section 8.2): Extract short factual core from
   verbose LLM outputs.  Ground-truth LoCoMo answers average 5.18 tokens.
   Expected: +5-10 F1 across all categories.
"""

from __future__ import annotations

import structlog

from ..utils.llm import LLMClient

logger = structlog.get_logger(__name__)

_VERIFY_PROMPT = """Given this question and retrieved context, determine:
Does the context ACTUALLY contain information that directly answers this
specific question? Or does it merely contain related but insufficient
information?

Context:
{context}

Question: {question}

Answer ONLY "ANSWERABLE" or "UNANSWERABLE" with a one-sentence reason.
Format: ANSWERABLE: <reason> or UNANSWERABLE: <reason>"""

_COMPRESS_PROMPT = """Given this full answer to a question, extract ONLY the
core factual answer in as few words as possible.

Question: {question}
Full answer: {verbose_answer}

Core answer (1-10 words):"""

_NO_INFO_RESPONSE = "I don't have information about that from our previous conversations."


class AdversarialVerifier:
    """Verifies whether retrieved context can actually answer the question."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def verify(self, question: str, context: str) -> bool:
        """Check if the context can answer the question.

        Returns True if answerable, False if the system should refuse.
        """
        if not question or not context or not context.strip():
            return False

        try:
            response = await self._llm.complete(
                _VERIFY_PROMPT.format(
                    context=context.strip()[:2000],
                    question=question.strip(),
                ),
                temperature=0.0,
                max_tokens=100,
            )
            result = response.strip().upper()
            return result.startswith("ANSWERABLE")
        except Exception as exc:
            logger.warning("adversarial_verification_failed", error=str(exc))
            # Default to answerable on failure (don't block legitimate queries)
            return True

    @staticmethod
    def unanswerable_response() -> str:
        """Standard response for unanswerable questions."""
        return _NO_INFO_RESPONSE


class AnswerCompressor:
    """Compresses verbose LLM answers to F1-friendly short format."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def compress(self, question: str, verbose_answer: str) -> str:
        """Extract the core factual answer from a verbose LLM response.

        Returns the compressed answer, or the original if compression fails.
        """
        if not verbose_answer or not verbose_answer.strip():
            return verbose_answer

        # Already short enough — don't compress
        word_count = len(verbose_answer.split())
        if word_count <= 10:
            return verbose_answer.strip().rstrip(".")

        try:
            response = await self._llm.complete(
                _COMPRESS_PROMPT.format(
                    question=question.strip(),
                    verbose_answer=verbose_answer.strip(),
                ),
                temperature=0.0,
                max_tokens=50,
            )
            short = response.strip().rstrip(".")
            if short and len(short) > 2:
                return short
        except Exception as exc:
            logger.warning("answer_compression_failed", error=str(exc))

        return verbose_answer.strip().rstrip(".")
