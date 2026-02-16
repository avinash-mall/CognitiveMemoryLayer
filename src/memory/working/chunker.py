"""Semantic chunking: LLM-based, rule-based, and Chonkie semantic."""

import hashlib
import json
import re
from datetime import UTC, datetime
from typing import Any

from ...utils.llm import LLMClient
from .models import ChunkType, SemanticChunk


class ChonkieUnavailableError(RuntimeError):
    """Raised when Chonkie is requested but chonkie[semantic] is not installed."""


CHUNKING_PROMPT = """Analyze the following text and extract semantically meaningful chunks.

For each chunk, identify:
1. The core statement or fact
2. The type (statement, preference, question, instruction, fact, event, opinion, constraint)
3. Key entities mentioned
4. Importance score (0.0-1.0) based on:
   - Explicit user preferences (high)
   - Personal information (high)
   - Task-relevant details (medium-high)
   - Constraints: goals, values, policies, states, causal reasons (high)
   - General conversation (low)

Text to analyze:
{text}

Context (previous chunks):
{context}

Return a JSON array of chunks:
[
  {{
    "text": "extracted chunk text",
    "type": "preference|statement|fact|event|question|instruction|opinion|constraint",
    "entities": ["entity1", "entity2"],
    "key_phrases": ["phrase1"],
    "salience": 0.8,
    "confidence": 0.9
  }}
]

Rules:
- Each chunk should be a single coherent idea
- Preserve the user's wording for preferences and facts
- Don't create chunks for filler/acknowledgments
- Combine related short statements if they form one idea
- Use type "constraint" for goals, values, policies, commitments, causal reasoning, or states that should govern future behavior"""


_CONSTRAINT_CUE_PHRASES = [
    "i'm trying to",
    "i don't want",
    "it's important that",
    "i need to avoid",
    "i'm anxious about",
    "i'm preparing for",
    "i must",
    "i should",
    "i can't",
    "i won't",
    "my goal is",
    "i'm focused on",
    "i'm committed to",
    "i value",
    "i believe",
    "because of",
    "in order to",
    "i never",
    "i always",
    "i'm working toward",
    "i care about",
    "so that",
    "that's why",
    "i'm dealing with",
]


def _compute_salience_boost_for_constraints(text: str) -> float:
    """Boost salience for constraint cues (goals, values, policies, causal). Cap at 0.4."""
    lower = text.lower()
    boost = 0.0
    matches = sum(1 for phrase in _CONSTRAINT_CUE_PHRASES if phrase in lower)
    if matches >= 2:
        boost = 0.4
    elif matches == 1:
        boost = 0.3
    return min(boost, 0.4)


def _compute_salience_boost_for_sentiment(text: str) -> float:
    """Boost salience for emotionally significant content. Cap at 0.3."""
    boost = 0.0
    # Excitement indicators
    if text.count("!") >= 2 or any(w.isupper() and len(w) > 2 for w in text.split()):
        boost += 0.15
    # Strong emotion words
    emotion_words = [
        "love",
        "hate",
        "amazing",
        "terrible",
        "excited",
        "worried",
        "thrilled",
        "devastated",
        "passionate",
    ]
    if any(word in text.lower() for word in emotion_words):
        boost += 0.1
    # Personal significance markers
    personal_markers = ["finally", "at last", "dream", "goal", "achieved"]
    if any(marker in text.lower() for marker in personal_markers):
        boost += 0.1
    return min(boost, 0.3)


class SemanticChunker:
    """
    Uses an LLM to break text into semantic chunks.
    Implements chunking similar to human working memory processing.
    """

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client

    async def chunk(
        self,
        text: str,
        context_chunks: list[SemanticChunk] | None = None,
        turn_id: str | None = None,
        role: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[SemanticChunk]:
        """
        Break text into semantic chunks.

        Args:
            text: Raw text to chunk
            context_chunks: Recent chunks for context
            turn_id: Source turn identifier
            role: "user" or "assistant"
            timestamp: Optional event timestamp (defaults to now)

        Returns:
            List of semantic chunks
        """
        if not text.strip():
            return []

        context_str = ""
        if context_chunks:
            context_str = "\n".join(
                [f"- [{c.chunk_type.value}] {c.text}" for c in context_chunks[-5:]]
            )

        prompt = CHUNKING_PROMPT.format(
            text=text,
            context=context_str or "None",
        )

        chunk_timestamp = timestamp or datetime.now(UTC)

        try:
            response = await self.llm.complete(
                prompt,
                temperature=0.0,
                max_tokens=1000,
            )
            chunks_data = json.loads(response)
            if not isinstance(chunks_data, list):
                chunks_data = [chunks_data]

            chunks = []
            for i, cd in enumerate(chunks_data):
                chunk_id = self._generate_chunk_id(text, i)
                raw_type = cd.get("type", "statement")
                try:
                    ctype = ChunkType(raw_type)
                except ValueError:
                    ctype = ChunkType.STATEMENT
                raw_text = cd.get("text", "")
                base_salience = float(cd.get("salience", 0.5))
                salience = min(
                    1.0,
                    base_salience
                    + _compute_salience_boost_for_sentiment(raw_text)
                    + _compute_salience_boost_for_constraints(raw_text),
                )
                chunks.append(
                    SemanticChunk(
                        id=chunk_id,
                        text=raw_text,
                        chunk_type=ctype,
                        source_turn_id=turn_id,
                        source_role=role,
                        entities=cd.get("entities", []),
                        key_phrases=cd.get("key_phrases", []),
                        salience=salience,
                        confidence=float(cd.get("confidence", 0.8)),
                        timestamp=chunk_timestamp,
                    )
                )
            return chunks

        except (json.JSONDecodeError, KeyError, TypeError):
            salience = (
                0.5
                + _compute_salience_boost_for_sentiment(text)
                + _compute_salience_boost_for_constraints(text)
            )
            return [
                SemanticChunk(
                    id=self._generate_chunk_id(text, 0),
                    text=text,
                    chunk_type=ChunkType.STATEMENT,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=min(1.0, salience),
                    confidence=0.5,
                    timestamp=chunk_timestamp,
                )
            ]

    def _generate_chunk_id(self, text: str, index: int) -> str:
        """Generate deterministic chunk ID."""
        content = f"{text}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]


class RuleBasedChunker:
    """
    Fast, rule-based chunker for when LLM is too slow or unavailable.
    """

    PREFERENCE_MARKERS = [
        "i prefer",
        "i like",
        "i love",
        "i hate",
        "i don't like",
        "i want",
    ]
    FACT_MARKERS = [
        "my name is",
        "i am",
        "i live",
        "i work",
        "i have",
    ]
    INSTRUCTION_MARKERS = [
        "please",
        "can you",
        "could you",
        "i need",
        "help me",
    ]
    CONSTRAINT_MARKERS = [
        "i'm trying to",
        "i don't want",
        "it's important that",
        "i need to avoid",
        "i'm anxious about",
        "i'm preparing for",
        "i must",
        "i should",
        "i can't",
        "i won't",
        "my goal is",
        "i'm focused on",
        "i'm committed to",
        "i value",
        "i believe",
        "because of",
        "in order to",
        "i never",
        "i always",
        "i'm working toward",
        "i care about",
    ]

    def chunk(
        self,
        text: str,
        turn_id: str | None = None,
        role: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[SemanticChunk]:
        """Rule-based chunking by sentences and markers."""
        # Keep trailing punctuation so we can detect questions
        sentences = re.findall(r"[^.!?]*[.!?]?", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunk_timestamp = timestamp or datetime.now(UTC)

        chunks = []
        for i, sentence in enumerate(sentences):
            lower = sentence.lower()
            chunk_type = ChunkType.STATEMENT
            salience = 0.3

            if any(m in lower for m in self.CONSTRAINT_MARKERS):
                chunk_type = ChunkType.CONSTRAINT
                salience = 0.85
            elif any(m in lower for m in self.PREFERENCE_MARKERS):
                chunk_type = ChunkType.PREFERENCE
                salience = 0.8
            elif any(m in lower for m in self.FACT_MARKERS):
                chunk_type = ChunkType.FACT
                salience = 0.7
            elif any(m in lower for m in self.INSTRUCTION_MARKERS):
                chunk_type = ChunkType.INSTRUCTION
                salience = 0.6
            elif "?" in sentence:
                chunk_type = ChunkType.QUESTION
                salience = 0.4

            salience = min(
                1.0,
                salience
                + _compute_salience_boost_for_sentiment(sentence)
                + _compute_salience_boost_for_constraints(sentence),
            )

            chunk_id = f"rule_{hashlib.sha256(sentence.encode()).hexdigest()[:8]}_{i}"
            chunks.append(
                SemanticChunk(
                    id=chunk_id,
                    text=sentence,
                    chunk_type=chunk_type,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=salience,
                    confidence=0.6,
                    timestamp=chunk_timestamp,
                )
            )
        return chunks


def _get_chonkie_semantic_chunker(
    chunk_size: int = 512,
    threshold: float = 0.7,
    embedding_model: str = "minishlab/potion-base-32M",
    **kwargs: Any,
) -> Any:
    """Lazy import Chonkie SemanticChunker. Raises ChonkieUnavailableError if not installed."""
    try:
        from chonkie import SemanticChunker as ChonkieSemanticChunker
    except ImportError as e:
        raise ChonkieUnavailableError(
            "chonkie[semantic] is not installed. Install with: pip install 'chonkie[semantic]'"
        ) from e
    return ChonkieSemanticChunker(
        embedding_model=embedding_model,
        threshold=threshold,
        chunk_size=chunk_size,
        **kwargs,
    )


class ChonkieChunkerAdapter:
    """
    Uses Chonkie SemanticChunker for semantic chunking (no LLM or rule-based logic).
    Maps Chonkie chunks to SemanticChunk for the rest of the pipeline.
    """

    def __init__(
        self,
        chunk_size: int = 512,
        threshold: float = 0.7,
        embedding_model: str = "minishlab/potion-base-32M",
        **kwargs: Any,
    ) -> None:
        self._chunk_size = chunk_size
        self._threshold = threshold
        self._embedding_model = embedding_model
        self._kwargs = kwargs
        self._chonkie: Any = None

    def _get_chonkie(self) -> Any:
        if self._chonkie is None:
            self._chonkie = _get_chonkie_semantic_chunker(
                chunk_size=self._chunk_size,
                threshold=self._threshold,
                embedding_model=self._embedding_model,
                **self._kwargs,
            )
        return self._chonkie

    def _generate_chunk_id(self, text: str, index: int) -> str:
        content = f"{text}:{index}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]

    def chunk(
        self,
        text: str,
        turn_id: str | None = None,
        role: str | None = None,
        timestamp: datetime | None = None,
    ) -> list[SemanticChunk]:
        """Chunk text with Chonkie SemanticChunker and map to SemanticChunk list."""
        if not text.strip():
            return []
        chunk_timestamp = timestamp or datetime.now(UTC)
        chonkie = self._get_chonkie()
        raw_chunks = chonkie.chunk(text)
        chunks = []
        for i, c in enumerate(raw_chunks):
            segment = getattr(c, "text", str(c)) if c else ""
            if not segment.strip():
                continue
            chunk_id = self._generate_chunk_id(segment, i)
            chunks.append(
                SemanticChunk(
                    id=chunk_id,
                    text=segment,
                    chunk_type=ChunkType.STATEMENT,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=0.5,
                    confidence=0.8,
                    timestamp=chunk_timestamp,
                )
            )
        if not chunks:
            chunks.append(
                SemanticChunk(
                    id=self._generate_chunk_id(text, 0),
                    text=text,
                    chunk_type=ChunkType.STATEMENT,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=0.5,
                    confidence=0.8,
                    timestamp=chunk_timestamp,
                )
            )
        return chunks
