"""Semantic chunking: LLM-based and rule-based."""
import hashlib
import json
import re
from typing import List, Optional

from .models import ChunkType, SemanticChunk

from ...utils.llm import LLMClient

CHUNKING_PROMPT = """Analyze the following text and extract semantically meaningful chunks.

For each chunk, identify:
1. The core statement or fact
2. The type (statement, preference, question, instruction, fact, event, opinion)
3. Key entities mentioned
4. Importance score (0.0-1.0) based on:
   - Explicit user preferences (high)
   - Personal information (high)
   - Task-relevant details (medium-high)
   - General conversation (low)

Text to analyze:
{text}

Context (previous chunks):
{context}

Return a JSON array of chunks:
[
  {{
    "text": "extracted chunk text",
    "type": "preference|statement|fact|event|question|instruction|opinion",
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
- Combine related short statements if they form one idea"""


def _compute_salience_boost_for_sentiment(text: str) -> float:
    """Boost salience for emotionally significant content. Cap at 0.3."""
    boost = 0.0
    # Excitement indicators
    if text.count("!") >= 2 or any(
        w.isupper() and len(w) > 2 for w in text.split()
    ):
        boost += 0.15
    # Strong emotion words
    emotion_words = [
        "love", "hate", "amazing", "terrible", "excited",
        "worried", "thrilled", "devastated", "passionate",
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
        context_chunks: Optional[List[SemanticChunk]] = None,
        turn_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[SemanticChunk]:
        """
        Break text into semantic chunks.

        Args:
            text: Raw text to chunk
            context_chunks: Recent chunks for context
            turn_id: Source turn identifier
            role: "user" or "assistant"

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
                salience = min(1.0, base_salience + _compute_salience_boost_for_sentiment(raw_text))
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
                    )
                )
            return chunks

        except (json.JSONDecodeError, KeyError, TypeError):
            salience = 0.5 + _compute_salience_boost_for_sentiment(text)
            return [
                SemanticChunk(
                    id=self._generate_chunk_id(text, 0),
                    text=text,
                    chunk_type=ChunkType.STATEMENT,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=min(1.0, salience),
                    confidence=0.5,
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
        "i prefer", "i like", "i love", "i hate",
        "i don't like", "i want",
    ]
    FACT_MARKERS = [
        "my name is", "i am", "i live", "i work", "i have",
    ]
    INSTRUCTION_MARKERS = [
        "please", "can you", "could you", "i need", "help me",
    ]

    def chunk(
        self,
        text: str,
        turn_id: Optional[str] = None,
        role: Optional[str] = None,
    ) -> List[SemanticChunk]:
        """Rule-based chunking by sentences and markers."""
        # Keep trailing punctuation so we can detect questions
        sentences = re.findall(r"[^.!?]*[.!?]?", text)
        sentences = [s.strip() for s in sentences if s.strip()]

        chunks = []
        for i, sentence in enumerate(sentences):
            lower = sentence.lower()
            chunk_type = ChunkType.STATEMENT
            salience = 0.3

            if any(m in lower for m in self.PREFERENCE_MARKERS):
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

            salience = min(1.0, salience + _compute_salience_boost_for_sentiment(sentence))

            chunk_id = f"rule_{hash(sentence) % 10000}_{i}"
            chunks.append(
                SemanticChunk(
                    id=chunk_id,
                    text=sentence,
                    chunk_type=chunk_type,
                    source_turn_id=turn_id,
                    source_role=role,
                    salience=salience,
                    confidence=0.6,
                )
            )
        return chunks
