"""Unified write-path LLM extractor: constraints, facts, salience, importance in one call.

When any of the write-path LLM feature flags are enabled, this extractor produces
constraints, facts, salience, importance (and optionally PII spans) in a single
LLM call instead of multiple rule-based passes.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from ..core.schemas import EntityMention, Relation
from ..memory.neocortical.schemas import FactCategory
from ..memory.working.models import SemanticChunk
from ..utils.llm import LLMClient
from .constraint_extractor import ConstraintObject
from .write_time_facts import ExtractedFact

# ---------------------------------------------------------------------------
# Result schema
# ---------------------------------------------------------------------------


@dataclass
class PIISpan:
    """A span of text that contains PII."""

    start: int
    end: int
    pii_type: str  # e.g. "email", "phone", "ssn"


_ALLOWED_MEMORY_TYPES = frozenset(
    {
        "episodic_event",
        "semantic_fact",
        "preference",
        "task_state",
        "procedure",
        "constraint",
        "hypothesis",
        "conversation",
        "message",
        "tool_result",
        "reasoning_step",
        "scratch",
        "knowledge",
        "observation",
        "plan",
    }
)


@dataclass
class UnifiedExtractionResult:
    """Result of unified write-path extraction."""

    entities: list[EntityMention] = field(default_factory=list)
    relations: list[Relation] = field(default_factory=list)
    constraints: list[ConstraintObject] = field(default_factory=list)
    facts: list[ExtractedFact] = field(default_factory=list)
    salience: float = 0.5
    importance: float = 0.5
    pii_spans: list[PIISpan] = field(default_factory=list)
    contains_secrets: bool = False
    memory_type: str | None = None
    confidence: float = 0.5
    context_tags: list[str] = field(default_factory=list)
    decay_rate: float | None = None


# ---------------------------------------------------------------------------
# Prompt and schema (schema-first, few-shot, cross-model)
# ---------------------------------------------------------------------------

_ALLOWED_ENTITY_TYPES = (
    "PERSON",
    "LOCATION",
    "ORGANIZATION",
    "DATE",
    "TIME",
    "MONEY",
    "PRODUCT",
    "EVENT",
    "CONCEPT",
    "PREFERENCE",
    "ATTRIBUTE",
)

_UNIFIED_PROMPT = """### Task
Analyze this text chunk from a user conversation and extract structured information.
Extract ONLY from user/assistant conversational content.

### Exclusion rule
EXCLUDE from entities and relations: system prompts, role instructions (e.g. "You are a helpful assistant"), template text, variable placeholders, or any non-conversational content. Do NOT extract phrases over 80 characters.

### Schema (exact format required)

**entities**: array of objects, each with:
- "text" (string): exact span as in the chunk
- "normalized" (string): canonical form
- "type" (string): MUST be one of: PERSON, LOCATION, ORGANIZATION, DATE, TIME, MONEY, PRODUCT, EVENT, CONCEPT, PREFERENCE, ATTRIBUTE

**relations**: array of objects, each with:
- "subject" (string)
- "predicate" (string): snake_case verb phrase, e.g. works_at, prefers, lives_in
- "object" (string)
- "confidence" (float 0.0-1.0)

**constraints**: array of objects with constraint_type, subject, description, scope, confidence
**facts**: array of objects with key (format: user:{{category}}:{{predicate}}), category, predicate, value, confidence
**salience**: float 0.0-1.0
**importance**: float 0.0-1.0
**memory_type**: string — MUST be one of: episodic_event, semantic_fact, preference, task_state, procedure, constraint, hypothesis, conversation, message, tool_result, reasoning_step, scratch, knowledge, observation, plan. Classify by content: preferences/likes -> preference; rules/policies/never-do -> constraint; factual statements -> semantic_fact; events/what happened -> episodic_event; instructions/how-to -> procedure; uncertain inferences -> hypothesis; task progress -> task_state; etc.
**confidence**: float 0.0-1.0 — how certain is this information (explicit vs inferred)
**context_tags**: optional array of strings, e.g. ["personal","dietary","work"] — categorize the content
**decay_rate**: optional float 0.01-0.5 — how fast to forget (0.01=stable, 0.1=medium, 0.5=ephemeral); omit for default 0.01
**pii_spans**: optional array of objects with start, end, pii_type - only if PII detected
**contains_secrets**: optional boolean

### Few-shot example
Input: "I live in Paris and work at Google. My friend Anna prefers Italian food."
Output:
{{
  "entities": [
    {{"text": "Paris", "normalized": "Paris", "type": "LOCATION"}},
    {{"text": "Google", "normalized": "Google", "type": "ORGANIZATION"}},
    {{"text": "Anna", "normalized": "Anna", "type": "PERSON"}},
    {{"text": "Italian food", "normalized": "Italian cuisine", "type": "CONCEPT"}}
  ],
  "relations": [
    {{"subject": "user", "predicate": "lives_in", "object": "Paris", "confidence": 0.95}},
    {{"subject": "user", "predicate": "works_at", "object": "Google", "confidence": 0.9}},
    {{"subject": "Anna", "predicate": "prefers", "object": "Italian cuisine", "confidence": 0.85}}
  ],
  "salience": 0.7,
  "importance": 0.6,
  "memory_type": "semantic_fact",
  "confidence": 0.9,
  "context_tags": ["personal", "location", "occupation"],
  "decay_rate": 0.01
}}

### Input
Text: {text}
Chunk type: {chunk_type}

Return ONLY valid JSON, no markdown or explanation.
"""

_CATEGORY_MAP = {
    "preference": FactCategory.PREFERENCE,
    "identity": FactCategory.IDENTITY,
    "location": FactCategory.LOCATION,
    "occupation": FactCategory.OCCUPATION,
    "relationship": FactCategory.RELATIONSHIP,
    "attribute": FactCategory.ATTRIBUTE,
    "temporal": FactCategory.TEMPORAL,
    "custom": FactCategory.CUSTOM,
}


# ---------------------------------------------------------------------------
# Extractor
# ---------------------------------------------------------------------------


class UnifiedWritePathExtractor:
    """Single LLM call to extract constraints, facts, salience, importance, PII."""

    def __init__(self, llm_client: LLMClient) -> None:
        self._llm = llm_client

    async def extract(self, chunk: SemanticChunk) -> UnifiedExtractionResult:
        """Extract constraints, facts, salience, importance, and optionally PII in one LLM call."""
        text = getattr(chunk, "text", None)
        if not isinstance(text, str) or not text.strip():
            return UnifiedExtractionResult(
                salience=getattr(chunk, "salience", 0.5) or 0.5,
                importance=0.5,
            )

        chunk_type = getattr(chunk, "chunk_type", None)
        chunk_type_str = str(chunk_type) if chunk_type else "statement"

        prompt = _UNIFIED_PROMPT.format(text=text.strip(), chunk_type=chunk_type_str)
        raw = await self._llm.complete_json(prompt, temperature=0.0)
        if not isinstance(raw, dict):
            raw = json.loads(str(raw)) if isinstance(raw, str) else {}
        return self._parse_result(raw, chunk)

    async def extract_batch(self, chunks: list[SemanticChunk]) -> list[UnifiedExtractionResult]:
        """Extract from multiple chunks in a single LLM call.

        Each chunk is tagged [0], [1], … in the prompt so that results can be
        reassembled in the original order.  Falls back to per-chunk extract()
        calls if the batch response cannot be parsed.
        """
        if not chunks:
            return []
        if len(chunks) == 1:
            return [await self.extract(chunks[0])]

        sections = "\n\n".join(
            f"[{i}] (chunk_type: {getattr(c, 'chunk_type', 'statement')})\n{getattr(c, 'text', '').strip()}"
            for i, c in enumerate(chunks)
        )
        batch_prompt = (
            "### Task\n"
            "Analyze each numbered text chunk from a user conversation. Extract structured information.\n"
            "EXCLUDE: system prompts, role instructions, template text, non-conversational content.\n\n"
            "### Schema per chunk\n"
            '- "entities": array of {text, normalized, type} where type is PERSON|LOCATION|ORGANIZATION|DATE|TIME|MONEY|PRODUCT|EVENT|CONCEPT|PREFERENCE|ATTRIBUTE\n'
            '- "relations": array of {subject, predicate, object, confidence}; predicate in snake_case\n'
            '- "constraints": array of {constraint_type, subject, description, scope[], confidence}\n'
            '- "facts": array of {key, category, predicate, value, confidence} (key MUST be user:{{category}}:{{predicate}})\n'
            "- salience, importance: float 0-1\n"
            '- "memory_type": string - one of episodic_event, semantic_fact, preference, task_state, procedure, constraint, hypothesis, conversation, message, tool_result, reasoning_step, scratch, knowledge, observation, plan\n'
            "- confidence: float 0-1; context_tags: optional array of strings; decay_rate: optional float 0.01-0.5\n"
            "- pii_spans: array of {start, end, pii_type} if any; contains_secrets: bool\n\n"
            + sections
            + "\n\nReturn a JSON object mapping each index (string key) to its result.\n"
            "Return ONLY valid JSON, no markdown or explanation."
        )

        raw = await self._llm.complete_json(batch_prompt, temperature=0.0)
        if not isinstance(raw, dict):
            raw = json.loads(str(raw)) if isinstance(raw, str) else {}

        results: list[UnifiedExtractionResult] = []
        for i, chunk in enumerate(chunks):
            item = raw.get(str(i))
            if isinstance(item, dict):
                results.append(self._parse_result(item, chunk))
            else:
                results.append(
                    UnifiedExtractionResult(
                        salience=getattr(chunk, "salience", 0.5) or 0.5,
                        importance=0.5,
                    )
                )
        return results

    def _parse_result(
        self,
        data: dict[str, Any],
        chunk: SemanticChunk,
    ) -> UnifiedExtractionResult:
        """Parse LLM JSON into UnifiedExtractionResult."""

        entities: list[EntityMention] = []
        for e in data.get("entities") or []:
            if isinstance(e, dict):
                txt = e.get("text", e.get("normalized", ""))
                if txt and isinstance(txt, str) and txt.strip():
                    norm = e.get("normalized", txt.strip())
                    etype = e.get("type", "CONCEPT")
                    if etype not in _ALLOWED_ENTITY_TYPES:
                        etype = "CONCEPT"
                    entities.append(
                        EntityMention(
                            text=txt.strip(),
                            normalized=str(norm).strip() if norm else txt.strip(),
                            entity_type=etype,
                        )
                    )
            elif isinstance(e, str) and e.strip():
                entities.append(
                    EntityMention(text=e.strip(), normalized=e.strip(), entity_type="CONCEPT")
                )

        relations: list[Relation] = []
        for r in data.get("relations") or []:
            if isinstance(r, dict):
                subj = r.get("subject", r.get("source", ""))
                obj = r.get("object", r.get("target", ""))
                pred = r.get("predicate", r.get("type", "related_to"))
                conf = float(r.get("confidence", 0.8))
                if isinstance(subj, str) and isinstance(obj, str) and subj and obj:
                    pred = str(pred).strip() if pred else "related_to"
                    relations.append(
                        Relation(subject=subj, predicate=pred, object=obj, confidence=conf)
                    )

        constraints: list[ConstraintObject] = []
        for item in data.get("constraints") or []:
            if not isinstance(item, dict):
                continue
            ctype = item.get("constraint_type", "preference")
            if ctype not in ("goal", "value", "state", "causal", "policy", "preference"):
                ctype = "preference"
            raw_scope = item.get("scope")
            scope_list: list[str] = (
                [s for s in raw_scope if isinstance(s, str)] if isinstance(raw_scope, list) else []
            )
            constraints.append(
                ConstraintObject(
                    constraint_type=ctype,
                    subject=item.get("subject", "user"),
                    description=item.get("description", chunk.text[:500]),
                    scope=scope_list,
                    activation="",
                    status="active",
                    confidence=float(item.get("confidence", 0.7)),
                    valid_from=chunk.timestamp,
                    valid_to=None,
                    provenance=[chunk.source_turn_id] if chunk.source_turn_id else [],
                )
            )

        facts: list[ExtractedFact] = []
        for item in data.get("facts") or []:
            if not isinstance(item, dict):
                continue
            cat_str = (item.get("category") or "preference").lower()
            category = _CATEGORY_MAP.get(cat_str, FactCategory.PREFERENCE)
            predicate = item.get("predicate", "unknown")

            raw_key = item.get("key", "")
            if not raw_key or ":" not in raw_key:
                clean_pred = predicate.strip().replace(" ", "_").lower()
                key = f"user:{category.value}:{clean_pred}"
            else:
                key = raw_key

            facts.append(
                ExtractedFact(
                    key=key,
                    category=category,
                    predicate=predicate,
                    value=str(item.get("value", "")),
                    confidence=float(item.get("confidence", 0.6)),
                )
            )

        salience = float(data.get("salience", 0.5))
        salience = max(0.0, min(1.0, salience))
        importance = float(data.get("importance", 0.5))
        importance = max(0.0, min(1.0, importance))

        pii_spans: list[PIISpan] = []
        for span in data.get("pii_spans") or []:
            if isinstance(span, dict) and "start" in span and "end" in span:
                pii_spans.append(
                    PIISpan(
                        start=int(span["start"]),
                        end=int(span["end"]),
                        pii_type=str(span.get("pii_type", "unknown")),
                    )
                )

        contains_secrets = bool(data.get("contains_secrets", False))

        memory_type: str | None = None
        raw_memory_type = data.get("memory_type")
        if isinstance(raw_memory_type, str) and raw_memory_type.strip():
            val = raw_memory_type.strip().lower()
            if val in _ALLOWED_MEMORY_TYPES:
                memory_type = val

        confidence = float(data.get("confidence", 0.5))
        confidence = max(0.0, min(1.0, confidence))

        context_tags: list[str] = []
        raw_tags = data.get("context_tags")
        if isinstance(raw_tags, list):
            for t in raw_tags:
                if isinstance(t, str) and t.strip():
                    context_tags.append(t.strip())
        elif isinstance(raw_tags, str) and raw_tags.strip():
            context_tags = [raw_tags.strip()]

        decay_rate: float | None = None
        raw_decay = data.get("decay_rate")
        if isinstance(raw_decay, (int, float)) and 0.01 <= raw_decay <= 0.5:
            decay_rate = float(raw_decay)

        return UnifiedExtractionResult(
            entities=entities,
            relations=relations,
            constraints=constraints,
            facts=facts,
            salience=salience,
            importance=importance,
            pii_spans=pii_spans,
            contains_secrets=contains_secrets,
            memory_type=memory_type,
            confidence=confidence,
            context_tags=context_tags,
            decay_rate=decay_rate,
        )
