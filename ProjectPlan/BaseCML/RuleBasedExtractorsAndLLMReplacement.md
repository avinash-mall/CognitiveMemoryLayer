# Rule-Based Extractors Audit and LLM Replacement Report

> [!NOTE]
> **Implementation Status:**  
> Implemented per plan. See BaseCMLStatus.md "Rule-Based Extractors LLM Replacement" and UsageDocumentation.md § Feature Flags.

> [!NOTE]
> **Validation Status:**  
> Verified against codebase (Commit/Version: `HEAD` as of Feb 18 2026).  
> All described locations, patterns, and logic match the implementation in `src/extraction/`, `src/memory/`, and `src/retrieval/`.


## Introduction

This document inventories all rule-based extractors and keyword-matching logic in the Cognitive Memory Layer (CML) codebase and describes how each can be made **dynamic** by calling the internal LLM (configured via `LLM_INTERNAL__*` in `.env`, e.g. Ollama). The goal is to allow implementers to replace fixed regex/marker rules with LLM-driven extraction without changing downstream contracts (hippocampal store, consolidation, retrieval). It covers extraction and classification (ConstraintExtractor, WriteTimeFactExtractor, RuleBasedChunker, QueryClassifier, salience boosts), security/risk logic (PIIRedactor, WriteGate PII/secret patterns), and other validation/utility rules.

**LLM configuration reference:** Set in `.env` (e.g. lines 22–25):

- `LLM_INTERNAL__PROVIDER=ollama`
- `LLM_INTERNAL__MODEL=llama3.2:3b`
- `LLM_INTERNAL__BASE_URL=http://host.docker.internal:11434/v1`
- `LLM_INTERNAL__API_KEY=` ()

When any `LLM_INTERNAL__*` is set, `get_internal_llm_client()` in `src/utils/llm.py` returns a client for that config; otherwise it falls back to the primary `LLM__*` client. Existing LLM consumers (SemanticChunker, EntityExtractor, RelationExtractor, GistExtractor, QueryClassifier) already use this pattern.

---

## 1. ConstraintExtractor

**Location:** `src/extraction/constraint_extractor.py`

**Role:** Rule-based extractor that identifies goals, values, states, causal reasoning, and policies from semantic chunks. Outputs `ConstraintObject` instances stored in `MemoryRecord.metadata["constraints"]`. Used on the write path (hippocampal store, consolidation worker, orchestrator).

### Pattern groups (regex + confidence boost)

Each type has a list of `(compiled_regex, confidence_boost)`. Minimum cumulative boost to consider a constraint detected: `_MIN_CONFIDENCE_THRESHOLD = 0.05`.

| Type    | Patterns (representative) | Boosts |
|---------|----------------------------|--------|
| **goal**  | `(?:i'?m trying to|i want to|my goal is|i'?m working toward|i aim to)` | 0.1 |
|         | `(?:i'?m preparing for|i'?m focused on|i'?m committed to|i plan to)` | 0.1 |
|         | `(?:i hope to|i intend to|i'?m striving|working on)` | 0.05 |
| **value** | `(?:i value|it'?s important (?:to me \|that )|i care about)` | 0.1 |
|         | `(?:i believe in|i strongly feel|matters? (?:a lot )?to me)` | 0.1 |
|         | `(?:i prioriti[sz]e|for me,? the most important)` | 0.05 |
| **state** | `(?:i'?m currently|i'?m dealing with|i'?m going through)` | 0.1 |
|         | `(?:i'?m anxious about|i'?m stressed about|i'?m worried about)` | 0.1 |
|         | `(?:i'?m struggling with|right now i|at the moment i)` | 0.05 |
| **causal** | `(?:because|since|that'?s why|the reason is)` | 0.05 |
|         | `(?:in order to|so that|to make sure|to avoid)` | 0.1 |
|         | `(?:due to|as a result|consequently)` | 0.05 |
| **policy** | `(?:i never|i always|i don'?t|i won'?t|i refuse to)` | 0.1 |
|         | `(?:i must|i have to|i need to avoid|i can'?t)` | 0.1 |
|         | `(?:i should|i shouldn'?t|i'?m not allowed to)` | 0.05 |

### [Recommended] Additional Patterns

| Type    | Patterns (New) | Boosts |
|---------|----------------|--------|
| **goal** | `(?:i aspire to|my ambition is|i dream of|i'?m determined to)` | 0.1 |
| **state** | `(?:i'?m feeling|my mood is|i feel|i'?m exhausted)` | 0.1 |
| **policy** | `(?:it is forbidden|off limits|strictly prohibited)` | 0.1 |

### Subject extraction (_extract_subject)

Rule-based logic to determine the constraint subject (usually `"user"`):

- Look for first colon in chunk text; if position is between 1 and 30, treat text before colon as candidate speaker.
- Accept candidate only if non-empty, first character uppercase, and `" said"` not in candidate (lowercased).
- Otherwise return `"user"`.

**LLM replacement:** When using `LLMConstraintExtractor`, include `subject` in the LLM output schema so the model returns the speaker/subject directly; omit or keep this heuristic as fallback when LLM does not return subject.

### Output schema

`ConstraintObject`: `constraint_type`, `subject`, `description`, `scope`, `activation`, `status`, `confidence`, `valid_from`, `valid_to`, `provenance`. Subject is derived from chunk text (e.g. "Speaker: ..." → speaker name, else `"user"`).

### How to replace with LLM calls

1. **New class (or feature-flagged path):** e.g. `LLMConstraintExtractor` that takes `LLMClient` from `get_internal_llm_client()`.
2. **Prompt:** Given chunk text, ask the LLM to return a JSON array of constraints, each with: `constraint_type` (one of goal, value, state, causal, policy, preference), `subject`, `description`, `scope` (list of strings), `confidence` (0.0–1.0). ly `activation`, `status`.
3. **Parsing:** Parse JSON response and map each item to `ConstraintObject`; use `chunk.timestamp` for `valid_from`, `chunk.source_turn_id` for `provenance`.
4. **Keep unchanged:** `ConstraintExtractor.detect_supersession()` and `ConstraintExtractor.constraint_fact_key()` so downstream logic (deduplication, fact keys) is unchanged.
5. **Feature flag:** e.g. `FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR`; when True, hippocampal store and consolidation use `LLMConstraintExtractor` instead of `ConstraintExtractor`. Default False for low latency on the hot write path.

---

## 2. WriteTimeFactExtractor

**Location:** `src/extraction/write_time_facts.py`

**Role:** Extracts structured facts at write-time from preference, fact, and constraint chunk types. Purely rule-based (no LLM). Used in the orchestrator for write-time facts (e.g. deep research path).

### Preference patterns

| Pattern (concept) | Regex | Key template | Category | Conf boost |
|-------------------|-------|--------------|----------|------------|
| I prefer/like/love/enjoy/hate/dislike X | `(?:i|my)\s+(?:prefer|like|love|enjoy|hate|dislike)\s+(.+)` | `user:preference:{pred}` | PREFERENCE | 0.7 |
| My favorite X is Y | `(?:my|the)\s+(?:favorite|favourite)\s+(\w+)\s+(?:is|are)\s+(.+)` | `user:preference:{pred}` | PREFERENCE | 0.75 |

### Identity patterns

| Pattern (concept) | Regex | Key | Category | Conf boost |
|-------------------|-------|-----|----------|------------|
| My name is X / I'm X / Call me X | `(?:my name is|i'?m|call me|i am)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)` | `user:identity:name` | IDENTITY | 0.85 |
| I live in X / I'm from X / I moved to X | `(?:i live in|i'?m from|i moved to|i'm based in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)` | `user:location:current_city` | LOCATION | 0.7 |
| I work as X / I'm a X / My job is X | `(?:i work as|i'?m a|my job is|my occupation is)\s+(.+)` | `user:occupation:role` | OCCUPATION | 0.7 |

### Predicate keywords (preference sub-categorisation)

Used by `_derive_predicate(value)` when the preference value does not come from a two-group pattern (e.g. "my favorite X is Y"). The first matching keyword group gives the predicate name; otherwise a 12-char hash of the value is used.

| Predicate | Keywords |
|-----------|----------|
| cuisine   | food, restaurant, eat, cook, meal, cuisine, dish |
| music     | music, song, band, listen, genre, artist |
| color     | color, colour |
| language  | language, speak |
| sport     | sport, play, team, game, exercise |
| movie     | movie, film, cinema, watch |
| book      | book, read, author, novel |

### [Recommended] New Categories (Relationships & Skills)

| Pattern (concept) | Regex | Key template | Category | Conf boost |
|-------------------|-------|--------------|----------|------------|
| **Relationship**<br>My X is Y | `(?:my|our)\s+(wife|husband|partner|spouse|mom|dad|mother|father|brother|sister|son|daughter|friend|colleague|boss)\s+(?:is|named?)\s+(.+)` | `user:relationship:{group1}` | RELATIONSHIP | 0.8 |
| **Skill/Ability**<br>I am good at X | `(?:i|i'?m)\s+(?:good at|skilled in|experienced with|an expert in)\s+(.+)` | `user:skill:{value_hash}` | ATTRIBUTE | 0.7 |
| **Allergy**<br>I am allergic to X | `(?:i|i'?m)\s+(?:allergic to|have a .* allergy to|intolerant to)\s+(.+)` | `user:health:allergy` | ATTRIBUTE | 0.9 |

### Output schema

`ExtractedFact`: `key`, `category` (FactCategory), `predicate`, `value`, `confidence`. Base confidence for write-time facts: `0.6`.

### How to replace with LLM calls

1. ** new class:** `LLMWriteTimeFactExtractor` taking `LLMClient` from `get_internal_llm_client()`.
2. **Prompt:** Given chunk text and chunk type (preference/fact/constraint), ask for a JSON array of facts: `key`, `category` (preference, identity, location, occupation), `predicate`, `value`, `confidence`. Align categories with `FactCategory` enum.
3. **Predicate derivation:** Either let the LLM output predicate names (and map to known predicates or keep as-is), or keep the existing `_PREDICATE_KEYWORDS` fallback for backward compatibility when LLM omits predicate.
4. **Integration:** Use in orchestrator write-time fact path behind a feature flag or config so rule-based remains default for latency.

---

## 3. RuleBasedChunker

**Location:** `src/memory/working/chunker.py`

**Role:** Fast, rule-based chunker used when `FEATURES__USE_FAST_CHUNKER=true` or when no LLM is available. Splits text by sentences and assigns chunk type and salience via marker lists.

### Marker lists (substring match on lowercased sentence)

| List | Markers |
|------|--------|
| **PREFERENCE_MARKERS** | i prefer, i like, i love, i hate, i don't like, i want |
| **FACT_MARKERS** | my name is, i am, i live, i work, i have |
| **INSTRUCTION_MARKERS** | please, can you, could you, i need, help me |
| **CONSTRAINT_MARKERS** | i'm trying to, i don't want, it's important that, i need to avoid, i'm anxious about, i'm preparing for, i must, i should, i can't, i won't, my goal is, i'm focused on, i'm committed to, i value, i believe, because of, in order to, i never, i always, i'm working toward, i care about |

### Chunk type and salience logic

- Sentence split: `re.findall(r"[^.!?]*[.!?]?", text)` then strip.
- Priority: CONSTRAINT (salience 0.85) > PREFERENCE (0.8) > FACT (0.7) > INSTRUCTION (0.6) > QUESTION (0.4 if `"?"` in sentence) > STATEMENT (0.3).
- Final salience is capped at 1.0 and adds salience boosts from constraint cues and sentiment (see Section 6).

### How to replace with LLM calls

**Replacement already exists:** **SemanticChunker** in the same file uses the LLM to produce semantic chunks (type, entities, key_phrases, salience, confidence). When `use_fast_chunker=False` and an LLM client is available, `WorkingMemoryManager` (`src/memory/working/manager.py`) uses `SemanticChunker(llm_client)` where `llm_client` is from `get_internal_llm_client()`.

**Making chunking dynamic:**

1. **Config:** Set `FEATURES__USE_FAST_CHUNKER=false` and ensure `LLM_INTERNAL__*` (or `LLM__*`) is set so `get_internal_llm_client()` returns a valid client.
2. **Docs:** Recommend in configuration docs that for dynamic chunking, use the LLM-based path (SemanticChunker) rather than RuleBasedChunker.
3. **:** Add a feature flag to completely hide RuleBasedChunker when LLM is required (e.g. fail fast if LLM unavailable instead of falling back to rules).

---

## 4. QueryClassifier

**Location:** `src/retrieval/classifier.py`

**Role:** Classifies user queries to determine retrieval strategy (intent, sources, top_k). Uses a **fast path** (pattern-based) first; if confidence &gt; 0.8 returns immediately. Otherwise uses **LLM fallback** (`_llm_classify`) when `llm_client` is provided.

### FAST_PATTERNS (intent → list of regex strings)

| Intent | Patterns |
|--------|----------|
| PREFERENCE_LOOKUP | `what (do|does) (i|my) (like|prefer|want|enjoy)`, `(my|i) (favorite|preferred)`, `do i (like|prefer|enjoy)` |
| IDENTITY_LOOKUP | `what('s\| is) my (name|email|phone|address|job|title)`, `who am i`, `my (name|email|phone)` |
| TASK_STATUS | `where (am i|are we) (in|on|at)`, `what('s\| is) (the|my) (status|progress)`, `(current|next) (step|task)` |
| TEMPORAL_QUERY | `(last|past) (week|month|day|year)`, `(yesterday|today|recently)`, `when did (i|we)`, `what happened` |
| PROCEDURAL | `how (do|can|should) (i|we)`, `what('s\| are) the steps`, `(procedure|process) for` |
| CONSTRAINT_CHECK | `should i`, `can i`, `is it ok (to|if)`, `would it be (ok|fine|good|bad)`, `what if i`, `do you think i should`, `recommend`, `is (this|that|it) (consistent|aligned|compatible)`, `would (this|that) (conflict|contradict|go against)` |

### [Recommended] Additional Intents

| Intent | Patterns |
|--------|----------|
| **RISK_ASSESSMENT** (maps to CONSTRAINT_CHECK) | `is it safe (to\|for)`, `(any\|what) (risk\|danger)`, `(is\|are) there (risks?\|dangers?)`, `potential downsides` |
| **OPINION_REQUEST** (maps to GENERAL_QUESTION or CONSTRAINT_CHECK) | `what do you think (about\|of)`, `your (thoughts\|opinion) on`, `how do you feel about` |

### _DECISION_PATTERNS (regex)

Used to set `analysis.is_decision_query = True` and to upgrade intent to CONSTRAINT_CHECK when intent is GENERAL_QUESTION or UNKNOWN.

- `\bshould i\b`, `\bcan i\b`, `\bis it ok\b`, `\bwould it be\b`, `\bwhat if i\b`, `\brecommend\b`, `\bstart (watching|doing|eating|buying)\b`, `\btry (this|that|the)\b`, `\bgo (out|for|to)\b`

### _CONSTRAINT_DIM_PATTERNS (dimension → list of compiled regex)

Used to set `analysis.constraint_dimensions` (list of dimension names).

| Dimension | Pattern |
|-----------|---------|
| goal | `\b(goal|objective|target|aim|ambition|milestone)\b` |
| value | `\b(value|principle|ethic|belief|priority)\b` |
| state | `\b(feel|mood|stress|anxious|tired|busy|sick)\b` |
| causal | `\b(because|reason|consequence|result|cause)\b` |
| policy | `\b(rule|policy|restriction|limit|boundary)\b` |

### Vague-query heuristic (_is_vague)

Used to decide whether to use `recent_context` to enrich the query before classification. Query is considered vague if:

- Length (stripped, lowercased) &lt; 15 characters, or
- Starts with (or equals) one of: `"any "`, `"what about"`, `"suggestions?"`, `"thoughts?"`, `"and?"`, `"so?"`, `"what do you think"`.

**LLM replacement:** Extend `CLASSIFICATION_PROMPT` to ask for a boolean `is_vague` or infer from intent/confidence; or add a lightweight LLM call "Is this query too vague to classify without context? (yes/no)" when context is available.

### Simple entity extraction (_extract_entities_simple)

Used only on the **fast path** when pattern-based classification succeeds. Rule-based logic:

- Split query into words; for each word (stripped of `?.,!`), treat as entity only if: (1) first character uppercase, (2) length &gt; 1, (3) not at sentence start (position 0 or after a word ending in `.!?`). This avoids "The", "What", "How" as entities.

**LLM replacement:** `_llm_classify()` already returns `entities` from the LLM. When using the fast path, either keep this heuristic or add an  small LLM call to extract entities for the query; or force LLM path when entity quality matters.

### How to replace / make dynamic

- **LLM path already exists:** `_llm_classify()` uses `CLASSIFICATION_PROMPT` and returns `QueryAnalysis` with intent, entities, time_reference, confidence. The retrieval path should pass `get_internal_llm_client()` (or the same internal client used by the orchestrator) into `QueryClassifier(llm_client=...)` so that when the fast path does not yield confidence &gt; 0.8, the LLM is used.
- **Making classification more dynamic:** Document that with LLM configured, the rule-based path is only a fast path for high-confidence matches. ly add a feature flag (e.g. `FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY`) to skip the fast path and always call the LLM for classification.

---

## 5. Salience boosts (chunker)

**Location:** `src/memory/working/chunker.py`

Used by both **SemanticChunker** (in its fallback when JSON parsing fails) and **RuleBasedChunker** to add salience boosts before capping at 1.0.

### _CONSTRAINT_CUE_PHRASES

Substring match (lowercased) on chunk/sentence text; boost capped at 0.4.

- i'm trying to, i don't want, it's important that, i need to avoid, i'm anxious about, i'm preparing for, i must, i should, i can't, i won't, my goal is, i'm focused on, i'm committed to, i value, i believe, because of, in order to, i never, i always, i'm working toward, i care about, so that, that's why, i'm dealing with

Logic: 2+ matches → +0.4; 1 match → +0.3.

### _compute_salience_boost_for_sentiment (cap 0.3)

- Excitement: 2+ `!` or any word &gt;2 chars all caps → +0.15
- Emotion words: love, hate, amazing, terrible, excited, worried, thrilled, devastated, passionate → +0.1
- Personal markers: finally, at last, dream, goal, achieved → +0.1

### How to replace with LLM

- **Recommendation:** Keep as lightweight post-processing after LLM chunking to avoid extra latency. The LLM (SemanticChunker) already returns a salience value per chunk; these boosts refine it.
- **:** Add an  "salience refinement" LLM call (e.g. "Rate importance 0–1 for this chunk and one reason") only when a product needs finer-grained salience; otherwise keep rule-based boosts.

---

## 6. Implementation notes

- **Single client factory:** All replacements should use `get_internal_llm_client()` from `src/utils/llm.py` so one `.env` block (e.g. `LLM_INTERNAL__*`) drives chunking, entity/relation extraction, consolidation, and the new LLM-based constraint/write-time fact/classification paths.
- **Preserve APIs:** New LLM-based extractors should expose the same method signatures (e.g. `extract(chunk)` returning `list[ConstraintObject]` or `list[ExtractedFact]`) so callers (hippocampal store, consolidation, orchestrator) do not need changes when toggling.
- **Feature flags:** Use flags such as `FEATURES__USE_LLM_CONSTRAINT_EXTRACTOR` or `FEATURES__USE_LLM_QUERY_CLASSIFIER_ONLY` so that rule-based remains the default for latency and cost; enable LLM path where dynamic behaviour is required.
- **Schema consistency:** ConstraintObject and ExtractedFact schemas must match between rule-based and LLM implementations so downstream storage and retrieval logic work unchanged.

---

## 7. Architecture diagram

```mermaid
flowchart LR
  subgraph current [Current rule-based]
    RBChunker[RuleBasedChunker]
    CE[ConstraintExtractor]
    WTF[WriteTimeFactExtractor]
    QC[QueryClassifier]
  end
  subgraph llm [LLM_INTERNAL]
    InternalClient[get_internal_llm_client]
  end
  subgraph replacement [LLM-based replacement]
    SemanticChunker[SemanticChunker]
    LLMCE[LLMConstraintExtractor]
    LLMWTF[LLMWriteTimeFactExtractor]
    QCLlm[QueryClassifier._llm_classify]
  end
  RBChunker -.->|"use_fast_chunker=false"| SemanticChunker
  CE -.->|"feature flag"| LLMCE
  WTF -.->|""| LLMWTF
  QC -->|"fast path"| QC
  QC -->|"fallback"| QCLlm
  SemanticChunker --> InternalClient
  LLMCE --> InternalClient
  LLMWTF --> InternalClient
  QCLlm --> InternalClient
```

---

## 8. PIIRedactor and WriteGate (security / risk)

### PIIRedactor

**Location:** `src/memory/hippocampal/redactor.py`

**Role:** Redacts PII from text before storage. Uses a fixed regex dictionary; supports additional patterns via constructor.

**PATTERNS (regex, case-insensitive):**

| Type         | Pattern |
|--------------|---------|
| EMAIL        | `\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z\|a-z]{2,}\b` |
| PHONE        | `\b\d{3}[-.]?\d{3}[-.]?\d{4}\b` |
| SSN          | `\b\d{3}-\d{2}-\d{4}\b` |
| CREDIT_CARD  | `\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b` |
| IP_ADDRESS   | `\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b` |

**LLM replacement:** . For stricter compliance or custom PII types, add an LLM step that returns span labels (e.g. "EMAIL", "PHONE") and merge with regex redaction. Keep regex as primary for latency and deterministic behaviour; LLM can run in parallel or as a second pass for high-risk content.

### WriteGate (PII and secret detection)

**Location:** `src/memory/hippocampal/write_gate.py`

**Role:** Decides whether to store incoming chunks. Uses configurable `pii_patterns` and `secret_patterns` (regex). Defaults when not provided:

**Default pii_patterns:** SSN, 16-digit number, email, phone (same style as redactor).

**Default secret_patterns:**

- `password\s*[:=]\s*\S+`
- `api[_-]?key\s*[:=]\s*\S+`
- `secret\s*[:=]\s*\S+`
- `token\s*[:=]\s*\S+`

If any secret pattern matches, the chunk is **skipped** (not stored). If any PII pattern matches, `redaction_required` is set and the chunk can be redacted then stored.

**LLM replacement:** . Use internal LLM to classify "contains_secrets" / "contains_pii" for edge cases (e.g. obfuscated secrets); keep regex as default for speed and auditability.

### WriteGate importance heuristic (_compute_importance)

**Location:** `src/memory/hippocampal/write_gate.py` (lines 141–176).

Used to compute an importance score (capped at 1.0) for the write decision. Rule-based boosts:

- **Type boosts:** ChunkType → fixed boost: PREFERENCE +0.3, CONSTRAINT +0.3, FACT +0.2, INSTRUCTION +0.1, EVENT +0.1.
- **Keyword boosts** (substring in `chunk.text.lower()`):
  - `["always", "never", "important", "remember"]` → +0.2
  - `["my name", "i am", "i live", "i work"]` → +0.15
- **Constraint cues** (12 phrases): `"i'm trying to"`, `"i don't want"`, `"it's important that"`, `"my goal is"`, `"i value"`, `"i believe"`, `"i must"`, `"i should"`, `"i'm preparing for"`, `"i'm focused on"`, `"in order to"`, `"because of"` → +0.2.
- **Entity count:** +0.1 per entity, cap 3. Final score is min(score, 1.0).

**LLM replacement:** . Use internal LLM to output an importance score (0–1) per chunk; keep rule-based as default for latency.


---

## 9. ConflictDetector (reconsolidation)

**Location:** `src/reconsolidation/conflict_detector.py` — `_fast_detect()` (lines 92–171).

**Role:** Fast heuristic conflict detection between existing memory and new statement. Used when confidence &gt; 0.8 before falling back to `_llm_detect()`.

### Keyword lists in _fast_detect

| List | Values | Effect |
|------|--------|--------|
| **correction_markers** | `["actually", "no,", "that's wrong", "i meant", "correction:", "not anymore", "changed"]` | If any substring in `new_statement.lower()` → return CORRECTION (confidence 0.85). |
| **negations** | `["not", "don't", "doesn't", "no longer", "never"]` | If negation in new but not in old and word overlap (after removing negation) &gt; 0.5 → DIRECT_CONTRADICTION (0.75). |
| **preference_words** | `["like", "prefer", "favorite", "enjoy", "love", "hate"]` | If both old and new contain one, and topic overlap (words minus preference_words and stopwords) &gt; 0.2 → TEMPORAL_CHANGE (0.6). |
| **stopwords** (topic overlap) | `{"i", "my", "a", "the", "is", "are"}` | Removed from word sets when computing topic overlap for TEMPORAL_CHANGE. |

**LLM replacement:** ConflictDetector already has `_llm_detect()`; the rule-based path is the fast path when confidence &gt; 0.8. Document that with LLM configured, rule-based is used first;  feature flag to force LLM-only for conflict detection.

---

## References

- **Config:** `src/core/config.py` — `LLMInternalSettings`, `use_fast_chunker`
- **LLM client:** `src/utils/llm.py` — `get_internal_llm_client()`
- **Orchestrator wiring:** `src/memory/orchestrator.py` — entity/relation extractors, fact extractor, chunker choice
- **Working memory:** `src/memory/working/manager.py` — chunker selection (SemanticChunker vs RuleBasedChunker)
- **PII redaction:** `src/memory/hippocampal/redactor.py` — PIIRedactor PATTERNS
- **Write gate:** `src/memory/hippocampal/write_gate.py` — WriteGateConfig pii_patterns, secret_patterns, _compute_importance
- **Neo4j validation:** `src/storage/neo4j.py` — `_REL_TYPE_ALLOWLIST`
- **Conflict detection:** `src/reconsolidation/conflict_detector.py` — _fast_detect correction_markers, negations, preference_words
- **Dashboard masking:** `src/api/dashboard_routes.py` — _SECRET_FIELD_TOKENS, _is_secret_field
- **Relation normalization:** `src/extraction/relation_extractor.py` — _normalize_predicate
- **Config URL:** `src/core/config.py` — postgres URL rewrite
- **.env example:** `.env.example` — `LLM_INTERNAL__PROVIDER`, `LLM_INTERNAL__MODEL`, `LLM_INTERNAL__BASE_URL`, `LLM_INTERNAL__API_KEY`
