"""
Judge prompts for six categories: multi-hop, temporal, common-sense,
single-hop, adversarial, Cognitive. Used by LLM-as-a-judge for evaluation.
"""

CONV_START_PROMPT = (
    "Below is a conversation between two people: {} and {}. "
    "The conversation takes place over multiple days, "
    "and the date of each conversation is written at the beginning of the conversation.\n\n"
)

PROMPT_TEMPLATES = {
    "multi-hop": """
You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (multi-hop fact QA).
Multi-hop questions require combining information from multiple parts of a conversation.

Labels:
- "correct": The answer conveys the same meaning as the reference. Semantic equivalence counts
  (e.g., "Lord of the Rings" and "LOTR trilogy" are equivalent; "2 times" and "twice" are equivalent).
- "partial": The answer gets some of the required facts right but misses others, or is vague but
  points in the right direction.
- "wrong": The answer is factually incorrect, hallucinates details not in the reference, or
  refuses to answer despite the information being available.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"partial"|"wrong", "reason": "<brief_reason>"}}
""",
    "single-hop": """
You are a Fact-Checking Judge.
Your task: Compare the model's prediction with the reference answer (single-hop fact QA).

Labels:
- "correct": The answer conveys the same meaning as the reference. Semantic equivalence counts
  (e.g., "running" and "jogging" are equivalent; paraphrases of the same fact are correct).
- "partial": The answer is on the right topic and partially correct but misses key details.
- "wrong": The answer is factually incorrect, hallucinates details not in the reference, or
  refuses to answer despite the information being available.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"partial"|"wrong", "reason": "<brief_reason>"}}
""",
    "temporal": """
You are a Temporal Logic Judge.
Your task: Check the calculation, duration, or sequence of events.

Labels:
- "correct": The calculated time, duration, or date matches the reference (semantic equivalents
  are allowed, e.g., "4 months" and "four months" are the same; "May 7" and "7 May 2023" match).
- "partial": The answer demonstrates correct temporal reasoning but has a minor error (e.g.,
  off by one day, or gives an approximate range that contains the correct answer).
- "wrong": The calculation is incorrect, the sequence is reversed, or the specific time is wrong.
  Also wrong if the model refuses to answer when temporal information is available.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"partial"|"wrong", "reason": "<brief_reason>"}}
""",
    "common-sense": """
You are a Knowledge Logic Judge.
Your task: Assess if the prediction applies correct commonsense/world knowledge consistent with the reference.
Common-sense questions require combining conversation context with general world knowledge.

Labels:
- "correct": The prediction reaches the same conclusion as the reference, even if worded differently.
  Equivalent answers expressed differently are still correct (e.g., "Likely yes" and "Probably" match).
- "partial": The reasoning is mostly correct and on topic but the final conclusion is vague,
  incomplete, or slightly off from the reference.
- "wrong": The reasoning contradicts commonsense or the reference, or the model refuses to answer
  when the context provides enough information to reason about the question.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"partial"|"wrong", "reason": "<brief_reason>"}}
""",
    "adversarial": """
You are a Skeptical Judge evaluating robustness.
The question is inherently misleading (e.g., asks about something not in the conversation).
Your task: Judge whether the model's answer conveys that "this was not mentioned in the conversation" (or equivalent refusal).

Labels:
- "correct": The prediction clearly conveys that the information was not mentioned / cannot be answered from the conversation. Score it.
- "wrong": The prediction does NOT convey that meaning—e.g., it gives a concrete answer or does not refuse. Do not score.

Model Prediction:
{pred}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"wrong", "reason": "<brief_reason>"}}
""",
    "Cognitive": """
You are a Memory Awareness Judge.
Your task: Judge whether the Model Prediction considers or is linked to the Evidence. If there is a clear connection, the answer is correct (score 1); if not, it is wrong (no score).

Labels:
- "correct": The prediction explicitly or implicitly reflects/uses the evidence (memory or constraint). Give 1 point.
- "wrong": The prediction does not show such a link to the evidence. No point.

Memory/Evidence:
{evidence}

Model Prediction:
{pred}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"wrong", "reason": "<brief_reason>"}}
""",
    "default": """
You are an expert evaluator.
Your task: Compare the prediction with the reference.

Labels:
- "correct": Factually consistent with the reference.
- "partial": Contains correct info but is incomplete.
- "wrong": Factually incorrect.

Reference Answer:
{gold}

Model Prediction:
{pred}

Relevant Evidence:
{evidence}

You MUST respond with ONLY a single JSON object. No explanation, no reasoning, no other text.
Format: {{"label": "correct"|"partial"|"wrong", "reason": "<brief_reason>"}}
""",
}
