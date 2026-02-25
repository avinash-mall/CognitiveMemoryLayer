"""
Dataset preparation for the 15-way memory-type classifier.

Pipeline:
1) Audit required dataset files.
2) Extract real samples with schema-aware dataset handlers.
3) Backfill missing labels with synthetic samples seeded from random real conversations.
4) Write train/test/eval splits and reports.
"""

import argparse
import json
import os
import random
import re
import sys
import zipfile
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List

import pandas as pd
try:
    from tqdm.auto import tqdm as _tqdm
except Exception:  # pragma: no cover - optional dependency fallback
    _tqdm = None


MODELS_ROOT = Path(__file__).resolve().parent.parent
DATASETS_DIR = MODELS_ROOT / "datasets"
PREPARED_DIR = MODELS_ROOT / "prepared"

MAX_TEXT_CHARS = 2000
DEFAULT_TARGET_PER_LABEL = 10_000
DEFAULT_MAX_REAL_PER_LABEL = 20_000
DEFAULT_SEED = 42
POOL_LIMIT = 80_000
DEFAULT_LLM_TIMEOUT_SECONDS = 60
DEFAULT_LLM_BATCH_SIZE = 64
DEFAULT_LLM_TEMPERATURE = 1.2

CANONICAL_LABELS = [
    "episodic_event",
    "semantic_fact",
    "preference",
    "constraint",
    "procedure",
    "hypothesis",
    "task_state",
    "conversation",
    "message",
    "tool_result",
    "reasoning_step",
    "scratch",
    "knowledge",
    "observation",
    "plan",
]

SPLIT_RATIOS = {"train": 0.8, "test": 0.1, "eval": 0.1}

DATASET_FILES = {
    "cooperbench": "cooperbench_train.parquet",
    "coqa_train": "coqa_train.parquet",
    "coqa_validation": "coqa_validation.parquet",
    "locomo10": "locomo_plus_locomo10.json",
    "locomoplus": "locomo_plus_locomo_plus.json",
    "personachat": "personachat_train_both_revised.txt",
    "wikihow_sep": "wikihow_sep.csv",
    "wikihow_cleaned": "wikihow_cleaned.csv",
    "reddit_0": "reddit_writing_prompts_train_00000.parquet",
    "reddit_1": "reddit_writing_prompts_train_00001.parquet",
    "chatgpt4o": "chatgpt4o_writing_prompts_sharegpt.jsonl",
    "wikipedia_cleaned": "wikipedia_cleaned_20231101.en",
    "structured_wikipedia_zip": "structured_wikipedia_enwiki_namespace_0.zip",
}

CONSTRAINT_RE = re.compile(r"\b(must|should|shall|required|never|always|policy|do not|don't)\b", re.IGNORECASE)
PREFERENCE_RE = re.compile(r"\b(i like|i love|i prefer|favorite|i enjoy|i dislike|i hate|rather)\b", re.IGNORECASE)
HYPOTHESIS_RE = re.compile(r"\b(maybe|might|possibly|perhaps|could be|i think|uncertain|likely)\b", re.IGNORECASE)
REASONING_RE = re.compile(r"\b(because|therefore|thus|so that|hence|as a result)\b", re.IGNORECASE)
PLAN_RE = re.compile(r"\b(goal|plan|first|then|next|finally|roadmap)\b", re.IGNORECASE)
PAST_EVENT_RE = re.compile(r"\b(yesterday|last|ago|went|was|were|did|happened|completed)\b", re.IGNORECASE)
TOOL_RESULT_RE = re.compile(
    r"\b(traceback|error|exception|exit code|stdout|stderr|diff --git|result|output)\b",
    re.IGNORECASE,
)

LLM_LABEL_GUIDANCE = {
    "episodic_event": "Write a past event as a concise narrative in 1-3 sentences.",
    "semantic_fact": "Write a stable factual statement (not a question), 1-2 sentences.",
    "preference": "Write a first-person preference statement (likes/dislikes/favorites), 1-2 sentences.",
    "constraint": "Write a rule/policy/requirement statement with prescriptive language (must/never/always).",
    "procedure": "Write a short how-to instruction sequence with explicit ordered steps.",
    "hypothesis": "Write an uncertain/speculative statement using terms like maybe/might/could.",
    "task_state": "Write a short current progress/status update with done/in-progress/next.",
    "conversation": "Write a short multi-turn dialogue (at least 3 turns) with speaker prefixes.",
    "message": "Write a single short chat message/utterance.",
    "tool_result": "Write one line of tool output/log/error/result with realistic formatting.",
    "reasoning_step": "Write a reasoning sentence using connectors like because/therefore/thus.",
    "scratch": "Write a temporary draft/scratchpad style note with partial ideas or TODOs.",
    "knowledge": "Write general world/domain knowledge, factual and non-personal, 1-2 sentences.",
    "observation": "Write an observation about visible/sensed state (screen/content/status).",
    "plan": "Write a short goal-oriented plan with ordered future steps.",
}


class _NoopProgress:
    def __init__(self, *args, **kwargs) -> None:
        self.total = kwargs.get("total")

    def update(self, n: int = 1) -> None:
        return

    def set_description(self, desc: str) -> None:
        return

    def set_postfix(self, ordered_dict=None, refresh=True, **kwargs) -> None:
        return

    def close(self) -> None:
        return


def _progress(*, total: int, desc: str, unit: str):
    if _tqdm is None:
        return _NoopProgress(total=total, desc=desc, unit=unit)
    return _tqdm(total=total, desc=desc, unit=unit, dynamic_ncols=True)


def dataset_path(key: str) -> Path:
    return DATASETS_DIR / DATASET_FILES[key]


def load_env_file() -> None:
    candidates = [
        MODELS_ROOT.parent.parent / ".env",
        MODELS_ROOT.parent / ".env",
        MODELS_ROOT / ".env",
    ]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            from dotenv import load_dotenv

            load_dotenv(env_path, override=False)
        except Exception:
            with open(env_path, encoding="utf-8", errors="replace") as f:
                for line in f:
                    line = line.strip()
                    if not line or line.startswith("#") or "=" not in line:
                        continue
                    key, _, val = line.partition("=")
                    key = key.strip()
                    val = val.strip().strip('"').strip("'")
                    os.environ.setdefault(key, val)
        break


def clean_text(value: object, max_chars: int = MAX_TEXT_CHARS) -> str:
    if value is None:
        return ""
    text = str(value).replace("\x00", " ")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > max_chars:
        text = text[:max_chars].rstrip()
    return text


def normalize_key(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def first_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    parts = re.split(r"(?<=[.!?])\s+", text, maxsplit=1)
    return clean_text(parts[0])


def split_steps(text: str, max_steps: int = 4) -> List[str]:
    parts = re.split(r"(?:\n|;|(?<=\.)\s+)", clean_text(text))
    steps: List[str] = []
    for p in parts:
        p = clean_text(p, 300).rstrip(".")
        if len(p) >= 12:
            steps.append(p)
        if len(steps) >= max_steps:
            break
    return steps


def find_reasoning_sentence(text: str) -> str:
    text = clean_text(text)
    if not text:
        return ""
    for sent in re.split(r"(?<=[.!?])\s+", text):
        if REASONING_RE.search(sent):
            return clean_text(sent)
    return first_sentence(text)


def split_chunks(text: str, max_chunks: int = 3, max_chars: int = 700) -> List[str]:
    lines = [clean_text(line, 400) for line in str(text).splitlines() if clean_text(line, 400)]
    if not lines:
        return []
    chunks: List[str] = []
    current: List[str] = []
    current_len = 0
    for line in lines:
        line_len = len(line) + 1
        if current and (current_len + line_len > max_chars):
            chunks.append(" ".join(current))
            current = []
            current_len = 0
            if len(chunks) >= max_chunks:
                break
        current.append(line)
        current_len += line_len
    if current and len(chunks) < max_chunks:
        chunks.append(" ".join(current))
    return [clean_text(c) for c in chunks if clean_text(c)]


def to_flat_strings(value: object) -> List[str]:
    if value is None:
        return []
    out: List[str] = []
    stack = [value]
    while stack:
        v = stack.pop()
        if v is None:
            continue
        if isinstance(v, str):
            t = clean_text(v)
            if t:
                out.append(t)
            continue
        if isinstance(v, dict):
            for vv in v.values():
                stack.append(vv)
            continue
        if hasattr(v, "tolist"):
            stack.append(v.tolist())
            continue
        if isinstance(v, (list, tuple, set)):
            for vv in v:
                stack.append(vv)
            continue
        t = clean_text(v)
        if t:
            out.append(t)
    return out


def parse_coqa_answers(raw_answers: object) -> List[str]:
    if isinstance(raw_answers, dict):
        return to_flat_strings(raw_answers.get("input_text"))
    if hasattr(raw_answers, "get"):
        try:
            return to_flat_strings(raw_answers.get("input_text"))
        except Exception:
            pass
    return to_flat_strings(raw_answers)


def iter_parquet_batches(path: Path) -> Iterator[pd.DataFrame]:
    try:
        import pyarrow.parquet as pq

        pf = pq.ParquetFile(path)
        for i in range(pf.metadata.num_row_groups):
            table = pf.read_row_group(i)
            yield table.to_pandas()
    except Exception:
        yield pd.read_parquet(path)


def is_constraint(text: str) -> bool:
    return CONSTRAINT_RE.search(text or "") is not None


def is_preference(text: str) -> bool:
    return PREFERENCE_RE.search(text or "") is not None


def is_hypothesis(text: str) -> bool:
    return HYPOTHESIS_RE.search(text or "") is not None


def is_reasoning(text: str) -> bool:
    return REASONING_RE.search(text or "") is not None


def is_plan(text: str) -> bool:
    return PLAN_RE.search(text or "") is not None


def is_past_event(text: str) -> bool:
    return PAST_EVENT_RE.search(text or "") is not None


def is_tool_result(text: str) -> bool:
    return TOOL_RESULT_RE.search(text or "") is not None


@dataclass
class PrepConfig:
    target_per_label: int = DEFAULT_TARGET_PER_LABEL
    max_real_per_label: int = DEFAULT_MAX_REAL_PER_LABEL
    split_ratios: Dict[str, float] = field(default_factory=lambda: dict(SPLIT_RATIOS))
    seed: int = DEFAULT_SEED
    llm_model: str = ""
    llm_base_url: str = ""
    llm_api_key: str = ""
    llm_temperature: float = DEFAULT_LLM_TEMPERATURE
    llm_timeout_seconds: int = DEFAULT_LLM_TIMEOUT_SECONDS
    llm_batch_size: int = DEFAULT_LLM_BATCH_SIZE


class ExampleStore:
    def __init__(self, max_real_per_label: int, seed: int) -> None:
        self.max_real_per_label = max_real_per_label
        self.rows_by_label: Dict[str, List[dict]] = {label: [] for label in CANONICAL_LABELS}
        self.real_counts: Dict[str, int] = {label: 0 for label in CANONICAL_LABELS}
        self.synthetic_counts: Dict[str, int] = {label: 0 for label in CANONICAL_LABELS}
        self.source_counts: Dict[str, int] = defaultdict(int)
        self._seen_by_label: Dict[str, set] = {label: set() for label in CANONICAL_LABELS}
        self._global_label_by_text: Dict[str, str] = {}
        self.conversation_pool: List[str] = []
        self.seed_pool_by_label: Dict[str, List[str]] = {label: [] for label in CANONICAL_LABELS}
        self.all_seed_pool: List[str] = []
        self.rng = random.Random(seed)

    def count(self, label: str) -> int:
        return len(self.rows_by_label[label])

    def real_full(self, label: str) -> bool:
        return self.real_counts[label] >= self.max_real_per_label

    def _push_pool(self, pool: List[str], text: str, limit: int) -> None:
        if len(pool) < limit:
            pool.append(text)
            return
        if self.rng.random() < 0.02:
            pool[self.rng.randrange(limit)] = text

    def add(self, label: str, text: object, source: str, is_synthetic: bool = False) -> bool:
        if label not in self.rows_by_label:
            return False
        text_str = clean_text(text)
        if len(text_str) < 8:
            return False

        text_key = normalize_key(text_str)
        if text_key in self._seen_by_label[label]:
            return False

        existing = self._global_label_by_text.get(text_key)
        if existing is not None and existing != label:
            return False

        if not is_synthetic and self.real_full(label):
            return False

        self.rows_by_label[label].append({"text": text_str, "label": label, "source": source})
        self._seen_by_label[label].add(text_key)
        self._global_label_by_text[text_key] = label
        self.source_counts[source] += 1
        if is_synthetic:
            self.synthetic_counts[label] += 1
        else:
            self.real_counts[label] += 1

        if len(text_str) >= 20 and label in {"conversation", "message"}:
            self._push_pool(self.conversation_pool, text_str, POOL_LIMIT)
        if len(text_str) >= 20:
            self._push_pool(self.seed_pool_by_label[label], text_str, POOL_LIMIT // 3)
            self._push_pool(self.all_seed_pool, text_str, POOL_LIMIT)
        return True

    def rows(self) -> List[dict]:
        combined: List[dict] = []
        for label in CANONICAL_LABELS:
            combined.extend(self.rows_by_label[label])
        return combined

    def label_counts(self) -> Dict[str, int]:
        return {label: len(self.rows_by_label[label]) for label in CANONICAL_LABELS}


def emit_marker_based_labels(store: ExampleStore, text: str, source: str) -> None:
    if is_constraint(text):
        store.add("constraint", text, source)
    if is_preference(text):
        store.add("preference", text, source)
    if is_hypothesis(text):
        store.add("hypothesis", text, source)
    if is_reasoning(text):
        store.add("reasoning_step", text, source)
    if is_plan(text):
        store.add("plan", text, source)
    if is_tool_result(text):
        store.add("tool_result", text, source)


def extract_locomo10(store: ExampleStore, path: Path) -> None:
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, list):
        return

    for item in data:
        conv = item.get("conversation") if isinstance(item, dict) else None
        if isinstance(conv, dict):
            speaker_a = clean_text(conv.get("speaker_a")) or "A"
            speaker_b = clean_text(conv.get("speaker_b")) or "B"
            for key, session in conv.items():
                if not key.startswith("session_") or not isinstance(session, list):
                    continue
                lines = []
                for turn in session:
                    if not isinstance(turn, dict):
                        continue
                    text = clean_text(turn.get("text"))
                    if not text:
                        continue
                    speaker = clean_text(turn.get("speaker"))
                    if not speaker:
                        speaker = speaker_a if len(lines) % 2 == 0 else speaker_b
                    lines.append(f"{speaker}: {text}")
                    store.add("message", text, "locomo10:message")
                    emit_marker_based_labels(store, text, "locomo10:derived")
                if len(lines) >= 2:
                    store.add("conversation", "\n".join(lines[:10]), "locomo10:conversation")

        for qa in item.get("qa", []) if isinstance(item, dict) else []:
            if not isinstance(qa, dict):
                continue
            q = clean_text(qa.get("question"))
            a = clean_text(qa.get("answer"))
            if q:
                store.add("message", q, "locomo10:qa")
            if a:
                store.add("semantic_fact", a, "locomo10:qa")
                if is_reasoning(a):
                    store.add("reasoning_step", a, "locomo10:qa")
            if q and a:
                store.add("conversation", f"User: {q}\nAssistant: {a}", "locomo10:qa")

        for text in to_flat_strings(item.get("event_summary") if isinstance(item, dict) else None):
            if len(text) < 12:
                continue
            store.add("episodic_event", text, "locomo10:event")
            store.add("semantic_fact", text, "locomo10:event")
            emit_marker_based_labels(store, text, "locomo10:event_derived")

        for text in to_flat_strings(item.get("session_summary") if isinstance(item, dict) else None):
            if len(text) < 12:
                continue
            store.add("semantic_fact", text, "locomo10:session_summary")
            store.add("knowledge", text, "locomo10:session_summary")
            emit_marker_based_labels(store, text, "locomo10:session_summary_derived")

        for text in to_flat_strings(item.get("observation") if isinstance(item, dict) else None):
            if len(text) < 12:
                continue
            store.add("observation", text, "locomo10:observation")


def extract_locomoplus(store: ExampleStore, path: Path) -> None:
    if not path.exists():
        return
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return
    if not isinstance(data, list):
        return

    for item in data:
        if not isinstance(item, dict):
            continue
        cue = clean_text(item.get("cue_dialogue"))
        trigger = clean_text(item.get("trigger_query"))
        relation = clean_text(item.get("relation_type")).lower()

        if cue:
            store.add("conversation", cue, "locomoplus:cue")
            store.add("message", cue, "locomoplus:cue")
            emit_marker_based_labels(store, cue, "locomoplus:cue_derived")
        if trigger:
            store.add("message", trigger, "locomoplus:trigger")
            emit_marker_based_labels(store, trigger, "locomoplus:trigger_derived")
            if trigger.endswith("?"):
                store.add("hypothesis", f"Maybe {trigger[:-1].strip()}.", "locomoplus:hypothesis")
        if cue and trigger:
            store.add("conversation", f"{cue}\n{trigger}", "locomoplus:pair")

        if relation in {"causal", "counterfactual", "hypothetical", "uncertain"} and trigger:
            store.add("hypothesis", f"It might be that {trigger.rstrip('?')}.", "locomoplus:relation")
        if relation in {"temporal", "event"} and trigger:
            store.add("episodic_event", trigger, "locomoplus:relation")


def extract_personachat(store: ExampleStore, path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            raw = clean_text(line, 4000)
            if not raw:
                continue
            raw = re.sub(r"^\d+\s+", "", raw)
            lower = raw.lower()

            if "persona:" in lower:
                persona = clean_text(raw.split(":", 1)[-1], 500)
                if not persona:
                    continue
                store.add("message", persona, "personachat:persona")
                store.add("semantic_fact", persona, "personachat:persona")
                if persona.lower().startswith(("i ", "my ")) or is_preference(persona):
                    store.add("preference", persona, "personachat:preference")
                emit_marker_based_labels(store, persona, "personachat:persona_derived")
            else:
                parts = [clean_text(p, 500) for p in raw.split("\t") if clean_text(p, 500)]
                if not parts:
                    continue
                for p in parts:
                    store.add("message", p, "personachat:message")
                    emit_marker_based_labels(store, p, "personachat:message_derived")
                if len(parts) >= 2:
                    lines = []
                    for idx, p in enumerate(parts[:8]):
                        speaker = "User" if idx % 2 == 0 else "Assistant"
                        lines.append(f"{speaker}: {p}")
                    store.add("conversation", "\n".join(lines), "personachat:conversation")

            if (
                store.real_full("message")
                and store.real_full("conversation")
                and store.real_full("preference")
            ):
                break


def extract_coqa(store: ExampleStore, path: Path, source_key: str) -> None:
    if not path.exists():
        return
    for df in iter_parquet_batches(path):
        for row in df.itertuples(index=False):
            story = clean_text(getattr(row, "story", ""))
            questions = to_flat_strings(getattr(row, "questions", None))
            answers = parse_coqa_answers(getattr(row, "answers", None))

            story_first = first_sentence(story)
            if story_first:
                store.add("semantic_fact", story_first, f"{source_key}:story")
                store.add("knowledge", story_first, f"{source_key}:story")
                if is_past_event(story_first):
                    store.add("episodic_event", story_first, f"{source_key}:story")

            conv_lines: List[str] = []
            for i, q_raw in enumerate(questions[:6]):
                q = clean_text(q_raw, 500)
                if not q:
                    continue
                store.add("message", q, f"{source_key}:question")
                conv_lines.append(f"User: {q}")
                emit_marker_based_labels(store, q, f"{source_key}:question_derived")

                a = clean_text(answers[i], 500) if i < len(answers) else ""
                if a:
                    store.add("message", a, f"{source_key}:answer")
                    store.add("semantic_fact", a, f"{source_key}:answer")
                    conv_lines.append(f"Assistant: {a}")
                    reason = (
                        f"Because {story_first.lower()}, {a}"
                        if story_first
                        else f"Because of the context, {a}"
                    )
                    store.add("reasoning_step", reason, f"{source_key}:reasoning")
                    emit_marker_based_labels(store, a, f"{source_key}:answer_derived")

            if len(conv_lines) >= 2:
                store.add("conversation", "\n".join(conv_lines[:10]), f"{source_key}:conversation")


def extract_cooperbench(store: ExampleStore, path: Path) -> None:
    if not path.exists():
        return
    try:
        df = pd.read_parquet(path)
    except Exception:
        return

    for row in df.itertuples(index=False):
        instance_id = clean_text(getattr(row, "instance_id", "task"))
        repo = clean_text(getattr(row, "repo", "repo"))
        base_commit = clean_text(getattr(row, "base_commit", ""))
        problem = clean_text(getattr(row, "problem_statement", ""), 2500)
        hints = clean_text(getattr(row, "hints_text", ""), 1200)
        patch = clean_text(getattr(row, "patch", ""), 6000)
        test_patch = clean_text(getattr(row, "test_patch", ""), 6000)

        if problem:
            p_first = first_sentence(problem)
            store.add("message", p_first, "cooperbench:problem")
            emit_marker_based_labels(store, problem, "cooperbench:problem_derived")
            task_state = f"Task state: issue {instance_id} in {repo} is in progress. Current problem: {p_first}"
            plan = f"Goal: resolve {instance_id} in {repo}. Plan: inspect code, apply patch, run tests."
            observation = (
                f"Observation: repository {repo} at commit {base_commit[:12]} has an unresolved issue."
            )
            store.add("task_state", task_state, "cooperbench:task_state")
            store.add("plan", plan, "cooperbench:plan")
            store.add("observation", observation, "cooperbench:observation")

        if hints:
            store.add("reasoning_step", hints, "cooperbench:hints")
            emit_marker_based_labels(store, hints, "cooperbench:hints_derived")

        for chunk in split_chunks(patch, max_chunks=2, max_chars=700):
            store.add("tool_result", chunk, "cooperbench:patch")
            store.add("scratch", f"Patch draft: {chunk}", "cooperbench:scratch")
        for chunk in split_chunks(test_patch, max_chunks=2, max_chars=700):
            store.add("tool_result", chunk, "cooperbench:test_patch")


def build_plan_text(title: str, steps: List[str], summary: str) -> str:
    goal = clean_text(title, 160) or clean_text(first_sentence(summary), 160) or "complete the task"
    if len(steps) >= 3:
        return (
            f"Goal: {goal}. Plan: first {steps[0].lower()}, then {steps[1].lower()}, "
            f"finally {steps[2].lower()}."
        )
    if len(steps) == 2:
        return f"Goal: {goal}. Plan: first {steps[0].lower()}, then {steps[1].lower()}."
    if len(steps) == 1:
        return f"Goal: {goal}. Plan: start with {steps[0].lower()}."
    return f"Goal: {goal}. Plan: define steps, execute carefully, and review outcomes."


def extract_wikihow_csv(store: ExampleStore, path: Path, source_key: str) -> None:
    if not path.exists():
        return
    try:
        for chunk in pd.read_csv(path, chunksize=8000, on_bad_lines="skip"):
            for row in chunk.itertuples(index=False):
                title = clean_text(getattr(row, "title", "") or getattr(row, "headline", ""), 180)
                summary = clean_text(getattr(row, "summary", "") or getattr(row, "overview", ""), 500)
                body = clean_text(getattr(row, "text", ""), 1800)
                if not body:
                    continue

                steps = split_steps(body, max_steps=4)
                if steps:
                    procedure = f"How to {title or 'complete the task'}: " + " ".join(
                        [f"{i + 1}) {s}." for i, s in enumerate(steps)]
                    )
                else:
                    procedure = f"How to {title or 'complete the task'}: {body}"
                store.add("procedure", procedure, f"{source_key}:procedure")
                store.add("plan", build_plan_text(title, steps, summary), f"{source_key}:plan")
                if summary:
                    store.add("semantic_fact", summary, f"{source_key}:summary")
                if is_reasoning(body):
                    store.add("reasoning_step", find_reasoning_sentence(body), f"{source_key}:reasoning")

            if store.real_full("procedure") and store.real_full("plan"):
                break
    except Exception:
        return


def extract_reddit_parquet(store: ExampleStore, path: Path, source_key: str) -> None:
    if not path.exists():
        return
    for df in iter_parquet_batches(path):
        for row in df.itertuples(index=False):
            prompt = clean_text(getattr(row, "prompt", ""), 500)
            story = clean_text(getattr(row, "story", ""), 2000)
            if prompt:
                store.add("message", prompt, f"{source_key}:prompt")
            if not story:
                continue
            store.add("scratch", story, f"{source_key}:scratch")
            first = first_sentence(story)
            if first and (is_past_event(first) or len(first) >= 40):
                store.add("episodic_event", first, f"{source_key}:event")
            if is_reasoning(story):
                store.add("reasoning_step", find_reasoning_sentence(story), f"{source_key}:reasoning")
            if prompt and first:
                store.add(
                    "conversation",
                    f"User: {prompt}\nAssistant: {first}",
                    f"{source_key}:conversation",
                )

        if store.real_full("scratch") and store.real_full("episodic_event"):
            break


def extract_chatgpt4o_jsonl(store: ExampleStore, path: Path) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue
            conv = obj.get("conversations")
            if not isinstance(conv, list):
                continue

            lines: List[str] = []
            for turn in conv[:8]:
                if not isinstance(turn, dict):
                    continue
                role = clean_text(turn.get("from", "speaker"), 40)
                value = clean_text(turn.get("value", ""), 700)
                if not value:
                    continue
                lines.append(f"{role}: {value}")
                store.add("message", value, "chatgpt4o:message")
                if role.lower() == "gpt" and len(value) > 40:
                    store.add("scratch", value, "chatgpt4o:scratch")
                emit_marker_based_labels(store, value, "chatgpt4o:derived")

            if len(lines) >= 2:
                store.add("conversation", "\n".join(lines), "chatgpt4o:conversation")

            if store.real_full("message") and store.real_full("conversation"):
                break


def extract_wikipedia_lines(store: ExampleStore, path: Path, source_key: str) -> None:
    if not path.exists():
        return
    with open(path, encoding="utf-8", errors="replace") as f:
        for line in f:
            text = clean_text(line)
            if len(text) < 20:
                continue
            store.add("knowledge", text, f"{source_key}:knowledge")
            store.add("semantic_fact", text, f"{source_key}:fact")
            if store.real_full("knowledge") and store.real_full("semantic_fact"):
                break


def extract_structured_wikipedia_zip(store: ExampleStore, path: Path) -> None:
    if not path.exists():
        return
    try:
        with zipfile.ZipFile(path, "r") as zf:
            names = [n for n in zf.namelist() if n.endswith(".jsonl") and not n.startswith("__MACOSX")]
            names.sort()
            for name in names:
                with zf.open(name) as f:
                    for raw in f:
                        try:
                            obj = json.loads(raw.decode("utf-8", errors="replace"))
                        except Exception:
                            continue
                        abstract = clean_text(obj.get("abstract", ""), 1200)
                        title = clean_text(obj.get("name", ""), 200)
                        if not abstract:
                            continue
                        text = f"{title}: {abstract}" if title else abstract
                        store.add("knowledge", text, "structured_wikipedia:knowledge")
                        store.add("semantic_fact", text, "structured_wikipedia:fact")
                        if store.real_full("knowledge") and store.real_full("semantic_fact"):
                            return
    except Exception:
        return


def extract_real_examples(store: ExampleStore) -> None:
    steps = [
        ("locomo10", lambda: extract_locomo10(store, dataset_path("locomo10"))),
        ("locomoplus", lambda: extract_locomoplus(store, dataset_path("locomoplus"))),
        ("personachat", lambda: extract_personachat(store, dataset_path("personachat"))),
        ("coqa_train", lambda: extract_coqa(store, dataset_path("coqa_train"), "coqa_train")),
        ("coqa_validation", lambda: extract_coqa(store, dataset_path("coqa_validation"), "coqa_validation")),
        ("cooperbench", lambda: extract_cooperbench(store, dataset_path("cooperbench"))),
        ("chatgpt4o", lambda: extract_chatgpt4o_jsonl(store, dataset_path("chatgpt4o"))),
        ("wikihow_sep", lambda: extract_wikihow_csv(store, dataset_path("wikihow_sep"), "wikihow_sep")),
        (
            "wikihow_cleaned",
            lambda: extract_wikihow_csv(store, dataset_path("wikihow_cleaned"), "wikihow_cleaned"),
        ),
        (
            "wikipedia_cleaned",
            lambda: extract_wikipedia_lines(store, dataset_path("wikipedia_cleaned"), "wikipedia_cleaned"),
        ),
        (
            "structured_wikipedia",
            lambda: extract_structured_wikipedia_zip(store, dataset_path("structured_wikipedia_zip")),
        ),
        ("reddit_00000", lambda: extract_reddit_parquet(store, dataset_path("reddit_0"), "reddit_00000")),
        ("reddit_00001", lambda: extract_reddit_parquet(store, dataset_path("reddit_1"), "reddit_00001")),
    ]

    pbar = _progress(total=len(steps), desc="Real extraction", unit="source")
    for name, fn in steps:
        before = sum(store.real_counts.values())
        fn()
        after = sum(store.real_counts.values())
        added = after - before
        pbar.set_description(f"Real extraction [{name}]")
        pbar.set_postfix({"added": added, "total": after})
        pbar.update(1)
    pbar.close()


def choose_seed_text(store: ExampleStore, label: str, rng: random.Random) -> str:
    if store.conversation_pool:
        base = rng.choice(store.conversation_pool)
    elif store.all_seed_pool:
        base = rng.choice(store.all_seed_pool)
    else:
        base = "User: please review the current task. Assistant: working on it now."

    label_pool = store.seed_pool_by_label.get(label) or []
    if label_pool and rng.random() < 0.35:
        return f"{base} {rng.choice(label_pool)}"
    return base


def _normalize_llm_output(text: str, label: str) -> str:
    t = str(text or "").strip()
    if not t:
        return ""
    t = re.sub(r"^```(?:json|text)?", "", t.strip(), flags=re.IGNORECASE)
    t = re.sub(r"```$", "", t.strip())
    t = clean_text(t)
    if label != "conversation":
        t = " ".join(line.strip() for line in t.splitlines() if line.strip())
    if t.lower().startswith(f"{label}:"):
        t = clean_text(t.split(":", 1)[-1])
    return clean_text(t)


def _synthetic_label_ok(label: str, text: str) -> bool:
    t = clean_text(text)
    if len(t) < 8:
        return False
    if label == "constraint":
        return is_constraint(t)
    if label == "preference":
        return is_preference(t) or t.lower().startswith(("i ", "my "))
    if label == "procedure":
        return bool(re.search(r"\b(step|first|then|next|finally|1\)|1\.)\b", t, re.IGNORECASE))
    if label == "hypothesis":
        return is_hypothesis(t)
    if label == "task_state":
        return bool(re.search(r"\b(progress|in progress|completed|blocked|next)\b", t, re.IGNORECASE))
    if label == "conversation":
        return t.count(":") >= 2 or t.count("\n") >= 2
    if label == "message":
        return len(t) <= 700
    if label == "tool_result":
        return is_tool_result(t)
    if label == "reasoning_step":
        return is_reasoning(t)
    if label == "scratch":
        return len(t) <= 1200
    if label == "knowledge":
        return not t.endswith("?")
    if label == "observation":
        return bool(re.search(r"\b(observation|screen|display|shows|visible|status)\b", t, re.IGNORECASE))
    if label == "plan":
        return is_plan(t)
    if label == "episodic_event":
        return is_past_event(t)
    if label == "semantic_fact":
        return not t.endswith("?")
    return True


class LLMSynthesizer:
    def __init__(self, config: PrepConfig, seed: int) -> None:
        self.config = config
        self.seed = seed
        self.model = config.llm_model or os.environ.get("LLM_EVAL__MODEL") or os.environ.get("OPENAI_MODEL") or "gpt-4o-mini"
        self.base_url = config.llm_base_url or os.environ.get("LLM_EVAL__BASE_URL") or os.environ.get("OPENAI_BASE_URL") or ""
        api_key = config.llm_api_key or os.environ.get("LLM_EVAL__API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
        if not api_key:
            if self.base_url:
                api_key = "ollama"
            else:
                raise RuntimeError(
                    "LLM synthetic mode requires OPENAI_API_KEY (or LLM_EVAL__API_KEY), "
                    "or a local --llm-base-url/LLM_EVAL__BASE_URL."
                )
        self.api_key = api_key
        self.temperature = float(config.llm_temperature)
        self.timeout_seconds = int(config.llm_timeout_seconds)
        self.client = self._create_client()

    def _create_client(self):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError("openai package is required for LLM synthetic generation.") from exc

        kwargs = {"api_key": self.api_key, "timeout": self.timeout_seconds}
        if self.base_url:
            kwargs["base_url"] = self.base_url.rstrip("/")
        return OpenAI(**kwargs)

    def _build_prompt(self, label: str, count: int, seeds: List[str]) -> str:
        guidance = LLM_LABEL_GUIDANCE.get(label, f"Write text for label {label}.")
        seed_lines = []
        for idx, seed in enumerate(seeds[:10], start=1):
            seed_lines.append(f"{idx}. {clean_text(seed, 280)}")
        seed_blob = "\n".join(seed_lines) if seed_lines else "1. User: please review this task.\n2. Assistant: working on it."

        return (
            "Generate synthetic data for a text classifier.\n"
            f"Target label: {label}\n"
            f"Guidance: {guidance}\n"
            f"Number of samples: {count}\n"
            "Use the base contexts below only as inspiration for topic/style diversity. "
            "Do not copy them verbatim.\n\n"
            f"Base contexts:\n{seed_blob}\n\n"
            "Rules:\n"
            f"- Return ONLY valid JSON array of exactly {count} strings.\n"
            "- No markdown, no code fences, no explanations.\n"
            "- Each string must be unique and under 220 characters.\n"
            f"- Every sample must match label `{label}` clearly.\n"
        )

    def _parse_response(self, raw: str) -> List[str]:
        text = (raw or "").strip()
        if not text:
            return []
        text = re.sub(r"^```(?:json|text)?", "", text.strip(), flags=re.IGNORECASE)
        text = re.sub(r"```$", "", text.strip())
        text = text.strip()

        # If model adds wrapper text around JSON, attempt bracket extraction first.
        if "[" in text and "]" in text:
            lidx = text.find("[")
            ridx = text.rfind("]")
            if lidx < ridx:
                bracket_payload = text[lidx : ridx + 1]
                try:
                    parsed = json.loads(bracket_payload)
                    if isinstance(parsed, dict) and isinstance(parsed.get("samples"), list):
                        parsed = parsed["samples"]
                    if isinstance(parsed, list):
                        return [str(x) for x in parsed if isinstance(x, (str, int, float))]
                except Exception:
                    pass

        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict) and isinstance(parsed.get("samples"), list):
                parsed = parsed["samples"]
            if isinstance(parsed, list):
                return [str(x) for x in parsed if isinstance(x, (str, int, float))]
        except Exception:
            pass

        quoted = re.findall(r'"((?:[^"\\\\]|\\\\.)*)"', text)
        if quoted:
            out = []
            for q in quoted:
                try:
                    out.append(json.loads(f'"{q}"'))
                except Exception:
                    out.append(q)
            if out:
                return out

        out: List[str] = []
        for line in text.splitlines():
            line = line.strip()
            if not line:
                continue
            line = re.sub(r"^\d+[\).\-\s]+", "", line)
            line = line.strip().strip('"').strip("'").strip(",")
            if line:
                out.append(line)
        return out

    def generate_batch(self, label: str, count: int, seeds: List[str]) -> List[str]:
        if count <= 0:
            return []
        prompt = self._build_prompt(label, count, seeds)
        max_tokens = max(800, min(8192, 320 * count))
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {
                    "role": "system",
                    "content": "You output only strict JSON arrays of synthetic training text.",
                },
                {"role": "user", "content": prompt},
            ],
            temperature=self.temperature,
            max_tokens=max_tokens,
        )
        text = (response.choices[0].message.content or "").strip()
        return self._parse_response(text)


def backfill_synthetic(store: ExampleStore, config: PrepConfig) -> List[dict]:
    rng = random.Random(config.seed + 7919)
    synthetic_rows: List[dict] = []
    llm = LLMSynthesizer(config, seed=config.seed)
    llm_desc = llm.base_url if llm.base_url else "OpenAI default endpoint"
    print(f"Synthetic generator: LLM model='{llm.model}' endpoint='{llm_desc}'")
    total_needed = sum(max(0, config.target_per_label - store.count(label)) for label in CANONICAL_LABELS)
    synth_pbar = _progress(total=total_needed, desc="LLM synthetic", unit="sample")

    try:
        for label in CANONICAL_LABELS:
            needed = max(0, config.target_per_label - store.count(label))
            attempts = 0
            max_attempts = max(config.target_per_label * 60, 20_000)
            stalled_rounds = 0
            synth_pbar.set_description(f"LLM synthetic [{label}]")

            while needed > 0 and attempts < max_attempts:
                batch_size = min(max(1, config.llm_batch_size), needed)
                attempts += batch_size
                added_this_round = 0

                seed_count = min(10, max(3, batch_size // 8))
                seeds = [choose_seed_text(store, label, rng) for _ in range(seed_count)]
                try:
                    generated = llm.generate_batch(label, batch_size, seeds)
                except Exception as exc:
                    print(f"LLM generation failed for {label}: {exc}", file=sys.stderr)
                    generated = []

                for text in generated:
                    cleaned = _normalize_llm_output(text, label)
                    if not _synthetic_label_ok(label, cleaned):
                        continue
                    source = f"synthetic_llm:{label}"
                    if store.add(label, cleaned, source, is_synthetic=True):
                        synthetic_rows.append({"text": cleaned, "label": label, "source": source})
                        needed -= 1
                        added_this_round += 1
                        synth_pbar.update(1)
                        if needed <= 0:
                            break

                synth_pbar.set_postfix({"remaining": needed, "attempts": attempts})
                if added_this_round == 0:
                    stalled_rounds += 1
                    if stalled_rounds >= 8:
                        raise RuntimeError(
                            f"LLM synthetic generation stalled for label '{label}'. "
                            "Check model/base-url/api-key or increase --llm-temperature/--llm-batch-size."
                        )
                else:
                    stalled_rounds = 0

            if needed > 0:
                raise RuntimeError(f"Unable to synthesize enough samples for label '{label}'. Missing: {needed}")
    finally:
        synth_pbar.close()

    return synthetic_rows


def stratified_split(df: pd.DataFrame, ratios: Dict[str, float], seed: int):
    if df.empty:
        return df.copy(), df.copy(), df.copy()

    train_ratio = float(ratios.get("train", 0.8))
    test_ratio = float(ratios.get("test", 0.1))
    eval_ratio = float(ratios.get("eval", 0.1))
    total = train_ratio + test_ratio + eval_ratio
    if abs(total - 1.0) > 1e-6:
        train_ratio, test_ratio, eval_ratio = 0.8, 0.1, 0.1

    rng = random.Random(seed)
    train_parts = []
    test_parts = []
    eval_parts = []

    for _label, group in df.groupby("label", sort=False):
        idx = list(group.index)
        rng.shuffle(idx)
        n = len(idx)

        n_train = int(n * train_ratio)
        n_test = int(n * test_ratio)
        n_eval = n - n_train - n_test

        if n >= 3:
            if n_train == 0:
                n_train = 1
                if n_eval > 1:
                    n_eval -= 1
            if n_test == 0:
                n_test = 1
                if n_eval > 1:
                    n_eval -= 1
            if n_eval == 0:
                n_eval = 1
                if n_train > n_test and n_train > 1:
                    n_train -= 1
                elif n_test > 1:
                    n_test -= 1

        train_parts.append(df.loc[idx[:n_train]])
        test_parts.append(df.loc[idx[n_train : n_train + n_test]])
        eval_parts.append(df.loc[idx[n_train + n_test : n_train + n_test + n_eval]])

    train_df = pd.concat(train_parts, ignore_index=True) if train_parts else pd.DataFrame(columns=df.columns)
    test_df = pd.concat(test_parts, ignore_index=True) if test_parts else pd.DataFrame(columns=df.columns)
    eval_df = pd.concat(eval_parts, ignore_index=True) if eval_parts else pd.DataFrame(columns=df.columns)

    train_df = train_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    test_df = test_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    eval_df = eval_df.sample(frac=1, random_state=seed).reset_index(drop=True)
    return train_df, test_df, eval_df


def run_audit() -> dict:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    datasets = []
    missing = []
    for key, filename in DATASET_FILES.items():
        path = DATASETS_DIR / filename
        exists = path.exists()
        size_bytes = path.stat().st_size if exists else 0
        datasets.append(
            {
                "key": key,
                "path": str(path),
                "exists": exists,
                "size_bytes": size_bytes,
            }
        )
        if not exists:
            missing.append(filename)

    report = {
        "dataset_dir": str(DATASETS_DIR),
        "datasets": datasets,
        "missing": missing,
    }
    with open(PREPARED_DIR / "dataset_audit.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print("Audit report written to", PREPARED_DIR / "dataset_audit.json")
    if missing:
        print("Missing datasets:", ", ".join(missing))
    return report


def write_validation_report(store: ExampleStore, config: PrepConfig, audit_report: dict) -> None:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    report = {
        "target_per_label": config.target_per_label,
        "max_real_per_label": config.max_real_per_label,
        "real_counts": store.real_counts,
        "synthetic_counts": store.synthetic_counts,
        "final_counts": store.label_counts(),
        "conversation_pool_size": len(store.conversation_pool),
        "missing_datasets": audit_report.get("missing", []),
        "top_sources": sorted(store.source_counts.items(), key=lambda kv: kv[1], reverse=True)[:50],
    }
    with open(PREPARED_DIR / "validation_report.json", "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    sample_rows = []
    for label in CANONICAL_LABELS:
        for row in store.rows_by_label[label][:20]:
            sample_rows.append(
                {
                    "label": label,
                    "source": row["source"],
                    "text": row["text"][:500],
                }
            )
    if sample_rows:
        pd.DataFrame(sample_rows).to_csv(PREPARED_DIR / "validated_sample.csv", index=False)

    print("Validation report written to", PREPARED_DIR / "validation_report.json")


def write_preparation_report(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    store: ExampleStore,
) -> None:
    lines = [
        "Label counts: " + json.dumps(store.label_counts()),
        "Real counts: " + json.dumps(store.real_counts),
        "Synthetic counts: " + json.dumps(store.synthetic_counts),
        f"train: {len(train_df)}",
        f"test: {len(test_df)}",
        f"eval: {len(eval_df)}",
    ]
    with open(PREPARED_DIR / "preparation_report.txt", "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print("Report written to", PREPARED_DIR / "preparation_report.txt")


def run_prepare(config: PrepConfig, no_synthetic: bool, audit_report: dict) -> int:
    PREPARED_DIR.mkdir(parents=True, exist_ok=True)
    store = ExampleStore(config.max_real_per_label, config.seed)
    extract_real_examples(store)

    if no_synthetic:
        missing = {
            label: config.target_per_label - store.count(label)
            for label in CANONICAL_LABELS
            if store.count(label) < config.target_per_label
        }
        if missing:
            print(
                "Synthetic generation is disabled; some labels are below target:",
                json.dumps(missing, indent=2),
            )
    else:
        try:
            backfill_synthetic(store, config)
        except RuntimeError as exc:
            print(str(exc), file=sys.stderr)
            return 1

    rows = store.rows()
    if not rows:
        print("No rows extracted.")
        return 1

    df = pd.DataFrame(rows)
    train_df, test_df, eval_df = stratified_split(df, config.split_ratios, config.seed)

    train_df.to_parquet(PREPARED_DIR / "train.parquet", index=False)
    test_df.to_parquet(PREPARED_DIR / "test.parquet", index=False)
    eval_df.to_parquet(PREPARED_DIR / "eval.parquet", index=False)

    synthetic_df = df[df["source"].str.startswith("synthetic:")].copy()
    synthetic_df.to_parquet(PREPARED_DIR / "synthetic.parquet", index=False)

    label_counts = store.label_counts()
    with open(PREPARED_DIR / "label_counts.json", "w", encoding="utf-8") as f:
        json.dump(label_counts, f, indent=2)

    write_validation_report(store, config, audit_report)
    write_preparation_report(train_df, test_df, eval_df, store)

    print("Prepared:", len(train_df), "train,", len(test_df), "test,", len(eval_df), "eval")
    print("Label counts:", label_counts)
    return 0


def run_stats() -> int:
    try:
        with open(PREPARED_DIR / "label_counts.json", encoding="utf-8") as f:
            counts = json.load(f)
        print("Label counts:", counts)
    except Exception:
        pass

    for split in ["train", "test", "eval"]:
        path = PREPARED_DIR / f"{split}.parquet"
        if not path.exists():
            continue
        try:
            n = len(pd.read_parquet(path))
            print(f"{split}: {n}")
        except Exception:
            continue
    return 0


def run_print_stats() -> int:
    if not PREPARED_DIR.exists():
        print("Prepared dir not found:", PREPARED_DIR)
        return 1

    for split in ["train", "test", "eval"]:
        path = PREPARED_DIR / f"{split}.parquet"
        if not path.exists():
            print(f"{split}: (missing)")
            continue
        df = pd.read_parquet(path)
        print(f"{split}: {len(df)} rows")
        counts = df["label"].value_counts()
        for label in CANONICAL_LABELS:
            value = int(counts.get(label, 0))
            if value > 0:
                print(f"  {label}: {value}")
        print()

    label_counts_path = PREPARED_DIR / "label_counts.json"
    if label_counts_path.exists():
        with open(label_counts_path, encoding="utf-8") as f:
            print("Overall label counts:", json.dumps(json.load(f), indent=2))
    return 0


def main() -> int:
    load_env_file()
    try:
        env_llm_temp = float(os.environ.get("LLM_EVAL__TEMPERATURE", DEFAULT_LLM_TEMPERATURE))
    except ValueError:
        env_llm_temp = DEFAULT_LLM_TEMPERATURE
    env_llm_temp = max(1.1, env_llm_temp)
    parser = argparse.ArgumentParser(description="Prepare datasets for 15-way memory classifier.")
    parser.add_argument("--no-synthetic", action="store_true", help="Skip synthetic backfill.")
    parser.add_argument("--stats-only", action="store_true", help="Print stats from prepared outputs and exit.")
    parser.add_argument(
        "--target-per-label",
        type=int,
        default=DEFAULT_TARGET_PER_LABEL,
        help="Minimum number of samples required per label (default: 10000).",
    )
    parser.add_argument(
        "--max-real-per-label",
        type=int,
        default=DEFAULT_MAX_REAL_PER_LABEL,
        help="Upper bound for real samples kept per label before synthetic backfill.",
    )
    parser.add_argument(
        "--llm-model",
        type=str,
        default=os.environ.get("LLM_EVAL__MODEL", ""),
        help="LLM model name for synthetic generation (env fallback: LLM_EVAL__MODEL).",
    )
    parser.add_argument(
        "--llm-base-url",
        type=str,
        default=os.environ.get("LLM_EVAL__BASE_URL", ""),
        help="OpenAI-compatible base URL (e.g., http://localhost:11434/v1).",
    )
    parser.add_argument(
        "--llm-api-key",
        type=str,
        default=os.environ.get("LLM_EVAL__API_KEY", os.environ.get("OPENAI_API_KEY", "")),
        help="API key for LLM provider.",
    )
    parser.add_argument(
        "--llm-temperature",
        type=float,
        default=env_llm_temp,
        help="Sampling temperature for synthetic generation.",
    )
    parser.add_argument(
        "--llm-timeout-seconds",
        type=int,
        default=DEFAULT_LLM_TIMEOUT_SECONDS,
        help="Timeout for each LLM request in seconds.",
    )
    parser.add_argument(
        "--llm-batch-size",
        type=int,
        default=DEFAULT_LLM_BATCH_SIZE,
        help="Number of samples requested per LLM call.",
    )
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED, help="Random seed.")
    args = parser.parse_args()

    if args.stats_only:
        return run_print_stats()

    config = PrepConfig(
        target_per_label=max(1, args.target_per_label),
        max_real_per_label=max(1, args.max_real_per_label),
        split_ratios=dict(SPLIT_RATIOS),
        seed=args.seed,
        llm_model=(args.llm_model or "").strip(),
        llm_base_url=(args.llm_base_url or "").strip(),
        llm_api_key=(args.llm_api_key or "").strip(),
        llm_temperature=float(args.llm_temperature),
        llm_timeout_seconds=max(5, int(args.llm_timeout_seconds)),
        llm_batch_size=max(1, int(args.llm_batch_size)),
    )

    audit_report = run_audit()
    status = run_prepare(config=config, no_synthetic=args.no_synthetic, audit_report=audit_report)
    run_stats()
    return status


if __name__ == "__main__":
    sys.exit(main())
