#!/usr/bin/env python3
"""
Cognitive Memory Layer — Full-Stack Quality Test
=================================================

A single, self-contained script that:
  1. Ingests a large, diverse corpus of user data into CML.
  2. Queries CML with semantically diverse probes (including cue-trigger disconnects).
  3. Uses an LLM-as-judge (via LLM_EVAL__* from .env) to score retrieval quality.
  4. Prints a detailed report: precision, recall, constraint consistency, latency.

Requirements:
  - CML API running (default [REDACTED]).
  - .env with LLM_EVAL__PROVIDER, LLM_EVAL__MODEL, LLM_EVAL__BASE_URL (and optionally
    LLM_EVAL__API_KEY / OPENAI_API_KEY).
  - ``pip install requests openai`` (both already in project deps).

Usage:
  python scripts/test_memory_quality.py                     # full run
  python scripts/test_memory_quality.py --skip-ingestion    # reuse existing data
  python scripts/test_memory_quality.py --verbose           # per-probe details

Note: The CML API enforces per-tenant rate limiting (default: 60 req/min).
For faster runs, set AUTH__RATE_LIMIT_REQUESTS_PER_MINUTE=600 in .env or
pass a higher value when starting the API server.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from textwrap import dedent

# ---------------------------------------------------------------------------
# Load .env
# ---------------------------------------------------------------------------
try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parent.parent / ".env")
except ImportError:
    pass

import requests

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
CML_BASE_URL = os.environ.get("CML_BASE_URL", "[REDACTED]").rstrip("/")
CML_API_KEY = os.environ.get("AUTH__API_KEY") or os.environ.get("CML_API_KEY") or "test-api-key"
TENANT_ID = f"quality-test-{int(time.time())}"
MAX_RETRIES = 5
RETRY_BACKOFF = 3

# ---------------------------------------------------------------------------
# Self-contained test corpus
# ---------------------------------------------------------------------------
# Each entry is (session_id, content, optional_timestamp, memory_type_hint).
# The corpus covers: identity, preferences, constraints (policy/value/goal/
# state/causal), episodic events, temporal facts, procedures, multi-hop chains,
# contradictions, and updates.

CORPUS: list[tuple[str, str, str | None]] = [
    # === IDENTITY ===
    ("s1", "My name is Elena Vasquez and I'm a marine biologist.", "2024-01-15T10:00:00Z"),
    ("s1", "I work at the Monterey Bay Aquarium Research Institute.", "2024-01-15T10:01:00Z"),
    ("s1", "I've been studying deep-sea bioluminescence for 12 years.", "2024-01-15T10:02:00Z"),
    ("s1", "I have a PhD from Scripps Institution of Oceanography.", "2024-01-15T10:03:00Z"),
    # === PREFERENCES ===
    ("s2", "I love Thai food, especially pad thai and green curry.", "2024-02-01T12:00:00Z"),
    ("s2", "My favorite programming language is Python.", "2024-02-01T12:05:00Z"),
    ("s2", "I prefer working early mornings, usually 5am to 1pm.", "2024-02-01T12:10:00Z"),
    ("s2", "I enjoy hiking and underwater photography as hobbies.", "2024-02-01T12:15:00Z"),
    ("s2", "My favorite book is 'The Silent World' by Jacques Cousteau.", "2024-02-01T12:20:00Z"),
    # === CONSTRAINTS — POLICY (hard rules) ===
    (
        "s3",
        "I never eat shellfish because I have a severe allergy — even trace amounts can cause anaphylaxis.",
        "2024-03-01T09:00:00Z",
    ),
    ("s3", "I always carry an EpiPen with me due to my shellfish allergy.", "2024-03-01T09:05:00Z"),
    (
        "s3",
        "I refuse to use single-use plastics because of their impact on ocean life.",
        "2024-03-01T09:10:00Z",
    ),
    ("s3", "I never share unpublished research data before peer review.", "2024-03-01T09:15:00Z"),
    # === CONSTRAINTS — VALUE ===
    ("s4", "I value environmental sustainability above convenience.", "2024-03-15T14:00:00Z"),
    (
        "s4",
        "Scientific integrity is the most important thing to me in my work.",
        "2024-03-15T14:05:00Z",
    ),
    (
        "s4",
        "I believe in open-access publishing so everyone can benefit from science.",
        "2024-03-15T14:10:00Z",
    ),
    # === CONSTRAINTS — GOAL ===
    (
        "s5",
        "I'm trying to publish my research on bioluminescent communication in Nature by end of year.",
        "2024-04-01T08:00:00Z",
    ),
    (
        "s5",
        "My goal is to secure a $2M NSF grant for the deep-sea lab expansion.",
        "2024-04-01T08:05:00Z",
    ),
    ("s5", "I'm working toward getting tenure at MBARI by 2026.", "2024-04-01T08:10:00Z"),
    # === CONSTRAINTS — STATE ===
    (
        "s6",
        "I'm currently dealing with budget cuts that threaten my research vessel access.",
        "2024-04-15T11:00:00Z",
    ),
    ("s6", "I'm stressed about the upcoming grant deadline on June 30th.", "2024-04-15T11:05:00Z"),
    ("s6", "Right now I'm mentoring three graduate students.", "2024-04-15T11:10:00Z"),
    # === CONSTRAINTS — CAUSAL ===
    (
        "s7",
        "Because of the budget cuts, I need to find alternative funding sources.",
        "2024-05-01T09:00:00Z",
    ),
    (
        "s7",
        "The reason I switched to ROV surveys is that manned submersibles are too expensive.",
        "2024-05-01T09:05:00Z",
    ),
    (
        "s7",
        "I prioritize Python over R because our lab's entire pipeline is built on it.",
        "2024-05-01T09:10:00Z",
    ),
    # === EPISODIC EVENTS ===
    (
        "s8",
        "Last Tuesday I presented my findings on deep-sea anglerfish bioluminescence at the MBARI seminar.",
        "2024-05-10T16:00:00Z",
    ),
    (
        "s8",
        "I discovered a new species of bioluminescent jellyfish at 3000m depth during the April expedition.",
        "2024-05-10T16:05:00Z",
    ),
    (
        "s8",
        "My research paper was rejected by Science last month, but the reviewers gave constructive feedback.",
        "2024-05-10T16:10:00Z",
    ),
    (
        "s8",
        "I had a meeting with the NSF program officer Dr. Chen about the grant application.",
        "2024-05-10T16:15:00Z",
    ),
    (
        "s8",
        "The lab's ROV broke down during the May expedition and we lost two days of data collection.",
        "2024-05-10T16:20:00Z",
    ),
    # === MULTI-HOP / RELATIONAL ===
    (
        "s9",
        "My graduate student Maria is working on the bioluminescence spectral analysis project.",
        "2024-05-15T10:00:00Z",
    ),
    (
        "s9",
        "Maria's thesis advisor before me was Dr. Tanaka at Woods Hole.",
        "2024-05-15T10:05:00Z",
    ),
    (
        "s9",
        "Dr. Chen from NSF recommended I collaborate with Dr. Patel at Scripps on the grant.",
        "2024-05-15T10:10:00Z",
    ),
    (
        "s9",
        "The new jellyfish species I discovered might be related to Atolla wyvillei.",
        "2024-05-15T10:15:00Z",
    ),
    # === TEMPORAL UPDATES (supersession) ===
    (
        "s10",
        "I moved from San Diego to Pacific Grove last year to be closer to MBARI.",
        "2024-06-01T09:00:00Z",
    ),
    (
        "s10",
        "I'm now considering moving back to San Diego because my partner got a job there.",
        "2024-07-15T09:00:00Z",
    ),
    # === PROCEDURES ===
    (
        "s11",
        "To calibrate the deep-sea spectrometer, first you need to run the dark-current baseline, then do a white reference with the standard lamp, and finally verify with the mercury line.",
        "2024-06-10T14:00:00Z",
    ),
    (
        "s11",
        "The ROV pre-dive checklist requires: pressure test, thruster check, camera calibration, ballast verification, and communication link test.",
        "2024-06-10T14:10:00Z",
    ),
    # === CONTRADICTION (to test belief revision) ===
    ("s12", "I don't drink coffee at all.", "2024-01-20T08:00:00Z"),
    (
        "s12",
        "Actually, I've started drinking one cup of coffee each morning since March.",
        "2024-04-01T08:00:00Z",
    ),
    # === DOMAIN-SPECIFIC KNOWLEDGE ===
    (
        "s13",
        "Bioluminescence in deep-sea organisms is primarily driven by the luciferin-luciferase reaction.",
        "2024-02-20T10:00:00Z",
    ),
    (
        "s13",
        "About 76% of deep-sea organisms produce their own light through bioluminescence.",
        "2024-02-20T10:05:00Z",
    ),
    (
        "s13",
        "The mesopelagic zone (200-1000m) has the highest density of bioluminescent organisms.",
        "2024-02-20T10:10:00Z",
    ),
    # === FINANCIAL / PRACTICAL ===
    ("s14", "My annual research budget is approximately $450,000.", "2024-03-20T09:00:00Z"),
    ("s14", "The ROV rental costs about $15,000 per day including crew.", "2024-03-20T09:05:00Z"),
    (
        "s14",
        "I need to save money for the conference in Tokyo next February.",
        "2024-03-20T09:10:00Z",
    ),
]

# ---------------------------------------------------------------------------
# Test probes: (query, expected_keywords, test_category, notes)
# expected_keywords: list of keyword groups — at least one keyword from each
# group must appear in retrieved context for the probe to count as a hit.
# ---------------------------------------------------------------------------


@dataclass
class Probe:
    query: str
    expected_keywords: list[list[str]]  # groups of keywords; need >=1 from each group
    category: str
    notes: str = ""
    failure_mode: str = ""  # which failure class this tests


PROBES: list[Probe] = [
    # --- IDENTITY RECALL ---
    Probe("What is my name?", [["elena", "vasquez"]], "identity", "Direct identity lookup"),
    Probe("Where do I work?", [["monterey", "mbari", "aquarium"]], "identity", "Workplace recall"),
    Probe("What are my qualifications?", [["phd", "scripps"]], "identity", "Education recall"),
    # --- PREFERENCE RECALL ---
    Probe(
        "What kind of food do I like?",
        [["thai", "pad thai", "curry"]],
        "preference",
        "Food preference",
    ),
    Probe(
        "When do I prefer to work?", [["morning", "5am", "early"]], "preference", "Work schedule"
    ),
    Probe("What are my hobbies?", [["hiking", "photography"]], "preference", "Hobby recall"),
    # --- CONSTRAINT RETRIEVAL (direct) ---
    Probe(
        "Do I have any food allergies?",
        [["shellfish", "allergy", "anaphylaxis"]],
        "constraint_direct",
        "Direct constraint query",
        "semantic_disconnect",
    ),
    Probe(
        "What are my personal rules about plastics?",
        [["single-use", "plastic", "ocean"]],
        "constraint_direct",
        "Policy constraint",
    ),
    Probe(
        "What are my research ethics rules?",
        [["unpublished", "peer review"]],
        "constraint_direct",
        "Research policy",
    ),
    # --- SEMANTIC DISCONNECT (trigger ≠ cue) ---
    Probe(
        "Recommend a seafood restaurant for dinner tonight.",
        [["shellfish", "allergy", "allergic", "anaphylaxis", "epipen"]],
        "semantic_disconnect",
        "Trigger (restaurant) should recall cue (shellfish allergy)",
        "semantic_disconnect",
    ),
    Probe(
        "What gift should I bring to a dinner party?",
        [["single-use", "plastic", "sustainability", "environmental"]],
        "semantic_disconnect",
        "Trigger (gift) should recall cue (no single-use plastics)",
        "semantic_disconnect",
    ),
    Probe(
        "What packaging should I use for shipping samples?",
        [["single-use", "plastic", "sustainability", "ocean"]],
        "semantic_disconnect",
        "Packaging query should recall plastics policy",
        "semantic_disconnect",
    ),
    # --- CONSTRAINT DILUTION ---
    Probe(
        "Should I take this job offer?",
        [["tenure", "mbari", "2026"]],
        "constraint_dilution",
        "Decision query should surface goal constraints not just random facts",
        "constraint_dilution",
    ),
    # --- TEMPORAL / RECENCY ---
    Probe(
        "Where do I live now?",
        [["san diego", "pacific grove", "moving"]],
        "temporal",
        "Should retrieve most recent location info",
        "temporal_recency",
    ),
    Probe(
        "Do I drink coffee?",
        [["coffee", "morning", "started", "march"]],
        "temporal",
        "Should retrieve updated belief (started drinking coffee)",
        "stale_constraint",
    ),
    # --- GOAL RETRIEVAL ---
    Probe(
        "What am I working toward in my career?",
        [["nature", "publish", "nsf", "grant", "tenure"]],
        "goal",
        "Should retrieve career goals",
    ),
    Probe(
        "What is my publication target?",
        [["nature", "bioluminescent", "end of year"]],
        "goal",
        "Specific goal recall",
    ),
    # --- STATE RETRIEVAL ---
    Probe(
        "What challenges am I dealing with right now?",
        [["budget", "cuts", "grant", "deadline", "stress"]],
        "state",
        "Current state retrieval",
    ),
    Probe(
        "Who am I mentoring?",
        [["graduate", "student", "three"]],
        "state",
        "Current mentoring state",
    ),
    # --- CAUSAL REASONING ---
    Probe(
        "Why did I switch to ROV surveys?",
        [["expensive", "submersible", "cost"]],
        "causal",
        "Causal reasoning retrieval",
    ),
    Probe(
        "Why do I need alternative funding?",
        [["budget", "cuts"]],
        "causal",
        "Causal chain retrieval",
    ),
    # --- MULTI-HOP ---
    Probe(
        "Who is Maria's previous advisor?",
        [["tanaka", "woods hole"]],
        "multi_hop",
        "Two-hop: Maria → thesis advisor → Tanaka",
    ),
    Probe(
        "Who recommended the Scripps collaboration?",
        [["chen", "nsf", "patel"]],
        "multi_hop",
        "NSF officer → collaboration recommendation",
    ),
    # --- PROCEDURAL ---
    Probe(
        "How do I calibrate the spectrometer?",
        [["dark-current", "baseline", "white reference", "mercury"]],
        "procedural",
        "Step-by-step procedure recall",
    ),
    Probe(
        "What's on the ROV pre-dive checklist?",
        [["pressure", "thruster", "camera", "ballast", "communication"]],
        "procedural",
        "Checklist procedure",
    ),
    # --- EPISODIC ---
    Probe(
        "What happened at the MBARI seminar?",
        [["anglerfish", "bioluminescence", "presented"]],
        "episodic",
        "Specific event recall",
    ),
    Probe(
        "What went wrong during the May expedition?",
        [["rov", "broke", "lost", "data"]],
        "episodic",
        "Failure event recall",
    ),
    # --- DOMAIN KNOWLEDGE ---
    Probe(
        "What drives bioluminescence in deep-sea organisms?",
        [["luciferin", "luciferase"]],
        "knowledge",
        "Scientific knowledge recall",
    ),
    Probe(
        "What percentage of deep-sea organisms produce light?",
        [["76"]],
        "knowledge",
        "Quantitative fact recall",
    ),
    # --- FINANCIAL ---
    Probe("What is my research budget?", [["450", "000"]], "financial", "Budget fact recall"),
    Probe(
        "How much does ROV rental cost?", [["15", "000", "day"]], "financial", "Cost fact recall"
    ),
]

# ---------------------------------------------------------------------------
# LLM helpers (read-only from LLM_EVAL__*)
# ---------------------------------------------------------------------------

_LLM_DEFAULT_BASE: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    "ollama": "[REDACTED]",
}


def _get_llm_config() -> tuple[str, str, str]:
    provider = (os.environ.get("LLM_EVAL__PROVIDER") or "openai").strip().lower()
    model = (os.environ.get("LLM_EVAL__MODEL") or "gpt-4o-mini").strip()
    api_key = (
        os.environ.get("LLM_EVAL__API_KEY") or os.environ.get("OPENAI_API_KEY") or ""
    ).strip()
    base_url = (os.environ.get("LLM_EVAL__BASE_URL") or "").strip()
    if not base_url:
        base_url = _LLM_DEFAULT_BASE.get(provider, "https://api.openai.com/v1")
    if provider != "openai" and not api_key:
        api_key = "dummy"
    return base_url.rstrip("/"), model, api_key


def _llm_call(prompt: str, max_tokens: int = 512, temperature: float = 0.0) -> str:
    from openai import OpenAI

    base_url, model, api_key = _get_llm_config()
    client = OpenAI(base_url=base_url, api_key=api_key)
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            max_tokens=max_tokens,
            temperature=temperature,
        )
        return (resp.choices[0].message.content or "").strip()
    except Exception as e:
        print(f"  [LLM error] {e}")
        return ""


# ---------------------------------------------------------------------------
# CML API helpers
# ---------------------------------------------------------------------------


def _api(method: str, path: str, payload: dict | None = None, tenant: str = TENANT_ID) -> dict:
    url = f"{CML_BASE_URL}/api/v1{path}"
    headers = {"X-API-Key": CML_API_KEY, "X-Tenant-ID": tenant, "Content-Type": "application/json"}
    for attempt in range(MAX_RETRIES):
        try:
            if method == "GET":
                resp = requests.get(url, headers=headers, timeout=60)
            else:
                resp = requests.post(url, json=payload or {}, headers=headers, timeout=120)
            if resp.status_code == 429:
                wait = min(60, RETRY_BACKOFF * 2**attempt)
                time.sleep(wait)
                continue
            if resp.status_code == 500:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            resp.raise_for_status()
            return resp.json()
        except requests.exceptions.ConnectionError:
            if attempt < MAX_RETRIES - 1:
                time.sleep(RETRY_BACKOFF * (attempt + 1))
                continue
            raise
    return {}


def cml_write(content: str, session_id: str, timestamp: str | None = None) -> dict:
    payload: dict = {"content": content, "session_id": session_id}
    if timestamp:
        payload["timestamp"] = timestamp
    return _api("POST", "/memory/write", payload)


def cml_read(query: str, max_results: int = 20) -> dict:
    return _api(
        "POST",
        "/memory/read",
        {
            "query": query,
            "format": "packet",
            "max_results": max_results,
        },
    )


def cml_read_context(query: str, max_results: int = 20) -> str:
    data = _api(
        "POST",
        "/memory/read",
        {
            "query": query,
            "format": "llm_context",
            "max_results": max_results,
        },
    )
    return data.get("llm_context") or ""


def cml_health() -> bool:
    try:
        resp = requests.get(f"{CML_BASE_URL}/api/v1/health", timeout=5)
        return resp.status_code == 200
    except Exception:
        return False


# ---------------------------------------------------------------------------
# Ingestion
# ---------------------------------------------------------------------------


def ingest_corpus() -> tuple[int, float]:
    print(f"\n{'=' * 70}")
    print(f"  PHASE 1: INGESTING {len(CORPUS)} memories into tenant '{TENANT_ID}'")
    print(f"{'=' * 70}\n")

    success = 0
    t0 = time.perf_counter()
    for i, (session_id, content, ts) in enumerate(CORPUS):
        try:
            result = cml_write(content, session_id, ts)
            if result.get("success"):
                success += 1
                status = "OK"
            else:
                status = "SKIP"
        except Exception as e:
            status = f"ERR: {e}"
        print(f"  [{i + 1:3d}/{len(CORPUS)}] {status:6s} | {content[:70]}...")
        time.sleep(1.2)  # respect per-tenant rate limit (default: 60 req/min)

    elapsed = time.perf_counter() - t0
    print(f"\n  Ingested {success}/{len(CORPUS)} memories in {elapsed:.1f}s")
    print(f"  Avg write latency: {elapsed / max(len(CORPUS), 1) * 1000:.0f}ms")
    return success, elapsed


# ---------------------------------------------------------------------------
# Retrieval probes
# ---------------------------------------------------------------------------


@dataclass
class ProbeResult:
    probe: Probe
    context: str
    raw_response: dict
    keyword_hits: list[bool]  # one per keyword group
    precision_keywords_found: int
    precision_keywords_total: int
    recall_groups_hit: int
    recall_groups_total: int
    latency_ms: float
    llm_judge_score: float = 0.0
    llm_judge_reason: str = ""
    constraint_relevant: bool = False


def run_probes(verbose: bool = False) -> list[ProbeResult]:
    print(f"\n{'=' * 70}")
    print(f"  PHASE 2: RUNNING {len(PROBES)} RETRIEVAL PROBES")
    print(f"{'=' * 70}\n")

    results: list[ProbeResult] = []
    for i, probe in enumerate(PROBES):
        t0 = time.perf_counter()
        raw = cml_read(probe.query)
        latency = (time.perf_counter() - t0) * 1000

        all_texts = " ".join(m.get("text", "") for m in raw.get("memories", [])).lower()
        constraint_texts = " ".join(c.get("text", "") for c in raw.get("constraints", [])).lower()
        combined = all_texts + " " + constraint_texts

        keyword_hits = []
        total_kw_found = 0
        total_kw = 0
        for group in probe.expected_keywords:
            group_hit = any(kw.lower() in combined for kw in group)
            keyword_hits.append(group_hit)
            total_kw += len(group)
            total_kw_found += sum(1 for kw in group if kw.lower() in combined)

        groups_hit = sum(keyword_hits)
        has_constraints = len(raw.get("constraints", [])) > 0

        result = ProbeResult(
            probe=probe,
            context=combined[:500],
            raw_response=raw,
            keyword_hits=keyword_hits,
            precision_keywords_found=total_kw_found,
            precision_keywords_total=total_kw,
            recall_groups_hit=groups_hit,
            recall_groups_total=len(probe.expected_keywords),
            latency_ms=latency,
            constraint_relevant=has_constraints,
        )
        results.append(result)

        status = "PASS" if all(keyword_hits) else "MISS"
        emoji = "\u2705" if status == "PASS" else "\u274c"
        print(
            f"  [{i + 1:2d}/{len(PROBES)}] {emoji} {status} | {probe.category:22s} | {latency:6.0f}ms | {probe.query[:55]}"
        )
        if verbose and status == "MISS":
            for j, (group, hit) in enumerate(
                zip(probe.expected_keywords, keyword_hits, strict=False)
            ):
                if not hit:
                    print(f"         Missing: {group}")
            print(f"         Retrieved: {combined[:200]}...")
            print()
        time.sleep(1.2)  # respect rate limit

    return results


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = dedent("""\
    You are evaluating a memory retrieval system. Given a user query, the expected
    information, and the actually retrieved context, score the retrieval quality.

    QUERY: {query}

    EXPECTED INFORMATION (at least one keyword from each group should appear):
    {expected}

    ACTUALLY RETRIEVED CONTEXT:
    {context}

    Score the retrieval on a scale of 0-10:
    - 10: Perfect retrieval — all expected information is present and relevant
    - 7-9: Good — most expected information present, minor gaps
    - 4-6: Partial — some expected information found but significant gaps
    - 1-3: Poor — very little expected information retrieved
    - 0: Complete failure — nothing relevant retrieved

    Also evaluate:
    - CONSTRAINT_CONSISTENCY: Did the system retrieve safety-critical constraints
      (allergies, policies, values) when they were relevant to the query? (yes/no/na)
    - RELEVANCE: How relevant was the retrieved context to the query? (high/medium/low)
    - NOISE: How much irrelevant information was included? (none/some/excessive)

    Return ONLY valid JSON:
    {{"score": 7, "constraint_consistency": "yes", "relevance": "high", "noise": "some", "reason": "Brief explanation"}}
""")


def run_llm_judge(results: list[ProbeResult]) -> list[ProbeResult]:
    print(f"\n{'=' * 70}")
    print(f"  PHASE 3: LLM-AS-JUDGE EVALUATION ({len(results)} probes)")
    print(f"{'=' * 70}\n")

    base_url, model, _ = _get_llm_config()
    print(f"  Judge model: {model}")
    print(f"  Judge endpoint: {base_url}\n")

    for i, r in enumerate(results):
        expected_str = "\n".join(
            f"  Group {j + 1}: {', '.join(group)}"
            for j, group in enumerate(r.probe.expected_keywords)
        )
        prompt = JUDGE_PROMPT.format(
            query=r.probe.query,
            expected=expected_str,
            context=r.context[:2000],
        )
        raw = _llm_call(prompt, max_tokens=300)
        try:
            if "```" in raw:
                raw = raw.split("```")[1].strip()
                if raw.startswith("json"):
                    raw = raw[4:].strip()
            data = json.loads(raw)
            r.llm_judge_score = float(data.get("score", 0))
            r.llm_judge_reason = data.get("reason", "")
            constraint_flag = data.get("constraint_consistency", "na")
            if constraint_flag == "yes":
                r.constraint_relevant = True
        except (json.JSONDecodeError, ValueError, TypeError):
            r.llm_judge_score = 0
            r.llm_judge_reason = f"Judge parse error: {raw[:100]}"

        emoji = (
            "\u2705"
            if r.llm_judge_score >= 7
            else ("\u26a0\ufe0f" if r.llm_judge_score >= 4 else "\u274c")
        )
        print(
            f"  [{i + 1:2d}/{len(results)}] {emoji} {r.llm_judge_score:4.1f}/10 | {r.probe.category:22s} | {r.probe.query[:50]}"
        )
        time.sleep(0.3)

    return results


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------


def print_report(results: list[ProbeResult], ingest_count: int, ingest_time: float) -> None:
    print(f"\n{'=' * 70}")
    print("  FINAL REPORT — CML Memory Quality Assessment")
    print(f"{'=' * 70}\n")

    # --- Ingestion summary ---
    print(f"  Corpus size:       {len(CORPUS)} memories")
    print(f"  Ingested:          {ingest_count}")
    print(f"  Ingestion time:    {ingest_time:.1f}s")
    print(f"  Tenant:            {TENANT_ID}")
    print()

    # --- Keyword-based recall ---
    total_groups = sum(r.recall_groups_total for r in results)
    hit_groups = sum(r.recall_groups_hit for r in results)
    keyword_recall = hit_groups / max(total_groups, 1)
    full_recall = sum(1 for r in results if all(r.keyword_hits)) / max(len(results), 1)

    print("  KEYWORD-BASED METRICS")
    print(f"  {'─' * 50}")
    print(
        f"  Group-level recall:  {keyword_recall:.1%} ({hit_groups}/{total_groups} keyword groups found)"
    )
    print(
        f"  Full-probe recall:   {full_recall:.1%} ({sum(1 for r in results if all(r.keyword_hits))}/{len(results)} probes fully satisfied)"
    )
    print()

    # --- LLM judge scores ---
    scores = [r.llm_judge_score for r in results]
    avg_score = sum(scores) / max(len(scores), 1)
    pass_count = sum(1 for s in scores if s >= 7)
    partial_count = sum(1 for s in scores if 4 <= s < 7)
    fail_count = sum(1 for s in scores if s < 4)

    print("  LLM JUDGE SCORES")
    print(f"  {'─' * 50}")
    print(f"  Average score:       {avg_score:.1f}/10")
    print(
        f"  Pass (\u22657):          {pass_count}/{len(results)} ({pass_count / max(len(results), 1):.0%})"
    )
    print(f"  Partial (4-6):       {partial_count}/{len(results)}")
    print(f"  Fail (<4):           {fail_count}/{len(results)}")
    print()

    # --- Per-category breakdown ---
    categories: dict[str, list[ProbeResult]] = {}
    for r in results:
        categories.setdefault(r.probe.category, []).append(r)

    print("  PER-CATEGORY BREAKDOWN")
    print(f"  {'─' * 50}")
    print(f"  {'Category':<25s} {'Recall':>8s} {'Judge':>8s} {'Latency':>10s}")
    print(f"  {'─' * 25} {'─' * 8} {'─' * 8} {'─' * 10}")
    for cat, cat_results in sorted(categories.items()):
        cat_recall = sum(1 for r in cat_results if all(r.keyword_hits)) / max(len(cat_results), 1)
        cat_judge = sum(r.llm_judge_score for r in cat_results) / max(len(cat_results), 1)
        cat_latency = sum(r.latency_ms for r in cat_results) / max(len(cat_results), 1)
        print(f"  {cat:<25s} {cat_recall:>7.0%} {cat_judge:>7.1f} {cat_latency:>8.0f}ms")
    print()

    # --- Failure mode analysis ---
    failure_modes: dict[str, list[ProbeResult]] = {}
    for r in results:
        if r.probe.failure_mode:
            fm = r.probe.failure_mode
            failure_modes.setdefault(fm, []).append(r)

    if failure_modes:
        print("  FAILURE MODE ANALYSIS")
        print(f"  {'─' * 50}")
        for fm, fm_results in sorted(failure_modes.items()):
            fm_pass = sum(1 for r in fm_results if all(r.keyword_hits))
            fm_judge = sum(r.llm_judge_score for r in fm_results) / max(len(fm_results), 1)
            print(f"  {fm:<30s} Recall: {fm_pass}/{len(fm_results)}  Judge: {fm_judge:.1f}/10")
        print()

    # --- Latency ---
    latencies = [r.latency_ms for r in results]
    avg_lat = sum(latencies) / max(len(latencies), 1)
    p50 = sorted(latencies)[len(latencies) // 2] if latencies else 0
    p95 = sorted(latencies)[int(len(latencies) * 0.95)] if latencies else 0

    print("  LATENCY")
    print(f"  {'─' * 50}")
    print(f"  Average:  {avg_lat:.0f}ms")
    print(f"  P50:      {p50:.0f}ms")
    print(f"  P95:      {p95:.0f}ms")
    print()

    # --- Constraint consistency ---
    constraint_probes = [
        r for r in results if r.probe.category in ("constraint_direct", "semantic_disconnect")
    ]
    if constraint_probes:
        cc_pass = sum(1 for r in constraint_probes if r.constraint_relevant and all(r.keyword_hits))
        print("  CONSTRAINT CONSISTENCY")
        print(f"  {'─' * 50}")
        print(f"  Constraint probes:     {len(constraint_probes)}")
        print(
            f"  Constraints retrieved: {cc_pass}/{len(constraint_probes)} ({cc_pass / max(len(constraint_probes), 1):.0%})"
        )
        print()

    # --- Worst performers ---
    print("  WORST PERFORMING PROBES (by judge score)")
    print(f"  {'─' * 50}")
    worst = sorted(results, key=lambda r: r.llm_judge_score)[:5]
    for r in worst:
        emoji = "\u274c" if r.llm_judge_score < 4 else "\u26a0\ufe0f"
        print(f"  {emoji} {r.llm_judge_score:.1f}/10 | {r.probe.category:22s} | {r.probe.query}")
        if r.llm_judge_reason:
            print(f"        Reason: {r.llm_judge_reason[:100]}")
    print()

    # --- Overall verdict ---
    print(f"  {'=' * 50}")
    if avg_score >= 7 and keyword_recall >= 0.8:
        print(f"  \u2705 OVERALL: PASS (score {avg_score:.1f}/10, recall {keyword_recall:.0%})")
    elif avg_score >= 4 and keyword_recall >= 0.5:
        print(
            f"  \u26a0\ufe0f  OVERALL: PARTIAL (score {avg_score:.1f}/10, recall {keyword_recall:.0%})"
        )
    else:
        print(
            f"  \u274c OVERALL: NEEDS IMPROVEMENT (score {avg_score:.1f}/10, recall {keyword_recall:.0%})"
        )
    print(f"  {'=' * 50}\n")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="CML Memory Quality Test")
    parser.add_argument("--skip-ingestion", action="store_true", help="Reuse existing data")
    parser.add_argument("--skip-judge", action="store_true", help="Skip LLM judge (keyword-only)")
    parser.add_argument("--verbose", action="store_true", help="Show per-probe details")
    parser.add_argument("--tenant", type=str, default=None, help="Override tenant ID")
    args = parser.parse_args()

    if args.tenant:
        global TENANT_ID
        TENANT_ID = args.tenant

    print("\n  CML Memory Quality Test")
    print(f"  Target: {CML_BASE_URL}")
    print(f"  Tenant: {TENANT_ID}")

    if not cml_health():
        print(f"\n  \u274c CML API not reachable at {CML_BASE_URL}")
        print("  Start the API first: uvicorn src.api.app:app --host 0.0.0.0 --port 8000")
        sys.exit(1)
    print("  \u2705 CML API healthy\n")

    # Phase 1: Ingest
    if args.skip_ingestion:
        ingest_count, ingest_time = len(CORPUS), 0.0
        print("  Skipping ingestion (--skip-ingestion)")
    else:
        ingest_count, ingest_time = ingest_corpus()

    # Phase 2: Retrieve
    results = run_probes(verbose=args.verbose)

    # Phase 3: Judge
    if not args.skip_judge:
        try:
            _get_llm_config()
            results = run_llm_judge(results)
        except Exception as e:
            print(f"\n  \u26a0\ufe0f  LLM judge skipped: {e}")
            print("  Set LLM_EVAL__PROVIDER, LLM_EVAL__MODEL, LLM_EVAL__BASE_URL in .env")

    # Phase 4: Report
    print_report(results, ingest_count, ingest_time)

    # Save JSON results
    out_path = Path(__file__).parent.parent / "scripts" / "quality_test_results.json"
    out_data = {
        "tenant": TENANT_ID,
        "timestamp": datetime.now(UTC).isoformat(),
        "corpus_size": len(CORPUS),
        "probes": [
            {
                "query": r.probe.query,
                "category": r.probe.category,
                "failure_mode": r.probe.failure_mode,
                "recall_groups_hit": r.recall_groups_hit,
                "recall_groups_total": r.recall_groups_total,
                "latency_ms": round(r.latency_ms, 1),
                "judge_score": r.llm_judge_score,
                "judge_reason": r.llm_judge_reason,
                "keyword_hits": r.keyword_hits,
            }
            for r in results
        ],
    }
    try:
        out_path.write_text(json.dumps(out_data, indent=2))
        print(f"  Results saved to {out_path}")
    except Exception:
        pass


if __name__ == "__main__":
    main()
