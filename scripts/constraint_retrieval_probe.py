#!/usr/bin/env python3
"""
Focused probe for latent-constraint retrieval and response generation.

What it does:
1. Optionally ingests a built-in scenario through the CML API.
2. Calls `/memory/read` and/or `/memory/turn`.
3. Prints retrieved memories and assembled context.
4. Optionally uses the repo's `LLM_EVAL__*` client to generate an answer.
"""

from __future__ import annotations

import argparse
import asyncio
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from textwrap import shorten
from typing import Any

import requests

ROOT = Path(__file__).resolve().parent.parent
SCRIPT_DIR = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from _shared import build_api_url, load_repo_env, normalize_bool_env, normalize_cml_base_url

load_repo_env()
normalize_bool_env("DEBUG")


@dataclass(frozen=True)
class WriteTurn:
    content: str
    timestamp: str | None = None
    context_tags: list[str] = field(default_factory=lambda: ["conversation"])


@dataclass(frozen=True)
class Scenario:
    name: str
    description: str
    writes: list[WriteTurn]
    query: str
    assistant_response: str | None = None
    expected_terms: list[str] = field(default_factory=list)


SCENARIOS: dict[str, Scenario] = {
    "shellfish_restaurant": Scenario(
        name="shellfish_restaurant",
        description="Classic cue-trigger semantic disconnect: allergy cue, restaurant trigger.",
        writes=[
            WriteTurn(
                "I never eat shellfish because I have a severe allergy. Even trace amounts are dangerous.",
                "2024-03-01T09:00:00Z",
                ["health", "diet"],
            ),
            WriteTurn(
                "I always carry an EpiPen due to the shellfish allergy.",
                "2024-03-01T09:05:00Z",
                ["health", "diet"],
            ),
            WriteTurn(
                "I love Thai food and quiet places with ocean views.",
                "2024-03-02T10:00:00Z",
                ["food", "preference"],
            ),
        ],
        query="Recommend a seafood restaurant for dinner tonight.",
        assistant_response="I can suggest a restaurant once I check whether any dietary constraints matter.",
        expected_terms=["shellfish", "allergy", "epipen"],
    ),
    "gift_plastics": Scenario(
        name="gift_plastics",
        description="Value and policy retrieval under a semantically distant gift query.",
        writes=[
            WriteTurn(
                "I refuse to use single-use plastics because of their impact on ocean life.",
                "2024-03-10T08:00:00Z",
                ["sustainability", "policy"],
            ),
            WriteTurn(
                "I value environmental sustainability above convenience.",
                "2024-03-10T08:05:00Z",
                ["sustainability", "value"],
            ),
            WriteTurn(
                "I like thoughtful handmade gifts.",
                "2024-03-12T12:00:00Z",
                ["gift", "preference"],
            ),
        ],
        query="What gift should I bring to a dinner party?",
        assistant_response="I should recommend something that fits your preferences and constraints.",
        expected_terms=["plastics", "sustainability", "handmade"],
    ),
    "budget_decision": Scenario(
        name="budget_decision",
        description="Goal and value retrieval for a decision query with spending pressure.",
        writes=[
            WriteTurn(
                "I need to save money for the conference in Tokyo next February.",
                "2024-04-01T08:00:00Z",
                ["finance", "goal"],
            ),
            WriteTurn(
                "I follow a strict monthly budget right now.",
                "2024-04-01T08:05:00Z",
                ["finance", "policy"],
            ),
            WriteTurn(
                "I still enjoy special dinners when they fit the plan.",
                "2024-04-01T08:10:00Z",
                ["food", "preference"],
            ),
        ],
        query="Can I afford dinner at a Michelin-star restaurant tonight?",
        assistant_response="I should weigh your current goals and spending rules before answering.",
        expected_terms=["save money", "budget", "plan"],
    ),
    "stale_update": Scenario(
        name="stale_update",
        description="Simple stale-vs-current probe for supersession behavior.",
        writes=[
            WriteTurn("I don't drink coffee at all.", "2024-01-20T08:00:00Z", ["habit", "state"]),
            WriteTurn(
                "Actually, I started drinking one cup of coffee every morning in March.",
                "2024-04-01T08:00:00Z",
                ["habit", "state"],
            ),
        ],
        query="Should you order me a coffee tomorrow morning?",
        assistant_response="I should answer using the most current version of your routine.",
        expected_terms=["coffee", "morning", "march"],
    ),
}


class CMLApi:
    def __init__(self, base_url: str, api_key: str, tenant_id: str, timeout: float = 30.0) -> None:
        self.base_url = normalize_cml_base_url(base_url)
        self.timeout = timeout
        self.session = requests.Session()
        self.headers = {
            "Content-Type": "application/json",
            "Accept": "application/json",
            "X-API-Key": api_key,
            "X-Tenant-ID": tenant_id,
        }

    def _post(self, path: str, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        url = build_api_url(self.base_url, path)
        start = time.perf_counter()
        response = self.session.post(url, json=payload, headers=self.headers, timeout=self.timeout)
        elapsed_ms = (time.perf_counter() - start) * 1000
        try:
            body = response.json()
        except ValueError:
            body = {"raw_text": response.text}
        if not response.ok:
            raise RuntimeError(f"{path} failed ({response.status_code}): {body}")
        return body, elapsed_ms

    def write(
        self, content: str, *, timestamp: str | None, context_tags: list[str], session_id: str
    ) -> dict[str, Any]:
        payload = {
            "content": content,
            "context_tags": context_tags,
            "session_id": session_id,
            "timestamp": timestamp,
        }
        body, _elapsed_ms = self._post("/memory/write", payload)
        return body

    def read(
        self,
        query: str,
        *,
        max_results: int,
        user_timezone: str | None,
        context_format: str,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "query": query,
            "max_results": max_results,
            "format": context_format,
            "user_timezone": user_timezone,
        }
        return self._post("/memory/read", payload)

    def turn(
        self,
        user_message: str,
        *,
        assistant_response: str | None,
        session_id: str,
        max_context_tokens: int,
        user_timezone: str | None,
    ) -> tuple[dict[str, Any], float]:
        payload = {
            "user_message": user_message,
            "assistant_response": assistant_response,
            "session_id": session_id,
            "max_context_tokens": max_context_tokens,
            "user_timezone": user_timezone,
        }
        return self._post("/memory/turn", payload)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Probe constraint retrieval and context assembly.")
    parser.add_argument("--scenario", choices=sorted(SCENARIOS), default="shellfish_restaurant")
    parser.add_argument("--mode", choices=("read", "turn", "compare"), default="compare")
    parser.add_argument("--query", help="Override the scenario query.")
    parser.add_argument(
        "--assistant-response", help="Optional assistant response for /memory/turn."
    )
    parser.add_argument(
        "--skip-ingestion", action="store_true", help="Reuse existing tenant state."
    )
    parser.add_argument(
        "--generate-answer",
        action="store_true",
        help="Use LLM_EVAL__* to generate an answer from the retrieved context.",
    )
    parser.add_argument("--list-scenarios", action="store_true", help="Print scenarios and exit.")
    parser.add_argument("--show-full-context", action="store_true", help="Do not truncate context.")
    parser.add_argument("--max-results", type=int, default=12)
    parser.add_argument("--max-context-tokens", type=int, default=800)
    parser.add_argument(
        "--base-url",
        default=normalize_cml_base_url(os.environ.get("CML_BASE_URL")),
        help="CML base URL. Accepts either the root URL or a /api/v1 URL.",
    )
    parser.add_argument(
        "--api-key",
        default=os.environ.get("CML_API_KEY") or os.environ.get("AUTH__API_KEY") or "test-api-key",
    )
    parser.add_argument("--tenant-id", help="Default: probe-<scenario>-<unix_ts>.")
    parser.add_argument("--session-id", default="constraint-probe-session")
    parser.add_argument("--user-timezone")
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument(
        "--expected-term",
        action="append",
        default=[],
        help="Additional term to require in retrieved memories/context summaries.",
    )
    return parser.parse_args()


def print_scenarios() -> None:
    for scenario in SCENARIOS.values():
        print(f"{scenario.name}: {scenario.description}")
        print(f"  query: {scenario.query}")
        print(f"  writes: {len(scenario.writes)}")
        print(f"  expected_terms: {', '.join(scenario.expected_terms) or '(none)'}")


def _fmt_float(value: Any) -> str:
    try:
        return f"{float(value):.2f}"
    except (TypeError, ValueError):
        return "n/a"


def truncate_block(text: str, *, max_chars: int = 1400, full: bool = False) -> str:
    if full or len(text) <= max_chars:
        return text
    return text[:max_chars] + "\n... [truncated]"


def format_memories(memories: list[dict[str, Any]], *, limit: int = 8) -> str:
    if not memories:
        return "(none)"
    lines: list[str] = []
    for idx, mem in enumerate(memories[:limit], start=1):
        mem_type = mem.get("type", "unknown")
        relevance = mem.get("relevance")
        confidence = mem.get("confidence")
        text = shorten(str(mem.get("text", "")), width=130, placeholder="...")
        lines.append(
            f"{idx:02d}. [{mem_type}] rel={_fmt_float(relevance)} conf={_fmt_float(confidence)} {text}"
        )
    return "\n".join(lines)


def context_summary(context: str) -> str:
    if not context:
        return "chars=0 first_section=(none)"
    first_section = "(none)"
    for line in context.splitlines():
        line = line.strip()
        if line.startswith("## ") or line.startswith("# "):
            first_section = line
            break
    return (
        f"chars={len(context)} "
        f"first_section={first_section!r} "
        f"contains_constraints={'Constraints' in context}"
    )


def evaluate_constraint_probe(
    memories: list[dict[str, Any]],
    context: str,
    *,
    expected_terms: list[str],
    top_k: int = 5,
) -> dict[str, Any]:
    """Compute deterministic hit metrics for constraint retrieval probes."""
    normalized_terms = [t.strip().lower() for t in expected_terms if t and t.strip()]
    if not normalized_terms:
        return {
            "top_k": int(top_k),
            "expected_terms": [],
            "memory_hit_terms": [],
            "context_hit_terms": [],
            "memory_hit_rate": 0.0,
            "context_hit_rate": 0.0,
            "all_terms_hit": True,
        }

    top_text = "\n".join(
        str(m.get("text", "") or "") for m in memories[: max(1, int(top_k))]
    ).lower()
    context_text = (context or "").lower()
    memory_hit_terms = [term for term in normalized_terms if term in top_text]
    context_hit_terms = [term for term in normalized_terms if term in context_text]
    union_hits = sorted(set(memory_hit_terms) | set(context_hit_terms))

    denom = float(len(normalized_terms))
    return {
        "top_k": int(top_k),
        "expected_terms": normalized_terms,
        "memory_hit_terms": sorted(memory_hit_terms),
        "context_hit_terms": sorted(context_hit_terms),
        "memory_hit_rate": round(len(memory_hit_terms) / denom, 4),
        "context_hit_rate": round(len(context_hit_terms) / denom, 4),
        "all_terms_hit": len(union_hits) == len(normalized_terms),
    }


async def generate_answer(query: str, memory_context: str) -> str:
    from src.utils.llm import get_eval_llm_client

    llm = get_eval_llm_client()
    system_prompt = (
        "Answer the user using the memory context when relevant. "
        "Prioritize active constraints, current state, and superseding updates over generic facts. "
        "If the memory context is insufficient, say so."
    )
    prompt = (
        f"User query:\n{query}\n\n"
        f"Memory context:\n{memory_context}\n\n"
        "Write a concise answer that is constraint-consistent."
    )
    return await llm.complete(prompt, temperature=0.0, max_tokens=220, system_prompt=system_prompt)


def ingest_scenario(api: CMLApi, scenario: Scenario, session_id: str) -> None:
    print("\n== Ingestion ==")
    for idx, turn in enumerate(scenario.writes, start=1):
        result = api.write(
            turn.content,
            timestamp=turn.timestamp,
            context_tags=turn.context_tags,
            session_id=session_id,
        )
        print(
            f"{idx:02d}. chunks_created={result.get('chunks_created', 0)} "
            f"memory_id={result.get('memory_id')}"
        )


def render_read_result(result: dict[str, Any], *, full_context: bool) -> None:
    print("\n== /memory/read ==")
    print(
        f"elapsed_ms={_fmt_float(result.get('elapsed_ms'))} "
        f"total_count={result.get('total_count', 0)}"
    )
    print("\nTop memories:")
    print(format_memories(result.get("memories", [])))
    llm_context = result.get("llm_context") or ""
    print("\nContext summary:")
    print(context_summary(llm_context))
    print("\nContext:")
    print(truncate_block(llm_context, full=full_context))


def render_turn_result(result: dict[str, Any], *, full_context: bool) -> None:
    print("\n== /memory/turn ==")
    print(
        f"memories_retrieved={result.get('memories_retrieved', 0)} "
        f"memories_stored={result.get('memories_stored', 0)} "
        f"reconsolidation_applied={result.get('reconsolidation_applied', False)}"
    )
    memory_context = result.get("memory_context") or ""
    print("\nContext summary:")
    print(context_summary(memory_context))
    print("\nContext:")
    print(truncate_block(memory_context, full=full_context))


def render_probe_metrics(label: str, metrics: dict[str, Any]) -> None:
    print(f"\n== {label} Metrics ==")
    print(f"expected_terms={metrics.get('expected_terms', [])}")
    print(f"memory_hit_terms={metrics.get('memory_hit_terms', [])}")
    print(f"context_hit_terms={metrics.get('context_hit_terms', [])}")
    print(
        f"memory_hit_rate={_fmt_float(metrics.get('memory_hit_rate'))} "
        f"context_hit_rate={_fmt_float(metrics.get('context_hit_rate'))} "
        f"all_terms_hit={metrics.get('all_terms_hit')}"
    )


def main() -> int:
    args = parse_args()

    if args.list_scenarios:
        print_scenarios()
        return 0

    scenario = SCENARIOS[args.scenario]
    query = args.query or scenario.query
    assistant_response = (
        args.assistant_response
        if args.assistant_response is not None
        else scenario.assistant_response
    )
    expected_terms = list(dict.fromkeys([*scenario.expected_terms, *args.expected_term]))
    tenant_id = args.tenant_id or f"probe-{args.scenario}-{int(time.time())}"

    print("== Probe ==")
    print(f"scenario={scenario.name}")
    print(f"description={scenario.description}")
    print(f"tenant_id={tenant_id}")
    print(f"session_id={args.session_id}")
    print(f"query={query}")
    print(f"mode={args.mode}")
    print(f"base_url={normalize_cml_base_url(args.base_url)}")
    print(f"expected_terms={expected_terms or '(none)'}")

    api = CMLApi(args.base_url, args.api_key, tenant_id, timeout=args.timeout)

    try:
        if not args.skip_ingestion:
            ingest_scenario(api, scenario, args.session_id)

        read_result: dict[str, Any] | None = None
        turn_result: dict[str, Any] | None = None

        if args.mode in ("read", "compare"):
            read_result, _elapsed_ms = api.read(
                query,
                max_results=args.max_results,
                user_timezone=args.user_timezone,
                context_format="llm_context",
            )
            render_read_result(read_result, full_context=args.show_full_context)
            render_probe_metrics(
                "/memory/read",
                evaluate_constraint_probe(
                    list(read_result.get("memories", [])),
                    read_result.get("llm_context") or "",
                    expected_terms=expected_terms,
                    top_k=args.max_results,
                ),
            )

        if args.mode in ("turn", "compare"):
            turn_result, _elapsed_ms = api.turn(
                query,
                assistant_response=assistant_response,
                session_id=args.session_id,
                max_context_tokens=args.max_context_tokens,
                user_timezone=args.user_timezone,
            )
            render_turn_result(turn_result, full_context=args.show_full_context)
            render_probe_metrics(
                "/memory/turn",
                evaluate_constraint_probe(
                    [],
                    turn_result.get("memory_context") or "",
                    expected_terms=expected_terms,
                    top_k=args.max_results,
                ),
            )

        if args.generate_answer:
            print("\n== Generated Answers ==")
            if read_result is not None and (read_result.get("llm_context") or ""):
                try:
                    answer = asyncio.run(generate_answer(query, read_result["llm_context"]))
                    print("\nFrom /memory/read context:")
                    print(answer)
                except Exception as exc:
                    print(f"\nFrom /memory/read context: generation failed: {exc}")
            if turn_result is not None and (turn_result.get("memory_context") or ""):
                try:
                    answer = asyncio.run(generate_answer(query, turn_result["memory_context"]))
                    print("\nFrom /memory/turn context:")
                    print(answer)
                except Exception as exc:
                    print(f"\nFrom /memory/turn context: generation failed: {exc}")

        return 0
    except requests.RequestException as exc:
        print(f"\nHTTP error: {exc}")
        return 1
    except RuntimeError as exc:
        print(f"\nAPI error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
