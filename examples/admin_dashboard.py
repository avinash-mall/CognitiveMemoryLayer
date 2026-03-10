"""Read-only admin/dashboard example for the py-cml client."""

from __future__ import annotations

from _shared import build_cml_config, explain_connection_failure, print_header

from cml import CognitiveMemoryLayer

EXAMPLE_META = {
    "name": "admin_dashboard",
    "kind": "python",
    "summary": "Read-only admin example covering overview, memories, facts, retrieval test, and jobs.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": True,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 60,
}


def main() -> int:
    print_header("CML Admin Dashboard")
    try:
        with CognitiveMemoryLayer(
            config=build_cml_config(require_admin_key=True, timeout=60.0)
        ) as memory:
            overview = memory.dashboard_overview()
            print(f"Overview total memories: {overview.total_memories}")
            print(f"Overview active memories: {overview.active_memories}")

            memories = memory.dashboard_memories(per_page=5)
            print(f"\nDashboard memories page size: {len(memories.items)}")
            for item in memories.items[:3]:
                print(f"  - [{item.type}] {item.text[:80]}")

            facts = memory.dashboard_facts(limit=5)
            print(f"\nFacts returned: {len(facts.items)}")

            retrieval = memory.test_retrieval("user preferences", max_results=5)
            print(f"\nRetrieval test count: {retrieval.total_count}")

            jobs = memory.get_jobs(limit=5)
            print(f"Recent jobs: {len(jobs.items)}")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
