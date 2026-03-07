"""Historical timestamp example for the py-cml client."""

from __future__ import annotations

import sys
from datetime import UTC, datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
EXAMPLES_ROOT = REPO_ROOT / "examples"
if str(EXAMPLES_ROOT) not in sys.path:
    sys.path.insert(0, str(EXAMPLES_ROOT))

from _shared import build_cml_config, explain_connection_failure, print_header  # noqa: E402

from cml import CognitiveMemoryLayer  # noqa: E402

EXAMPLE_META = {
    "name": "temporal_fidelity",
    "kind": "python",
    "summary": "Historical timestamp example for replay and temporal reasoning.",
    "requires_api": True,
    "requires_api_key": True,
    "requires_base_url": True,
    "requires_admin_key": False,
    "requires_embedded": False,
    "requires_openai": False,
    "requires_anthropic": False,
    "interactive": False,
    "timeout_sec": 120,
}


def main() -> int:
    print_header("CML Temporal Fidelity")
    try:
        with CognitiveMemoryLayer(config=build_cml_config(timeout=120.0)) as memory:
            six_months_ago = datetime.now(UTC) - timedelta(days=180)
            three_months_ago = datetime.now(UTC) - timedelta(days=90)
            one_month_ago = datetime.now(UTC) - timedelta(days=30)

            memory.write(
                "User used to prefer dark mode in all applications.",
                timestamp=six_months_ago,
                session_id="historical-theme",
                context_tags=["preference", "theme"],
            )
            memory.write(
                "User switched to light mode because of eye strain.",
                timestamp=three_months_ago,
                session_id="historical-theme",
                context_tags=["preference", "theme", "health"],
            )
            turn = memory.turn(
                user_message="What is my preferred theme setting?",
                assistant_response="You recently prefer light mode because it reduces eye strain.",
                timestamp=one_month_ago,
                session_id="historical-theme",
            )
            print(f"Turn stored {turn.memories_stored} memories")

            current = memory.write(
                "User is currently validating the temporal fidelity example.",
                session_id="historical-theme",
                context_tags=["testing"],
            )
            print(f"Current write success={current.success}")

            result = memory.read("theme preference", max_results=5)
            print(f"\nRetrieved {result.total_count} memories")
            for item in result.memories:
                print(f"  - [{item.timestamp.date()}] {item.text}")
        return 0
    except Exception as exc:
        print(f"Example failed: {exc}")
        print(explain_connection_failure())
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
