#!/usr/bin/env python3
"""
Run full LoCoMo evaluation: optionally tear down and rebuild Docker, wait for API health,
run eval_locomo_plus.py (unified LoCoMo + Locomo-Plus), then generate and display the
performance table matching the paper format.

Table columns: Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap

Run from project root:
  python evaluation/scripts/run_full_eval.py
  python evaluation/scripts/run_full_eval.py --skip-docker    # API already running
  python evaluation/scripts/run_full_eval.py --limit-samples 50  # Quick test

Requires: OPENAI_API_KEY (for LLM-as-judge), OLLAMA_QA_MODEL (optional, default gpt-oss:20b).
"""

from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

# Project root: evaluation/scripts -> evaluation -> project root
_SCRIPT_DIR = Path(__file__).resolve().parent
_ROOT = _SCRIPT_DIR.parent.parent
_COMPOSE_FILE = _ROOT / "docker" / "docker-compose.yml"
_EVAL_LOCOMO_PLUS = _ROOT / "evaluation" / "scripts" / "eval_locomo_plus.py"
_GENERATE_REPORT = _ROOT / "evaluation" / "scripts" / "generate_locomo_report.py"
_UNIFIED_FILE = _ROOT / "evaluation" / "locomo_plus" / "data" / "unified_input_samples_v2.json"
_OUT_DIR = _ROOT / "evaluation" / "outputs"
_LOCOMO_PLUS_ROOT = _ROOT / "evaluation" / "locomo_plus"
_HEALTH_URL = "http://localhost:8000/api/v1/health"
_JUDGE_SUMMARY = _OUT_DIR / "locomo_plus_qa_cml_judge_summary.json"
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT_SEC = 180


def _banner(text: str, char: str = "=") -> None:
    """Print a visible step banner."""
    width = max(70, len(text) + 4)
    print(flush=True)
    print(char * width, flush=True)
    print(f"  {text}", flush=True)
    print(char * width, flush=True)


def _run(cmd: list[str], step_name: str) -> bool:
    """Run command, return True on success."""
    print(f"  Running: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(_ROOT))
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        print(
            f"  FAILED: {step_name} (exit code {result.returncode}, {elapsed:.1f}s)",
            file=sys.stderr,
        )
        return False
    print(f"  OK ({elapsed:.1f}s)", flush=True)
    return True


def main() -> None:
    import argparse

    p = argparse.ArgumentParser(description="Run full LoCoMo evaluation pipeline")
    p.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker tear-down/rebuild and API wait (API must already be running)",
    )
    p.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        metavar="N",
        help="Run only first N samples (for quick testing)",
    )
    p.add_argument(
        "--ingestion-workers",
        type=int,
        default=5,
        metavar="N",
        help="Number of concurrent workers for Phase A ingestion (default 5)",
    )
    args = p.parse_args()

    _banner("LoCoMo Full Evaluation Pipeline", "=")
    print(f"  Project root: {_ROOT}", flush=True)
    if args.skip_docker:
        print("  Mode: skip-docker (API assumed running)", flush=True)
    if args.limit_samples:
        print(f"  Limit: {args.limit_samples} samples", flush=True)
    print(f"  Workers: {args.ingestion_workers}", flush=True)
    print(flush=True)

    ollama_model = os.environ.get("OLLAMA_QA_MODEL", "gpt-oss:20b")
    step_count = 5 if not args.skip_docker else 2  # Docker steps + eval + table
    current_step = 0

    if not args.skip_docker:
        # Step 1
        current_step += 1
        _banner(f"Step {current_step}/{step_count}: Tear down containers and volumes", "-")
        if not _run(
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "down", "-v"],
            "docker compose down -v",
        ):
            sys.exit(1)

        # Step 2
        current_step += 1
        _banner(f"Step {current_step}/{step_count}: Build and start services", "-")
        if not _run(
            [
                "docker",
                "compose",
                "-f",
                str(_COMPOSE_FILE),
                "up",
                "-d",
                "--build",
                "postgres",
                "neo4j",
                "redis",
                "api",
            ],
            "docker compose up",
        ):
            sys.exit(1)

        # Step 3
        current_step += 1
        _banner(f"Step {current_step}/{step_count}: Wait for CML API health", "-")
        print(
            f"  Polling {_HEALTH_URL} every {HEALTH_POLL_INTERVAL}s (timeout {HEALTH_TIMEOUT_SEC}s)",
            flush=True,
        )
        t0 = time.monotonic()
        deadline = t0 + HEALTH_TIMEOUT_SEC
        while time.monotonic() < deadline:
            try:
                import urllib.request

                req = urllib.request.Request(_HEALTH_URL, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        elapsed = time.monotonic() - t0
                        print(f"  API healthy ({elapsed:.1f}s)", flush=True)
                        break
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"  Waiting... {elapsed:.0f}s ({e})", flush=True)
                time.sleep(HEALTH_POLL_INTERVAL)
        else:
            print("  FAILED: API did not become healthy within timeout.", file=sys.stderr)
            sys.exit(1)

    # Step 4: Run evaluation
    current_step += 1
    _banner(f"Step {current_step}/{step_count}: Locomo-Plus evaluation (ingest, QA, judge)", "-")
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_LOCOMO_PLUS_ROOT)
    cmd = [
        sys.executable,
        str(_EVAL_LOCOMO_PLUS),
        "--unified-file",
        str(_UNIFIED_FILE),
        "--out-dir",
        str(_OUT_DIR),
        "--ollama-model",
        ollama_model,
    ]
    if args.limit_samples:
        cmd.extend(["--limit-samples", str(args.limit_samples)])
    cmd.extend(["--ingestion-workers", str(args.ingestion_workers)])
    print(f"  PYTHONPATH={env['PYTHONPATH']}", flush=True)
    print(f"  OLLAMA_QA_MODEL={ollama_model}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        print(
            f"  FAILED: eval_locomo_plus.py (exit {result.returncode}, {elapsed:.1f}s)",
            file=sys.stderr,
        )
        sys.exit(result.returncode)
    print(f"  OK ({elapsed:.1f}s)", flush=True)

    # Step 5: Generate and display performance table
    current_step += 1
    _banner(f"Step {current_step}/{step_count}: Performance table", "-")
    method = f"CML+{ollama_model}"
    if _JUDGE_SUMMARY.exists():
        report_cmd = [
            sys.executable,
            str(_GENERATE_REPORT),
            "--summary",
            str(_JUDGE_SUMMARY),
            "--method",
            method,
        ]
        subprocess.run(report_cmd, cwd=str(_ROOT))
        print(f"\n  Outputs: {_OUT_DIR}", flush=True)
        print(f"  - {_JUDGE_SUMMARY.name}", flush=True)
        print("  - locomo_plus_qa_cml_predictions.json", flush=True)
        print("  - locomo_plus_qa_cml_judged.json", flush=True)
    else:
        print(
            "  No judge summary found; evaluation may have failed before judge phase.",
            file=sys.stderr,
        )
        print(f"  Expected: {_JUDGE_SUMMARY}", file=sys.stderr)
        sys.exit(1)

    _banner("Pipeline complete", "=")
    print(flush=True)


if __name__ == "__main__":
    main()
