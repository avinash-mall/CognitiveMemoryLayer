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
  python evaluation/scripts/run_full_eval.py --resume         # Continue from last state (use with --skip-docker)

Resume: Use --resume (with --skip-docker) to continue after a crash or interrupt. Do not tear down
Docker/DB between runs when resuming; the same CML API must still be running.

Requires: OPENAI_API_KEY (for LLM-as-judge), OLLAMA_QA_MODEL (optional, default gpt-oss:20b).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from datetime import UTC, datetime
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
_STATE_FILE = _OUT_DIR / "full_eval_state.json"
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT_SEC = 180


def _load_state() -> dict | None:
    """Load pipeline state from disk. Returns None if missing or invalid."""
    if not _STATE_FILE.exists():
        return None
    try:
        data = json.loads(_STATE_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else None
    except (json.JSONDecodeError, OSError):
        return None


def _write_state(pipeline_step: int, eval_phase: str, ingestion_completed_indices: list[int] | None = None) -> None:
    """Write pipeline state to disk."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    payload: dict = {
        "pipeline_step": pipeline_step,
        "eval_phase": eval_phase,
        "last_updated_iso": datetime.now(UTC).isoformat(),
    }
    if ingestion_completed_indices is not None:
        payload["ingestion_completed_indices"] = ingestion_completed_indices
    _STATE_FILE.write_text(json.dumps(payload, indent=2), encoding="utf-8")


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
    p.add_argument(
        "--resume",
        action="store_true",
        help="Continue from last saved state; skip Docker steps if state indicates step 4+. Use with --skip-docker; do not tear down Docker between runs.",
    )
    args = p.parse_args()

    # When not resuming, clear any existing state so we start clean.
    if not args.resume and _STATE_FILE.exists():
        _STATE_FILE.unlink(missing_ok=True)

    # Skip Docker steps if user passed --skip-docker or if resuming and state says we're past step 3.
    state = _load_state() if args.resume else None
    skip_docker = args.skip_docker or (
        args.resume and state is not None and state.get("pipeline_step", 0) >= 4
    )

    _banner("LoCoMo Full Evaluation Pipeline", "=")
    print(f"  Project root: {_ROOT}", flush=True)
    if skip_docker:
        print("  Mode: skip-docker (API assumed running)", flush=True)
    if args.resume:
        print("  Mode: resume (continuing from last state)", flush=True)
    if args.limit_samples:
        print(f"  Limit: {args.limit_samples} samples", flush=True)
    print(f"  Workers: {args.ingestion_workers}", flush=True)
    print(flush=True)

    ollama_model = os.environ.get("OLLAMA_QA_MODEL", "gpt-oss:20b")
    step_count = 5 if not skip_docker else 2  # Docker steps + eval + table
    current_step = 0

    if not skip_docker:
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
    if not _STATE_FILE.exists():
        _write_state(pipeline_step=4, eval_phase="ingestion", ingestion_completed_indices=[])
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
    if args.resume:
        cmd.append("--resume")
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
    done_state = _load_state() or {}
    _write_state(
        pipeline_step=5,
        eval_phase="done",
        ingestion_completed_indices=done_state.get("ingestion_completed_indices") or [],
    )

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
