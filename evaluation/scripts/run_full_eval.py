#!/usr/bin/env python3
"""
Run full LoCoMo evaluation: optionally tear down and rebuild Docker, wait for API health,
run eval_locomo_plus.py (unified LoCoMo + Locomo-Plus), then generate and display the
performance table matching the paper format.

Outputs are validated after each relevant step. On failure, state is written to
evaluation/outputs/run_full_eval_state.json; --resume continues from the failed step
(and from the next sample for step 4). --resume implies --skip-docker (no need to pass both).

Table columns: Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap

Run from project root:
  python evaluation/scripts/run_full_eval.py
  python evaluation/scripts/run_full_eval.py --skip-docker    # API already running
  python evaluation/scripts/run_full_eval.py --limit-samples 50  # Quick test
  python evaluation/scripts/run_full_eval.py --resume          # Resume from last failure (implies --skip-docker)
  python evaluation/scripts/run_full_eval.py --score-only    # Judge + table only; requires predictions JSON.

Requires: OPENAI_API_KEY (for LLM-as-judge). QA uses LLM from project .env (LLM__PROVIDER, LLM__MODEL, LLM__BASE_URL).
"""

from __future__ import annotations

import json
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
_VALIDATE_OUTPUTS = _ROOT / "evaluation" / "scripts" / "validate_outputs.py"
_UNIFIED_FILE = _ROOT / "evaluation" / "locomo_plus" / "data" / "unified_input_samples_v2.json"
_OUT_DIR = _ROOT / "evaluation" / "outputs"
_LOCOMO_PLUS_ROOT = _ROOT / "evaluation" / "locomo_plus"
_HEALTH_URL = "http://localhost:8000/api/v1/health"
_JUDGE_SUMMARY = _OUT_DIR / "locomo_plus_qa_cml_judge_summary.json"
_PREDICTIONS_FILE = _OUT_DIR / "locomo_plus_qa_cml_predictions.json"
_STATE_FILE = _OUT_DIR / "run_full_eval_state.json"
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


def _load_state() -> dict | None:
    """Load run state from evaluation/outputs/run_full_eval_state.json."""
    if not _STATE_FILE.exists():
        return None
    try:
        with open(_STATE_FILE, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(
    *,
    last_completed_step: int = 0,
    failure_step: int | None = None,
    failure_message: str = "",
    last_completed_sample: int | None = None,
    step_count: int = 5,
    skip_docker: bool = False,
) -> None:
    """Write run state so --resume can continue from failure."""
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    data = {
        "last_completed_step": last_completed_step,
        "failure_step": failure_step,
        "failure_message": failure_message,
        "step_count": step_count,
        "skip_docker": skip_docker,
    }
    if last_completed_sample is not None:
        data["last_completed_sample"] = last_completed_sample
    with open(_STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _run_validation() -> tuple[bool, str]:
    """Run validate_outputs.py; return (success, error_message)."""
    proc = subprocess.run(
        [sys.executable, str(_VALIDATE_OUTPUTS), "--outputs-dir", str(_OUT_DIR)],
        cwd=str(_ROOT),
        capture_output=True,
        text=True,
    )
    if proc.returncode == 0:
        return True, ""
    return False, (proc.stderr or proc.stdout or f"Validation exited with code {proc.returncode}").strip()


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
        help="Resume from last failure (implies --skip-docker).",
    )
    p.add_argument(
        "--score-only",
        action="store_true",
        help="Run only Phase C (judge) and performance table; requires existing predictions JSON.",
    )
    args = p.parse_args()

    # Resume implies skip-docker so API/DB state is preserved
    if args.resume:
        args.skip_docker = True

    _banner("LoCoMo Full Evaluation Pipeline", "=")
    print(f"  Project root: {_ROOT}", flush=True)
    if args.skip_docker:
        print("  Mode: skip-docker (API assumed running)", flush=True)
    if args.resume:
        print("  Mode: resume (skip ingestion + consolidation)", flush=True)
    if args.score_only:
        print("  Mode: score-only (judge + table)", flush=True)
    if args.limit_samples:
        print(f"  Limit: {args.limit_samples} samples", flush=True)
    print(f"  Workers: {args.ingestion_workers}", flush=True)
    print(flush=True)

    # Step start: when resuming, run from failed step; otherwise from 1 or 4
    step_start = 4 if args.skip_docker else 1
    if args.resume:
        state = _load_state()
        if state and state.get("failure_step") is not None:
            step_start = state["failure_step"]
            msg = state.get("failure_message") or ""
            print(f"  Resuming from step {step_start}. Previous failure: {msg}", flush=True)
            if state.get("last_completed_sample") is not None:
                n = state["last_completed_sample"] + 1
                print(f"  Resuming evaluation from sample {n}.", flush=True)
        print(flush=True)

    # QA model label for table (from .env; eval_locomo_plus reads LLM__* itself)
    llm_model = os.environ.get("LLM__MODEL", "gpt-4o-mini")
    step_count = 5 if not args.skip_docker else 2  # Docker steps + eval + table
    current_step = 0

    if not args.skip_docker and step_start <= 3:
        # Step 1
        current_step = 1
        _banner(f"Step {current_step}/{step_count}: Tear down containers and volumes", "-")
        if not _run(
            ["docker", "compose", "-f", str(_COMPOSE_FILE), "down", "-v"],
            "docker compose down -v",
        ):
            _save_state(
                last_completed_step=0,
                failure_step=1,
                failure_message="docker compose down -v failed",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            sys.exit(1)
        _save_state(
            last_completed_step=1,
            failure_step=None,
            failure_message="",
            step_count=step_count,
            skip_docker=args.skip_docker,
        )

        # Step 2
        current_step = 2
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
            _save_state(
                last_completed_step=1,
                failure_step=2,
                failure_message="docker compose up failed",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            sys.exit(1)
        _save_state(
            last_completed_step=2,
            failure_step=None,
            failure_message="",
            step_count=step_count,
            skip_docker=args.skip_docker,
        )

        # Step 3
        current_step = 3
        _banner(f"Step {current_step}/{step_count}: Wait for CML API health", "-")
        print(
            f"  Polling {_HEALTH_URL} every {HEALTH_POLL_INTERVAL}s (timeout {HEALTH_TIMEOUT_SEC}s)",
            flush=True,
        )
        t0 = time.monotonic()
        deadline = t0 + HEALTH_TIMEOUT_SEC
        health_ok = False
        health_body: dict | None = None
        while time.monotonic() < deadline:
            try:
                import urllib.request

                req = urllib.request.Request(_HEALTH_URL, method="GET")
                with urllib.request.urlopen(req, timeout=10) as resp:
                    if resp.status == 200:
                        elapsed = time.monotonic() - t0
                        print(f"  API healthy ({elapsed:.1f}s)", flush=True)
                        health_ok = True
                        try:
                            health_body = json.loads(resp.read().decode())
                        except (json.JSONDecodeError, ValueError):
                            health_body = {}
                        break
            except Exception as e:
                elapsed = time.monotonic() - t0
                print(f"  Waiting... {elapsed:.0f}s ({e})", flush=True)
                time.sleep(HEALTH_POLL_INTERVAL)
        if not health_ok:
            _save_state(
                last_completed_step=2,
                failure_step=3,
                failure_message="API did not become healthy within timeout",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            print("  FAILED: API did not become healthy within timeout.", file=sys.stderr)
            sys.exit(1)
        if (health_body or {}).get("status") != "healthy":
            _save_state(
                last_completed_step=2,
                failure_step=3,
                failure_message="Health check returned unexpected body (expected status=healthy)",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            print(
                "  FAILED: Health check returned unexpected body (expected status=healthy).",
                file=sys.stderr,
            )
            sys.exit(1)
        _save_state(
            last_completed_step=3,
            failure_step=None,
            failure_message="",
            step_count=step_count,
            skip_docker=args.skip_docker,
        )

    # Step 4: Run evaluation
    current_step = 4
    if current_step >= step_start:
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
        ]
        if args.limit_samples:
            cmd.extend(["--limit-samples", str(args.limit_samples)])
        if args.resume:
            cmd.extend(["--skip-ingestion", "--skip-consolidation"])
        if args.score_only:
            cmd.append("--score-only")
        cmd.extend(["--ingestion-workers", str(args.ingestion_workers)])
        print(f"  PYTHONPATH={env['PYTHONPATH']}", flush=True)
        print(f"  QA LLM (from .env): LLM__MODEL={llm_model}", flush=True)
        t0 = time.monotonic()
        result = subprocess.run(cmd, cwd=str(_ROOT), env=env)
        elapsed = time.monotonic() - t0
        if result.returncode != 0:
            last_sample: int | None = None
            if _PREDICTIONS_FILE.exists():
                try:
                    with open(_PREDICTIONS_FILE, encoding="utf-8") as f:
                        preds = json.load(f)
                    if isinstance(preds, list) and preds:
                        last_sample = len(preds) - 1
                except (json.JSONDecodeError, OSError):
                    pass
            msg = f"eval_locomo_plus.py exited with code {result.returncode}"
            if last_sample is not None:
                msg = f"Failed after sample {last_sample + 1} (exit code {result.returncode})"
            _save_state(
                last_completed_step=3,
                failure_step=4,
                failure_message=msg,
                last_completed_sample=last_sample,
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            print(f"  FAILED: {msg} ({elapsed:.1f}s)", file=sys.stderr)
            sys.exit(1)
        print(f"  OK ({elapsed:.1f}s)", flush=True)
        # Validate outputs after step 4
        ok, err = _run_validation()
        if not ok:
            _save_state(
                last_completed_step=3,
                failure_step=4,
                failure_message=f"Validation failed after Step 4 (evaluation):\n{err}",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            print("  Validation failed after Step 4 (evaluation):", file=sys.stderr)
            print(err, file=sys.stderr)
            sys.exit(1)
        _save_state(
            last_completed_step=4,
            failure_step=None,
            failure_message="",
            last_completed_sample=None,
            step_count=step_count,
            skip_docker=args.skip_docker,
        )

    # Step 5: Generate and display performance table
    current_step = 5
    if current_step >= step_start:
        _banner(f"Step {current_step}/{step_count}: Performance table", "-")
        # Validate before generating table
        ok, err = _run_validation()
        if not ok:
            _save_state(
                last_completed_step=4,
                failure_step=5,
                failure_message=f"Validation failed before table:\n{err}",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            print("  Validation failed before Step 5:", file=sys.stderr)
            print(err, file=sys.stderr)
            sys.exit(1)
        method = f"CML+{llm_model}"
        if _JUDGE_SUMMARY.exists():
            report_cmd = [
                sys.executable,
                str(_GENERATE_REPORT),
                "--summary",
                str(_JUDGE_SUMMARY),
                "--method",
                method,
            ]
            r = subprocess.run(report_cmd, cwd=str(_ROOT))
            if r.returncode != 0:
                _save_state(
                    last_completed_step=4,
                    failure_step=5,
                    failure_message=f"generate_locomo_report.py exited with code {r.returncode}",
                    step_count=step_count,
                    skip_docker=args.skip_docker,
                )
                sys.exit(r.returncode)
            print(f"\n  Outputs: {_OUT_DIR}", flush=True)
            print(f"  - {_JUDGE_SUMMARY.name}", flush=True)
            print("  - locomo_plus_qa_cml_predictions.json", flush=True)
            print("  - locomo_plus_qa_cml_judged.json", flush=True)
        else:
            _save_state(
                last_completed_step=4,
                failure_step=5,
                failure_message="No judge summary found; evaluation may have failed before judge phase",
                step_count=step_count,
                skip_docker=args.skip_docker,
            )
            print(
                "  No judge summary found; evaluation may have failed before judge phase.",
                file=sys.stderr,
            )
            print(f"  Expected: {_JUDGE_SUMMARY}", file=sys.stderr)
            sys.exit(1)
        _save_state(
            last_completed_step=5,
            failure_step=None,
            failure_message="",
            step_count=step_count,
            skip_docker=args.skip_docker,
        )

    _banner("Pipeline complete", "=")
    print(flush=True)


if __name__ == "__main__":
    main()
