#!/usr/bin/env python3
"""
Run all LoCoMo evaluation phases in order: tear down Docker + volumes,
rebuild and start services, wait for API health, run eval_locomo.py, then
print a result comparison table (CML + LoCoMo baselines if present).

Prints the current step and phase so progress is visible. Run from project root:
  python evaluation/scripts/run_full_eval.py

To run without monitoring (background), see ProjectPlan/LocomoEval/RunEvaluation.md Section 8.2.
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
_EVAL_SCRIPT = _ROOT / "evaluation" / "scripts" / "eval_locomo.py"
_COMPARE_SCRIPT = _ROOT / "evaluation" / "scripts" / "compare_results.py"
_DATA_FILE = _ROOT / "evaluation" / "locomo" / "data" / "locomo10.json"
_OUT_DIR = _ROOT / "evaluation" / "outputs"
_LOCOMO_OUTPUTS = _ROOT / "evaluation" / "locomo" / "outputs"
_LOCOMO_ROOT = _ROOT / "evaluation" / "locomo"
HEALTH_URL = "http://localhost:8000/api/v1/health"
_CML_STATS = _OUT_DIR / "locomo10_qa_cml_stats.json"
_LOCOMO_BASELINE_STATS = _LOCOMO_OUTPUTS / "locomo10_qa_stats.json"
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT_SEC = 180


def _run(cmd: list[str], step_name: str) -> None:
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"FAILED: {step_name} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    print("LoCoMo full evaluation pipeline (4 steps + comparison)", flush=True)
    print("Project root:", _ROOT, flush=True)

    # Step 1
    print("\n--- Step 1/4: Tearing down containers and volumes ---", flush=True)
    _run(
        ["docker", "compose", "-f", str(_COMPOSE_FILE), "down", "-v"],
        "Step 1: docker compose down -v",
    )

    # Step 2
    print("\n--- Step 2/4: Building and starting postgres, neo4j, redis, api ---", flush=True)
    _run(
        [
            "docker", "compose", "-f", str(_COMPOSE_FILE),
            "up", "-d", "--build", "postgres", "neo4j", "redis", "api",
        ],
        "Step 2: docker compose up",
    )

    # Step 3
    print("\n--- Step 3/4: Waiting for CML API health ---", flush=True)
    deadline = time.monotonic() + HEALTH_TIMEOUT_SEC
    while time.monotonic() < deadline:
        try:
            import urllib.request
            req = urllib.request.Request(HEALTH_URL, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    print("  API is healthy.", flush=True)
                    break
        except Exception as e:
            print(f"  Waiting for API... ({e})", flush=True)
            time.sleep(HEALTH_POLL_INTERVAL)
    else:
        print("FAILED: Step 3: API did not become healthy within timeout.", file=sys.stderr)
        sys.exit(1)

    # Step 4
    print("\n--- Step 4/4: Running LoCoMo evaluation (ingestion, QA, scoring) ---", flush=True)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_LOCOMO_ROOT)
    cmd = [
        sys.executable,
        str(_EVAL_SCRIPT),
        "--data-file", str(_DATA_FILE),
        "--out-dir", str(_OUT_DIR),
    ]
    print(f"  PYTHONPATH={env['PYTHONPATH']}", flush=True)
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if result.returncode != 0:
        print(f"FAILED: Step 4: eval_locomo.py (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print("\n--- All steps completed successfully. ---", flush=True)
    print(f"Outputs: {_OUT_DIR / 'locomo10_qa_cml.json'}, {_CML_STATS}", flush=True)

    # Comparison: CML stats + LoCoMo baseline stats (if present); print table
    stats_files = [_CML_STATS]
    if _LOCOMO_BASELINE_STATS.exists():
        stats_files.append(_LOCOMO_BASELINE_STATS)
    if _CML_STATS.exists():
        print("\n--- Result comparison (overall accuracy & recall) ---", flush=True)
        compare_cmd = [sys.executable, str(_COMPARE_SCRIPT)] + [str(p) for p in stats_files]
        subprocess.run(compare_cmd, cwd=str(_ROOT))
    else:
        print("  (No CML stats file; skip comparison.)", flush=True)


if __name__ == "__main__":
    main()
