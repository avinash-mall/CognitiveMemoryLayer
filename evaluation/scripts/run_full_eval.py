#!/usr/bin/env python3
"""
Run full LoCoMo evaluation: tear down Docker + volumes, rebuild and start services,
wait for API health, run eval_locomo_plus.py (unified LoCoMo + Locomo-Plus), then
generate and print the performance table.

Table columns: Method | single-hop | multi-hop | temporal | commonsense | adversarial | average | LoCoMo-Plus | Gap

Run from project root:
  python evaluation/scripts/run_full_eval.py

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


def _run(cmd: list[str], step_name: str) -> None:
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(_ROOT))
    if result.returncode != 0:
        print(f"FAILED: {step_name} (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)


def main() -> None:
    print("LoCoMo full evaluation pipeline (4 steps + table)", flush=True)
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
            req = urllib.request.Request(_HEALTH_URL, method="GET")
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
    print("\n--- Step 4/4: Running Locomo-Plus evaluation (ingestion, QA, LLM-as-judge) ---", flush=True)
    _OUT_DIR.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["PYTHONPATH"] = str(_LOCOMO_PLUS_ROOT)
    ollama_model = os.environ.get("OLLAMA_QA_MODEL", "gpt-oss:20b")
    cmd = [
        sys.executable,
        str(_EVAL_LOCOMO_PLUS),
        "--unified-file", str(_UNIFIED_FILE),
        "--out-dir", str(_OUT_DIR),
        "--ollama-model", ollama_model,
    ]
    print(f"  PYTHONPATH={env['PYTHONPATH']}", flush=True)
    print(f"  Running: {' '.join(cmd)}", flush=True)
    result = subprocess.run(cmd, cwd=str(_ROOT), env=env)
    if result.returncode != 0:
        print(f"FAILED: Step 4: eval_locomo_plus.py (exit code {result.returncode})", file=sys.stderr)
        sys.exit(result.returncode)

    print("\n--- All steps completed successfully. ---", flush=True)
    print(f"Outputs: {_OUT_DIR / 'locomo_plus_qa_cml_judge_summary.json'}", flush=True)

    # Generate and print performance table
    if _JUDGE_SUMMARY.exists():
        method = f"CML+{ollama_model}"
        report_cmd = [
            sys.executable,
            str(_GENERATE_REPORT),
            "--summary", str(_JUDGE_SUMMARY),
            "--method", method,
        ]
        subprocess.run(report_cmd, cwd=str(_ROOT))
    else:
        print("  (No judge summary; skip table.)", flush=True)


if __name__ == "__main__":
    main()
