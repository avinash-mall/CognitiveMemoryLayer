"""Run full LoCoMo/Locomo-Plus evaluation pipeline."""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path

from cml.eval.locomo import run_locomo_plus
from cml.eval.report import generate_locomo_report
from cml.eval.types import FullEvalConfig, LocomoEvalConfig
from cml.eval.validate import validate_outputs

HEALTH_URL = "http://localhost:8000/api/v1/health"
HEALTH_POLL_INTERVAL = 5
HEALTH_TIMEOUT_SEC = 180


@dataclass(slots=True)
class _Paths:
    repo_root: Path
    compose_file: Path
    unified_file: Path
    out_dir: Path
    judge_summary: Path
    predictions_file: Path
    state_file: Path


def _resolve_paths(repo_root: Path) -> _Paths:
    out_dir = repo_root / "evaluation" / "outputs"
    return _Paths(
        repo_root=repo_root,
        compose_file=repo_root / "docker" / "docker-compose.yml",
        unified_file=repo_root / "evaluation" / "locomo_plus" / "data" / "unified_input_samples_v2.json",
        out_dir=out_dir,
        judge_summary=out_dir / "locomo_plus_qa_cml_judge_summary.json",
        predictions_file=out_dir / "locomo_plus_qa_cml_predictions.json",
        state_file=out_dir / "run_full_eval_state.json",
    )


def _banner(text: str, char: str = "=") -> None:
    width = max(70, len(text) + 4)
    print(flush=True)
    print(char * width, flush=True)
    print(f"  {text}", flush=True)
    print(char * width, flush=True)


def _run(cmd: list[str], cwd: Path, step_name: str) -> bool:
    print(f"  Running: {' '.join(cmd)}", flush=True)
    t0 = time.monotonic()
    result = subprocess.run(cmd, cwd=str(cwd))
    elapsed = time.monotonic() - t0
    if result.returncode != 0:
        print(
            f"  FAILED: {step_name} (exit code {result.returncode}, {elapsed:.1f}s)",
            file=sys.stderr,
            flush=True,
        )
        return False
    print(f"  OK ({elapsed:.1f}s)", flush=True)
    return True


def _load_state(state_file: Path) -> dict | None:
    if not state_file.exists():
        return None
    try:
        with open(state_file, encoding="utf-8") as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _save_state(
    state_file: Path,
    *,
    last_completed_step: int = 0,
    failure_step: int | None = None,
    failure_message: str = "",
    last_completed_sample: int | None = None,
    step_count: int = 5,
    skip_docker: bool = False,
) -> None:
    state_file.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "last_completed_step": last_completed_step,
        "failure_step": failure_step,
        "failure_message": failure_message,
        "step_count": step_count,
        "skip_docker": skip_docker,
    }
    if last_completed_sample is not None:
        data["last_completed_sample"] = last_completed_sample
    with open(state_file, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)


def _wait_for_health(url: str, *, poll_interval: int, timeout_sec: int) -> tuple[bool, dict | None]:
    print(
        f"  Polling {url} every {poll_interval}s (timeout {timeout_sec}s)",
        flush=True,
    )
    t0 = time.monotonic()
    deadline = t0 + timeout_sec
    while time.monotonic() < deadline:
        try:
            req = urllib.request.Request(url, method="GET")
            with urllib.request.urlopen(req, timeout=10) as resp:
                if resp.status == 200:
                    elapsed = time.monotonic() - t0
                    print(f"  API healthy ({elapsed:.1f}s)", flush=True)
                    body: dict | None = None
                    try:
                        body = json.loads(resp.read().decode())
                    except (json.JSONDecodeError, ValueError):
                        body = {}
                    return True, body
        except Exception as exc:
            elapsed = time.monotonic() - t0
            print(f"  Waiting... {elapsed:.0f}s ({exc})", flush=True)
            time.sleep(poll_interval)
    return False, None


def run_full_eval(config: FullEvalConfig) -> int:
    repo_root = Path(config.repo_root).resolve()
    paths = _resolve_paths(repo_root)
    required = [paths.compose_file, paths.unified_file, repo_root / "evaluation" / "locomo_plus"]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(
            "Repository root is missing required evaluation files. "
            f"Pass --repo-root explicitly. Missing: {missing}"
        )

    skip_docker = bool(config.skip_docker)
    if config.resume:
        skip_docker = True

    _banner("LoCoMo Full Evaluation Pipeline", "=")
    print(f"  Project root: {repo_root}", flush=True)
    if skip_docker:
        print("  Mode: skip-docker (API assumed running)", flush=True)
    if config.resume:
        print("  Mode: resume (skip ingestion + consolidation)", flush=True)
    if config.score_only:
        print("  Mode: score-only (judge + table)", flush=True)
    if config.limit_samples:
        print(f"  Limit: {config.limit_samples} samples", flush=True)
    print(f"  Workers: {config.ingestion_workers}", flush=True)
    print(flush=True)

    step_start = 4 if skip_docker else 1
    if config.resume:
        state = _load_state(paths.state_file)
        if state and state.get("failure_step") is not None:
            step_start = int(state["failure_step"])
            msg = state.get("failure_message") or ""
            print(f"  Resuming from step {step_start}. Previous failure: {msg}", flush=True)
            if state.get("last_completed_sample") is not None:
                n = int(state["last_completed_sample"]) + 1
                print(f"  Resuming evaluation from sample {n}.", flush=True)
        print(flush=True)

    llm_model = os.environ.get("LLM_EVAL__MODEL") or os.environ.get(
        "LLM_INTERNAL__MODEL", "gpt-4o-mini"
    )
    step_count = 5 if not skip_docker else 2

    if not skip_docker and step_start <= 3:
        _banner(f"Step 1/{step_count}: Tear down containers and volumes", "-")
        if not _run(
            ["docker", "compose", "-f", str(paths.compose_file), "down", "-v"],
            cwd=repo_root,
            step_name="docker compose down -v",
        ):
            _save_state(
                paths.state_file,
                last_completed_step=0,
                failure_step=1,
                failure_message="docker compose down -v failed",
                step_count=step_count,
                skip_docker=skip_docker,
            )
            return 1
        _save_state(
            paths.state_file,
            last_completed_step=1,
            step_count=step_count,
            skip_docker=skip_docker,
        )

        _banner(f"Step 2/{step_count}: Build and start services", "-")
        if not _run(
            [
                "docker",
                "compose",
                "-f",
                str(paths.compose_file),
                "up",
                "-d",
                "--build",
                "postgres",
                "neo4j",
                "redis",
                "api",
            ],
            cwd=repo_root,
            step_name="docker compose up",
        ):
            _save_state(
                paths.state_file,
                last_completed_step=1,
                failure_step=2,
                failure_message="docker compose up failed",
                step_count=step_count,
                skip_docker=skip_docker,
            )
            return 1
        _save_state(
            paths.state_file,
            last_completed_step=2,
            step_count=step_count,
            skip_docker=skip_docker,
        )

        _banner(f"Step 3/{step_count}: Wait for CML API health", "-")
        health_ok, health_body = _wait_for_health(
            HEALTH_URL,
            poll_interval=config.health_poll_interval_sec,
            timeout_sec=config.health_timeout_sec,
        )
        if not health_ok:
            _save_state(
                paths.state_file,
                last_completed_step=2,
                failure_step=3,
                failure_message="API did not become healthy within timeout",
                step_count=step_count,
                skip_docker=skip_docker,
            )
            return 1
        if (health_body or {}).get("status") != "healthy":
            _save_state(
                paths.state_file,
                last_completed_step=2,
                failure_step=3,
                failure_message="Health check returned unexpected body (expected status=healthy)",
                step_count=step_count,
                skip_docker=skip_docker,
            )
            return 1
        _save_state(
            paths.state_file,
            last_completed_step=3,
            step_count=step_count,
            skip_docker=skip_docker,
        )

    if step_start <= 4:
        _banner(f"Step 4/{step_count}: Locomo-Plus evaluation (ingest, QA, judge)", "-")
        paths.out_dir.mkdir(parents=True, exist_ok=True)
        cml_url = os.environ.get("CML_BASE_URL", "http://localhost:8000")
        cml_api_key = os.environ.get("CML_API_KEY", "test-key")

        locomo_cfg = LocomoEvalConfig(
            unified_file=paths.unified_file,
            out_dir=paths.out_dir,
            cml_url=cml_url,
            cml_api_key=cml_api_key,
            limit_samples=config.limit_samples,
            ingestion_workers=config.ingestion_workers,
            skip_ingestion=config.resume,
            skip_consolidation=config.resume,
            score_only=config.score_only,
            judge_model=llm_model,
        )
        print(f"  QA LLM (from .env): LLM_EVAL__MODEL/LLM_INTERNAL__MODEL={llm_model}", flush=True)
        t0 = time.monotonic()
        try:
            run_locomo_plus(locomo_cfg)
        except Exception as exc:
            elapsed = time.monotonic() - t0
            last_sample: int | None = None
            if paths.predictions_file.exists():
                try:
                    with open(paths.predictions_file, encoding="utf-8") as f:
                        preds = json.load(f)
                    if isinstance(preds, list) and preds:
                        last_sample = len(preds) - 1
                except (json.JSONDecodeError, OSError):
                    pass
            msg = str(exc)
            if last_sample is not None:
                msg = f"Failed after sample {last_sample + 1}: {exc}"
            _save_state(
                paths.state_file,
                last_completed_step=3,
                failure_step=4,
                failure_message=msg,
                last_completed_sample=last_sample,
                step_count=step_count,
                skip_docker=skip_docker,
            )
            print(f"  FAILED: {msg} ({elapsed:.1f}s)", file=sys.stderr, flush=True)
            return 1
        elapsed = time.monotonic() - t0
        print(f"  OK ({elapsed:.1f}s)", flush=True)

        errors = validate_outputs(paths.out_dir)
        if errors:
            _save_state(
                paths.state_file,
                last_completed_step=3,
                failure_step=4,
                failure_message="Validation failed after Step 4:\n" + "\n".join(errors),
                step_count=step_count,
                skip_docker=skip_docker,
            )
            print("  Validation failed after Step 4:", file=sys.stderr, flush=True)
            for err in errors:
                print(f"    - {err}", file=sys.stderr, flush=True)
            return 1

        _save_state(
            paths.state_file,
            last_completed_step=4,
            step_count=step_count,
            skip_docker=skip_docker,
        )

    if step_start <= 5:
        _banner(f"Step 5/{step_count}: Performance table", "-")
        errors = validate_outputs(paths.out_dir)
        if errors:
            _save_state(
                paths.state_file,
                last_completed_step=4,
                failure_step=5,
                failure_message="Validation failed before table:\n" + "\n".join(errors),
                step_count=step_count,
                skip_docker=skip_docker,
            )
            print("  Validation failed before Step 5:", file=sys.stderr, flush=True)
            for err in errors:
                print(f"    - {err}", file=sys.stderr, flush=True)
            return 1

        method = f"CML+{llm_model}"
        if not paths.judge_summary.exists():
            _save_state(
                paths.state_file,
                last_completed_step=4,
                failure_step=5,
                failure_message="No judge summary found; evaluation may have failed before judge phase",
                step_count=step_count,
                skip_docker=skip_docker,
            )
            print("  No judge summary found.", file=sys.stderr, flush=True)
            return 1

        print(flush=True)
        print(generate_locomo_report(paths.judge_summary, method=method, no_title=False), flush=True)
        print(flush=True)
        print(f"  Outputs: {paths.out_dir}", flush=True)
        print(f"  - {paths.judge_summary.name}", flush=True)
        print("  - locomo_plus_qa_cml_predictions.json", flush=True)
        print("  - locomo_plus_qa_cml_judged.json", flush=True)

        _save_state(
            paths.state_file,
            last_completed_step=5,
            step_count=step_count,
            skip_docker=skip_docker,
        )

    _banner("Pipeline complete", "=")
    print(flush=True)
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run full LoCoMo evaluation pipeline")
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path.cwd(),
        help="Repository root (default: current working directory)",
    )
    parser.add_argument(
        "--skip-docker",
        action="store_true",
        help="Skip Docker tear-down/rebuild and API wait (API must already be running)",
    )
    parser.add_argument(
        "--limit-samples",
        type=int,
        default=None,
        metavar="N",
        help="Run only first N samples (for quick testing)",
    )
    parser.add_argument(
        "--ingestion-workers",
        type=int,
        default=5,
        metavar="N",
        help="Number of concurrent workers for Phase A ingestion (default 5)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume from last failure (implies --skip-docker).",
    )
    parser.add_argument(
        "--score-only",
        action="store_true",
        help="Run only Phase C (judge) and performance table; requires existing predictions JSON.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = FullEvalConfig(
        repo_root=Path(args.repo_root),
        skip_docker=bool(args.skip_docker),
        limit_samples=args.limit_samples,
        ingestion_workers=int(args.ingestion_workers),
        resume=bool(args.resume),
        score_only=bool(args.score_only),
    )
    try:
        return run_full_eval(cfg)
    except FileNotFoundError as exc:
        print(str(exc), file=sys.stderr, flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
