#!/usr/bin/env python3
"""End-to-end LongMemEval-V2 evaluation driver for CognitiveMemoryLayer (CML) using a local model.

Runs both ``web`` (240 questions) and ``enterprise`` (211 questions) domains for the selected
LongMemEval-V2 tier (``small`` or ``medium``), combines domain metrics with example-count-weighted
averaging, computes exact Latency-Accuracy Frontier Score (LAFS) gain against the official
LongMemEval-V2 reference frontier, and writes summary JSON artifacts under
``evaluation/results/longmemeval_v2/``.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import urllib.request
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
LME_V2_ROOT = REPO_ROOT / "evaluation" / "longmemeval_v2"
if str(LME_V2_ROOT) not in sys.path:
    sys.path.insert(0, str(LME_V2_ROOT))

from leaderboard.combine_aggregated_metrics import combine_metrics
from leaderboard.compute_lafs import Point, lafs_summary_for_submission


def _check_endpoint(url: str, label: str) -> None:
    req = urllib.request.Request(url, method="GET")
    try:
        with urllib.request.urlopen(req, timeout=15) as resp:
            if resp.status >= 400:
                raise RuntimeError(f"{label} returned HTTP {resp.status} at {url}")
    except Exception as exc:
        raise RuntimeError(f"{label} health check failed ({url}): {exc}") from exc


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Run full LongMemEval-V2 evaluation with CML and local LLM/VLM endpoint."
    )
    p.add_argument(
        "--data-root",
        type=Path,
        default=LME_V2_ROOT / "data" / "longmemeval-v2",
        help="Path to prepared LongMemEval-V2 dataset",
    )
    p.add_argument(
        "--output-root",
        type=Path,
        default=REPO_ROOT / "evaluation" / "outputs" / "longmemeval_v2",
        help="Root directory for per-domain run outputs",
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=REPO_ROOT / "evaluation" / "results" / "longmemeval_v2",
        help="Directory for committed summary JSON artifacts",
    )
    p.add_argument("--tier", choices=["small", "medium"], default="small")
    p.add_argument(
        "--methods",
        nargs="+",
        default=["cml"],
        choices=["cml", "cml_fast", "no_retrieval", "rag_query_to_slice", "rag_query_to_slice_notes", "agentrunbook_r"],
        help="One or more operating points / methods to evaluate and combine into LAFS",
    )
    p.add_argument(
        "--domains",
        nargs="+",
        default=["web", "enterprise"],
        choices=["web", "enterprise"],
    )
    p.add_argument("--limit", type=int, default=None, help="Optional per-domain question limit")
    p.add_argument("--cml-base-url", default=os.getenv("CML_BASE_URL", "http://127.0.0.1:8000"))
    p.add_argument("--cml-tenant-prefix", default=os.getenv("CML_TENANT_PREFIX", "lme2v2"))

    p.add_argument("--local-base-url", default=os.getenv("READER_BASE_URL", "http://127.0.0.1:8023/v1"))
    p.add_argument("--local-model", default=os.getenv("READER_MODEL", "qwen3.8-27b"))
    p.add_argument("--reader-temperature", type=float, default=0.0)
    p.add_argument("--reader-top-p", type=float, default=0.95)
    p.add_argument("--reader-top-k", type=int, default=20)
    p.add_argument("--reader-reasoning-effort", choices=["low", "medium", "high"], default="low")
    p.add_argument("--reader-enable-thinking", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--max-completion-tokens", type=int, default=2048)
    p.add_argument("--memory-context-max-tokens", type=int, default=16000)
    p.add_argument("--prompt-build-max-workers", type=int, default=1)
    p.add_argument("--reader-max-concurrent-requests", type=int, default=2)
    p.add_argument("--evaluator-reasoning-effort", choices=["low", "medium", "high"], default="low")
    p.add_argument("--evaluator-max-completion-tokens", type=int, default=1024)
    p.add_argument("--run-tag", default=None, help="Optional suffix tag for output folders and summary JSON")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--skip-existing", action="store_true", help="Reuse aggregated_metrics.json if already present")
    p.add_argument(
        "--preingest-only",
        action="store_true",
        help="Batch-ingest all unique tier trajectories into CML (ideal on GPU before starting local LLM) and exit",
    )
    return p.parse_args()


def _build_dashboard_compatible_summary(
    *,
    method_label: str,
    tier: str,
    model_name: str,
    combined: dict[str, Any],
    domain_metrics: dict[str, dict[str, Any]],
    lafs_info: dict[str, Any] | None,
) -> dict[str, Any]:
    overall = combined.get("overall") or {}
    total_q = int(overall.get("count_all_questions") or 0)
    overall_acc = float(overall.get("overall_full_set") or 0.0)
    non_abst = combined.get("non_abstention_by_category") or {}
    abst_by_cat = combined.get("abstention_by_category") or {}

    by_category: dict[str, Any] = {}
    for cat, item in non_abst.items():
        cnt = int(item.get("count") or 0)
        avg = float(item.get("pct_correct") or 0.0)
        by_category[cat] = {
            "score": round(avg * cnt, 2),
            "count": cnt,
            "valid": cnt,
            "errors": 0,
            "avg": round(avg, 4),
            "pct_unknown": round(float(item.get("pct_unknown") or 0.0), 4),
        }
    for cat, item in abst_by_cat.items():
        cnt = int(item.get("count") or 0)
        avg = float(item.get("pct_correct") or 0.0)
        by_category[cat] = {
            "score": round(avg * cnt, 2),
            "count": cnt,
            "valid": cnt,
            "errors": 0,
            "avg": round(avg, 4),
            "pct_unknown": round(float(item.get("pct_unknown") or 0.0), 4),
        }

    return {
        "benchmark": "LongMemEval-V2",
        "tier": tier,
        "method": method_label,
        "model": model_name,
        "api_cost_usd": 0.0,
        "total_score": round(overall_acc * total_q, 2),
        "total_samples": total_q,
        "valid_samples": total_q,
        "error_count": 0,
        "max_possible": total_q,
        "overall_avg": round(overall_acc, 4),
        "overall_non_abstention": round(float(overall.get("overall_non_abstention_only") or 0.0), 4),
        "overall_abstention": round(float(overall.get("overall_abstention_only") or 0.0), 4),
        "memory_query_latency": combined.get("memory_query"),
        "memory_context_tokens": combined.get("memory_context"),
        "reader_tokens": combined.get("tokens"),
        "by_category": by_category,
        "combined_abstention_by_category": combined.get("combined_abstention_by_category"),
        "domains": {
            dom: {
                "overall": dm.get("overall"),
                "non_abstention_by_category": dm.get("non_abstention_by_category"),
                "abstention_by_category": dm.get("abstention_by_category"),
                "memory_query": dm.get("memory_query"),
                "memory_context": dm.get("memory_context"),
                "tokens": dm.get("tokens"),
            }
            for dom, dm in domain_metrics.items()
        },
        "lafs": lafs_info,
        "generated_at_utc": datetime.now(UTC).isoformat(),
    }


def main() -> int:
    args = parse_args()
    local_root = args.local_base_url.rstrip("/")
    if local_root.endswith("/v1"):
        local_health = f"{local_root[:-3]}/health"
    else:
        local_health = f"{local_root}/health"

    if any(m in {"cml", "cml_fast"} for m in args.methods) or args.preingest_only:
        _check_endpoint(f"{args.cml_base_url.rstrip('/')}/api/v1/health", "CML API")

    if args.preingest_only:
        import tempfile

        from data.public_data import load_haystack, load_questions, load_trajectories
        from memory_modules.cml_memory import CMLMemory
        from tqdm import tqdm

        questions = load_questions(args.data_root)
        haystack = load_haystack(args.data_root, args.tier)
        trajectories = load_trajectories(args.data_root)
        for domain in args.domains:
            dom_qids = [str(q["id"]) for q in questions if q.get("domain") == domain]
            unique_tids = sorted({tid for qid in dom_qids for tid in haystack.get(qid, [])})
            print(
                f"[preingest] Domain={domain} | Tier={args.tier} | Unique trajectories={len(unique_tids)}",
                flush=True,
            )
            with tempfile.TemporaryDirectory(prefix=f"cml_preingest_{domain}_") as tmp_ws:
                mem = CMLMemory(
                    {
                        "workspace_dir": tmp_ws,
                        "trajectories_root_dir": str(args.data_root),
                        "cml_params": {
                            "base_url": args.cml_base_url,
                            "tenant_prefix": args.cml_tenant_prefix,
                            "batch_size": 128,
                        },
                    }
                )
                for tid in tqdm(unique_tids, desc=f"Ingesting {domain}", unit="traj"):
                    mem.insert(trajectories[tid])
                mem._flush_pending_cml_turns()
        print("[preingest] Completed CML pre-ingestion for all requested domains!", flush=True)
        return 0

    _check_endpoint(local_health, "Local LLM/VLM server")

    args.output_root.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    env = os.environ.copy()
    env["OPENAI_API_KEY"] = env.get("OPENAI_API_KEY") or "local"
    env["OPENAI_BASE_URL"] = args.local_base_url
    env["CML_BASE_URL"] = args.cml_base_url

    submission_points: list[Point] = []
    per_method_combined: dict[str, dict[str, Any]] = {}
    per_method_domains: dict[str, dict[str, dict[str, Any]]] = {}

    for method in args.methods:
        domain_metrics: dict[str, dict[str, Any]] = {}
        domain_paths: dict[str, Path] = {}
        suffix = f"_{args.run_tag}" if args.run_tag else ""

        for domain in args.domains:
            out_dir = args.output_root / f"{method}_{domain}_{args.tier}{suffix}"
            agg_path = out_dir / "aggregated_metrics.json"
            if args.skip_existing and agg_path.exists():
                print(f"[skip-existing] Reusing {agg_path}", flush=True)
            else:
                cmd = [
                    sys.executable,
                    str(LME_V2_ROOT / "evaluation" / "run_eval.py"),
                    "--method",
                    method,
                    "--data-root",
                    str(args.data_root),
                    "--domain",
                    domain,
                    "--tier",
                    args.tier,
                    "--output-dir",
                    str(out_dir),
                    "--cml-base-url",
                    args.cml_base_url,
                    "--cml-tenant-prefix",
                    args.cml_tenant_prefix,
                    "--reader-model",
                    args.local_model,
                    "--reader-base-url",
                    args.local_base_url,
                    "--reader-temperature",
                    str(args.reader_temperature),
                    "--reader-top-p",
                    str(args.reader_top_p),
                    "--reader-top-k",
                    str(args.reader_top_k),
                    "--reader-reasoning-effort",
                    args.reader_reasoning_effort,
                    "--max-completion-tokens",
                    str(args.max_completion_tokens),
                    "--memory-context-max-tokens",
                    str(args.memory_context_max_tokens),
                    "--prompt-build-max-workers",
                    str(args.prompt_build_max_workers),
                    "--reader-max-concurrent-requests",
                    str(args.reader_max_concurrent_requests),
                    "--controller-model",
                    args.local_model,
                    "--controller-base-url",
                    args.local_base_url,
                    "--evaluator-model",
                    args.local_model,
                    "--evaluator-base-url",
                    args.local_base_url,
                    "--evaluator-reasoning-effort",
                    args.evaluator_reasoning_effort,
                    "--evaluator-max-completion-tokens",
                    str(args.evaluator_max_completion_tokens),
                    "--evaluator-temperature",
                    "0.0",
                ]
                if not args.reader_enable_thinking:
                    cmd.append("--no-reader-enable-thinking")
                if args.limit is not None:
                    cmd.extend(["--limit", str(args.limit)])
                if args.resume:
                    cmd.append("--resume")

                print(f"\n=== Running {method} | domain={domain} | tier={args.tier} ===", flush=True)
                subprocess.run(cmd, check=True, env=env)

            dm = json.loads(agg_path.read_text(encoding="utf-8"))
            domain_metrics[domain] = dm
            domain_paths[domain] = agg_path

        if "web" in domain_metrics and "enterprise" in domain_metrics:
            combined = combine_metrics(
                domain_metrics["enterprise"],
                domain_metrics["web"],
                domain_paths["enterprise"],
                domain_paths["web"],
            )
            combined_out = args.output_root / f"{method}_{args.tier}{suffix}_combined_metrics.json"
            combined_out.write_text(json.dumps(combined, indent=2) + "\n", encoding="utf-8")
        else:
            only_dom = next(iter(domain_metrics))
            combined = domain_metrics[only_dom]

        per_method_combined[method] = combined
        per_method_domains[method] = domain_metrics

        acc_pct = float((combined.get("overall") or {}).get("overall_full_set") or 0.0) * 100.0
        lat_sec = max(
            0.001,
            float((combined.get("memory_query") or {}).get("avg_seconds") or 0.1),
        )
        submission_points.append(Point(name=f"CML ({method})", acc=acc_pct, latency=lat_sec))

    date_str = datetime.now(UTC).strftime("%Y-%m-%d")
    model_slug = args.local_model.lower().replace("/", "_").replace(".", "").replace("-", "")
    tag_part = f"_{args.run_tag}" if args.run_tag else ""

    for method, combined in per_method_combined.items():
        single_pt = [p for p in submission_points if p.name == f"CML ({method})"]
        single_lafs = (
            lafs_summary_for_submission(args.tier, single_pt)
            if len(args.domains) == 2 and args.limit is None
            else None
        )
        summary_payload = _build_dashboard_compatible_summary(
            method_label=method,
            tier=args.tier,
            model_name=args.local_model,
            combined=combined,
            domain_metrics=per_method_domains[method],
            lafs_info=single_lafs,
        )
        if args.limit is None and len(args.domains) == 2:
            res_path = (
                args.results_dir
                / f"longmemeval_v2_{date_str}_{method}_{model_slug}_{args.tier}_full_451{tag_part}_summary.json"
            )
            res_path.write_text(json.dumps(summary_payload, indent=2) + "\n", encoding="utf-8")
            print(f"[results] Wrote summary artifact: {res_path}", flush=True)

        print(
            f"\n[{method} | {args.tier}] Overall={summary_payload['overall_avg'] * 100:.2f}% "
            f"(Non-Abst={summary_payload['overall_non_abstention'] * 100:.2f}%, "
            f"Abst={summary_payload['overall_abstention'] * 100:.2f}%) | "
            f"Avg Query Latency={((combined.get('memory_query') or {}).get('avg_seconds') or 0.0):.2f}s",
            flush=True,
        )
        if single_lafs is not None:
            print(
                f"[{method} | LAFS] Ref={single_lafs['reference_lafs']:.2f} -> "
                f"Submission={single_lafs['submission_lafs']:.2f} "
                f"(Gain={single_lafs['lafs_gain']:+.2f})",
                flush=True,
            )

    if len(submission_points) > 1 and args.limit is None and len(args.domains) == 2:
        multi_lafs = lafs_summary_for_submission(args.tier, submission_points)
        multi_path = (
            args.results_dir
            / f"longmemeval_v2_{date_str}_cml_multi_point_{model_slug}_{args.tier}_full_451{tag_part}_lafs.json"
        )
        multi_path.write_text(json.dumps(multi_lafs, indent=2) + "\n", encoding="utf-8")
        print(
            f"\n[Multi-Point LAFS ({len(submission_points)} operating points)] "
            f"Ref={multi_lafs['reference_lafs']:.2f} -> "
            f"Submission={multi_lafs['submission_lafs']:.2f} "
            f"(Gain={multi_lafs['lafs_gain']:+.2f})\nSaved: {multi_path}",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
