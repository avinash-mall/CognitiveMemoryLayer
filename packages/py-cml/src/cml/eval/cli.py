"""CLI entrypoints for cml.eval."""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from cml.eval.config import find_repo_root


def _import_eval_module(name: str):
    """Lazy-import an eval submodule's main function, with a clear error on missing deps."""
    try:
        if name == "compare":
            from cml.eval.compare import main

            return main
        if name == "locomo":
            from cml.eval.locomo import main

            return main
        if name == "pipeline":
            from cml.eval.pipeline import main

            return main
        if name == "report":
            from cml.eval.report import main

            return main
        if name == "validate":
            from cml.eval.validate import main

            return main
    except ImportError as exc:
        print(
            f"Evaluation dependency missing: {exc}\n"
            'Install with: pip install "cognitive-memory-layer[eval]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    raise ValueError(f"Unknown eval module: {name}")


def _default_repo_root() -> Path:
    return find_repo_root(Path.cwd()) or Path.cwd()


def _default_unified_file() -> Path:
    root = _default_repo_root()
    return root / "evaluation" / "locomo_plus" / "data" / "unified_input_samples_v2.json"


def _default_out_dir() -> Path:
    return _default_repo_root() / "evaluation" / "outputs"


def _add_run_full_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run-full", help="Run full LoCoMo evaluation pipeline")
    parser.add_argument("--repo-root", type=Path, default=_default_repo_root())
    parser.add_argument("--skip-docker", action="store_true")
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--ingestion-workers", type=int, default=5)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--score-only", action="store_true")


def _add_run_locomo_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("run-locomo", help="Run Locomo-Plus CML-backed evaluation")
    parser.add_argument("--unified-file", type=Path, default=_default_unified_file())
    parser.add_argument("--out-dir", type=Path, default=_default_out_dir())
    parser.add_argument("--cml-url", type=str, default=os.environ.get("CML_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--cml-api-key", type=str, default=os.environ.get("CML_API_KEY", "test-key"))
    parser.add_argument("--max-results", type=int, default=25)
    parser.add_argument("--limit-samples", type=int, default=None)
    parser.add_argument("--skip-ingestion", action="store_true")
    parser.add_argument("--skip-consolidation", action="store_true")
    parser.add_argument("--score-only", action="store_true")
    parser.add_argument(
        "--judge-model",
        type=str,
        default=os.environ.get("LLM_EVAL__MODEL") or os.environ.get("LLM_INTERNAL__MODEL", "gpt-4o-mini"),
    )
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument("--ingestion-workers", type=int, default=10)


def _add_validate_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("validate", help="Validate eval outputs")
    parser.add_argument("--outputs-dir", type=Path, default=_default_out_dir())


def _add_report_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("report", help="Generate LoCoMo / Locomo-Plus report")
    parser.add_argument("--summary", type=Path, default=_default_out_dir() / "locomo_plus_qa_cml_judge_summary.json")
    parser.add_argument("--method", type=str, default="CML")
    parser.add_argument("--no-title", action="store_true")


def _add_compare_parser(subparsers: argparse._SubParsersAction) -> None:
    parser = subparsers.add_parser("compare", help="Compare scores with paper baselines")
    parser.add_argument("--summary", type=Path, default=_default_out_dir() / "locomo_plus_qa_cml_judge_summary.json")
    parser.add_argument("--method", type=str, default="CML+gpt-oss:20b")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="cml-eval", description="Evaluation tools for CML")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_run_full_parser(subparsers)
    _add_run_locomo_parser(subparsers)
    _add_validate_parser(subparsers)
    _add_report_parser(subparsers)
    _add_compare_parser(subparsers)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "run-full":
        pipeline_main = _import_eval_module("pipeline")
        return pipeline_main(
            [
                "--repo-root",
                str(args.repo_root),
                *(["--skip-docker"] if args.skip_docker else []),
                *(["--limit-samples", str(args.limit_samples)] if args.limit_samples is not None else []),
                "--ingestion-workers",
                str(args.ingestion_workers),
                *(["--resume"] if args.resume else []),
                *(["--score-only"] if args.score_only else []),
            ]
        )

    if args.command == "run-locomo":
        locomo_main = _import_eval_module("locomo")
        return locomo_main(
            [
                "--unified-file",
                str(args.unified_file),
                "--out-dir",
                str(args.out_dir),
                "--cml-url",
                str(args.cml_url),
                "--cml-api-key",
                str(args.cml_api_key),
                "--max-results",
                str(args.max_results),
                *(["--limit-samples", str(args.limit_samples)] if args.limit_samples is not None else []),
                *(["--skip-ingestion"] if args.skip_ingestion else []),
                *(["--skip-consolidation"] if args.skip_consolidation else []),
                *(["--score-only"] if args.score_only else []),
                "--judge-model",
                str(args.judge_model),
                *(["--verbose"] if args.verbose else []),
                "--ingestion-workers",
                str(args.ingestion_workers),
            ]
        )

    if args.command == "validate":
        validate_main = _import_eval_module("validate")
        return validate_main(["--outputs-dir", str(args.outputs_dir)])

    if args.command == "report":
        report_main = _import_eval_module("report")
        report_argv = ["--summary", str(args.summary), "--method", str(args.method)]
        if args.no_title:
            report_argv.append("--no-title")
        return report_main(report_argv)

    if args.command == "compare":
        compare_main = _import_eval_module("compare")
        return compare_main(["--summary", str(args.summary), "--method", str(args.method)])

    parser.print_help()
    return 1


def main_legacy_eval_locomo(argv: list[str] | None = None) -> int:
    return _import_eval_module("locomo")(argv)


def main_legacy_run_full(argv: list[str] | None = None) -> int:
    return _import_eval_module("pipeline")(argv)


def main_legacy_validate(argv: list[str] | None = None) -> int:
    return _import_eval_module("validate")(argv)


def main_legacy_report(argv: list[str] | None = None) -> int:
    return _import_eval_module("report")(argv)


def main_legacy_compare(argv: list[str] | None = None) -> int:
    return _import_eval_module("compare")(argv)


if __name__ == "__main__":
    raise SystemExit(main())
