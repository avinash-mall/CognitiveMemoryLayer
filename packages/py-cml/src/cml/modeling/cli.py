"""CLI entrypoints for cml.modeling."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from cml.modeling.config import default_model_config_path
from cml.modeling.types import PrepareConfig, TrainConfig


def _import_prepare():
    try:
        from cml.modeling.prepare import main as prepare_main
        from cml.modeling.prepare import prepare_data
    except ImportError as exc:
        print(
            f"Modeling dependency missing: {exc}\n"
            'Install with: pip install "cognitive-memory-layer[modeling]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return prepare_main, prepare_data


def _import_train():
    try:
        from cml.modeling.train import main as train_main
        from cml.modeling.train import train_models
    except ImportError as exc:
        print(
            f"Modeling dependency missing: {exc}\n"
            'Install with: pip install "cognitive-memory-layer[modeling]"',
            file=sys.stderr,
        )
        raise SystemExit(1) from exc
    return train_main, train_models


def _default_config_path() -> Path:
    try:
        return default_model_config_path(Path.cwd())
    except FileNotFoundError:
        return Path("packages/models/model_pipeline.toml")


def _add_prepare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-rows-per-source", type=int, default=None)
    parser.add_argument("--max-per-task-label", type=int, default=None)
    parser.add_argument("--target-per-task-label", type=int, default=None)
    parser.add_argument("--llm-temperature", type=float, default=None)
    parser.add_argument("--llm-concurrency", type=int, default=None)
    parser.add_argument("--disable-download", action="store_true")
    parser.add_argument("--allow-missing-datasets-package", action="store_true")
    parser.add_argument("--force-full", action="store_true")
    parser.add_argument("--no-multilingual", action="store_true")


def _add_train_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--config", type=Path, default=_default_config_path())
    parser.add_argument("--families", type=str, default="")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--max-iter", type=int, default=None)
    parser.add_argument("--max-features", type=int, default=None)
    parser.add_argument("--predict-batch-size", type=int, default=None)
    parser.add_argument("--early-stopping", type=str, default=None)
    parser.add_argument("--early-stopping-patience", type=int, default=None)
    parser.add_argument("--early-stopping-metric", type=str, default=None)
    parser.add_argument("--early-stopping-min-delta", type=float, default=None)
    parser.add_argument("--calibration-method", type=str, default=None)
    parser.add_argument("--prepared-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--tasks", type=str, default="")
    parser.add_argument("--objective-types", type=str, default="")
    parser.add_argument("--max-seq-length", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--token-model-name-or-path", type=str, default=None)
    parser.add_argument("--token-num-train-epochs", type=int, default=None)
    parser.add_argument("--token-per-device-train-batch-size", type=int, default=None)
    parser.add_argument("--token-per-device-eval-batch-size", type=int, default=None)
    parser.add_argument("--token-stride", type=int, default=None)
    parser.add_argument("--token-warmup-ratio", type=float, default=None)
    parser.add_argument("--token-weight-decay", type=float, default=None)
    parser.add_argument("--token-gradient-accumulation-steps", type=int, default=None)
    parser.add_argument("--calibration-split", type=str, default=None)
    parser.add_argument("--export-thresholds", action="store_true")
    strict_group = parser.add_mutually_exclusive_group()
    strict_group.add_argument(
        "--strict",
        dest="strict",
        action="store_true",
        help="Fail when configured tasks/objectives are unsupported or missing training rows.",
    )
    strict_group.add_argument(
        "--allow-skips",
        dest="strict",
        action="store_false",
        help="Allow unsupported or missing tasks to be skipped during training.",
    )
    parser.set_defaults(strict=True)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="cml-models", description="Model prep/train tools for CML"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    prepare_parser = subparsers.add_parser("prepare", help="Prepare modeling datasets")
    _add_prepare_args(prepare_parser)

    train_parser = subparsers.add_parser("train", help="Train custom models")
    _add_train_args(train_parser)

    pipeline_parser = subparsers.add_parser("pipeline", help="Run prepare+train pipeline")
    pipeline_parser.add_argument("--config", type=Path, default=_default_config_path())
    pipeline_parser.add_argument("--skip-prepare", action="store_true")
    pipeline_parser.add_argument("--skip-train", action="store_true")
    pipeline_parser.add_argument(
        "pipeline_args",
        nargs=argparse.REMAINDER,
        help="Optional pass-through args applied to both prepare and train commands",
    )

    return parser


def _prepare_config_from_args(args: argparse.Namespace) -> PrepareConfig:
    return PrepareConfig(
        config_path=Path(args.config),
        seed=args.seed,
        max_rows_per_source=args.max_rows_per_source,
        max_per_task_label=args.max_per_task_label,
        target_per_task_label=args.target_per_task_label,
        llm_temperature=args.llm_temperature,
        llm_concurrency=args.llm_concurrency,
        disable_download=bool(args.disable_download),
        allow_missing_datasets_package=bool(args.allow_missing_datasets_package),
        force_full=bool(args.force_full),
        no_multilingual=bool(args.no_multilingual),
    )


def _train_config_from_args(args: argparse.Namespace) -> TrainConfig:
    return TrainConfig(
        config_path=Path(args.config),
        families=str(args.families),
        seed=args.seed,
        max_iter=args.max_iter,
        max_features=args.max_features,
        predict_batch_size=args.predict_batch_size,
        early_stopping=(
            None
            if args.early_stopping is None
            else str(args.early_stopping).strip().lower() in {"1", "true", "yes", "on"}
        ),
        early_stopping_patience=args.early_stopping_patience,
        early_stopping_metric=args.early_stopping_metric,
        early_stopping_min_delta=args.early_stopping_min_delta,
        calibration_method=args.calibration_method,
        prepared_dir=args.prepared_dir,
        output_dir=args.output_dir,
        tasks=str(args.tasks),
        objective_types=str(args.objective_types),
        max_seq_length=args.max_seq_length,
        learning_rate=args.learning_rate,
        token_model_name_or_path=args.token_model_name_or_path,
        token_num_train_epochs=args.token_num_train_epochs,
        token_per_device_train_batch_size=args.token_per_device_train_batch_size,
        token_per_device_eval_batch_size=args.token_per_device_eval_batch_size,
        token_stride=args.token_stride,
        token_warmup_ratio=args.token_warmup_ratio,
        token_weight_decay=args.token_weight_decay,
        token_gradient_accumulation_steps=args.token_gradient_accumulation_steps,
        calibration_split=args.calibration_split,
        export_thresholds=bool(args.export_thresholds),
        strict=bool(args.strict),
    )


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.command == "prepare":
        _, prepare_data = _import_prepare()
        return prepare_data(_prepare_config_from_args(args))

    if args.command == "train":
        _, train_models = _import_train()
        return train_models(_train_config_from_args(args))

    if args.command == "pipeline":
        passthrough = list(args.pipeline_args or [])
        if passthrough and passthrough[0] == "--":
            passthrough = passthrough[1:]

        prep_parser = argparse.ArgumentParser(add_help=False)
        _add_prepare_args(prep_parser)
        train_parser = argparse.ArgumentParser(add_help=False)
        _add_train_args(train_parser)

        prep_ns, _ = prep_parser.parse_known_args(["--config", str(args.config), *passthrough])
        train_ns, _ = train_parser.parse_known_args(["--config", str(args.config), *passthrough])

        from cml.modeling.pipeline import run_pipeline

        return run_pipeline(
            _prepare_config_from_args(prep_ns) if not args.skip_prepare else None,
            _train_config_from_args(train_ns) if not args.skip_train else None,
        )

    parser.print_help()
    return 1


def main_legacy_prepare(argv: list[str] | None = None) -> int:
    prepare_main, _ = _import_prepare()
    return prepare_main(argv)


def main_legacy_train(argv: list[str] | None = None) -> int:
    train_main, _ = _import_train()
    return train_main(argv)


if __name__ == "__main__":
    raise SystemExit(main())
