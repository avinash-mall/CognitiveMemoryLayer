"""Evaluation helpers and CLI for CML."""

from cml.eval.types import FullEvalConfig, LocomoEvalConfig


def compare_locomo_scores(*args, **kwargs):
    from cml.eval.compare import compare_locomo_scores as _fn

    return _fn(*args, **kwargs)


def run_locomo_plus(*args, **kwargs):
    from cml.eval.locomo import run_locomo_plus as _fn

    return _fn(*args, **kwargs)


def run_full_eval(*args, **kwargs):
    from cml.eval.pipeline import run_full_eval as _fn

    return _fn(*args, **kwargs)


def generate_locomo_report(*args, **kwargs):
    from cml.eval.report import generate_locomo_report as _fn

    return _fn(*args, **kwargs)


def validate_outputs(*args, **kwargs):
    from cml.eval.validate import validate_outputs as _fn

    return _fn(*args, **kwargs)


__all__ = [
    "FullEvalConfig",
    "LocomoEvalConfig",
    "compare_locomo_scores",
    "generate_locomo_report",
    "run_full_eval",
    "run_locomo_plus",
    "validate_outputs",
]
