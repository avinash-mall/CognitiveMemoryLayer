"""Typed configs for evaluation workflows."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(slots=True)
class LocomoEvalConfig:
    unified_file: Path
    out_dir: Path
    cml_url: str
    cml_api_key: str
    max_results: int = 25
    limit_samples: int | None = None
    skip_ingestion: bool = False
    skip_consolidation: bool = False
    score_only: bool = False
    judge_model: str = "gpt-4o-mini"
    verbose: bool = False
    ingestion_workers: int = 10


@dataclass(slots=True)
class FullEvalConfig:
    repo_root: Path
    skip_docker: bool = False
    limit_samples: int | None = None
    ingestion_workers: int = 5
    resume: bool = False
    score_only: bool = False
    health_timeout_sec: int = 180
    health_poll_interval_sec: int = 5
