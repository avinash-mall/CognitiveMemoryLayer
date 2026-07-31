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
    # Tenant namespace for this run. Tenants are named {tenant_prefix}-{canonical_idx},
    # and the index is relative to whichever --unified-file was passed. Two runs over
    # different files therefore reuse the same tenant IDs for entirely different
    # conversations, and the second run retrieves against the first one's memories.
    # Give every run its own prefix unless you intend to reuse a corpus.
    tenant_prefix: str = "lp"
    skip_ingestion: bool = False
    skip_consolidation: bool = False
    score_only: bool = False
    judge_model: str = "gpt-4o-mini"
    verbose: bool = False
    ingestion_workers: int = 10
    qa_backend: str = "openai_compatible"
    judge_backend: str = "call_llm"


@dataclass(slots=True)
class FullEvalConfig:
    repo_root: Path
    skip_docker: bool = False
    limit_samples: int | None = None
    ingestion_workers: int = 10
    resume: bool = False
    score_only: bool = False
    health_timeout_sec: int = 180
    health_poll_interval_sec: int = 5
