"""Shared helpers for evaluation modules."""

from __future__ import annotations

import importlib.util
import json
import sys
import urllib.request
from contextlib import contextmanager
from pathlib import Path

LOCOMO10_URL = "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json"
UNIFIED_INPUT_FILENAME = "unified_input_samples_v2.json"
LOCOMO_PLUS_DIRNAME = "locomo_plus"


def find_repo_root(start: Path | None = None) -> Path | None:
    """Detect repo root by scanning parents for known project markers."""
    current = (start or Path.cwd()).resolve()
    for candidate in [current, *current.parents]:
        markers = (
            candidate / "docker" / "docker-compose.yml",
            candidate / "evaluation" / "locomo_plus",
            candidate / "packages" / "models" / "model_pipeline.toml",
        )
        if all(marker.exists() for marker in markers):
            return candidate
    return None


def load_repo_dotenv(repo_root: Path) -> None:
    """Load repo .env if python-dotenv is available."""
    try:
        from dotenv import load_dotenv
    except ImportError:
        return

    env_file = repo_root / ".env"
    if env_file.exists():
        load_dotenv(env_file)


def eval_data_dir(repo_root: Path) -> Path:
    return repo_root / "evaluation" / LOCOMO_PLUS_DIRNAME / "data"


def _download_file(url: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with urllib.request.urlopen(url, timeout=60) as response:
        destination.write_bytes(response.read())


@contextmanager
def _prepend_sys_path(path: Path):
    path_str = str(path)
    sys.path.insert(0, path_str)
    try:
        yield
    finally:
        try:
            sys.path.remove(path_str)
        except ValueError:
            pass


def _load_build_unified_samples(data_dir: Path):
    unified_input_path = data_dir / "unified_input.py"
    spec = importlib.util.spec_from_file_location("cml_eval_unified_input", unified_input_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load unified input builder from {unified_input_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    build_fn = getattr(module, "build_unified_samples", None)
    if not callable(build_fn):
        raise AttributeError(f"{unified_input_path} does not expose build_unified_samples()")
    return build_fn


def ensure_unified_eval_data(unified_file: Path, repo_root: Path | None = None) -> Path:
    """Ensure the unified Locomo input exists, downloading/building it when possible."""
    target = Path(unified_file)
    if target.exists():
        return target

    root = repo_root or find_repo_root(target.parent) or find_repo_root(Path.cwd())
    if root is None:
        raise FileNotFoundError(
            f"Unified input file not found: {target}. "
            "Run from a CML checkout or pass an existing --unified-file."
        )

    data_dir = eval_data_dir(root)
    expected_default = data_dir / UNIFIED_INPUT_FILENAME
    if target.resolve(strict=False) != expected_default.resolve(strict=False):
        raise FileNotFoundError(
            f"Unified input file not found: {target}. "
            f"Automatic download/build currently supports only {expected_default}."
        )

    locomo_plus_path = data_dir / "locomo_plus.json"
    if not locomo_plus_path.exists():
        raise FileNotFoundError(
            f"Required Locomo-Plus data is missing: {locomo_plus_path}. "
            "Automatic rebuild needs the repository evaluation assets."
        )

    locomo10_path = data_dir / "locomo10.json"
    if not locomo10_path.exists():
        print(f"Missing {locomo10_path.name}; downloading from upstream LoCoMo repo...", flush=True)
        _download_file(LOCOMO10_URL, locomo10_path)
        print(f"Downloaded {locomo10_path}", flush=True)

    print(f"Building {target.name} from repository evaluation data...", flush=True)
    with _prepend_sys_path(data_dir):
        build_unified_samples = _load_build_unified_samples(data_dir)
        samples = build_unified_samples(data_dir)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(samples, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Wrote {len(samples)} samples to {target}", flush=True)
    return target
