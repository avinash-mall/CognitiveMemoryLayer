"""Repo-root detection for the eval CLI.

`find_repo_root` requires *every* marker to exist, so one stale entry disables
detection entirely and silently: `_default_out_dir()` and
`_default_unified_file()` both become None and the runner loses its checkpoint
and input paths with no error. That is exactly what happened between 51afd15
(which removed `packages/models/model_pipeline.toml`) and the fix these tests
guard — which is why the real repo is asserted against, not just a synthetic tree.
"""

from pathlib import Path

from cml.eval.config import find_repo_root


def _repo_root() -> Path:
    # tests/unit/ -> tests/ -> py-cml/ -> packages/ -> repo root
    return Path(__file__).resolve().parents[4]


def test_detects_the_real_repo_root_from_within_it():
    root = _repo_root()
    assert find_repo_root(root) == root


def test_detects_from_a_nested_directory():
    root = _repo_root()
    assert find_repo_root(root / "packages" / "py-cml") == root


def test_every_marker_actually_exists_in_this_repo():
    """Guards the failure mode directly: a marker that no longer ships."""
    root = _repo_root()
    assert (root / "docker" / "docker-compose.yml").exists()
    assert (root / "evaluation" / "locomo_plus").exists()


def test_returns_none_outside_any_repo(tmp_path):
    assert find_repo_root(tmp_path) is None


def test_a_partial_tree_is_not_a_repo_root(tmp_path):
    # one marker present, the other missing -> not a match
    (tmp_path / "docker").mkdir()
    (tmp_path / "docker" / "docker-compose.yml").write_text("services: {}\n")
    assert find_repo_root(tmp_path) is None
