from __future__ import annotations

import hashlib
import json
import os
import signal
import shutil
import subprocess
import threading
import time
import traceback
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ASSET_ROOT = Path(__file__).resolve().parent / "assets" / "agentrunbook_online_learning"
CONSOLIDATION_INSTRUCTION_PATH = ASSET_ROOT / "CONSOLIDATE_STRATEGY.md"
STRATEGY_SKELETON_PATH = ASSET_ROOT / "LEARNED_RETRIEVAL_STRATEGY_SKELETON.md"
STRATEGY_FILENAME = "LEARNED_RETRIEVAL_STRATEGY.md"
DEFAULT_CODEX_BINARY = Path(os.getenv("CODEX_BINARY", "codex"))
DEFAULT_CODEX_TIMEOUT_SECONDS = float(os.getenv("CODEX_TIMEOUT_SECONDS", "1800"))
PROCESS_POLL_INTERVAL_SECONDS = 0.25
TERMINATION_GRACE_SECONDS = 5.0


def require(condition: bool, message: str) -> None:
    if not condition:
        raise RuntimeError(message)


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def relative_symlink(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    relative_target = os.path.relpath(src, start=dst.parent)
    dst.symlink_to(relative_target)


def _json_load(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _file_sha256(path: Path) -> str | None:
    if not path.exists():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as fp:
        for chunk in iter(lambda: fp.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _file_size(path: Path) -> int | None:
    if not path.exists():
        return None
    return path.stat().st_size


def _parse_codex_json_events(raw_stdout: str) -> tuple[list[dict[str, Any]], dict[str, int] | None]:
    events: list[dict[str, Any]] = []
    usage: dict[str, int] | None = None
    for line in raw_stdout.splitlines():
        stripped = line.strip()
        if not stripped.startswith("{"):
            continue
        try:
            payload = json.loads(stripped)
        except json.JSONDecodeError:
            continue
        if isinstance(payload, dict):
            events.append(payload)
            if payload.get("type") == "turn.completed" and isinstance(payload.get("usage"), dict):
                usage = payload["usage"]
    return events, usage


def _terminate_process_group(process: subprocess.Popen[str], *, reason: str) -> tuple[str, str]:
    if process.poll() is not None:
        return process.communicate()
    try:
        if os.name == "posix":
            os.killpg(process.pid, signal.SIGTERM)
        else:
            process.terminate()
        return process.communicate(timeout=TERMINATION_GRACE_SECONDS)
    except ProcessLookupError:
        return process.communicate()
    except subprocess.TimeoutExpired:
        try:
            if os.name == "posix":
                os.killpg(process.pid, signal.SIGKILL)
            else:
                process.kill()
        except ProcessLookupError:
            return process.communicate()
        return process.communicate()


def _event_item(event: dict[str, Any]) -> dict[str, Any] | None:
    item = event.get("item")
    return item if isinstance(item, dict) else None


def _codex_tool_event_count(events: list[dict[str, Any]]) -> int:
    count = 0
    for event in events:
        item = _event_item(event)
        if item is None:
            continue
        item_type = str(item.get("type", ""))
        if "tool" in item_type or "command" in item_type or item_type == "apply_patch":
            count += 1
    return count


def _failed_codex_apply_patch_events(events: list[dict[str, Any]]) -> list[dict[str, Any]]:
    failed: list[dict[str, Any]] = []
    for event in events:
        item = _event_item(event)
        if item is None:
            continue
        item_type = str(item.get("type", ""))
        name = str(item.get("name", ""))
        if "apply_patch" not in item_type and "apply_patch" not in name:
            continue
        status = item.get("status")
        exit_code = item.get("exit_code")
        if status == "failed" or (isinstance(exit_code, int) and exit_code != 0):
            failed.append(event)
    return failed


def _strategy_validation_errors(strategy_path: Path) -> list[str]:
    errors: list[str] = []
    if not strategy_path.exists():
        return ["strategy_file_missing"]
    text = strategy_path.read_text(encoding="utf-8")
    if not text.strip():
        errors.append("strategy_file_empty")
    required_sections = [
        "# Learned Retrieval Strategy",
        "## Past Queries",
        "## Strategies",
    ]
    for section in required_sections:
        if section not in text:
            errors.append(f"missing_section:{section}")
    required_table_headers = [
        "| Looking for | Evidence status | Evidence found | Applicability and guard | Fast path |",
        "| When | Evidence status | Try first | Guard |",
    ]
    for header in required_table_headers:
        if header not in text:
            errors.append(f"missing_table_header:{header}")
    return errors


@dataclass(frozen=True)
class AgentRunbookOnlineLearningCodexConfig:
    binary: Path
    model: str
    reasoning_effort: str
    timeout_seconds: float


@dataclass
class AgentRunbookOnlineLearningCodexRunResult:
    final_output: str = ""
    stdout_text: str = ""
    stderr_text: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    usage: dict[str, int] | None = None
    error_detail: str | None = None
    error_traceback: str = ""
    timed_out: bool = False
    interrupted: bool = False
    returncode: int | None = None
    command: list[str] = field(default_factory=list)


class AgentRunbookOnlineLearningCodexRunner:
    def __init__(self, config: AgentRunbookOnlineLearningCodexConfig) -> None:
        self.config = config
        require(config.model.strip(), "online-learning Codex model must be non-empty")
        require(
            config.reasoning_effort.strip(),
            "online-learning Codex reasoning_effort must be non-empty",
        )
        require(
            isinstance(config.timeout_seconds, (int, float))
            and not isinstance(config.timeout_seconds, bool)
            and float(config.timeout_seconds) > 0.0,
            "online-learning Codex timeout_seconds must be a positive number",
        )
        self.binary = self._resolve_binary(config.binary)

    def _resolve_binary(self, binary: Path) -> Path:
        binary_text = str(binary).strip()
        require(binary_text, "online-learning consolidation_codex_binary must be non-empty")
        resolved = shutil.which(binary_text) if os.sep not in binary_text else None
        path = Path(resolved).resolve() if resolved else Path(binary_text).expanduser().resolve()
        require(path.exists(), f"online-learning Codex binary does not exist: {path}")
        return path

    def _build_command(self, *, attempt_dir: Path, last_message_path: Path) -> list[str]:
        prompt = (
            "Read CONSOLIDATE_STRATEGY.md and update "
            "LEARNED_RETRIEVAL_STRATEGY.md. Do not modify "
            "sandbox/memory_module_output.json."
        )
        return [
            str(self.binary),
            "exec",
            "-C",
            str(attempt_dir),
            "--skip-git-repo-check",
            "--ephemeral",
            "--dangerously-bypass-approvals-and-sandbox",
            "--json",
            "-o",
            str(last_message_path),
            "-m",
            self.config.model,
            "-c",
            f"model_reasoning_effort={json.dumps(self.config.reasoning_effort)}",
            prompt,
        ]

    def run(
        self,
        *,
        attempt_dir: Path,
        last_message_path: Path,
        is_cancelled: Any | None = None,
    ) -> AgentRunbookOnlineLearningCodexRunResult:
        command = self._build_command(
            attempt_dir=attempt_dir,
            last_message_path=last_message_path,
        )
        started_at_ts = time.time()
        stdout_text = ""
        stderr_text = ""
        timed_out = False
        interrupted = False
        returncode: int | None = None
        process: subprocess.Popen[str] | None = None

        try:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                start_new_session=(os.name == "posix"),
            )
            while True:
                elapsed_seconds = time.time() - started_at_ts
                remaining_seconds = float(self.config.timeout_seconds) - elapsed_seconds
                if is_cancelled is not None and is_cancelled():
                    interrupted = True
                    stdout_text, stderr_text = _terminate_process_group(
                        process,
                        reason="cancel_event",
                    )
                    break
                if remaining_seconds <= 0:
                    timed_out = True
                    stdout_text, stderr_text = _terminate_process_group(
                        process,
                        reason="timeout",
                    )
                    break
                try:
                    stdout_text, stderr_text = process.communicate(
                        timeout=min(PROCESS_POLL_INTERVAL_SECONDS, remaining_seconds),
                    )
                    break
                except subprocess.TimeoutExpired:
                    continue
            returncode = process.returncode
        except Exception as exc:
            return AgentRunbookOnlineLearningCodexRunResult(
                stdout_text=stdout_text,
                stderr_text=stderr_text,
                error_detail=str(exc),
                error_traceback="".join(
                    traceback.format_exception(type(exc), exc, exc.__traceback__)
                ),
                returncode=returncode,
                command=command,
            )

        events, usage = _parse_codex_json_events(stdout_text)
        final_output = (
            last_message_path.read_text(encoding="utf-8")
            if last_message_path.exists()
            else ""
        )
        error_detail = None
        if interrupted:
            error_detail = "Codex consolidation cancelled before completion"
        elif timed_out:
            error_detail = (
                f"Codex consolidation timed out after {self.config.timeout_seconds}s"
            )
        elif returncode != 0:
            error_detail = f"Codex consolidation exited with code {returncode}"

        return AgentRunbookOnlineLearningCodexRunResult(
            final_output=final_output,
            stdout_text=stdout_text,
            stderr_text=stderr_text,
            events=events,
            usage=usage,
            error_detail=error_detail,
            timed_out=timed_out,
            interrupted=interrupted,
            returncode=returncode,
            command=command,
        )


@dataclass(frozen=True)
class AgentRunbookOnlineLearningConfig:
    enabled: bool
    strategy_memory_dir: Path | None = None
    consolidation_codex_binary: Path = DEFAULT_CODEX_BINARY
    timeout_seconds: float = DEFAULT_CODEX_TIMEOUT_SECONDS

    @classmethod
    def from_params(cls, params_obj: object) -> "AgentRunbookOnlineLearningConfig":
        if params_obj is None:
            return cls(enabled=False)
        require(
            isinstance(params_obj, dict),
            "agentrunbook_c_v2 online_learning_params must be an object",
        )
        params = dict(params_obj)
        enabled = params.get("enabled", False)
        require(isinstance(enabled, bool), "online_learning_params.enabled must be a boolean")
        if not enabled:
            return cls(enabled=False)

        strategy_memory_dir_obj = params.get("strategy_memory_dir")
        strategy_memory_dir = (
            Path(strategy_memory_dir_obj).expanduser().resolve()
            if isinstance(strategy_memory_dir_obj, str) and strategy_memory_dir_obj.strip()
            else None
        )
        codex_binary_obj = params.get("consolidation_codex_binary", str(DEFAULT_CODEX_BINARY))
        require(
            isinstance(codex_binary_obj, str) and codex_binary_obj.strip(),
            "online_learning_params.consolidation_codex_binary must be a non-empty string",
        )
        consolidation_codex_binary = Path(codex_binary_obj).expanduser()
        timeout_seconds_obj = params.get("timeout_seconds", DEFAULT_CODEX_TIMEOUT_SECONDS)
        require(
            isinstance(timeout_seconds_obj, (int, float))
            and not isinstance(timeout_seconds_obj, bool)
            and float(timeout_seconds_obj) > 0.0,
            "online_learning_params.timeout_seconds must be a positive number",
        )
        return cls(
            enabled=True,
            strategy_memory_dir=strategy_memory_dir,
            consolidation_codex_binary=consolidation_codex_binary,
            timeout_seconds=float(timeout_seconds_obj),
        )

    def to_params(self) -> dict[str, object]:
        out: dict[str, object] = {"enabled": self.enabled}
        if self.enabled:
            out["strategy_memory_dir"] = (
                str(self.strategy_memory_dir) if self.strategy_memory_dir is not None else None
            )
            out["consolidation_codex_binary"] = str(self.consolidation_codex_binary)
            out["timeout_seconds"] = self.timeout_seconds
        return out


class AgentRunbookOnlineLearning:
    def __init__(self, config: AgentRunbookOnlineLearningConfig) -> None:
        self.config = config
        self._lock = threading.Lock()
        self._attempt_metadata: dict[str, dict[str, Any]] = {}
        self._latest_successful_attempt_by_invocation: dict[str, Path] = {}
        self._loaded_strategy_memory_dir: Path | None = None
        if config.enabled:
            require(
                CONSOLIDATION_INSTRUCTION_PATH.exists(),
                f"missing online-learning consolidation instruction: {CONSOLIDATION_INSTRUCTION_PATH}",
            )
            require(
                STRATEGY_SKELETON_PATH.exists(),
                f"missing online-learning strategy skeleton: {STRATEGY_SKELETON_PATH}",
            )

    @property
    def enabled(self) -> bool:
        return self.config.enabled

    def bind_loaded_strategy_memory_dir(self, strategy_memory_dir: Path) -> None:
        if not self.enabled or self.config.strategy_memory_dir is not None:
            return None
        strategy_file = strategy_memory_dir.resolve() / STRATEGY_FILENAME
        if strategy_file.exists():
            self._loaded_strategy_memory_dir = strategy_file.parent
        return None

    def strategy_memory_dir(
        self,
        *,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> Path:
        if self.config.strategy_memory_dir is not None:
            return self.config.strategy_memory_dir
        if self._loaded_strategy_memory_dir is not None:
            return self._loaded_strategy_memory_dir
        if query_trace_dir is not None:
            return query_trace_dir.parent / "strategy_memory"
        require(
            workspace_dir is not None,
            "online learning requires workspace_dir or query_trace_dir for strategy memory",
        )
        return workspace_dir / "strategy_memory"

    def strategy_file(
        self,
        *,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> Path:
        return self.strategy_memory_dir(
            query_trace_dir=query_trace_dir,
            workspace_dir=workspace_dir,
        ) / STRATEGY_FILENAME

    def ensure_strategy_file(
        self,
        *,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> Path:
        strategy_file = self.strategy_file(
            query_trace_dir=query_trace_dir,
            workspace_dir=workspace_dir,
        )
        strategy_file.parent.mkdir(parents=True, exist_ok=True)
        if not strategy_file.exists():
            shutil.copy2(STRATEGY_SKELETON_PATH, strategy_file)
        return strategy_file

    def snapshot_strategy_file(self, strategy_file: Path, snapshot_path: Path) -> None:
        snapshot_path.parent.mkdir(parents=True, exist_ok=True)
        if strategy_file.exists():
            shutil.copy2(strategy_file, snapshot_path)
        else:
            snapshot_path.write_text("", encoding="utf-8")

    def expose_to_sandbox(
        self,
        *,
        sandbox_dir: Path,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> None:
        if not self.enabled:
            return None

        attempt_dir = sandbox_dir.parent
        sandbox_strategy_path = sandbox_dir / STRATEGY_FILENAME
        before_snapshot_path = attempt_dir / "strategy_before.md"
        metadata: dict[str, Any] = {
            "strategy_enabled": False,
            "strategy_file": None,
            "sandbox_strategy_path": str(sandbox_strategy_path),
            "strategy_exposure_mode": "copy",
            "before_snapshot_path": str(before_snapshot_path),
            "before_size_bytes": None,
            "before_sha256": None,
            "setup_error": None,
        }

        try:
            with self._lock:
                strategy_file = self.ensure_strategy_file(
                    query_trace_dir=query_trace_dir,
                    workspace_dir=workspace_dir,
                )
                self.snapshot_strategy_file(strategy_file, before_snapshot_path)
                shutil.copy2(strategy_file, sandbox_strategy_path)
                sandbox_strategy_path.chmod(0o444)
                metadata.update(
                    {
                        "strategy_enabled": True,
                        "strategy_file": str(strategy_file),
                        "before_size_bytes": _file_size(strategy_file),
                        "before_sha256": _file_sha256(strategy_file),
                    }
                )
        except Exception as exc:
            metadata["setup_error"] = str(exc)

        self._attempt_metadata[str(attempt_dir.resolve())] = metadata
        return None

    def latest_attempt_dir(self, *, query_trace_dir: Path | None, query_invocation_id: str) -> Path | None:
        if query_trace_dir is None:
            return None
        invocation_trace_dir = query_trace_dir / query_invocation_id
        if not invocation_trace_dir.exists():
            return None
        attempt_dirs = sorted(
            path
            for path in invocation_trace_dir.iterdir()
            if path.is_dir() and path.name.startswith("attempt_")
        )
        return attempt_dirs[-1] if attempt_dirs else None

    def finalize_attempt(
        self,
        *,
        query_trace_dir: Path | None,
        query_invocation_id: str,
        attempt_result: dict[str, Any],
    ) -> None:
        if not self.enabled:
            return None
        attempt_dir = self.latest_attempt_dir(
            query_trace_dir=query_trace_dir,
            query_invocation_id=query_invocation_id,
        )
        if attempt_dir is None:
            return None
        summary_path = attempt_dir / "summary.json"
        metadata = self._attempt_metadata.pop(str(attempt_dir.resolve()), {})
        if not metadata:
            return None

        sandbox_strategy_path = Path(str(metadata["sandbox_strategy_path"]))
        before_snapshot_path = Path(str(metadata["before_snapshot_path"]))
        after_snapshot_path = attempt_dir / "strategy_after.md"
        restored = False
        restore_reason: str | None = None

        with self._lock:
            try:
                self.snapshot_strategy_file(sandbox_strategy_path, after_snapshot_path)
            except Exception as exc:
                metadata["after_snapshot_error"] = str(exc)

            before_size = _file_size(before_snapshot_path) or 0
            after_size = _file_size(after_snapshot_path) or 0
            before_sha256 = _file_sha256(before_snapshot_path)
            after_sha256 = _file_sha256(after_snapshot_path)
            query_phase_changed_strategy = before_sha256 != after_sha256
            if not attempt_result.get("success"):
                restore_reason = "attempt_failed"
            elif before_size > 0 and after_size == 0:
                restore_reason = "empty_after_nonempty_before"
            elif query_phase_changed_strategy:
                restore_reason = "query_phase_strategy_edit"

            if restore_reason is not None and before_snapshot_path.exists():
                sandbox_strategy_path.chmod(0o644) if sandbox_strategy_path.exists() else None
                shutil.copy2(before_snapshot_path, sandbox_strategy_path)
                sandbox_strategy_path.chmod(0o444)
                restored = True

        strategy_file_value = metadata.get("strategy_file")
        strategy_file = Path(strategy_file_value) if isinstance(strategy_file_value, str) else None
        metadata.update(
            {
                "after_snapshot_path": str(after_snapshot_path),
                "after_size_bytes": _file_size(after_snapshot_path),
                "after_sha256": _file_sha256(after_snapshot_path),
                "shared_final_size_bytes": _file_size(strategy_file) if strategy_file else None,
                "shared_final_sha256": _file_sha256(strategy_file) if strategy_file else None,
                "changed": metadata.get("before_sha256") != _file_sha256(after_snapshot_path),
                "query_phase_changed_strategy": query_phase_changed_strategy,
                "restored_from_before": restored,
                "restore_reason": restore_reason,
            }
        )
        self.update_summary_with_attempt(summary_path, metadata)
        if attempt_result.get("success"):
            self._latest_successful_attempt_by_invocation[query_invocation_id] = attempt_dir
        return None

    def update_summary_with_attempt(self, summary_path: Path, metadata: dict[str, Any]) -> None:
        summary_payload = self._summary_payload(summary_path)
        summary_payload["strategy_memory"] = metadata
        save_json(summary_path, summary_payload)

    def update_summary_with_consolidation(self, summary_path: Path, metadata: dict[str, Any]) -> None:
        summary_payload = self._summary_payload(summary_path)
        summary_payload["strategy_consolidation"] = metadata
        save_json(summary_path, summary_payload)

    def _summary_payload(self, summary_path: Path) -> dict[str, Any]:
        if summary_path.exists():
            try:
                summary_payload = _json_load(summary_path)
            except Exception:
                summary_payload = {}
        else:
            summary_payload = {}
        if not isinstance(summary_payload, dict):
            summary_payload = {}
        return summary_payload

    def expose_consolidation_strategy_file(
        self,
        *,
        strategy_file: Path,
        attempt_dir: Path,
    ) -> tuple[Path, str, str | None]:
        link_path = attempt_dir / STRATEGY_FILENAME
        if link_path.exists() or link_path.is_symlink():
            link_path.unlink()
        try:
            relative_symlink(strategy_file, link_path)
            return link_path, "symlink", None
        except Exception as exc:
            shutil.copy2(strategy_file, link_path)
            return link_path, "copy", str(exc)

    def run_consolidation(
        self,
        *,
        query_invocation_id: str,
        attempt_dir: Path,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
        consolidation_runner: AgentRunbookOnlineLearningCodexRunner,
        is_cancelled: Any | None = None,
    ) -> dict[str, Any]:
        if not self.enabled:
            return {"status": "skipped", "reason": "strategy_memory_disabled"}
        if is_cancelled is not None and is_cancelled():
            raise KeyboardInterrupt("online-learning consolidation cancelled")

        summary_path = attempt_dir / "summary.json"
        consolidation_summary_path = attempt_dir / "strategy_consolidation_summary.json"
        instruction_path = attempt_dir / "CONSOLIDATE_STRATEGY.md"
        stdout_path = attempt_dir / "strategy_consolidation_stdout.log"
        stderr_path = attempt_dir / "strategy_consolidation_stderr.log"
        events_path = attempt_dir / "strategy_consolidation_events.json"
        last_message_path = attempt_dir / "strategy_consolidation_last_message.txt"
        before_snapshot_path = attempt_dir / "strategy_consolidation_before.md"
        after_snapshot_path = attempt_dir / "strategy_consolidation_after.md"
        exposed_strategy_path = attempt_dir / STRATEGY_FILENAME
        sandbox_output_path = attempt_dir / "sandbox" / "memory_module_output.json"
        sandbox_strategy_path = attempt_dir / "sandbox" / STRATEGY_FILENAME
        asset_strategy_path = ASSET_ROOT / STRATEGY_FILENAME

        metadata: dict[str, Any] = {
            "query_invocation_id": query_invocation_id,
            "attempt_dir": str(attempt_dir),
            "summary_path": str(summary_path),
            "status": "not_started",
            "started_at_utc": None,
            "completed_at_utc": None,
            "duration_seconds": None,
            "strategy_file": None,
            "strategy_link_path": str(exposed_strategy_path),
            "strategy_link_mode": None,
            "strategy_link_error": None,
            "before_snapshot_path": str(before_snapshot_path),
            "before_size_bytes": None,
            "before_sha256": None,
            "after_snapshot_path": str(after_snapshot_path),
            "after_size_bytes": None,
            "after_sha256": None,
            "final_size_bytes": None,
            "final_sha256": None,
            "changed": False,
            "copied_back": False,
            "restored_from_before": False,
            "restore_reason": None,
            "timed_out": False,
            "runner_error_detail": None,
            "tool_call_count": 0,
            "apply_patch_failure_count": 0,
            "tool_failure_count": 0,
            "validation_errors": [],
            "sandbox_output_before_sha256": _file_sha256(sandbox_output_path),
            "sandbox_output_after_sha256": None,
            "sandbox_strategy_before_sha256": _file_sha256(sandbox_strategy_path),
            "sandbox_strategy_after_sha256": None,
            "asset_strategy_before_sha256": _file_sha256(asset_strategy_path),
            "asset_strategy_after_sha256": None,
            "usage": None,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path),
            "last_message_path": str(last_message_path),
            "instruction_path": str(instruction_path),
            "consolidation_summary_path": str(consolidation_summary_path),
        }

        strategy_file: Path | None = None
        run_result = AgentRunbookOnlineLearningCodexRunResult()
        started_at_ts = time.time()
        try:
            strategy_file = self.ensure_strategy_file(
                query_trace_dir=query_trace_dir,
                workspace_dir=workspace_dir,
            )
            metadata["strategy_file"] = str(strategy_file)
            with self._lock:
                self.snapshot_strategy_file(strategy_file, before_snapshot_path)
                metadata["before_size_bytes"] = _file_size(strategy_file)
                metadata["before_sha256"] = _file_sha256(strategy_file)
                link_path, link_mode, link_error = self.expose_consolidation_strategy_file(
                    strategy_file=strategy_file,
                    attempt_dir=attempt_dir,
                )
                metadata["strategy_link_path"] = str(link_path)
                metadata["strategy_link_mode"] = link_mode
                metadata["strategy_link_error"] = link_error

            instruction_text = CONSOLIDATION_INSTRUCTION_PATH.read_text(encoding="utf-8")
            instruction_path.write_text(instruction_text, encoding="utf-8")
            metadata["instruction_sources"] = [str(instruction_path)]
            metadata["instruction_delivery"] = "attempt_file_and_task_prompt"
            metadata["started_at_utc"] = datetime.fromtimestamp(
                started_at_ts,
                timezone.utc,
            ).isoformat()
            run_result = consolidation_runner.run(
                attempt_dir=attempt_dir,
                last_message_path=last_message_path,
                is_cancelled=is_cancelled,
            )
            if not last_message_path.exists():
                last_message_path.write_text(run_result.final_output, encoding="utf-8")
            stdout_path.write_text(run_result.stdout_text, encoding="utf-8")
            save_json(events_path, run_result.events)
            stderr_text = run_result.stderr_text
            if run_result.error_traceback:
                stderr_text = f"{stderr_text}\n{run_result.error_traceback}".lstrip()
            stderr_path.write_text(stderr_text, encoding="utf-8")

            failed_apply_patch_calls = _failed_codex_apply_patch_events(run_result.events)
            with self._lock:
                validation_errors: list[str] = []
                run_failed = run_result.error_detail is not None
                restore_reason = None
                link_path_value = metadata.get("strategy_link_path")
                link_path = (
                    Path(link_path_value)
                    if isinstance(link_path_value, str) and link_path_value
                    else None
                )
                candidate_strategy_path = (
                    link_path
                    if metadata.get("strategy_link_mode") == "copy"
                    and link_path is not None
                    and link_path.exists()
                    else strategy_file
                )

                if run_failed:
                    restore_reason = (
                        "consolidation_timeout"
                        if run_result.timed_out
                        else "consolidation_failed"
                    )
                else:
                    before_size = _file_size(before_snapshot_path) or 0
                    after_size = _file_size(candidate_strategy_path) or 0
                    if before_size > 0 and after_size == 0:
                        validation_errors.append("empty_after_nonempty_before")
                    validation_errors.extend(_strategy_validation_errors(candidate_strategy_path))

                    metadata["sandbox_output_after_sha256"] = _file_sha256(sandbox_output_path)
                    metadata["sandbox_strategy_after_sha256"] = _file_sha256(sandbox_strategy_path)
                    metadata["asset_strategy_after_sha256"] = _file_sha256(asset_strategy_path)
                    if (
                        metadata["sandbox_output_before_sha256"]
                        != metadata["sandbox_output_after_sha256"]
                    ):
                        validation_errors.append("sandbox_memory_module_output_changed")
                    if (
                        metadata["sandbox_strategy_before_sha256"]
                        != metadata["sandbox_strategy_after_sha256"]
                    ):
                        validation_errors.append("sandbox_strategy_changed")
                    if (
                        metadata["asset_strategy_before_sha256"]
                        != metadata["asset_strategy_after_sha256"]
                    ):
                        validation_errors.append("asset_strategy_changed")
                    if validation_errors:
                        restore_reason = "validation_failed"

                metadata["validation_errors"] = validation_errors
                metadata["apply_patch_failure_count"] = len(failed_apply_patch_calls)
                metadata["tool_failure_count"] = len(failed_apply_patch_calls)
                if restore_reason is not None:
                    shutil.copy2(before_snapshot_path, strategy_file)
                    metadata["restored_from_before"] = True
                    metadata["restore_reason"] = restore_reason
                elif metadata.get("strategy_link_mode") == "copy":
                    shutil.copy2(candidate_strategy_path, strategy_file)
                    metadata["copied_back"] = True

                self.snapshot_strategy_file(strategy_file, after_snapshot_path)

            if metadata["restore_reason"] is not None:
                status = "failed"
            elif failed_apply_patch_calls:
                status = "finished_with_editor_retries"
            else:
                status = "finished"

            metadata.update(
                {
                    "status": status,
                    "completed_at_utc": utc_now_iso(),
                    "duration_seconds": time.time() - started_at_ts,
                    "timed_out": run_result.timed_out,
                    "interrupted": run_result.interrupted,
                    "runner_error_detail": run_result.error_detail,
                    "runner": "codex_cli",
                    "command": run_result.command,
                    "returncode": run_result.returncode,
                    "event_count": len(run_result.events),
                    "tool_call_count": _codex_tool_event_count(run_result.events),
                    "usage": run_result.usage,
                    "after_size_bytes": _file_size(after_snapshot_path),
                    "after_sha256": _file_sha256(after_snapshot_path),
                    "final_size_bytes": _file_size(strategy_file),
                    "final_sha256": _file_sha256(strategy_file),
                    "changed": metadata.get("before_sha256") != _file_sha256(after_snapshot_path),
                }
            )
        except Exception as exc:
            if strategy_file is not None and before_snapshot_path.exists():
                try:
                    shutil.copy2(before_snapshot_path, strategy_file)
                    metadata["restored_from_before"] = True
                    metadata["restore_reason"] = "internal_error"
                except Exception as restore_exc:
                    metadata["restore_error"] = str(restore_exc)
            metadata.update(
                {
                    "status": "internal_error",
                    "completed_at_utc": utc_now_iso(),
                    "duration_seconds": time.time() - started_at_ts,
                    "error": str(exc),
                    "timed_out": run_result.timed_out,
                    "interrupted": run_result.interrupted,
                    "runner_error_detail": run_result.error_detail,
                    "runner": "codex_cli",
                    "command": run_result.command,
                    "returncode": run_result.returncode,
                    "event_count": len(run_result.events),
                    "tool_call_count": _codex_tool_event_count(run_result.events),
                    "usage": run_result.usage,
                }
            )

        save_json(consolidation_summary_path, metadata)
        self.update_summary_with_consolidation(summary_path, metadata)
        return metadata

    def attempt_for_post_query(
        self,
        *,
        query_trace_dir: Path | None,
        query_invocation_id: str,
    ) -> Path | None:
        return self._latest_successful_attempt_by_invocation.get(query_invocation_id) or self.latest_attempt_dir(
            query_trace_dir=query_trace_dir,
            query_invocation_id=query_invocation_id,
        )

    def copy_strategy_to(
        self,
        *,
        output_dir: Path,
        query_trace_dir: Path | None,
        workspace_dir: Path | None,
    ) -> None:
        if not self.enabled:
            return None
        try:
            strategy_file = self.strategy_file(
                query_trace_dir=query_trace_dir,
                workspace_dir=workspace_dir,
            )
        except Exception:
            return None
        if strategy_file.exists():
            target_dir = output_dir / "strategy_memory"
            target_dir.mkdir(parents=True, exist_ok=True)
            target_file = target_dir / STRATEGY_FILENAME
            if strategy_file.resolve() != target_file.resolve():
                shutil.copy2(strategy_file, target_file)
        return None
