from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .agentrunbook_c import AgentRunbookC
from .codex import (
    read_memory_output_status,
    relative_symlink,
    save_json,
    utc_now_iso,
)
from .memory import MemoryConfig, register_memory, require
from .agentrunbook_online_learning import (
    AgentRunbookOnlineLearning,
    AgentRunbookOnlineLearningCodexConfig,
    AgentRunbookOnlineLearningCodexRunner,
    AgentRunbookOnlineLearningConfig,
)
from .oai_agents_sdk import (
    OaiAgentsSDKRunner,
    OaiAgentsSDKRunnerConfig,
    ensure_non_negative_int,
    ensure_positive_float,
    ensure_positive_int,
    ensure_string,
)


QUERY_INSTRUCTION_APPENDIX = """

## Learned Retrieval Strategy

If `LEARNED_RETRIEVAL_STRATEGY.md` exists, use it as private retrieval
guidance. It may contain notes learned from previous queries across task types.
You may read it before broad trajectory exploration and revisit it after
inspecting the current working directory to find the most relevant note. Do not
edit the strategy file.

Each learned row's `Evidence status` describes how the previous task's cited
evidence fit the previous task. It is provenance for that old note, not the
evidence status for the current question, and not proof that the note should be
used now.

Use relevant learned rows as candidate search leads:

- `directly_supported` means the previous task had exact positive support. For
  the current question, use it only after current cited evidence is non-empty,
  directly proves the requested target, matches the exact entity / actor-view /
  page / section / field-control / workflow stage / pre-post state, and resolves
  any UI-count, label-mapping, section-boundary, or before-after ambiguity.
- `contradicts_premise` means the previous task had exact contradictory or
  closed-set absence evidence. For the current question, use it only after the
  current scoped span directly shows the named field, control, workflow, page, or
  premise is absent or wrong. If a current closed set of options, fields, tabs,
  buttons, or related records is visible and the requested target is absent,
  surface that absence clearly.
- `near_match_only` means the previous task found only a similar workflow, page,
  actor/view, entity, field, section, time, or state. Use it only as navigation
  or contrast; it is not answer evidence for the current question.
- `insufficient` means the previous task lacked exact support or contradiction.
  Use it only as a warning about missing evidence or a repeatable search trap;
  still continue current-task exploration.

Do not treat learned notes as absolute truth. Still inspect the current working
directory and current trajectory evidence to decide whether any note is helpful
for this question.

This appendix does not change the output JSON schema. `memory_markdown` and
`trajectory_spans` are the final filtered memory context:

- Include only current verified evidence and concise reasoning that should be
  used for the current question. If a learned note guided the search, mention it
  only when useful and only after current cited evidence verifies the point.
- Omit rejected notes.
- Usually omit `near_match_only` notes and spans; include them only when useful
  as clearly labeled reference or contrast.
- For `contradicts_premise`, clearly state the false premise and cite the exact
  scoped absence or contradiction. If a closed set proves the requested target is
  absent, describe that absence directly instead of treating the evidence as
  missing.
- Never pass along answer-like text from a learned note unless current cited
  evidence independently verifies it.
"""


@register_memory
class AgentRunbookCV2(AgentRunbookC):
    memory_type = "agentrunbook_c_v2"

    def __init__(self, memory_params: dict[str, object]) -> None:
        online_learning_config = AgentRunbookOnlineLearningConfig.from_params(
            memory_params.get("online_learning_params")
        )
        sdk_params_obj = memory_params.get(
            "query_openai_sdk_params",
            memory_params.get("query_codex_params", memory_params.get("codex_params", {})),
        )
        require(
            isinstance(sdk_params_obj, dict),
            "agentrunbook_c_v2 query_openai_sdk_params must be an object",
        )
        sdk_params = dict(sdk_params_obj)

        model = ensure_string(sdk_params.get("model"), field_name="query_openai_sdk_params.model")
        reasoning_effort = ensure_string(
            sdk_params.get("reasoning_effort"),
            field_name="query_openai_sdk_params.reasoning_effort",
        )
        timeout_seconds = ensure_positive_float(
            sdk_params.get("timeout_seconds"),
            field_name="query_openai_sdk_params.timeout_seconds",
        )
        max_attempts = ensure_positive_int(
            sdk_params.get("max_attempts", sdk_params.get("max_retries")),
            field_name="query_openai_sdk_params.max_attempts/max_retries",
        )

        base_params = dict(memory_params)
        base_params["query_codex_params"] = {
            # Satisfies the inherited CodexMemory initializer; v2 executes the SDK runner instead.
            "binary": sys.executable,
            "model": model,
            "reasoning_effort": reasoning_effort,
            "timeout_seconds": timeout_seconds,
            "max_retries": max_attempts,
            "prompt": sdk_params.get("prompt", memory_params.get("prompt", "")) or None,
            "require_evidence_gate": False,
            "extra_config": [],
            "extra_args": [],
        }
        if base_params["query_codex_params"]["prompt"] is None:
            base_params["query_codex_params"].pop("prompt")

        super().__init__(base_params)
        self.online_learning = AgentRunbookOnlineLearning(online_learning_config)
        if self.online_learning.enabled:
            strategy_prompt_suffix = (
                "Also read LEARNED_RETRIEVAL_STRATEGY.md for reusable retrieval leads; verify exact current evidence before treating any note as support."
            )
            if strategy_prompt_suffix not in self.codex_prompt:
                self.codex_prompt = f"{self.codex_prompt.rstrip()} {strategy_prompt_suffix}"
            step_one_marker = "\n## Step 1: do a quick triage of the query so that you have an expectation of what to do."
            if step_one_marker in self.query_instruction_text:
                self.query_instruction_text = self.query_instruction_text.replace(
                    step_one_marker,
                    QUERY_INSTRUCTION_APPENDIX + step_one_marker,
                    1,
                )
            else:
                self.query_instruction_text = (
                    self.query_instruction_text.rstrip() + QUERY_INSTRUCTION_APPENDIX
                )

        self.sdk_model = model
        self.sdk_reasoning_effort = reasoning_effort
        self.sdk_timeout_seconds = timeout_seconds
        self.sdk_max_attempts = max_attempts
        self.codex_max_attempts = max_attempts
        self.sdk_max_turns = ensure_positive_int(
            sdk_params.get("max_turns"),
            field_name="query_openai_sdk_params.max_turns",
        )
        self.sdk_api_key_env = ensure_string(
            sdk_params.get("api_key_env"),
            field_name="query_openai_sdk_params.api_key_env",
        )
        self.tool_timeout_seconds = ensure_positive_float(
            sdk_params.get("tool_timeout_seconds"),
            field_name="query_openai_sdk_params.tool_timeout_seconds",
        )
        self.max_tool_output_chars = ensure_positive_int(
            sdk_params.get("max_tool_output_chars"),
            field_name="query_openai_sdk_params.max_tool_output_chars",
        )
        self.responses_transport = ensure_string(
            sdk_params.get("responses_transport", "websocket"),
            field_name="query_openai_sdk_params.responses_transport",
        )
        require(
            self.responses_transport == "websocket",
            "agentrunbook_c_v2 only supports responses_transport='websocket'",
        )
        self.api_connect_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_connect_timeout_seconds", 15.0),
            field_name="query_openai_sdk_params.api_connect_timeout_seconds",
        )
        self.api_read_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_read_timeout_seconds", 300.0),
            field_name="query_openai_sdk_params.api_read_timeout_seconds",
        )
        self.api_write_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_write_timeout_seconds", 300.0),
            field_name="query_openai_sdk_params.api_write_timeout_seconds",
        )
        self.api_pool_timeout_seconds = ensure_positive_float(
            sdk_params.get("api_pool_timeout_seconds", 300.0),
            field_name="query_openai_sdk_params.api_pool_timeout_seconds",
        )
        self.api_max_retries = ensure_non_negative_int(
            sdk_params.get("api_max_retries", 0),
            field_name="query_openai_sdk_params.api_max_retries",
        )
        self.consolidation_codex_runner = (
            AgentRunbookOnlineLearningCodexRunner(
                AgentRunbookOnlineLearningCodexConfig(
                    binary=self.online_learning.config.consolidation_codex_binary,
                    model=self.sdk_model,
                    reasoning_effort=self.sdk_reasoning_effort,
                    timeout_seconds=self.online_learning.config.timeout_seconds,
                )
            )
            if self.online_learning.enabled
            else None
        )
        self.sdk_runner = OaiAgentsSDKRunner(
            OaiAgentsSDKRunnerConfig(
                model=self.sdk_model,
                reasoning_effort=self.sdk_reasoning_effort,
                timeout_seconds=self.sdk_timeout_seconds,
                max_turns=self.sdk_max_turns,
                api_key_env=self.sdk_api_key_env,
                tool_timeout_seconds=self.tool_timeout_seconds,
                max_tool_output_chars=self.max_tool_output_chars,
                agent_name="AgentRunbookCOpenAISDK",
                responses_transport=self.responses_transport,
                api_connect_timeout_seconds=self.api_connect_timeout_seconds,
                api_read_timeout_seconds=self.api_read_timeout_seconds,
                api_write_timeout_seconds=self.api_write_timeout_seconds,
                api_pool_timeout_seconds=self.api_pool_timeout_seconds,
                api_max_retries=self.api_max_retries,
            )
        )

    @property
    def memory_config(self) -> MemoryConfig:
        memory_params: dict[str, object] = {
            "evidence_mode": self.evidence_mode,
            "query_openai_sdk_params": {
                "model": self.sdk_model,
                "reasoning_effort": self.sdk_reasoning_effort,
                "timeout_seconds": self.sdk_timeout_seconds,
                "max_retries": self.sdk_max_attempts,
                "max_turns": self.sdk_max_turns,
                "api_key_env": self.sdk_api_key_env,
                "tool_timeout_seconds": self.tool_timeout_seconds,
                "max_tool_output_chars": self.max_tool_output_chars,
                "responses_transport": self.responses_transport,
                "api_connect_timeout_seconds": self.api_connect_timeout_seconds,
                "api_read_timeout_seconds": self.api_read_timeout_seconds,
                "api_write_timeout_seconds": self.api_write_timeout_seconds,
                "api_pool_timeout_seconds": self.api_pool_timeout_seconds,
                "api_max_retries": self.api_max_retries,
                "prompt": self.codex_prompt,
            },
        }
        if self.trajectory_pool_root is not None:
            memory_params["trajectory_pool_root"] = str(self.trajectory_pool_root)
        if self.online_learning.enabled:
            memory_params["online_learning_params"] = self.online_learning.config.to_params()
        return {
            "memory_type": self.memory_type,
            "memory_params": memory_params,
        }

    def _runner_summary_fields(self) -> dict[str, Any]:
        return {
            "runner": "openai_agents_sdk",
            "model": self.sdk_model,
            "reasoning_effort": self.sdk_reasoning_effort,
            "max_turns": self.sdk_max_turns,
            "tool_timeout_seconds": self.tool_timeout_seconds,
            "max_tool_output_chars": self.max_tool_output_chars,
            "responses_transport": self.responses_transport,
            "api_connect_timeout_seconds": self.api_connect_timeout_seconds,
            "api_read_timeout_seconds": self.api_read_timeout_seconds,
            "api_write_timeout_seconds": self.api_write_timeout_seconds,
            "api_pool_timeout_seconds": self.api_pool_timeout_seconds,
            "api_max_retries": self.api_max_retries,
            "online_learning_enabled": self.online_learning.enabled,
        }

    def _populate_sandbox_scripts(self, sandbox_dir: Path) -> list[str]:
        scripts = super()._populate_sandbox_scripts(sandbox_dir)
        self.online_learning.expose_to_sandbox(
            sandbox_dir=sandbox_dir,
            query_trace_dir=self.query_trace_dir,
            workspace_dir=self.workspace_dir,
        )
        return scripts

    def _estimate_sdk_token_throughput(
        self,
        *,
        duration_seconds: float,
        tool_calls: list[dict[str, Any]],
        usage: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        if not usage:
            return None

        output_tokens = int(usage.get("output_tokens", 0) or 0)
        total_tokens = int(usage.get("total_tokens", 0) or 0)
        reasoning_output_tokens = int(usage.get("reasoning_output_tokens", 0) or 0)
        tool_duration_seconds = sum(
            float(call.get("duration_seconds", 0.0) or 0.0) for call in tool_calls
        )
        estimated_model_wait_seconds = max(duration_seconds - tool_duration_seconds, 0.0)

        def rate(tokens: int, seconds: float) -> float | None:
            if tokens <= 0 or seconds <= 0.0:
                return None
            return tokens / seconds

        return {
            "model": self.sdk_model,
            "basis": "Agents SDK aggregate usage tokens divided by attempt elapsed seconds",
            "note": "Estimate only: the SDK exposes token usage but not per-response latency here.",
            "output_tokens": output_tokens,
            "reasoning_output_tokens": reasoning_output_tokens,
            "total_tokens": total_tokens,
            "elapsed_seconds": duration_seconds,
            "tool_duration_seconds": tool_duration_seconds,
            "estimated_model_wait_seconds": estimated_model_wait_seconds,
            "output_tokens_per_elapsed_second": rate(output_tokens, duration_seconds),
            "total_tokens_per_elapsed_second": rate(total_tokens, duration_seconds),
            "reasoning_output_tokens_per_elapsed_second": rate(
                reasoning_output_tokens,
                duration_seconds,
            ),
            "output_tokens_per_estimated_model_wait_second": rate(
                output_tokens,
                estimated_model_wait_seconds,
            ),
            "total_tokens_per_estimated_model_wait_second": rate(
                total_tokens,
                estimated_model_wait_seconds,
            ),
        }

    def _execute_prepared_attempt(
        self,
        *,
        sandbox_dir: Path,
        attempt_dir: Path,
        stdout_path: Path,
        stderr_path: Path,
        last_message_path: Path,
        events_path: Path,
    ) -> dict[str, Any]:
        started_at_ts = time.time()
        run_result = self.sdk_runner.run(
            sandbox_dir=sandbox_dir,
            user_prompt=self.codex_prompt,
        )
        last_message_path.write_text(run_result.final_output, encoding="utf-8")

        duration_seconds = time.time() - started_at_ts
        stdout_payload = {
            "runner": "openai_agents_sdk",
            "model": self.sdk_model,
            "reasoning_effort": self.sdk_reasoning_effort,
            "final_output": run_result.final_output,
            "tool_calls": run_result.tool_calls,
            "usage": run_result.usage,
        }
        stdout_path.write_text(
            json.dumps(stdout_payload, indent=2, ensure_ascii=True) + "\n",
            encoding="utf-8",
        )
        save_json(events_path, stdout_payload)
        stderr_path.write_text(run_result.error_traceback, encoding="utf-8")

        if run_result.error_detail is not None:
            status_after = "timeout" if run_result.timed_out else "openai_sdk_error"
            status_detail = run_result.error_detail
        else:
            status_after = None
            status_detail = None

        return {
            "started_at_ts": started_at_ts,
            "duration_seconds": duration_seconds,
            "fatal_error": run_result.error_detail is not None,
            "status_after": status_after,
            "status_after_detail": status_detail,
            "summary_fields": {
                "timed_out": run_result.timed_out,
                "runner_error_detail": run_result.error_detail,
                "tool_call_count": len(run_result.tool_calls),
                "usage": run_result.usage,
                "estimated_token_throughput": self._estimate_sdk_token_throughput(
                    duration_seconds=duration_seconds,
                    tool_calls=run_result.tool_calls,
                    usage=run_result.usage,
                ),
            },
        }

    def _run_query_attempt(
        self,
        *,
        query_invocation_id: str,
        query_text: str,
        query_image: str | None,
    ) -> dict[str, Any]:
        require(
            self.workspace_dir is not None,
            "agentrunbook_c_v2 workspace_dir is not configured",
        )
        attempt_index, attempt_dir = self._next_attempt_dir(query_invocation_id)
        sandbox_dir = attempt_dir / "sandbox"
        sandbox_dir.mkdir(parents=True, exist_ok=True)

        question_payload = self._build_question_payload(
            query_text=query_text,
            query_image=query_image,
            sandbox_dir=sandbox_dir,
        )
        save_json(sandbox_dir / "question.json", question_payload)
        (sandbox_dir / "INSTRUCTION.md").write_text(self.query_instruction_text, encoding="utf-8")
        summary_result = self._ensure_trajectory_summary(attempt_dir=attempt_dir)
        if not summary_result["success"]:
            self.online_learning.expose_to_sandbox(
                sandbox_dir=sandbox_dir,
                query_trace_dir=self.query_trace_dir,
                workspace_dir=self.workspace_dir,
            )
            summary: dict[str, Any] = {
                "query_invocation_id": query_invocation_id,
                "attempt_index": attempt_index,
                "completed_at_utc": utc_now_iso(),
                **self._runner_summary_fields(),
                "question_has_image": query_image is not None,
                "question_payload": question_payload,
                "sandbox_dir": str(sandbox_dir),
                "scripts_dir": str(sandbox_dir / "scripts"),
                "scripts_dir_mode": "single_helper",
                "sandbox_helper_scripts": [],
                "trajectory_summary_concise_path": summary_result["trajectory_summary_concise_path"],
                "trajectory_summary_full_path": summary_result["trajectory_summary_full_path"],
                "summary_renderer_path": summary_result["summary_renderer_path"],
                "summary_stdout_path": summary_result["summary_stdout_path"],
                "summary_stderr_path": summary_result["summary_stderr_path"],
                "summary_rendered": summary_result["summary_rendered"],
                "status_after": summary_result["status"],
                "status_after_detail": summary_result["detail"],
            }
            save_json(attempt_dir / "summary.json", summary)
            attempt_result = {
                "success": False,
                "status": summary_result["status"],
                "detail": summary_result["detail"],
                "memory_context": [],
            }
            self.online_learning.finalize_attempt(
                query_trace_dir=self.query_trace_dir,
                query_invocation_id=query_invocation_id,
                attempt_result=attempt_result,
            )
            return attempt_result

        relative_symlink(self.workspace_dir / "trajectories", sandbox_dir / "trajectories")
        sandbox_helper_scripts = self._populate_sandbox_scripts(sandbox_dir)

        output_path = sandbox_dir / "memory_module_output.json"
        last_message_path = attempt_dir / "last_message.txt"
        stdout_path = attempt_dir / "stdout.log"
        stderr_path = attempt_dir / "stderr.log"
        events_path = attempt_dir / "events.json"
        summary_path = attempt_dir / "summary.json"
        execution_result = self._execute_prepared_attempt(
            sandbox_dir=sandbox_dir,
            attempt_dir=attempt_dir,
            stdout_path=stdout_path,
            stderr_path=stderr_path,
            last_message_path=last_message_path,
            events_path=events_path,
        )
        status = read_memory_output_status(
            output_path,
            require_evidence_gate=self.require_evidence_gate,
        )
        raw_output_text = output_path.read_text(encoding="utf-8") if output_path.exists() else None
        status_after = execution_result["status_after"] or status.state
        status_detail = execution_result["status_after_detail"]
        if status_detail is None:
            status_detail = status.detail
        summary: dict[str, Any] = {
            "query_invocation_id": query_invocation_id,
            "attempt_index": attempt_index,
            "started_at_utc": datetime.fromtimestamp(
                execution_result["started_at_ts"],
                timezone.utc,
            ).isoformat(),
            "completed_at_utc": utc_now_iso(),
            "duration_seconds": execution_result["duration_seconds"],
            **self._runner_summary_fields(),
            "question_has_image": query_image is not None,
            "question_payload": question_payload,
            "sandbox_dir": str(sandbox_dir),
            "scripts_dir": str(sandbox_dir / "scripts"),
            "scripts_dir_mode": "single_helper",
            "sandbox_helper_scripts": sandbox_helper_scripts,
            "trajectory_summary_concise_path": summary_result["trajectory_summary_concise_path"],
            "trajectory_summary_full_path": summary_result["trajectory_summary_full_path"],
            "summary_renderer_path": summary_result["summary_renderer_path"],
            "summary_stdout_path": summary_result["summary_stdout_path"],
            "summary_stderr_path": summary_result["summary_stderr_path"],
            "summary_rendered": summary_result["summary_rendered"],
            **execution_result["summary_fields"],
            "status_after": status_after,
            "status_after_detail": status_detail,
            "stdout_path": str(stdout_path),
            "stderr_path": str(stderr_path),
            "events_path": str(events_path),
            "last_message_path": str(last_message_path),
            "output_path": str(output_path),
            "agent_output_raw_text": raw_output_text,
        }

        if execution_result["fatal_error"] and not status.is_finished:
            save_json(summary_path, summary)
            attempt_result = {
                "success": False,
                "status": status_after,
                "detail": status_detail,
                "memory_context": [],
            }
            self.online_learning.finalize_attempt(
                query_trace_dir=self.query_trace_dir,
                query_invocation_id=query_invocation_id,
                attempt_result=attempt_result,
            )
            return attempt_result
        if not status.is_finished:
            save_json(summary_path, summary)
            attempt_result = {
                "success": False,
                "status": status.state,
                "detail": status.detail,
                "memory_context": [],
            }
            self.online_learning.finalize_attempt(
                query_trace_dir=self.query_trace_dir,
                query_invocation_id=query_invocation_id,
                attempt_result=attempt_result,
            )
            return attempt_result

        try:
            normalized_output = self._normalize_output_for_query(output_path)
            memory_context = self._build_memory_context_from_output(normalized_output)
        except Exception as exc:
            summary["status_after"] = "internal_postprocess_error"
            summary["status_after_detail"] = str(exc)
            save_json(summary_path, summary)
            attempt_result = {
                "success": False,
                "status": "internal_postprocess_error",
                "detail": str(exc),
                "memory_context": [],
            }
            self.online_learning.finalize_attempt(
                query_trace_dir=self.query_trace_dir,
                query_invocation_id=query_invocation_id,
                attempt_result=attempt_result,
            )
            return attempt_result

        summary_fields = {
            "memory_markdown": normalized_output["memory_markdown"],
            "trajectory_spans_raw": normalized_output["trajectory_spans_raw"],
            "trajectory_spans_valid": normalized_output["trajectory_spans_valid"],
            "trajectory_spans_invalid": normalized_output["trajectory_spans_invalid"],
            "memory_context_item_count": len(memory_context),
        }
        if "evidence_status" in normalized_output:
            summary_fields.update(
                {
                    "evidence_status": normalized_output["evidence_status"],
                    "evidence_status_reason": normalized_output["evidence_status_reason"],
                    "answer_policy": normalized_output["answer_policy"],
                }
            )
        summary.update(summary_fields)
        save_json(summary_path, summary)
        attempt_result = {
            "success": True,
            "status": status.state,
            "detail": status.detail,
            "memory_context": memory_context,
        }
        self.online_learning.finalize_attempt(
            query_trace_dir=self.query_trace_dir,
            query_invocation_id=query_invocation_id,
            attempt_result=attempt_result,
        )
        return attempt_result

    def post_query_hook(
        self,
        *,
        query: str,
        query_image: str | None,
        memory_context: list[dict[str, str]],
    ) -> dict[str, object] | None:
        if not self.online_learning.enabled:
            return None
        query_context = self.get_query_context()
        query_invocation_id = query_context.get("query_invocation_id")
        if not isinstance(query_invocation_id, str) or not query_invocation_id.strip():
            return {
                "status": "skipped",
                "reason": "unknown_query_invocation_id",
            }

        attempt_dir = self.online_learning.attempt_for_post_query(
            query_trace_dir=self.query_trace_dir,
            query_invocation_id=query_invocation_id,
        )
        if attempt_dir is None:
            return {
                "status": "skipped",
                "reason": "missing_attempt_dir",
                "query_invocation_id": query_invocation_id,
            }

        summary_path = attempt_dir / "summary.json"
        if not summary_path.exists():
            return {
                "status": "skipped",
                "reason": "missing_attempt_summary",
                "query_invocation_id": query_invocation_id,
                "attempt_dir": str(attempt_dir),
            }
        try:
            summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            return {
                "status": "skipped",
                "reason": "invalid_attempt_summary",
                "query_invocation_id": query_invocation_id,
                "attempt_dir": str(attempt_dir),
                "detail": str(exc),
            }
        if not isinstance(summary_payload, dict) or summary_payload.get("status_after") != "finished":
            return {
                "status": "skipped",
                "reason": "query_not_finished",
                "query_invocation_id": query_invocation_id,
                "attempt_dir": str(attempt_dir),
                "status_after": (
                    summary_payload.get("status_after")
                    if isinstance(summary_payload, dict)
                    else None
                ),
            }

        require(
            self.consolidation_codex_runner is not None,
            "online-learning consolidation runner is not configured",
        )
        return self.online_learning.run_consolidation(
            query_invocation_id=query_invocation_id,
            attempt_dir=attempt_dir,
            query_trace_dir=self.query_trace_dir,
            workspace_dir=self.workspace_dir,
            consolidation_runner=self.consolidation_codex_runner,
            is_cancelled=self._is_cancelled,
        )

    def _save_backend(self, output_dir: Path) -> None:
        super()._save_backend(output_dir)
        self.online_learning.copy_strategy_to(
            output_dir=output_dir,
            query_trace_dir=self.query_trace_dir,
            workspace_dir=self.workspace_dir,
        )
        return None

    def _load_backend(self, input_dir: Path) -> None:
        super()._load_backend(input_dir)
        self.online_learning.bind_loaded_strategy_memory_dir(input_dir / "strategy_memory")
        return None
