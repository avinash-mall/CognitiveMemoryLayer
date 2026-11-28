import json
from pathlib import Path
import tempfile
import unittest

from evaluation.harness import build_prompt_row, compact_nonshared_memory_workspace
from memory_modules.agentrunbook_r import AgentRunbookR
from memory_modules.codex import CodexMemory
from memory_modules.memory import Memory


class QueryContextSpyMemory(Memory):
    def __init__(self, memory_type: str) -> None:
        self.memory_type = memory_type
        self.seen_query_context: dict[str, object] | None = None
        super().__init__({})

    def insert(self, trajectory: dict[str, object]) -> None:
        _ = trajectory

    def query(self, query: str, query_image: str | None = None) -> list[dict[str, str]]:
        _ = query, query_image
        self.seen_query_context = self.get_query_context()
        return []


def prompt_item() -> dict[str, object]:
    return {
        "index": 0,
        "question_item": {
            "id": "secret-question-id",
            "question_type": "procedure-abs",
            "question": "How do I complete the task?",
            "answer": "secret-answer",
            "eval_function": "secret-evaluator",
            "metadata": {"original_goal": ["secret-original-goal"]},
        },
        "question_id": "secret-question-id",
        "query_invocation_id": "opaque-invocation-id",
        "question_type": "procedure-abs",
        "category": "procedure-abs",
        "eval_function": "secret-evaluator",
        "eval_name": "llm_abstention_checker",
        "question_text": "How do I complete the task?",
        "question_image": None,
        "answer_gold": "secret-answer",
    }


class QueryPrivacyTest(unittest.TestCase):
    def test_all_backends_receive_only_opaque_query_context(self) -> None:
        for memory_type in (
            "no_retrieval",
            "rag",
            "agentrunbook_r",
            "codex",
            "agentrunbook_c",
            "agentrunbook_c_v2",
            "cml",
        ):
            with self.subTest(memory_type=memory_type):
                memory = QueryContextSpyMemory(memory_type)
                build_prompt_row(
                    prompt_item(),
                    haystack_ids=[],
                    memory=memory,
                    system_prompt="Answer using memory.",
                    memory_context_max_tokens=1,
                )
                self.assertEqual(
                    memory.seen_query_context,
                    {"query_invocation_id": "opaque-invocation-id"},
                )
                self.assertEqual(memory.get_query_context(), {})

    def test_query_context_api_rejects_benchmark_metadata(self) -> None:
        memory = QueryContextSpyMemory("spy")
        with self.assertRaises(TypeError):
            memory.set_query_context(  # type: ignore[call-arg]
                query_invocation_id="opaque-invocation-id",
                question_id="secret-question-id",
            )

    def test_agentrunbook_r_planner_uses_only_question_text(self) -> None:
        memory = object.__new__(AgentRunbookR)
        memory._runtime_query_summary = "safe memory summary"

        messages = memory._build_query_generation_messages(
            query="How do I complete the task?",
        )
        prompt = messages[1]["content"]

        self.assertIn("Question text: How do I complete the task?", prompt)
        for forbidden in (
            "Question ID:",
            "Question type:",
            "Question image path:",
            "Original goals attached",
        ):
            self.assertNotIn(forbidden, prompt)

    def test_agentrunbook_r_fallback_is_metadata_independent(self) -> None:
        memory = object.__new__(AgentRunbookR)
        self.assertEqual(
            memory._fallback_query_bundle(query="  Find   the relevant workflow  "),
            {
                "raw_state_queries": ["Find the relevant workflow"],
                "event_query": "Find the relevant workflow",
                "note_query": "Find the relevant workflow",
            },
        )

    def test_codex_question_payload_contains_only_query_inputs(self) -> None:
        memory = object.__new__(CodexMemory)
        with tempfile.TemporaryDirectory() as temp_dir:
            payload = memory._build_question_payload(
                query_text="How do I complete the task?",
                query_image=None,
                sandbox_dir=Path(temp_dir),
            )
        self.assertEqual(payload, {"question": "How do I complete the task?"})

    def test_compaction_uses_opaque_query_invocation_id(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            workspace_dir = root / "workspace"
            query_trace_dir = root / "query_traces"
            trajectories_dir = workspace_dir / "trajectories"
            (trajectories_dir / "keep").mkdir(parents=True)
            (trajectories_dir / "drop").mkdir()

            (workspace_dir / "index.json").write_text(
                json.dumps(
                    {
                        "memory_type": "codex",
                        "inserted_trajectory_ids": ["keep", "drop"],
                    }
                ),
                encoding="utf-8",
            )
            (workspace_dir / "haystack_manifest.json").write_text(
                json.dumps(
                    {
                        "id": "test-haystack",
                        "variant": "test",
                        "trajectory_ids": ["keep", "drop"],
                        "metadata": {},
                    }
                ),
                encoding="utf-8",
            )

            opaque_summary_dir = query_trace_dir / "opaque-invocation-id" / "attempt_001"
            opaque_summary_dir.mkdir(parents=True)
            (opaque_summary_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status_after": "finished",
                        "trajectory_spans_valid": [{"trajectory_id": "keep"}],
                    }
                ),
                encoding="utf-8",
            )
            private_id_summary_dir = query_trace_dir / "secret-question-id" / "attempt_001"
            private_id_summary_dir.mkdir(parents=True)
            (private_id_summary_dir / "summary.json").write_text(
                json.dumps(
                    {
                        "status_after": "finished",
                        "trajectory_spans_valid": [{"trajectory_id": "drop"}],
                    }
                ),
                encoding="utf-8",
            )

            compact_nonshared_memory_workspace(
                workspace_dir=workspace_dir,
                query_trace_dir=query_trace_dir,
                query_invocation_id="opaque-invocation-id",
            )

            self.assertTrue((trajectories_dir / "keep").exists())
            self.assertFalse((trajectories_dir / "drop").exists())

    def test_coding_backend_configs_do_not_reference_questions_file(self) -> None:
        repo_root = Path(__file__).resolve().parents[1]
        for config_name in ("codex.json", "agentrunbook_c.json"):
            with self.subTest(config_name=config_name):
                config = json.loads(
                    (repo_root / "evaluation" / "memory_configs" / config_name).read_text(
                        encoding="utf-8"
                    )
                )
                self.assertNotIn("questions_path", config["memory_params"])


if __name__ == "__main__":
    unittest.main()
