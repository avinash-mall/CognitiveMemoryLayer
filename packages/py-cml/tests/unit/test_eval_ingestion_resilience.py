"""Ingestion resilience for the LoCoMo runner.

Guards the failure that zombied a six-hour run: one conversation's batch hit the
client read timeout, `ReadTimeout` was not in the retry list, the exception
propagated out of the `as_completed` loop, and `ThreadPoolExecutor.__exit__`
(shutdown(wait=True)) kept executing every queued conversation with the
checkpoint loop dead — hours of LLM burn with no checkpoint and no QA phase.
"""

from unittest.mock import MagicMock, patch

import pytest
import requests

from cml.eval import locomo


def _samples(n):
    # distinct prompts -> distinct conversation groups
    return [
        {"input_prompt": f"Speaker A: hello number {i}\n\nQuestion: q{i}", "trigger": f"q{i}"}
        for i in range(n)
    ]


class TestPhaseAContinuesPastFailures:
    def _run(self, tmp_path, failing: set[int]):
        checkpoint = tmp_path / "ck.json"

        def fake_ingest(url, key, idx, sample, delay, pbar=None):
            if idx in failing:
                raise requests.exceptions.ReadTimeout("boom")

        with patch.object(locomo, "_ingest_sample", side_effect=fake_ingest):
            locomo.phase_a_ingestion(
                _samples(5),
                "http://x",
                "k",
                None,
                ingestion_workers=3,
                checkpoint_file=checkpoint,
            )
        return checkpoint

    def test_all_success_checkpoints_everything(self, tmp_path):
        ck = self._run(tmp_path, failing=set())
        import json

        assert set(json.loads(ck.read_text())["completed_indices"]) == {0, 1, 2, 3, 4}

    def test_one_failure_still_checkpoints_the_rest_then_raises(self, tmp_path):
        import json

        with pytest.raises(RuntimeError, match="1 of 5 conversations failed"):
            self._run(tmp_path, failing={2})
        # the healthy conversations were ingested AND checkpointed
        assert set(json.loads((tmp_path / "ck.json").read_text())["completed_indices"]) == {
            0,
            1,
            3,
            4,
        }

    def test_rerun_after_failure_retries_only_the_failure(self, tmp_path):
        with pytest.raises(RuntimeError):
            self._run(tmp_path, failing={2})
        calls: list[int] = []

        def recording(url, key, idx, sample, delay, pbar=None):
            calls.append(idx)

        with patch.object(locomo, "_ingest_sample", side_effect=recording):
            locomo.phase_a_ingestion(
                _samples(5),
                "http://x",
                "k",
                None,
                ingestion_workers=3,
                checkpoint_file=tmp_path / "ck.json",
            )
        assert calls == [2]


class TestBatchWriteRetriesTimeouts:
    def test_read_timeout_is_retried_then_succeeds(self):
        ok = MagicMock(status_code=200)
        session = MagicMock()
        session.post.side_effect = [requests.exceptions.ReadTimeout("t"), ok]
        with (
            patch.object(locomo, "_get_session", return_value=session),
            patch.object(locomo.time, "sleep"),
        ):
            locomo._cml_write_batch("http://x", "k", "t1", [{"content": "c"}])
        assert session.post.call_count == 2

    def test_timeout_still_raises_after_exhausting_retries(self):
        session = MagicMock()
        session.post.side_effect = requests.exceptions.ReadTimeout("t")
        with (
            patch.object(locomo, "_get_session", return_value=session),
            patch.object(locomo.time, "sleep"),
            pytest.raises(requests.exceptions.ReadTimeout),
        ):
            locomo._cml_write_batch("http://x", "k", "t1", [{"content": "c"}])
        assert session.post.call_count == 3
