"""Unit tests for utils timing and metrics."""

from unittest.mock import patch

import pytest

from src.utils.metrics import track_retrieval_latency
from src.utils.timing import timed


class TestTimed:
    """Tests for timed context manager."""

    def test_timed_yields_and_completes(self):
        with timed("test_op"):
            pass

    def test_timed_measures_elapsed(self):
        with patch("src.utils.timing.logger") as mock_logger:
            with timed("test_op"):
                pass
            mock_logger.info.assert_called_once()
            call_kw = mock_logger.info.call_args[1]
            assert call_kw["operation"] == "test_op"
            assert "elapsed_ms" in call_kw
            assert call_kw["elapsed_ms"] >= 0

    def test_timed_uses_warning_when_over_threshold(self):
        with patch("src.utils.timing.logger") as mock_logger:
            with patch("src.utils.timing.time") as mock_time:
                mock_time.perf_counter.side_effect = [0.0, 1.0]  # 1000 ms
                with timed("slow_op", warn_ms=200):
                    pass
            mock_logger.warning.assert_called_once()
            call_kw = mock_logger.warning.call_args[1]
            assert call_kw["operation"] == "slow_op"
            assert call_kw["elapsed_ms"] == 1000.0


class TestTrackRetrievalLatency:
    """Tests for track_retrieval_latency decorator."""

    def test_sync_wrapped_returns_value(self):
        @track_retrieval_latency(tenant_id="t1")
        def get_value():
            return 42
        assert get_value() == 42

    @pytest.mark.asyncio
    async def test_async_wrapped_returns_value(self):
        @track_retrieval_latency(tenant_id="t2")
        async def get_async():
            return 99
        result = await get_async()
        assert result == 99
