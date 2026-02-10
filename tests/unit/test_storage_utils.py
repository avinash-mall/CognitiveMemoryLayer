"""Unit tests for storage utilities."""

from datetime import datetime, timezone

from src.storage.utils import naive_utc


class TestNaiveUtc:
    """Tests for naive_utc."""

    def test_none_returns_none(self):
        assert naive_utc(None) is None

    def test_naive_datetime_unchanged(self):
        dt = datetime(2025, 2, 10, 12, 0, 0)  # no tzinfo
        assert naive_utc(dt) is dt
        assert naive_utc(dt).tzinfo is None

    def test_aware_utc_converted_to_naive_utc(self):
        dt = datetime(2025, 2, 10, 12, 0, 0, tzinfo=timezone.utc)
        result = naive_utc(dt)
        assert result.tzinfo is None
        assert result.year == 2025
        assert result.month == 2
        assert result.day == 10
        assert result.hour == 12

    def test_aware_non_utc_converted_to_naive_utc(self):
        from datetime import timedelta
        # UTC+2
        tz = timezone(timedelta(hours=2))
        dt = datetime(2025, 2, 10, 14, 0, 0, tzinfo=tz)  # 14:00 UTC+2 = 12:00 UTC
        result = naive_utc(dt)
        assert result.tzinfo is None
        assert result.hour == 12
        assert result.minute == 0
