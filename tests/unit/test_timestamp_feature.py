"""
Unit tests for the timestamp feature.

Tests that the optional timestamp parameter is correctly threaded through
the entire write pipeline from API to storage.
"""

from datetime import datetime, timezone
import pytest

from src.api.schemas import WriteMemoryRequest, ProcessTurnRequest
from src.memory.working.models import SemanticChunk, ChunkType


class TestTimestampInSchemas:
    """Test timestamp field in API schemas."""

    def test_write_memory_request_accepts_timestamp(self):
        """WriteMemoryRequest should accept optional timestamp."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        req = WriteMemoryRequest(content="Test memory", timestamp=ts)
        assert req.timestamp == ts

    def test_write_memory_request_timestamp_defaults_to_none(self):
        """WriteMemoryRequest timestamp should default to None."""
        req = WriteMemoryRequest(content="Test memory")
        assert req.timestamp is None

    def test_process_turn_request_accepts_timestamp(self):
        """ProcessTurnRequest should accept optional timestamp."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        req = ProcessTurnRequest(user_message="Hello", timestamp=ts)
        assert req.timestamp == ts

    def test_process_turn_request_timestamp_defaults_to_none(self):
        """ProcessTurnRequest timestamp should default to None."""
        req = ProcessTurnRequest(user_message="Hello")
        assert req.timestamp is None


class TestTimestampInChunker:
    """Test timestamp handling in semantic chunker."""

    def test_semantic_chunk_uses_provided_timestamp(self):
        """SemanticChunk should use provided timestamp."""
        ts = datetime(2025, 1, 15, 10, 30, 0, tzinfo=timezone.utc)
        chunk = SemanticChunk(
            id="test-chunk", text="Test content", chunk_type=ChunkType.STATEMENT, timestamp=ts
        )
        assert chunk.timestamp == ts

    def test_semantic_chunk_defaults_to_now_when_no_timestamp(self):
        """SemanticChunk should default to current time when no timestamp provided."""
        before = datetime.now(timezone.utc)
        chunk = SemanticChunk(id="test-chunk", text="Test content", chunk_type=ChunkType.STATEMENT)
        after = datetime.now(timezone.utc)

        # Timestamp should be between before and after
        assert before <= chunk.timestamp <= after


class TestTimestampInOrchestrator:
    """Test timestamp threading through orchestrator."""

    @pytest.mark.asyncio
    async def test_orchestrator_write_method_signature_accepts_timestamp(self):
        """Orchestrator.write method should have timestamp parameter."""
        from src.memory.orchestrator import MemoryOrchestrator
        import inspect

        # Check that write method accepts timestamp parameter
        sig = inspect.signature(MemoryOrchestrator.write)
        params = sig.parameters

        assert "timestamp" in params
        # Should be optional (has default value)
        assert params["timestamp"].default is not inspect.Parameter.empty


class TestTimestampInSeamlessProvider:
    """Test timestamp threading through seamless provider."""

    @pytest.mark.asyncio
    async def test_seamless_provider_process_turn_method_signature_accepts_timestamp(self):
        """SeamlessMemoryProvider.process_turn method should have timestamp parameter."""
        from src.memory.seamless_provider import SeamlessMemoryProvider
        import inspect

        # Check that process_turn method accepts timestamp parameter
        sig = inspect.signature(SeamlessMemoryProvider.process_turn)
        params = sig.parameters

        assert "timestamp" in params
        # Should be optional (has default value)
        assert params["timestamp"].default is not inspect.Parameter.empty


class TestTimestampBackwardCompatibility:
    """Test that timestamp feature is backward compatible."""

    def test_api_schema_without_timestamp_is_valid(self):
        """API schemas should be valid without timestamp field."""
        # WriteMemoryRequest
        write_req = WriteMemoryRequest(content="Test")
        assert write_req.model_dump() is not None

        # ProcessTurnRequest
        turn_req = ProcessTurnRequest(user_message="Hello")
        assert turn_req.model_dump() is not None

    def test_timestamp_parameter_is_optional_in_all_layers(self):
        """Timestamp parameter should be optional throughout the stack."""
        from src.memory.orchestrator import MemoryOrchestrator
        from src.memory.short_term import ShortTermMemory
        from src.memory.working.manager import WorkingMemoryManager
        from src.memory.working.chunker import RuleBasedChunker, SemanticChunker
        from src.memory.hippocampal.store import HippocampalStore
        from src.memory.seamless_provider import SeamlessMemoryProvider
        import inspect

        # Check all key methods have optional timestamp parameter
        methods_to_check = [
            (MemoryOrchestrator, "write"),
            (ShortTermMemory, "ingest_turn"),
            (WorkingMemoryManager, "process_input"),
            (RuleBasedChunker, "chunk"),
            (SemanticChunker, "chunk"),
            (HippocampalStore, "encode_chunk"),
            (HippocampalStore, "encode_batch"),
            (SeamlessMemoryProvider, "process_turn"),
        ]

        for cls, method_name in methods_to_check:
            method = getattr(cls, method_name)
            sig = inspect.signature(method)
            params = sig.parameters

            # timestamp should exist and be optional
            assert (
                "timestamp" in params
            ), f"{cls.__name__}.{method_name} missing timestamp parameter"
            assert (
                params["timestamp"].default is not inspect.Parameter.empty
            ), f"{cls.__name__}.{method_name} timestamp should be optional"


class TestTimestampTemporalFidelity:
    """Test that timestamp enables temporal fidelity for historical data."""

    @pytest.mark.asyncio
    async def test_historical_timestamp_preserved(self):
        """Historical timestamps should be preserved through the pipeline."""
        from src.memory.working.chunker import RuleBasedChunker

        # Use a historical timestamp (e.g., from Locomo benchmark)
        historical_ts = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

        chunker = RuleBasedChunker()
        chunks = chunker.chunk(text="I prefer dark mode.", timestamp=historical_ts)

        # All chunks should have the historical timestamp
        assert len(chunks) > 0
        for chunk in chunks:
            assert chunk.timestamp == historical_ts

    def test_different_timestamps_for_different_events(self):
        """Different events can have different timestamps."""
        ts1 = datetime(2023, 1, 1, 10, 0, 0, tzinfo=timezone.utc)
        ts2 = datetime(2023, 6, 1, 15, 30, 0, tzinfo=timezone.utc)

        chunk1 = SemanticChunk(
            id="chunk-1", text="Event 1", chunk_type=ChunkType.STATEMENT, timestamp=ts1
        )

        chunk2 = SemanticChunk(
            id="chunk-2", text="Event 2", chunk_type=ChunkType.STATEMENT, timestamp=ts2
        )

        assert chunk1.timestamp == ts1
        assert chunk2.timestamp == ts2
        assert chunk1.timestamp != chunk2.timestamp
