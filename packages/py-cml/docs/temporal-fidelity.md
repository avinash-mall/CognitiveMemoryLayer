# Temporal Fidelity

The `timestamp` parameter enables storing memories with their original event times, providing **temporal fidelity** for historical data replay and accurate temporal reasoning.

## Overview

By default, memories are timestamped with the current time when they're stored. However, when working with historical data, benchmarks, or data migration, you often need to preserve the original event time. The optional `timestamp` parameter allows you to specify exactly when an event occurred.

## Basic Usage

### Storing Historical Memories

```python
from datetime import datetime, timezone
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer()

# Store a memory with a specific historical timestamp
historical_time = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

memory.write(
    "User mentioned they prefer dark mode",
    timestamp=historical_time,
    session_id="session_1"
)
```

### Processing Historical Turns

```python
# Process a conversation turn with a specific timestamp
memory.turn(
    user_message="What's the weather like?",
    assistant_response="It's sunny today!",
    timestamp=historical_time,
    session_id="session_1"
)
```

### Default Behavior (Current Time)

```python
# When timestamp is omitted, it defaults to "now"
memory.write("This memory gets the current timestamp")
```

## Use Cases

### 1. Benchmark Evaluations

Temporal fidelity is essential for benchmark evaluations like [Locomo](https://github.com/google/locomo) that test long-term memory capabilities by replaying historical conversations:

```python
from datetime import datetime, timezone

# Locomo provides session dates like "2023-06-15 14:30:00"
session_date = datetime.fromisoformat("2023-06-15T14:30:00+00:00")

# Store each dialog with the correct session timestamp
for dialog in session_dialogs:
    memory.write(
        content=f"{dialog['speaker']}: {dialog['text']}",
        timestamp=session_date,
        session_id=session_id,
        metadata={
            "speaker": dialog["speaker"],
            "dia_id": dialog["dia_id"]
        }
    )
```

### 2. Data Migration

When migrating from another system, preserve original timestamps:

```python
# Import historical data from another system
for record in legacy_system_records:
    memory.write(
        content=record["content"],
        timestamp=record["created_at"],  # Original timestamp
        metadata={"source": "legacy_system", "original_id": record["id"]}
    )
```

### 3. Testing Temporal Reasoning

Test how your system handles memories from different time periods:

```python
from datetime import timedelta

now = datetime.now(timezone.utc)

# Store memories at different times
memory.write("User preferred dark mode", timestamp=now - timedelta(days=180))
memory.write("User switched to light mode", timestamp=now - timedelta(days=90))
memory.write("User confirmed light mode preference", timestamp=now - timedelta(days=30))

# Test retrieval and temporal reasoning
result = memory.read("user theme preference")
# System should prioritize more recent preferences
```

## Complete Example: Historical Conversation Replay

```python
from datetime import datetime, timezone, timedelta
from cml import CognitiveMemoryLayer

memory = CognitiveMemoryLayer()

# Simulate a conversation from 6 months ago
base_time = datetime.now(timezone.utc) - timedelta(days=180)

conversation = [
    ("Hello, I'm new here", "Welcome! How can I help you?"),
    ("I need help with settings", "I'd be happy to help. What would you like to configure?"),
    ("I prefer dark themes", "Got it! I'll remember that."),
]

# Store each turn with incremental timestamps
for i, (user_msg, assistant_msg) in enumerate(conversation):
    turn_time = base_time + timedelta(minutes=i * 5)
    
    memory.turn(
        user_message=user_msg,
        assistant_response=assistant_msg,
        timestamp=turn_time,
        session_id="historical_session"
    )

# Verify temporal ordering
memories = memory.read("user preferences", max_results=10)
for mem in memories.memories:
    print(f"[{mem.timestamp}] {mem.text}")
```

## API Reference

### Methods Supporting Timestamp

- **`write(content, ..., timestamp=None)`** — Store a memory with optional event timestamp
- **`turn(user_message, ..., timestamp=None)`** — Process a turn with optional event timestamp  
- **`remember(content, ..., timestamp=None)`** — Alias for `write()`, also supports timestamp

### Timestamp Format

The `timestamp` parameter accepts Python `datetime` objects. Always use timezone-aware datetimes:

```python
from datetime import datetime, timezone

# ✅ Good: timezone-aware
timestamp = datetime(2023, 6, 15, 14, 30, 0, tzinfo=timezone.utc)

# ✅ Good: current time with timezone
timestamp = datetime.now(timezone.utc)

# ❌ Avoid: naive datetime (no timezone)
timestamp = datetime(2023, 6, 15, 14, 30, 0)  # Missing timezone!
```

### Backward Compatibility

The `timestamp` parameter is **optional** and **backward compatible**:

- When provided: Uses the specified timestamp
- When omitted: Defaults to current time (`datetime.now(timezone.utc)`)
- Existing code without `timestamp` continues to work unchanged

## Implementation Details

The timestamp is threaded through the entire CML pipeline:

1. **API Layer** — `WriteMemoryRequest` and `ProcessTurnRequest` accept optional `timestamp`
2. **Orchestrator** — Passes timestamp to short-term memory
3. **Working Memory** — Chunker uses timestamp when creating semantic chunks
4. **Hippocampal Store** — Encodes chunks with the specified timestamp
5. **Storage** — PostgreSQL stores the timestamp in the `timestamp` column

When no timestamp is provided, each layer defaults to the current time, ensuring backward compatibility.

## Best Practices

1. **Always use timezone-aware datetimes** — Use `timezone.utc` or another explicit timezone
2. **Validate historical timestamps** — Ensure timestamps are reasonable (not in the future, not too far in the past)
3. **Document timestamp source** — Use metadata to track where timestamps came from (e.g., `{"timestamp_source": "benchmark_data"}`)
4. **Test temporal ordering** — Verify that retrieval respects temporal relationships
5. **Consider time zones** — Store all timestamps in UTC and convert for display

## Related Documentation

- [API Reference](api-reference.md) — Full method signatures
- [Examples](examples.md) — Runnable examples including `temporal_fidelity.py`
- [Getting Started](getting-started.md) — Quick introduction to py-cml

## Example Script

See [`examples/temporal_fidelity.py`](../examples/temporal_fidelity.py) for a complete, runnable example demonstrating:
- Historical memory storage
- Conversation replay
- Benchmark evaluation scenarios
- Temporal ordering verification
