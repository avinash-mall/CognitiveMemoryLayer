"""Internal request models for constructing API payloads."""

from __future__ import annotations

from cml_contracts.models import (
    BulkActionRequest,
    ConfigUpdateRequest,
    CreateSessionRequest,
    DashboardConsolidateRequest,
    DashboardForgetRequest,
    DashboardReconsolidateRequest,
    DashboardRetrievalRequest,
    FeedbackRequest,
    ForgetRequest,
    ProcessTurnRequest,
    ReadMemoryRequest,
    SimulateRequest,
    UpdateMemoryRequest,
    WindowConsolidateRequest,
    WriteMemoryRequest,
)

__all__ = [
    "BulkActionRequest",
    "ConfigUpdateRequest",
    "CreateSessionRequest",
    "DashboardConsolidateRequest",
    "DashboardForgetRequest",
    "DashboardReconsolidateRequest",
    "DashboardRetrievalRequest",
    "FeedbackRequest",
    "ForgetRequest",
    "ReadRequest",
    "SimulateRequest",
    "TurnRequest",
    "UpdateRequest",
    "WindowConsolidateRequest",
    "WriteRequest",
]

# Renamed SDK twins of the canonical server contracts, unified in
# ``cml_contracts.models``. Kept as importable aliases so the public
# ``cml.models`` surface is unchanged.
WriteRequest = WriteMemoryRequest
TurnRequest = ProcessTurnRequest
UpdateRequest = UpdateMemoryRequest
ReadRequest = ReadMemoryRequest
