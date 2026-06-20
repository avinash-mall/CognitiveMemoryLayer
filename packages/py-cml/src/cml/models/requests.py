"""Internal request models for constructing API payloads."""

from __future__ import annotations

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from cml_contracts.models import (
    BulkActionRequest,
    ConfigUpdateRequest,
    CreateSessionRequest,
    DashboardConsolidateRequest,
    DashboardForgetRequest,
    DashboardReconsolidateRequest,
    DashboardRetrievalRequest,
    ForgetRequest,
    ProcessTurnRequest,
    UpdateMemoryRequest,
    WriteMemoryRequest,
)

from .enums import MemoryType

__all__ = [
    "BulkActionRequest",
    "ConfigUpdateRequest",
    "CreateSessionRequest",
    "DashboardConsolidateRequest",
    "DashboardForgetRequest",
    "DashboardReconsolidateRequest",
    "DashboardRetrievalRequest",
    "ForgetRequest",
    "ReadRequest",
    "TurnRequest",
    "UpdateRequest",
    "WriteRequest",
]

# Renamed SDK twins of the canonical server contracts, unified in
# ``cml_contracts.models`` (field-identical). Kept as importable aliases so the
# public ``cml.models`` surface is unchanged.
WriteRequest = WriteMemoryRequest
TurnRequest = ProcessTurnRequest
UpdateRequest = UpdateMemoryRequest


class ReadRequest(BaseModel):
    """Read memory request payload."""

    model_config = ConfigDict(populate_by_name=True)

    query: str
    max_results: int = Field(default=10, ge=1, le=50)
    context_filter: list[str] | None = None
    memory_types: list[MemoryType] | None = None
    since: datetime | None = None
    until: datetime | None = None
    response_format: Literal["packet", "list", "llm_context"] = Field(
        default="packet", alias="format"
    )
    user_timezone: str | None = (
        None  # IANA timezone (e.g. "America/New_York") for "today"/"yesterday"
    )
