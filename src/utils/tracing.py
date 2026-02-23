"""OpenTelemetry tracing integration (A-07).

Provides lightweight tracing helpers that work when OpenTelemetry is installed,
and silently no-op when it's not. This avoids hard-coding a dependency while
enabling distributed tracing for production deployments.

Usage:
    from src.utils.tracing import trace_span

    async with trace_span("memory.write", tenant_id=tenant_id):
        result = await orchestrator.write(...)
"""

from __future__ import annotations

from contextlib import asynccontextmanager, contextmanager
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

# Attempt to import OpenTelemetry; fall back to no-ops if not installed
try:
    from opentelemetry import trace
    from opentelemetry.trace import StatusCode

    _tracer = trace.get_tracer("cognitive-memory-layer")
    _OTEL_AVAILABLE = True
except ImportError:
    _OTEL_AVAILABLE = False
    _tracer = None  # type: ignore[assignment]
    StatusCode = None  # type: ignore[assignment,misc]


def is_tracing_enabled() -> bool:
    """True when OpenTelemetry SDK is installed and configured."""
    return _OTEL_AVAILABLE


@contextmanager
def trace_span(name: str, **attributes: Any):
    """Synchronous context manager for tracing a code block.

    When OpenTelemetry is not installed, this is a no-op.
    """
    if not _OTEL_AVAILABLE or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attributes.items():
            span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


@asynccontextmanager
async def async_trace_span(name: str, **attributes: Any):
    """Async context manager for tracing a code block.

    When OpenTelemetry is not installed, this is a no-op.
    """
    if not _OTEL_AVAILABLE or _tracer is None:
        yield None
        return
    with _tracer.start_as_current_span(name) as span:
        for k, v in attributes.items():
            span.set_attribute(k, str(v))
        try:
            yield span
        except Exception as exc:
            span.set_status(StatusCode.ERROR, str(exc))
            span.record_exception(exc)
            raise


def configure_tracing(service_name: str = "cognitive-memory-layer") -> None:
    """Configure OpenTelemetry tracing with sensible defaults.

    Call this once at application startup (e.g., in lifespan).
    Requires ``opentelemetry-sdk`` and an exporter to be installed.
    """
    if not _OTEL_AVAILABLE:
        logger.info("otel_not_installed", msg="OpenTelemetry not installed; tracing disabled")
        return

    try:
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor, SpanExporter

        # Try OTLP exporter first, fall back to console
        exporter: SpanExporter
        try:
            from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

            exporter = OTLPSpanExporter()
        except ImportError:
            from opentelemetry.sdk.trace.export import ConsoleSpanExporter

            exporter = ConsoleSpanExporter()

        provider = TracerProvider(
            resource=Resource.create({"service.name": service_name}),
        )
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info(
            "otel_configured",
            service_name=service_name,
            exporter=type(exporter).__name__,
        )
    except Exception as e:
        logger.warning("otel_configure_failed", error=str(e))
