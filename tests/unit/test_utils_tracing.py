from __future__ import annotations

import asyncio
import sys
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock

import pytest

import src.utils.tracing as tracing


def _run_coro(coro):
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class _FakeSpan:
    def __init__(self) -> None:
        self.attributes: dict[str, str] = {}
        self.status: tuple[object, str] | None = None
        self.recorded: list[Exception] = []

    def set_attribute(self, key: str, value: str) -> None:
        self.attributes[key] = value

    def set_status(self, code: object, description: str) -> None:
        self.status = (code, description)

    def record_exception(self, exc: Exception) -> None:
        self.recorded.append(exc)


class _FakeTracer:
    def __init__(self, span: _FakeSpan) -> None:
        self.span = span
        self.names: list[str] = []

    def start_as_current_span(self, name: str):
        self.names.append(name)
        span = self.span

        class _ContextManager:
            def __enter__(self):
                return span

            def __exit__(self, exc_type, exc, tb):
                return False

        return _ContextManager()


def _install_fake_otel_sdk(
    monkeypatch: pytest.MonkeyPatch,
    *,
    with_otlp: bool = False,
    broken_provider: bool = False,
) -> None:
    otel_mod = ModuleType("opentelemetry")
    otel_mod.__path__ = []  # type: ignore[attr-defined]
    sdk_mod = ModuleType("opentelemetry.sdk")
    sdk_mod.__path__ = []  # type: ignore[attr-defined]
    resources_mod = ModuleType("opentelemetry.sdk.resources")
    trace_mod = ModuleType("opentelemetry.sdk.trace")
    export_mod = ModuleType("opentelemetry.sdk.trace.export")

    class Resource:
        @staticmethod
        def create(attributes: dict[str, str]) -> dict[str, str]:
            return attributes

    class TracerProvider:
        def __init__(self, resource):
            if broken_provider:
                raise RuntimeError("provider boom")
            self.resource = resource
            self.processors: list[object] = []

        def add_span_processor(self, processor) -> None:
            self.processors.append(processor)

    class BatchSpanProcessor:
        def __init__(self, exporter) -> None:
            self.exporter = exporter

    class SpanExporter:
        pass

    class ConsoleSpanExporter:
        pass

    resources_mod.Resource = Resource
    trace_mod.TracerProvider = TracerProvider
    export_mod.BatchSpanProcessor = BatchSpanProcessor
    export_mod.SpanExporter = SpanExporter
    export_mod.ConsoleSpanExporter = ConsoleSpanExporter

    monkeypatch.setitem(sys.modules, "opentelemetry", otel_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk", sdk_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.resources", resources_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace", trace_mod)
    monkeypatch.setitem(sys.modules, "opentelemetry.sdk.trace.export", export_mod)

    if with_otlp:
        exporter_mod = ModuleType("opentelemetry.exporter")
        exporter_mod.__path__ = []  # type: ignore[attr-defined]
        otlp_mod = ModuleType("opentelemetry.exporter.otlp")
        otlp_mod.__path__ = []  # type: ignore[attr-defined]
        proto_mod = ModuleType("opentelemetry.exporter.otlp.proto")
        proto_mod.__path__ = []  # type: ignore[attr-defined]
        grpc_mod = ModuleType("opentelemetry.exporter.otlp.proto.grpc")
        grpc_mod.__path__ = []  # type: ignore[attr-defined]
        trace_exporter_mod = ModuleType("opentelemetry.exporter.otlp.proto.grpc.trace_exporter")

        class OTLPSpanExporter:
            pass

        trace_exporter_mod.OTLPSpanExporter = OTLPSpanExporter
        monkeypatch.setitem(sys.modules, "opentelemetry.exporter", exporter_mod)
        monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp", otlp_mod)
        monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto", proto_mod)
        monkeypatch.setitem(sys.modules, "opentelemetry.exporter.otlp.proto.grpc", grpc_mod)
        monkeypatch.setitem(
            sys.modules,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
            trace_exporter_mod,
        )
    else:
        monkeypatch.delitem(
            sys.modules,
            "opentelemetry.exporter.otlp.proto.grpc.trace_exporter",
            raising=False,
        )


def test_is_tracing_enabled_reflects_module_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    assert tracing.is_tracing_enabled() is True
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    assert tracing.is_tracing_enabled() is False


def test_trace_span_noops_without_otel(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(tracing, "_tracer", None)

    with tracing.trace_span("memory.write", tenant_id="tenant-a") as span:
        assert span is None

    async def _run() -> None:
        async with tracing.async_trace_span("memory.write", tenant_id="tenant-a") as span:
            assert span is None

    _run_coro(_run())


def test_trace_span_sets_attributes_and_records_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    tracer = _FakeTracer(span)
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_tracer", tracer)
    monkeypatch.setattr(tracing, "StatusCode", SimpleNamespace(ERROR="ERROR"))

    with tracing.trace_span("memory.write", tenant_id="tenant-a", attempts=2) as current:
        assert current is span

    assert tracer.names == ["memory.write"]
    assert span.attributes == {"tenant_id": "tenant-a", "attempts": "2"}

    with (
        pytest.raises(RuntimeError, match="boom"),
        tracing.trace_span("memory.write", tenant_id="tenant-a"),
    ):
        raise RuntimeError("boom")

    assert span.status == ("ERROR", "boom")
    assert isinstance(span.recorded[0], RuntimeError)


@pytest.mark.asyncio
async def test_async_trace_span_sets_attributes_and_records_exceptions(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    span = _FakeSpan()
    tracer = _FakeTracer(span)
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "_tracer", tracer)
    monkeypatch.setattr(tracing, "StatusCode", SimpleNamespace(ERROR="ERROR"))

    async with tracing.async_trace_span("memory.read", tenant_id="tenant-a") as current:
        assert current is span

    assert tracer.names == ["memory.read"]
    assert span.attributes == {"tenant_id": "tenant-a"}

    with pytest.raises(RuntimeError, match="async boom"):
        async with tracing.async_trace_span("memory.read", tenant_id="tenant-a"):
            raise RuntimeError("async boom")

    assert span.status == ("ERROR", "async boom")
    assert isinstance(span.recorded[0], RuntimeError)


def test_configure_tracing_logs_when_otel_missing(monkeypatch: pytest.MonkeyPatch) -> None:
    logger = MagicMock()
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", False)
    monkeypatch.setattr(tracing, "logger", logger)

    tracing.configure_tracing("svc-a")

    logger.info.assert_called_once()


def test_configure_tracing_logs_when_trace_api_is_unavailable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    logger = MagicMock()
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "trace", None)
    monkeypatch.setattr(tracing, "logger", logger)

    tracing.configure_tracing("svc-b")

    logger.warning.assert_called_once_with("otel_trace_unavailable")


def test_configure_tracing_uses_console_exporter_when_otlp_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_api = SimpleNamespace(set_tracer_provider=MagicMock())
    logger = MagicMock()
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "trace", trace_api)
    monkeypatch.setattr(tracing, "logger", logger)
    _install_fake_otel_sdk(monkeypatch, with_otlp=False)

    tracing.configure_tracing("svc-c")

    provider = trace_api.set_tracer_provider.call_args.args[0]
    assert provider.resource == {"service.name": "svc-c"}
    assert provider.processors[0].exporter.__class__.__name__ == "ConsoleSpanExporter"
    logger.info.assert_called_once()


def test_configure_tracing_uses_otlp_exporter_when_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_api = SimpleNamespace(set_tracer_provider=MagicMock())
    logger = MagicMock()
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "trace", trace_api)
    monkeypatch.setattr(tracing, "logger", logger)
    _install_fake_otel_sdk(monkeypatch, with_otlp=True)

    tracing.configure_tracing("svc-d")

    provider = trace_api.set_tracer_provider.call_args.args[0]
    assert provider.resource == {"service.name": "svc-d"}
    assert provider.processors[0].exporter.__class__.__name__ == "OTLPSpanExporter"
    logger.info.assert_called_once()


def test_configure_tracing_logs_warning_when_sdk_setup_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    trace_api = SimpleNamespace(set_tracer_provider=MagicMock())
    logger = MagicMock()
    monkeypatch.setattr(tracing, "_OTEL_AVAILABLE", True)
    monkeypatch.setattr(tracing, "trace", trace_api)
    monkeypatch.setattr(tracing, "logger", logger)
    _install_fake_otel_sdk(monkeypatch, broken_provider=True)

    tracing.configure_tracing("svc-e")

    logger.warning.assert_called_once()
