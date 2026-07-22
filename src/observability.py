"""Tracing helpers with Langfuse support and a local fallback.

The project must run from a fresh clone without external services. This module
therefore attempts to use Langfuse only when credentials and the package are
available; otherwise it records the same logical spans locally and prints short
duration lines for demos and tests.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
import os
import socket
from time import perf_counter
from typing import Any, Iterator, Mapping
from urllib.parse import urlparse
from uuid import uuid4


SENSITIVE_KEYS = {"key", "secret", "token", "password", "authorization"}


def _safe_metadata(metadata: Mapping[str, Any] | None) -> dict[str, Any]:
    """Return metadata safe for tracing by dropping obvious secret fields."""
    safe: dict[str, Any] = {}
    for key, value in (metadata or {}).items():
        lowered = key.casefold()
        if any(marker in lowered for marker in SENSITIVE_KEYS):
            continue
        if isinstance(value, str) and len(value) > 500:
            safe[key] = value[:500] + "...[truncated]"
        else:
            safe[key] = value
    return safe


@dataclass
class TraceEvent:
    """Local record of one completed span."""

    name: str
    duration_ms: float
    metadata: dict[str, Any] = field(default_factory=dict)
    error_type: str | None = None


class Tracer:
    """Span recorder backed by Langfuse when configured, otherwise local logs."""

    def __init__(self, trace_name: str = "agent.run", metadata: Mapping[str, Any] | None = None):
        self.agent_version = os.getenv("AGENT_VERSION", "local-dev") or "local-dev"
        self.trace_id = str(uuid4())
        self.provider = "local"
        self.events: list[TraceEvent] = []
        self._client = None
        self._trace = None
        self._otel_tracer = None

        base_metadata = {
            "agent_version": self.agent_version,
            **_safe_metadata(metadata),
        }
        self._configure_langfuse(trace_name, base_metadata)

    def _configure_langfuse(self, trace_name: str, metadata: Mapping[str, Any]) -> None:
        public_key = os.getenv("LANGFUSE_PUBLIC_KEY")
        secret_key = os.getenv("LANGFUSE_SECRET_KEY")
        host = os.getenv("LANGFUSE_HOST") or "https://cloud.langfuse.com"
        if not public_key or not secret_key:
            return
        if not _host_reachable(host):
            self.events.append(
                TraceEvent(
                    name="observability.langfuse_unreachable",
                    duration_ms=0.0,
                    metadata={"host": host},
                    error_type="LangfuseUnreachable",
                )
            )
            return

        try:
            from langfuse import Langfuse  # type: ignore

            kwargs: dict[str, str] = {
                "public_key": public_key,
                "secret_key": secret_key,
            }
            kwargs["host"] = host
            self._client = Langfuse(**kwargs)
            if hasattr(self._client, "trace"):
                self._trace = self._client.trace(name=trace_name, metadata=dict(metadata))
                self.trace_id = str(getattr(self._trace, "id", self.trace_id))
            else:
                # Langfuse v4 exposes an OpenTelemetry tracer instead of the
                # older trace/span helper API. We use it directly so configured
                # deployments still emit real spans.
                self._otel_tracer = getattr(self._client, "_otel_tracer", None)
                create_trace_id = getattr(self._client, "create_trace_id", None)
                if callable(create_trace_id):
                    self.trace_id = str(create_trace_id())
            self.provider = "langfuse"
        except Exception as exc:  # pragma: no cover - depends on optional package/config
            self.events.append(
                TraceEvent(
                    name="observability.langfuse_unavailable",
                    duration_ms=0.0,
                    metadata={"error_type": exc.__class__.__name__},
                    error_type=exc.__class__.__name__,
                )
            )

    @contextmanager
    def span(self, name: str, metadata: Mapping[str, Any] | None = None) -> Iterator[None]:
        """Record a duration span and never let tracing break the pipeline."""
        safe = {
            "agent_version": self.agent_version,
            **_safe_metadata(metadata),
        }
        start = perf_counter()
        remote_span = self._start_remote_span(name, safe)
        print(f"[span:start] {name} provider={self.provider} version={self.agent_version}")
        error_type: str | None = None
        try:
            yield
        except Exception as exc:
            error_type = exc.__class__.__name__
            raise
        finally:
            duration_ms = (perf_counter() - start) * 1000
            event_metadata = {**safe, "latency_ms": round(duration_ms, 3)}
            if error_type:
                event_metadata["error_type"] = error_type
            self.events.append(TraceEvent(name, duration_ms, event_metadata, error_type))
            self._end_remote_span(remote_span, event_metadata)
            print(f"[span:end] {name} duration_ms={duration_ms:.1f}")

    def _start_remote_span(self, name: str, metadata: Mapping[str, Any]):
        if self._trace is None:
            if self._otel_tracer is None:
                return None
            try:  # pragma: no cover - depends on Langfuse runtime
                context_manager = self._otel_tracer.start_as_current_span(
                    name,
                    attributes=_otel_attributes(metadata),
                )
                span = context_manager.__enter__()
                return ("otel", context_manager, span)
            except Exception:
                self.provider = "local"
                self._otel_tracer = None
                return None
        try:  # pragma: no cover - depends on Langfuse runtime
            return self._trace.span(name=name, metadata=dict(metadata))
        except Exception:
            self.provider = "local"
            self._trace = None
            return None

    def _end_remote_span(self, remote_span: Any, metadata: Mapping[str, Any]) -> None:
        if remote_span is None:
            return
        if isinstance(remote_span, tuple) and remote_span and remote_span[0] == "otel":
            _, context_manager, span = remote_span
            try:  # pragma: no cover - depends on Langfuse runtime
                for key, value in _otel_attributes(metadata).items():
                    span.set_attribute(key, value)
                context_manager.__exit__(None, None, None)
            except Exception:
                self.provider = "local"
            return
        try:  # pragma: no cover - depends on Langfuse runtime
            remote_span.end(metadata=dict(metadata))
        except Exception:
            self.provider = "local"

    def flush(self) -> None:
        """Flush Langfuse events when the optional client supports it."""
        if self._client is None:
            return
        flush = getattr(self._client, "flush", None)
        if callable(flush):
            try:  # pragma: no cover - depends on Langfuse runtime
                flush()
            except Exception:
                self.provider = "local"


def create_tracer(trace_name: str = "agent.run", metadata: Mapping[str, Any] | None = None) -> Tracer:
    """Factory used by agent and MCP tools to share the same fallback behavior."""
    return Tracer(trace_name=trace_name, metadata=metadata)


def _otel_attributes(metadata: Mapping[str, Any]) -> dict[str, str | bool | int | float]:
    """Convert metadata to OpenTelemetry-safe scalar attributes."""
    attributes: dict[str, str | bool | int | float] = {}
    for key, value in metadata.items():
        if isinstance(value, (str, bool, int, float)):
            attributes[key] = value
        elif value is None:
            attributes[key] = ""
        else:
            attributes[key] = str(value)
    return attributes


def _host_reachable(host: str, timeout_s: float = 0.75) -> bool:
    """Check the tracing endpoint before enabling noisy async exporters."""
    parsed = urlparse(host if "://" in host else f"https://{host}")
    hostname = parsed.hostname
    if not hostname:
        return False
    port = parsed.port or (443 if parsed.scheme == "https" else 80)
    try:
        with socket.create_connection((hostname, port), timeout=timeout_s):
            return True
    except OSError:
        return False
