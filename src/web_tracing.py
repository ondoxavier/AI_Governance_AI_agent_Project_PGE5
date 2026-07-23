"""Tracer bridge that mirrors real span lifecycle events to web clients."""

from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from queue import Queue
from time import perf_counter
from typing import Any, Iterator, Mapping

try:
    from src.observability import Tracer
except ModuleNotFoundError:  # CLI-compatible imports
    from observability import Tracer


class WebTracer:
    """Preserve the existing tracer while publishing its genuine span events."""

    def __init__(self, base: Tracer, events: Queue[dict[str, Any]]):
        self._base = base
        self._events = events
        self.trace_id = base.trace_id
        self.agent_version = base.agent_version
        self.provider = base.provider

    def _emit(self, event_type: str, **payload: Any) -> None:
        self._events.put({
            "type": event_type,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "trace_id": self.trace_id,
            **payload,
        })

    @contextmanager
    def span(self, name: str, metadata: Mapping[str, Any] | None = None) -> Iterator[None]:
        safe_metadata = {
            key: value for key, value in (metadata or {}).items()
            if not any(marker in key.casefold() for marker in ("key", "secret", "token", "password"))
        }
        started = perf_counter()
        self._emit("span.started", name=name, metadata=safe_metadata)
        error_type = None
        try:
            with self._base.span(name, safe_metadata):
                yield
        except Exception as exc:
            error_type = exc.__class__.__name__
            raise
        finally:
            self._emit(
                "span.completed",
                name=name,
                duration_ms=round((perf_counter() - started) * 1000, 3),
                status="failed" if error_type else "completed",
                error_type=error_type,
            )

    def flush(self) -> None:
        self._base.flush()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._base, name)
