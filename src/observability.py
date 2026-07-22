"""Local observability helpers with Langfuse-style span names."""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import asdict, dataclass
from pathlib import Path
import json
import os
import time
from typing import Iterator


@dataclass(frozen=True)
class SpanRecord:
    name: str
    duration_s: float
    status: str
    agent_version: str
    metadata: dict[str, str | int | float]


class ObservabilityTracer:
    """Collect spans locally and export them as JSONL."""

    def __init__(self, verbose: bool = True, agent_version: str | None = None) -> None:
        self.verbose = verbose
        self.agent_version = agent_version or os.getenv("AGENT_VERSION", "local-dev")
        self.records: list[SpanRecord] = []

    @contextmanager
    def span(self, name: str, **metadata: str | int | float) -> Iterator[None]:
        start = time.perf_counter()
        if self.verbose:
            print(f"[span:start] {name} version={self.agent_version}")
        status = "ok"
        try:
            yield
        except Exception:
            status = "error"
            raise
        finally:
            duration = time.perf_counter() - start
            self.records.append(
                SpanRecord(
                    name=name,
                    duration_s=round(duration, 6),
                    status=status,
                    agent_version=self.agent_version,
                    metadata=dict(metadata),
                )
            )
            if self.verbose:
                print(f"[span:end] {name} duration_s={duration:.3f}")

    def count_by_name(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            counts[record.name] = counts.get(record.name, 0) + 1
        return counts

    def count_tools(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for record in self.records:
            if record.name.startswith("tool."):
                counts[record.name] = counts.get(record.name, 0) + 1
        return counts

    def export_jsonl(self, path: str | Path) -> Path:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
        return output
