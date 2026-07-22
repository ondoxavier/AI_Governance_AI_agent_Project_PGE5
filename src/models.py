"""Shared lightweight models for guardrail decisions."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class FilterResult:
    allowed: bool
    normalized_text: str = ""
    reason: str = ""
    reasons: list[str] = field(default_factory=list)
    risk_score: float = 0.0


@dataclass(frozen=True)
class GateDecision:
    allowed: bool
    reason: str = ""
