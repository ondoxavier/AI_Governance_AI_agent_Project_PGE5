"""Runtime settings used by guardrails.

The values are intentionally conservative defaults so local tests and demos do
not depend on an external configuration system.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Settings:
    max_input_chars: int = 4_000
    max_llm_calls: int = 8
    max_tool_calls: int = 20
    max_estimated_tokens: int = 40_000


settings = Settings()
