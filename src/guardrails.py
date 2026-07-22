"""Guardrails L1/L4 and token budget utilities."""

from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


class SecurityError(ValueError):
    """Raised when a guardrail blocks an input or action."""


INJECTION_PATTERNS = [
    r"\bignore\s+(all\s+)?(previous|above|prior)\s+instructions\b",
    r"\boublie\s+(toutes\s+)?(les\s+)?instructions\b",
    r"\br[eé]v[eè]le\s+(le\s+)?(prompt|syst[eè]me|secret|token|cl[eé])\b",
    r"\bdeveloper\s+message\b",
    r"\bsystem\s+prompt\b",
    r"\bexfiltrat(e|ion)\b",
    r"\bapi[_ -]?key\b",
    r"\bdelete\s+all\b",
    r"\brm\s+-rf\b",
]


ACTION_RISK_MATRIX = {
    "hybrid_search": {"risk": "low", "requires_approval": False},
    "classify_ai_act_risk": {"risk": "low", "requires_approval": False},
    "security_screen": {"risk": "low", "requires_approval": False},
    "read_local_corpus": {"risk": "low", "requires_approval": False},
    "external_request": {"risk": "medium", "requires_approval": True},
    "write_file": {"risk": "high", "requires_approval": True},
    "send_email": {"risk": "high", "requires_approval": True},
    "delete_data": {"risk": "critical", "requires_approval": True},
}


def normalize_text(text: str) -> str:
    """Normalize user text before security checks."""
    return unicodedata.normalize("NFKC", text or "").casefold()


def l1_filter(user_input: str) -> str:
    """Apply L1 input filtering and return normalized text.

    Raises:
        SecurityError: when a known prompt-injection pattern is detected.
    """
    normalized = normalize_text(user_input)
    for pattern in INJECTION_PATTERNS:
        if re.search(pattern, normalized, flags=re.IGNORECASE):
            raise SecurityError(f"Entrée bloquée par le filtre L1: {pattern}")
    return normalized


def authorize_action(action_name: str, approved: bool = False) -> bool:
    """Authorize an action with the L4 risk matrix."""
    action = ACTION_RISK_MATRIX.get(action_name)
    if action is None:
        raise SecurityError(f"Action inconnue refusée par L4: {action_name}")
    if action["requires_approval"] and not approved:
        raise SecurityError(f"Action {action_name} requiert une approbation humaine")
    return True


@dataclass
class TokenBudget:
    """Simple token budget approximation used before retrieval and synthesis."""

    max_tokens: int = 3500
    used_tokens: int = 0

    def estimate(self, text: str) -> int:
        return max(1, len((text or "").split()))

    def consume(self, text: str) -> None:
        tokens = self.estimate(text)
        if self.used_tokens + tokens > self.max_tokens:
            raise SecurityError(
                f"Budget de tokens dépassé: {self.used_tokens + tokens}/{self.max_tokens}"
            )
        self.used_tokens += tokens

    @property
    def remaining(self) -> int:
        return max(0, self.max_tokens - self.used_tokens)
