"""Shared LLM client for the reasoning layer (DeepInfra, OpenAI-compatible API).

Two model tiers, following the model_pro/model_flash pattern:
    - synthesis model: fast/cheap, called k times per question (self-consistency)
    - critic model: stronger, called once per question (anti-hallucination review)

Returns None everywhere when the package or the API key is missing so callers
can fall back to the deterministic reasoning path — the project must keep
running from a fresh clone without any LLM key configured.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover - optional dependency
    pass

DEFAULT_BASE_URL = "https://api.deepinfra.com/v1/openai"
DEFAULT_SYNTHESIS_MODEL = "meta-llama/Meta-Llama-3.1-8B-Instruct"
DEFAULT_CRITIC_MODEL = "meta-llama/Llama-3.3-70B-Instruct"

# Indicative DeepInfra pricing, USD per million tokens (input, output).
# Source: third-party trackers (pricepertoken.com, aipricing.guru), July 2026 --
# verify against https://deepinfra.com/pricing before quoting exact figures.
PRICING_USD_PER_M_TOKENS = {
    "meta-llama/Meta-Llama-3.1-8B-Instruct": (0.05, 0.08),
    "meta-llama/Llama-3.3-70B-Instruct": (0.23, 0.40),
}

# Cumulative token usage across every chat() call in this process, keyed by
# model. Reset per evaluation run via reset_usage() so cost figures reflect
# one run, not the whole process lifetime.
_usage_by_model: dict[str, dict[str, int]] = {}


@lru_cache(maxsize=1)
def get_client() -> Any | None:
    """Build a single shared OpenAI-compatible client pointed at DeepInfra.

    Cached: the underlying HTTP client is reused across all synthesis and
    critic calls in a run instead of reconnecting each time.
    """
    api_key = os.getenv("DEEPINFRA_API_KEY")
    if not api_key:
        return None
    try:
        from openai import OpenAI
    except ImportError:
        return None
    base_url = os.getenv("LLM_BASE_URL", DEFAULT_BASE_URL)
    try:
        return OpenAI(api_key=api_key, base_url=base_url)
    except Exception:
        return None


def get_synthesis_model() -> str:
    return os.getenv("LLM_MODEL_SYNTHESIS", DEFAULT_SYNTHESIS_MODEL)


def get_critic_model() -> str:
    return os.getenv("LLM_MODEL_CRITIC", DEFAULT_CRITIC_MODEL)


def is_available() -> bool:
    return get_client() is not None


def chat(
    system_prompt: str,
    user_prompt: str,
    *,
    model: str,
    temperature: float = 0.7,
    max_tokens: int = 700,
) -> str | None:
    """Single chat completion; returns None on any client/network/API failure
    so the caller can fall back to the deterministic path instead of crashing."""
    client = get_client()
    if client is None:
        return None
    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = getattr(response, "usage", None)
        if usage is not None:
            entry = _usage_by_model.setdefault(model, {"prompt_tokens": 0, "completion_tokens": 0})
            entry["prompt_tokens"] += getattr(usage, "prompt_tokens", 0) or 0
            entry["completion_tokens"] += getattr(usage, "completion_tokens", 0) or 0
        return response.choices[0].message.content
    except Exception:
        return None


def reset_usage() -> None:
    """Clear accumulated token usage — call before measuring a single run's cost."""
    _usage_by_model.clear()


def get_usage_totals() -> dict[str, dict[str, int]]:
    """Return {model: {prompt_tokens, completion_tokens}} accumulated since the
    last reset_usage() call."""
    return {model: dict(counts) for model, counts in _usage_by_model.items()}


def estimate_cost_usd() -> tuple[float, list[str]]:
    """Estimate USD cost from accumulated usage and the indicative pricing table.

    Returns (total_cost, notes) where notes lists any model missing from the
    pricing table (cost for that model is then excluded, not guessed at).
    """
    total = 0.0
    notes: list[str] = []
    for model, counts in _usage_by_model.items():
        pricing = PRICING_USD_PER_M_TOKENS.get(model)
        if pricing is None:
            notes.append(f"Tarif inconnu pour {model}: cout exclu de l'estimation.")
            continue
        price_in, price_out = pricing
        total += (
            counts["prompt_tokens"] * price_in + counts["completion_tokens"] * price_out
        ) / 1_000_000
    return round(total, 6), notes


def reset_cache() -> None:
    """Test helper: clear the cached client (e.g. after monkeypatching env vars)."""
    get_client.cache_clear()
