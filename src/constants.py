"""Small shared constants for agent and MCP modules.

Also loads .env once here, since this module is imported first by
agent.py, reasoning.py, mcp_server.py, ingest.py and retrieval.py — every
os.getenv() call in the project (DEEPINFRA_API_KEY, LANGFUSE_*,
AGENT_VERSION...) depends on this having run.
"""

from __future__ import annotations

from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).resolve().parents[1] / ".env")
except ImportError:  # pragma: no cover - optional dependency
    pass


DISCLAIMER = (
    "Cette analyse est generee par IA et doit etre validee par un juriste "
    "avant toute decision."
)

UNKNOWN_DATE = "date non renseignée dans le corpus"
