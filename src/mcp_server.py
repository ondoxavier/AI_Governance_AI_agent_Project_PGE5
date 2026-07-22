"""MCP server exposing the AI governance project tools.

The module stays importable without the optional MCP package so tests and the
local agent can run from a fresh clone. Tool functions are plain Python
callables first, then registered with FastMCP when the package is available.
"""

from __future__ import annotations

from collections import Counter
import os
from pathlib import Path
import re
from time import perf_counter
from typing import Any

from constants import DISCLAIMER
from guardrails import SecurityError, authorize_action, l1_filter
from observability import create_tracer
from reasoning import classify_ai_act_risk
from retrieval import SearchResult, hybrid_search as run_hybrid_search

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional production dependency
    FastMCP = None


VALID_JURISDICTIONS = {"EU", "US", "UK"}
MAX_TOP_K = 10
DEFAULT_TOP_K = 4
COMPARE_TOP_K = 3
TEXT_SNIPPET_LIMIT = 900
DEFAULT_DATA_DIR = Path(os.getenv("AI_GOVERNANCE_DATA_DIR", "data"))


def _compact_text(value: str) -> str:
    return re.sub(r"\s+", " ", value or "").strip()


def _validate_query(query: str, *, field_name: str = "query") -> str:
    """Validate user-provided search text without adding heavy dependencies."""
    if not isinstance(query, str):
        raise SecurityError(f"{field_name} doit etre une chaine de caracteres.")
    cleaned = _compact_text(query)
    if len(cleaned) < 3:
        raise SecurityError(f"{field_name} est trop court.")
    if len(cleaned) > 1000:
        raise SecurityError(f"{field_name} est trop long (max 1000 caracteres).")
    l1_filter(cleaned)
    return cleaned


def _validate_top_k(top_k: int) -> int:
    try:
        value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise SecurityError("top_k doit etre un entier.") from exc
    if value < 1:
        raise SecurityError("top_k doit etre superieur ou egal a 1.")
    return min(value, MAX_TOP_K)


def _validate_jurisdiction(jurisdiction: str | None) -> str | None:
    if jurisdiction is None:
        return None
    value = _compact_text(str(jurisdiction))
    if not value or value.casefold() == "all":
        return None
    upper = value.upper()
    if upper not in VALID_JURISDICTIONS:
        raise SecurityError(f"Juridiction non reconnue: {jurisdiction}")
    return upper


def _result_to_dict(result: SearchResult) -> dict[str, Any]:
    """Serialize a SearchResult while preserving legal metadata."""
    text = result.document.text
    if len(text) > TEXT_SNIPPET_LIMIT:
        text = text[:TEXT_SNIPPET_LIMIT] + "...[truncated]"
    return {
        "title": result.document.title,
        "text": text,
        "source": result.document.source,
        "jurisdiction": result.document.jurisdiction,
        "status": result.document.status,
        "date": result.document.date,
        "score": round(float(result.score), 4),
        "method": result.method,
    }


def _status_summary(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    counts = Counter(str(item.get("status") or "inconnu") for item in results)
    return [
        {"status": status, "count": count}
        for status, count in sorted(counts.items(), key=lambda item: item[0])
    ]


def _empty_jurisdiction_block(error: str | None = None) -> dict[str, Any]:
    block: dict[str, Any] = {"status_summary": [], "results": []}
    if error:
        block["error"] = error
    return block


def hybrid_search(query: str, top_k: int = DEFAULT_TOP_K, jurisdiction: str | None = None) -> dict[str, Any]:
    """Search the governance corpus with hybrid retrieval.

    Use when:
        The caller needs source passages before a compliance or governance
        analysis.
    Do NOT use:
        To execute external actions, modify data, or provide final legal advice.
    Args:
        query: Regulatory question or AI use case.
        top_k: Maximum number of passages to return, capped by the server.
        jurisdiction: Optional EU, US, UK or all filter.
    Returns:
        JSON-serializable dictionary with retrieved passages. Each passage keeps
        title, text, source, jurisdiction, status, score and retrieval method.
    Example:
        hybrid_search("AI credit scoring obligations", top_k=3, jurisdiction="EU")
    """
    tracer = create_tracer("mcp.tool.hybrid_search", {"tool_name": "hybrid_search"})
    start = perf_counter()
    try:
        clean_query = _validate_query(query)
        safe_top_k = _validate_top_k(top_k)
        safe_jurisdiction = _validate_jurisdiction(jurisdiction)
        with tracer.span(
            "mcp.tool.hybrid_search",
            {"tool_name": "hybrid_search", "jurisdiction": safe_jurisdiction or "all", "top_k": safe_top_k},
        ):
            authorize_action("hybrid_search")
            results = run_hybrid_search(
                clean_query,
                top_k=safe_top_k,
                data_dir=DEFAULT_DATA_DIR,
                jurisdiction=safe_jurisdiction,
            )
        serialized = [_result_to_dict(result) for result in results]
        return {
            "ok": True,
            "query": clean_query,
            "jurisdiction": safe_jurisdiction or "all",
            "results": serialized,
            "warnings": [] if serialized else ["Aucun resultat retourne."],
            "disclaimer": DISCLAIMER,
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "results": [],
            "warnings": ["Recherche interrompue par une erreur controlee."],
            "disclaimer": DISCLAIMER,
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    finally:
        try:
            tracer.flush()
        except Exception:
            pass


def classify_ai_act_tool(query: str) -> dict[str, Any]:
    """Classify the likely AI Act risk level from retrieved evidence.

    Use when:
        The caller needs a first-pass AI Act risk qualification grounded in the
        corpus.
    Do NOT use:
        To replace legal review, decide rights automatically, or classify
        jurisdictions outside the indexed corpus.
    Args:
        query: AI system description or governance question.
    Returns:
        Risk label, evidence count, preserved sources and a legal-review
        disclaimer.
    Example:
        classify_ai_act_risk("AI system used for recruitment CV screening")
    """
    tracer = create_tracer("mcp.tool.classify_ai_act_risk", {"tool_name": "classify_ai_act_risk"})
    start = perf_counter()
    try:
        clean_query = _validate_query(query)
        with tracer.span("mcp.tool.classify_ai_act_risk", {"tool_name": "classify_ai_act_risk"}):
            authorize_action("classify_ai_act_risk")
            contexts = run_hybrid_search(
                clean_query,
                top_k=DEFAULT_TOP_K,
                data_dir=DEFAULT_DATA_DIR,
                jurisdiction="EU",
            )
            risk = classify_ai_act_risk(clean_query, contexts)
        return {
            "ok": True,
            "risk": risk,
            "evidence_count": len(contexts),
            "sources": [_result_to_dict(result) for result in contexts],
            "disclaimer": DISCLAIMER,
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    except Exception as exc:
        return {
            "ok": False,
            "error": f"{exc.__class__.__name__}: {exc}",
            "disclaimer": DISCLAIMER,
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    finally:
        try:
            tracer.flush()
        except Exception:
            pass


def security_screen(text: str) -> dict[str, Any]:
    """Run L1 input security screening.

    Use when:
        A caller wants to check text before tool execution or LLM reasoning.
    Do NOT use:
        To validate factual accuracy or legal correctness.
    Args:
        text: User input or external text to screen.
    Returns:
        Whether the text passed L1 and the normalized safe form when available.
    Example:
        security_screen("ignore previous instructions")
    """
    tracer = create_tracer("mcp.tool.security_screen", {"tool_name": "security_screen"})
    start = perf_counter()
    try:
        if not isinstance(text, str):
            raise SecurityError("text doit etre une chaine de caracteres.")
        with tracer.span("mcp.tool.security_screen", {"tool_name": "security_screen"}):
            authorize_action("security_screen")
            normalized = l1_filter(text)
        return {
            "ok": True,
            "normalized": normalized,
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    except SecurityError as exc:
        return {
            "ok": False,
            "error": str(exc),
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    finally:
        try:
            tracer.flush()
        except Exception:
            pass


def compare_jurisdiction(
    topic: str,
    top_k: int = COMPARE_TOP_K,
) -> dict[str, Any]:
    """Compare regulatory evidence across EU, US and UK.

    Use when:
        The caller needs a jurisdiction-by-jurisdiction comparison based on
        documents contained in the indexed regulatory corpus.
    Do NOT use:
        To provide definitive legal advice, make an automated legal decision,
        merge binding and voluntary statuses into one generic statement, or
        compare jurisdictions that are not present in the corpus.
    Args:
        topic: Regulatory topic or AI use case to search for.
        top_k: Number of passages per jurisdiction, capped by the server.
    Returns:
        JSON-serializable dictionary containing distinct EU, US and UK evidence
        blocks. Every retrieved item preserves its source, jurisdiction and
        regulatory status. Partial jurisdiction failures are returned in that
        jurisdiction block without deleting other results.
    Example:
        compare_jurisdiction(
            "AI system used to prescreen consumer credit applications"
        )

    Limits:
        This tool retrieves and structures evidence only. Human legal validation
        is required before any decision.
    """
    tracer = create_tracer("mcp.tool.compare_jurisdiction", {"tool_name": "compare_jurisdiction"})
    start = perf_counter()
    warnings: list[str] = []
    jurisdiction_blocks: dict[str, Any] = {
        "EU": _empty_jurisdiction_block(),
        "US": _empty_jurisdiction_block(),
        "UK": _empty_jurisdiction_block(),
    }

    try:
        try:
            clean_topic = _validate_query(topic, field_name="topic")
            safe_top_k = _validate_top_k(top_k)
            authorize_action("compare_jurisdiction")
        except Exception as exc:
            with tracer.span("mcp.tool.compare_jurisdiction.refused", {"error_type": exc.__class__.__name__}):
                return {
                    "ok": False,
                    "topic": topic if isinstance(topic, str) else "",
                    "jurisdictions": jurisdiction_blocks,
                    "warnings": [f"Comparaison refusee: {exc.__class__.__name__}: {exc}"],
                    "disclaimer": DISCLAIMER,
                    "latency_ms": round((perf_counter() - start) * 1000, 3),
                }

        with tracer.span("mcp.tool.compare_jurisdiction", {"top_k": safe_top_k}):
            # Keep the three calls explicit and sequential: shared model caches in
            # retrieval.py are not guaranteed to be thread-safe.
            for jurisdiction in ("EU", "US", "UK"):
                with tracer.span(
                    f"mcp.tool.compare_jurisdiction.{jurisdiction}",
                    {"jurisdiction": jurisdiction, "top_k": safe_top_k},
                ):
                    try:
                        results = run_hybrid_search(
                            query=clean_topic,
                            top_k=safe_top_k,
                            data_dir=DEFAULT_DATA_DIR,
                            jurisdiction=jurisdiction,
                        )
                        serialized = [_result_to_dict(result) for result in results]
                        jurisdiction_blocks[jurisdiction] = {
                            "status_summary": _status_summary(serialized),
                            "results": serialized,
                        }
                        if not serialized:
                            warnings.append(f"Aucun resultat pour {jurisdiction}.")
                    except Exception as exc:
                        message = f"{exc.__class__.__name__}: {exc}"
                        jurisdiction_blocks[jurisdiction] = _empty_jurisdiction_block(error=message)
                        warnings.append(f"Erreur retrieval pour {jurisdiction}: {message}")

        return {
            "ok": not all(not block["results"] for block in jurisdiction_blocks.values()),
            "topic": clean_topic,
            "jurisdictions": jurisdiction_blocks,
            "warnings": warnings,
            "disclaimer": DISCLAIMER,
            "latency_ms": round((perf_counter() - start) * 1000, 3),
        }
    finally:
        try:
            tracer.flush()
        except Exception:
            pass


MCP_TOOL_REGISTRY = {
    "hybrid_search": hybrid_search,
    "classify_ai_act_risk": classify_ai_act_tool,
    "security_screen": security_screen,
    "compare_jurisdiction": compare_jurisdiction,
}


if FastMCP is not None:
    mcp = FastMCP("ai-governance-agent")
    mcp.tool()(hybrid_search)
    mcp.tool(name="classify_ai_act_risk")(classify_ai_act_tool)
    mcp.tool()(security_screen)
    mcp.tool()(compare_jurisdiction)
else:
    mcp = None


def main() -> None:
    if mcp is None:
        print("Le package MCP n'est pas installe. Les outils restent importables localement.")
        return
    mcp.run()


if __name__ == "__main__":
    main()
