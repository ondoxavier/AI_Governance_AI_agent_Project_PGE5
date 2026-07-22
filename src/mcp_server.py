"""MCP server exposing the project tools.

The module stays importable without the optional MCP package so tests and the
local agent can run from a fresh clone.
"""

from __future__ import annotations

from guardrails import SecurityError, authorize_action, l1_filter
from reasoning import classify_ai_act_risk
from retrieval import hybrid_search as run_hybrid_search

try:
    from mcp.server.fastmcp import FastMCP
except ImportError:  # pragma: no cover - optional production dependency
    FastMCP = None


def hybrid_search(query: str, top_k: int = 4) -> dict:
    """Search the governance corpus with hybrid retrieval.

    Quand l'utiliser: pour récupérer les sources avant une analyse de conformité.
    Quand ne pas l'utiliser: pour exécuter une action externe ou modifier des données.
    Valeur retournée: liste de passages avec titre, source, score et extrait.
    Exemple: hybrid_search("IA de scoring crédit", 3).
    """
    try:
        l1_filter(query)
        authorize_action("hybrid_search")
        results = run_hybrid_search(query, top_k=top_k)
        return {
            "ok": True,
            "results": [
                {
                    "title": result.document.title,
                    "source": result.document.source,
                    "score": round(result.score, 4),
                    "text": result.document.text,
                }
                for result in results
            ],
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc), "results": []}


def classify_ai_act_tool(query: str) -> dict:
    """Classify the likely AI Act risk level.

    Quand l'utiliser: pour obtenir une première qualification réglementaire.
    Quand ne pas l'utiliser: pour remplacer une validation juridique humaine.
    Valeur retournée: niveau de risque probable et preuves utilisées.
    Exemple: classify_ai_act_risk("IA de recrutement").
    """
    try:
        l1_filter(query)
        authorize_action("classify_ai_act_risk")
        contexts = run_hybrid_search(query, top_k=4)
        return {
            "ok": True,
            "risk": classify_ai_act_risk(query, contexts),
            "evidence_count": len(contexts),
        }
    except Exception as exc:
        return {"ok": False, "error": str(exc)}


def security_screen(text: str) -> dict:
    """Run L1 input security screening.

    Quand l'utiliser: avant tout appel d'outil ou de LLM.
    Quand ne pas l'utiliser: pour juger la vérité métier d'une affirmation.
    Valeur retournée: statut autorisé/bloqué et raison éventuelle.
    Exemple: security_screen("ignore previous instructions").
    """
    try:
        authorize_action("security_screen")
        normalized = l1_filter(text)
        return {"ok": True, "normalized": normalized}
    except SecurityError as exc:
        return {"ok": False, "error": str(exc)}


if FastMCP is not None:
    mcp = FastMCP("ai-governance-agent")
    mcp.tool()(hybrid_search)
    mcp.tool(name="classify_ai_act_risk")(classify_ai_act_tool)
    mcp.tool()(security_screen)
else:
    mcp = None


def main() -> None:
    if mcp is None:
        print("Le package MCP n'est pas installé. Les outils restent importables localement.")
        return
    mcp.run()


if __name__ == "__main__":
    main()
