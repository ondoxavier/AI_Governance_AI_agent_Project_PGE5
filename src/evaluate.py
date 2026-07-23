"""Run the local evaluation for retrieval, latency, cost and observability.

This is a deterministic evaluation harness for the report. It is intentionally
dependency-light so it can run in a fresh clone even before Langfuse or a paid
LLM is configured.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json
import re
import time
import unicodedata

import llm_client
from agent import run_agent
from guardrails import SecurityError, TokenBudget
from observability import create_tracer
from reasoning import format_answer, self_consistency
from retrieval import SearchResult, hybrid_search, tokenize


ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = ROOT / "evaluation"


@dataclass(frozen=True)
class EvalCase:
    id: str
    question: str
    jurisdiction: str | None
    expected_terms: list[str]
    expected_answer_terms: list[str]


EVAL_CASES = [
    EvalCase(
        "eu_art5_prohibited",
        "Quelles pratiques d'IA sont interdites selon l'Article 5 de l'AI Act ?",
        "EU",
        ["interdites", "manipulation", "social", "biometrique"],
        ["interdit", "manipulation", "biometrique"],
    ),
    EvalCase(
        "eu_recruitment_high_risk",
        "Un outil de tri de CV est-il a haut risque selon l'AI Act ?",
        "EU",
        ["emploi", "recrutement", "selection", "haut risque"],
        ["haut", "risque", "emploi"],
    ),
    EvalCase(
        "eu_provider_obligations",
        "Quelles obligations s'appliquent aux fournisseurs de systemes d'IA a haut risque ?",
        "EU",
        ["gestion des risques", "documentation", "transparence", "supervision"],
        ["gestion", "documentation", "supervision"],
    ),
    EvalCase(
        "eu_technical_documentation",
        "Que doit contenir la documentation technique d'un systeme d'IA a haut risque ?",
        "EU",
        ["documentation technique", "systeme", "risque", "conformite"],
        ["documentation", "risque", "conformite"],
    ),
    EvalCase(
        "eu_human_oversight",
        "Qu'exige la supervision humaine pour un systeme d'IA a haut risque ?",
        "EU",
        ["supervision humaine", "controle humain", "risque"],
        ["supervision", "humaine", "risque"],
    ),
    EvalCase(
        "gdpr_dpia",
        "Quand une analyse d'impact relative a la protection des donnees est-elle obligatoire sous le RGPD ?",
        "EU",
        ["analyse d'impact", "protection des donnees", "risque eleve"],
        ["donnees", "risque", "protection"],
    ),
    EvalCase(
        "us_nist_functions",
        "Quelles sont les quatre fonctions du NIST AI Risk Management Framework ?",
        "US",
        ["govern", "map", "measure", "manage"],
        ["govern", "map", "measure", "manage"],
    ),
    EvalCase(
        "us_colorado_credit",
        "Que regule la loi Colorado SB24-205 pour les decisions de credit assistees par IA ?",
        "US",
        ["colorado", "credit", "decision", "consumer"],
        ["colorado", "credit", "decision"],
    ),
    EvalCase(
        "uk_principles",
        "Quels sont les principes britanniques de regulation de l'IA et sont-ils contraignants ?",
        "UK",
        ["safety", "security", "transparency", "fairness", "accountability"],
        ["principes", "transparency", "fairness"],
    ),
    EvalCase(
        "uk_vs_eu",
        "Comment l'approche britannique differe-t-elle de l'AI Act sur la classification des risques ?",
        "UK",
        ["principles", "sector", "risk", "ai act"],
        ["uk", "risque", "ai act"],
    ),
]


def _strip_boilerplate(text: str) -> str:
    """Cut the fixed AVERTISSEMENTS/DISCLAIMER trailer added by agent.py.

    Without this, `final` (which always carries this trailer) and
    `baseline_answer` (which never does, since it comes straight out of
    format_answer()) are not scored on comparable content: the generic
    disclaimer sentence cannot match the retrieved legal text, so it
    mechanically drags faithfulness down regardless of answer quality.
    """
    for header in ("\n\nAVERTISSEMENTS\n", "\n\nDISCLAIMER\n"):
        index = text.find(header)
        if index != -1:
            text = text[:index]
    return text


def _plain(text: str) -> str:
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch)).casefold()


def _contains_term(text: str, term: str) -> bool:
    return _plain(term) in _plain(text)


def context_recall(results: list[SearchResult], expected_terms: list[str]) -> float:
    context = " ".join(result.document.text for result in results)
    hits = sum(1 for term in expected_terms if _contains_term(context, term))
    return hits / max(1, len(expected_terms))


def context_precision(results: list[SearchResult], expected_terms: list[str]) -> float:
    if not results:
        return 0.0
    relevant = 0
    for result in results:
        text = result.document.text
        if any(_contains_term(text, term) for term in expected_terms):
            relevant += 1
    return relevant / len(results)


def answer_relevancy(answer: str, expected_terms: list[str]) -> float:
    hits = sum(1 for term in expected_terms if _contains_term(answer, term))
    return hits / max(1, len(expected_terms))


def faithfulness(answer: str, results: list[SearchResult]) -> float:
    context_tokens = set(tokenize(" ".join(result.document.text for result in results)))
    answer_tokens = [token for token in tokenize(answer) if len(token) > 4]
    if not answer_tokens:
        return 0.0
    supported = sum(1 for token in answer_tokens if token in context_tokens)
    return supported / len(answer_tokens)


# ══════════════════════════════════════════════════════════════════════════════
#  LLM-judge for the two semantic answer-quality metrics (faithfulness +
#  answer_relevancy). The word-overlap proxies above systematically under-score
#  correct French answers grounded in a mostly-English corpus (the terms match
#  in meaning, not in surface form). A judge model reads for meaning instead.
#  Falls back to the deterministic proxies when no LLM key is set or a call
#  fails, so the harness still runs end-to-end from a fresh clone.
# ══════════════════════════════════════════════════════════════════════════════

JUDGE_SYSTEM_PROMPT = """Tu es un juge d'evaluation RAG rigoureux et impartial.
On te donne une QUESTION, une REPONSE produite par un agent, et le CONTEXTE
documentaire reellement recupere.

Evalue deux criteres, chacun sur une echelle continue de 0.0 a 1.0 :

1. faithfulness (fidelite) : chaque affirmation factuelle de la REPONSE est-elle
   soutenue par le CONTEXTE ? 1.0 = tout est appuye par le contexte ; 0.0 = la
   reponse invente des faits absents du contexte. Ignore les phrases de
   disclaimer generiques (validation humaine, etc.).

2. answer_relevancy (pertinence) : la REPONSE repond-elle directement et
   completement a la QUESTION ? 1.0 = reponse ciblee et complete ; 0.0 = hors
   sujet ou vide.

Reponds UNIQUEMENT avec un objet JSON sur une seule ligne, sans aucun texte
autour :
{"faithfulness": <nombre 0.0-1.0>, "answer_relevancy": <nombre 0.0-1.0>, "justification": "<une phrase courte>"}
"""


def _clamp_unit(value: float) -> float:
    return min(1.0, max(0.0, float(value)))


def llm_judge_answer(question: str, answer: str, results: list[SearchResult]) -> dict | None:
    """Score (faithfulness, answer_relevancy) with the critic LLM in one call.

    Returns a dict with both scores in [0, 1] plus the judge's one-line
    justification, or None on any failure (no key, network error, unparsable
    output) so the caller falls back to the deterministic proxies.
    """
    if not llm_client.is_available():
        return None
    context = "\n".join(f"- {result.document.text[:600]}" for result in results) or "(aucun contexte)"
    user_prompt = f"QUESTION:\n{question}\n\nREPONSE:\n{answer}\n\nCONTEXTE:\n{context}"
    raw = llm_client.chat(
        JUDGE_SYSTEM_PROMPT,
        user_prompt,
        model=llm_client.get_critic_model(),
        temperature=0.0,  # judge should be as reproducible as the model allows
        max_tokens=200,
    )
    if not raw:
        return None
    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        faith = _clamp_unit(data["faithfulness"])
        relevancy = _clamp_unit(data["answer_relevancy"])
    except (ValueError, KeyError, TypeError):
        return None
    return {
        "faithfulness": round(faith, 4),
        "answer_relevancy": round(relevancy, 4),
        "justification": str(data.get("justification", ""))[:200],
    }


def average(rows: list[dict], key: str) -> float:
    return sum(float(row[key]) for row in rows) / max(1, len(rows))


def evaluate() -> dict:
    rows: list[dict] = []
    total_tool_calls: dict[str, int] = {}
    total_latency = 0.0
    token_budget_triggered = False
    judge_rows_llm = 0  # how many rows got their answer-quality scores from the LLM judge
    llm_client.reset_usage()  # isolate this run's cost from any prior calls in-process

    for case in EVAL_CASES:
        baseline = hybrid_search(
            case.question,
            top_k=1,
            jurisdiction=case.jurisdiction,
            mode="baseline",
        )
        final = hybrid_search(case.question, top_k=3, jurisdiction=case.jurisdiction)
        baseline_answer = format_answer(self_consistency(case.question, baseline, k=1))

        tracer = create_tracer("evaluation.run", {"case_id": case.id})
        start = time.perf_counter()
        response = run_agent(case.question, jurisdiction=case.jurisdiction or "all", tracer=tracer)
        # Scored on the same content shape as baseline_answer (no disclaimer
        # trailer) so the comparison isolates reasoning quality, not the
        # presence of a fixed compliance sentence. See _strip_boilerplate().
        answer = _strip_boilerplate(response.answer)
        latency = time.perf_counter() - start
        total_latency += latency

        for event in tracer.events:
            if event.name in {"retrieval.search", "guardrail.l4"}:
                total_tool_calls[event.name] = total_tool_calls.get(event.name, 0) + 1

        # Answer-quality metrics: prefer the LLM judge, fall back per side to the
        # deterministic word-overlap proxies when the judge is unavailable/fails.
        baseline_judge = llm_judge_answer(case.question, baseline_answer, baseline)
        final_judge = llm_judge_answer(case.question, answer, final)
        if baseline_judge is not None and final_judge is not None:
            judge_rows_llm += 1

        if baseline_judge is not None:
            b_faith, b_rel = baseline_judge["faithfulness"], baseline_judge["answer_relevancy"]
        else:
            b_faith = round(faithfulness(baseline_answer, baseline), 4)
            b_rel = round(answer_relevancy(baseline_answer, case.expected_answer_terms), 4)

        if final_judge is not None:
            f_faith, f_rel = final_judge["faithfulness"], final_judge["answer_relevancy"]
        else:
            f_faith = round(faithfulness(answer, final), 4)
            f_rel = round(answer_relevancy(answer, case.expected_answer_terms), 4)

        rows.append(
            {
                "id": case.id,
                "jurisdiction": case.jurisdiction or "all",
                "latency_s": round(latency, 4),
                "baseline_context_recall": round(context_recall(baseline, case.expected_terms), 4),
                "baseline_context_precision": round(context_precision(baseline, case.expected_terms), 4),
                "final_context_recall": round(context_recall(final, case.expected_terms), 4),
                "final_context_precision": round(context_precision(final, case.expected_terms), 4),
                "baseline_faithfulness": b_faith,
                "baseline_answer_relevancy": b_rel,
                "final_faithfulness": f_faith,
                "final_answer_relevancy": f_rel,
                "answer_metrics_judge": "llm" if final_judge is not None else "deterministic-proxy",
                "judge_justification": final_judge["justification"] if final_judge is not None else "",
                "top_method": final[0].method if final else "none",
            }
        )

    try:
        tiny_budget = TokenBudget(max_tokens=3)
        tiny_budget.consume("one two three four")
    except SecurityError:
        token_budget_triggered = True

    llm_used = llm_client.is_available()
    total_cost_usd, cost_notes = llm_client.estimate_cost_usd() if llm_used else (0.0, [])
    usage_totals = llm_client.get_usage_totals() if llm_used else {}
    if llm_used:
        cost_note = (
            "Cout estime a partir de l'usage reel DeepInfra (tarifs indicatifs "
            "juillet 2026, a verifier sur deepinfra.com/pricing). "
            + (" ".join(cost_notes) if cost_notes else "")
        ).strip()
    else:
        cost_note = "Aucune cle DEEPINFRA_API_KEY configuree: fallback deterministe, aucun appel LLM paye."

    if judge_rows_llm == len(EVAL_CASES):
        judge_backend = "llm"
    elif judge_rows_llm == 0:
        judge_backend = "deterministic-proxy"
    else:
        judge_backend = f"mixed ({judge_rows_llm}/{len(EVAL_CASES)} rows via LLM judge)"

    summary = {
        "num_questions": len(EVAL_CASES),
        "llm_used": llm_used,
        "answer_metrics_judge": judge_backend,
        "cost_average_usd": round(total_cost_usd / len(EVAL_CASES), 6),
        "cost_total_usd": total_cost_usd,
        "cost_note": cost_note,
        "token_usage_by_model": usage_totals,
        "latency_average_s": round(total_latency / len(EVAL_CASES), 4),
        "baseline": {
            "context_recall": round(average(rows, "baseline_context_recall"), 4),
            "context_precision": round(average(rows, "baseline_context_precision"), 4),
            "faithfulness": round(average(rows, "baseline_faithfulness"), 4),
            "answer_relevancy": round(average(rows, "baseline_answer_relevancy"), 4),
        },
        "final": {
            "context_recall": round(average(rows, "final_context_recall"), 4),
            "context_precision": round(average(rows, "final_context_precision"), 4),
            "faithfulness": round(average(rows, "final_faithfulness"), 4),
            "answer_relevancy": round(average(rows, "final_answer_relevancy"), 4),
        },
        "tool_calls_10_runs": total_tool_calls,
        "token_budget_triggered": token_budget_triggered,
        "rows": rows,
    }
    return summary


def main() -> int:
    results = evaluate()
    RESULTS_DIR.mkdir(exist_ok=True)
    output = RESULTS_DIR / "latest_results.json"
    output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(results, ensure_ascii=False, indent=2))
    print(f"\nSaved: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
