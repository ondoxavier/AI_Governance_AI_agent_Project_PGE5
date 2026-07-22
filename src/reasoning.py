"""Reasoning and final synthesis utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from retrieval import SearchResult


RISK_KEYWORDS = {
    "interdit": ["manipulation subliminale", "notation sociale", "biométrique", "vulnérabilités"],
    "élevé": ["emploi", "éducation", "crédit", "services essentiels", "migration", "justice", "infrastructures"],
    "limité": ["transparence", "interagit", "contenu synthétique", "chatbot"],
    "minimal": ["minimal", "faible impact", "bonnes pratiques"],
}


@dataclass(frozen=True)
class ReasonedAnswer:
    evidence: list[str]
    analysis: str
    conclusion: str
    confidence: str
    critic_verdict: str


def classify_ai_act_risk(question: str, contexts: list[SearchResult]) -> str:
    text = f"{question} " + " ".join(result.document.text for result in contexts)
    lowered = text.casefold()
    scores = {
        risk: sum(1 for keyword in keywords if keyword in lowered)
        for risk, keywords in RISK_KEYWORDS.items()
    }
    winner, score = Counter(scores).most_common(1)[0]
    return winner if score > 0 else "minimal"


def synthesize_once(question: str, contexts: list[SearchResult], variant: int = 0) -> ReasonedAnswer:
    risk = classify_ai_act_risk(question, contexts)
    evidence = [
        f"{result.document.title} ({result.document.source})"
        for result in contexts[:3]
    ]
    obligations = {
        "interdit": "cesser ou refuser le cas d'usage, puis documenter le motif d'interdiction",
        "élevé": "mettre en place gestion des risques, documentation, traçabilité et supervision humaine",
        "limité": "informer clairement l'utilisateur et tracer les limites du système",
        "minimal": "appliquer des bonnes pratiques de gouvernance et de surveillance",
    }
    confidence = "élevée" if contexts and contexts[0].score > 0.6 else "moyenne"
    analysis = (
        f"Le cas décrit est rapproché du niveau {risk}, car les preuves récupérées "
        f"mentionnent les critères réglementaires associés. Variante de synthèse {variant + 1}/3."
    )
    conclusion = f"Niveau de risque probable: {risk}. Obligation principale: {obligations[risk]}."
    verdict = critic_review(question, evidence, conclusion)
    return ReasonedAnswer(evidence, analysis, conclusion, confidence, verdict)


def self_consistency(question: str, contexts: list[SearchResult], k: int = 3) -> ReasonedAnswer:
    candidates = [synthesize_once(question, contexts, variant=index) for index in range(k)]
    risk_votes = [
        candidate.conclusion.split(":", 1)[1].split(".", 1)[0].strip()
        for candidate in candidates
    ]
    selected_risk = Counter(risk_votes).most_common(1)[0][0]
    for candidate in candidates:
        if selected_risk in candidate.conclusion:
            return candidate
    return candidates[0]


def critic_review(question: str, evidence: list[str], conclusion: str) -> str:
    if not evidence:
        return "CRITIQUE: réponse fragile, aucune preuve documentaire récupérée."
    if "Niveau de risque probable" not in conclusion:
        return "CRITIQUE: conclusion incomplète, niveau de risque absent."
    if len(question.split()) < 4:
        return "CRITIQUE: question trop courte, demander plus de contexte métier."
    return "CRITIQUE: réponse acceptable, preuves présentes et conclusion actionnable."


def format_answer(answer: ReasonedAnswer) -> str:
    evidence = "\n".join(f"- {item}" for item in answer.evidence)
    return (
        "PREUVES\n"
        f"{evidence}\n\n"
        "ANALYSE\n"
        f"{answer.analysis}\n\n"
        "CONCLUSION\n"
        f"{answer.conclusion}\n\n"
        "CONFIANCE\n"
        f"{answer.confidence}\n\n"
        f"{answer.critic_verdict}"
    )
