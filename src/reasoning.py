"""Reasoning and final synthesis utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from constants import UNKNOWN_DATE
from retrieval import SearchResult


RISK_KEYWORDS = {
    "interdit": ["manipulation subliminale", "notation sociale", "biométrique", "vulnérabilités"],
    "élevé": ["emploi", "éducation", "crédit", "services essentiels", "migration", "justice", "infrastructures"],
    "limité": ["transparence", "interagit", "contenu synthétique", "chatbot"],
    "minimal": ["minimal", "faible impact", "bonnes pratiques"],
}

REQUIRED_INPUT_FIELDS = [
    "objectif du système",
    "personnes concernées",
    "données utilisées",
    "décisions produites",
    "degré d’autonomie",
    "secteur d’activité",
    "pays de déploiement",
    "présence de biométrie",
    "fournisseur du modèle",
    "possibilité d’intervention humaine",
]

FIELD_MARKERS = {
    "objectif du système": ["evaluer", "évaluer", "classer", "scorer", "preselectionner", "présélectionner", "surveiller"],
    "personnes concernées": ["candidat", "client", "travailleur", "employe", "employé", "eleve", "élève", "patient"],
    "données utilisées": ["donnee", "donnée", "data", "cv", "credit", "crédit", "biomet", "biomét", "sante", "santé"],
    "décisions produites": ["decision", "décision", "refus", "acceptation", "score", "classement", "recommandation"],
    "degré d’autonomie": ["automatique", "autonome", "humain", "supervision", "intervention"],
    "secteur d’activité": ["banque", "credit", "crédit", "emploi", "education", "éducation", "sante", "santé", "police"],
    "pays de déploiement": ["eu", "ue", "europe", "france", "us", "usa", "uk", "royaume-uni", "etats-unis", "états-unis"],
    "présence de biométrie": ["biometr", "biométr", "visage", "empreinte", "identification"],
    "fournisseur du modèle": ["fournisseur", "modele", "modèle", "openai", "anthropic", "mistral", "azure"],
    "possibilité d’intervention humaine": ["intervention humaine", "supervision humaine", "humain", "recours", "revue humaine"],
}


class CriticVerdict(str, Enum):
    APPROVE = "APPROVE"
    REVISE = "REVISE"


@dataclass(frozen=True)
class ReasonedAnswer:
    evidence: list[str]
    analysis: str
    conclusion: str
    confidence: float
    critic_verdict: str
    vote_count: int = 1
    total_votes: int = 1
    confidence_justification: str = "1/1 conclusion concordante"
    input_fields: dict[str, str] = field(default_factory=dict)
    missing_information: list[str] = field(default_factory=list)
    candidate_conclusions: list[str] = field(default_factory=list)


def classify_ai_act_risk(question: str, contexts: list[SearchResult]) -> str:
    text = f"{question} " + " ".join(result.document.text for result in contexts)
    lowered = text.casefold()
    scores = {
        risk: sum(1 for keyword in keywords if keyword in lowered)
        for risk, keywords in RISK_KEYWORDS.items()
    }
    winner, score = Counter(scores).most_common(1)[0]
    return winner if score > 0 else "minimal"


def parse_critic_verdict(raw_verdict: str) -> str:
    """Parse an exact critic verdict; ambiguous text always asks for revision."""
    if not isinstance(raw_verdict, str):
        return CriticVerdict.REVISE.value
    first_line = next((line.strip() for line in raw_verdict.splitlines() if line.strip()), "")
    normalized = first_line.upper()
    if normalized == CriticVerdict.APPROVE.value:
        return CriticVerdict.APPROVE.value
    if normalized == CriticVerdict.REVISE.value:
        return CriticVerdict.REVISE.value
    return CriticVerdict.REVISE.value


def extract_input_fields(question: str) -> dict[str, str]:
    """Deterministically expose the 10 required fields and explicit gaps."""
    lowered = (question or "").casefold()
    fields: dict[str, str] = {}
    for label in REQUIRED_INPUT_FIELDS:
        markers = FIELD_MARKERS[label]
        matched = [marker for marker in markers if marker.casefold() in lowered]
        if matched:
            fields[label] = f"indice détecté: {', '.join(matched[:3])}"
        else:
            fields[label] = "non renseigné (extraction déterministe limitée)"
    return fields


def missing_information_from_fields(fields: dict[str, str]) -> list[str]:
    return [
        label for label in REQUIRED_INPUT_FIELDS
        if fields.get(label, "").casefold().startswith("non renseigné")
    ]


def synthesize_once(question: str, contexts: list[SearchResult], variant: int = 0) -> ReasonedAnswer:
    risk = classify_ai_act_risk(question, contexts)
    evidence = [
        (
            f"{result.document.title} "
            f"({result.document.source} | date: {result.document.date or UNKNOWN_DATE} "
            f"| statut: {result.document.status or 'statut non renseigné'})"
        )
        for result in contexts[:3]
    ]
    obligations = {
        "interdit": "cesser ou refuser le cas d'usage, puis documenter le motif d'interdiction",
        "élevé": "mettre en place gestion des risques, documentation, traçabilité et supervision humaine",
        "limité": "informer clairement l'utilisateur et tracer les limites du système",
        "minimal": "appliquer des bonnes pratiques de gouvernance et de surveillance",
    }
    input_fields = extract_input_fields(question)
    missing_information = missing_information_from_fields(input_fields)
    analysis = (
        f"Le cas décrit est rapproché du niveau {risk}, car les preuves récupérées "
        f"mentionnent les critères réglementaires associés. Variante de synthèse {variant + 1}. "
        "Extraction des champs en mode déterministe: les champs absents sont signalés."
    )
    conclusion = f"Niveau de risque probable: {risk}. Obligation principale: {obligations[risk]}."
    verdict = critic_review(question, evidence, conclusion)
    return ReasonedAnswer(
        evidence=evidence,
        analysis=analysis,
        conclusion=conclusion,
        confidence=0.3,
        critic_verdict=verdict,
        input_fields=input_fields,
        missing_information=missing_information,
        candidate_conclusions=[conclusion],
    )


def _conclusion_signature(conclusion: str) -> str:
    if "Niveau de risque probable:" not in conclusion:
        return conclusion.strip().casefold()
    return conclusion.split("Niveau de risque probable:", 1)[1].split(".", 1)[0].strip().casefold()


def _confidence_from_vote(vote_count: int, total_votes: int) -> float:
    if vote_count == total_votes and total_votes > 0:
        return 0.9
    if vote_count >= 2:
        return 0.6
    return 0.3


def self_consistency(
    question: str,
    contexts: list[SearchResult],
    k: int = 3,
    *,
    tracer: Any | None = None,
) -> ReasonedAnswer:
    k = max(1, int(k))
    candidates: list[ReasonedAnswer] = []
    for index in range(k):
        span_name = f"reasoning.synthesis.{index + 1}"
        if tracer is None:
            candidates.append(synthesize_once(question, contexts, variant=index))
            continue
        with tracer.span(span_name, {"k": k, "variant": index + 1}):
            candidates.append(synthesize_once(question, contexts, variant=index))

    signatures = [_conclusion_signature(candidate.conclusion) for candidate in candidates]
    selected_signature, vote_count = Counter(signatures).most_common(1)[0]
    selected = next(
        candidate for candidate in candidates
        if _conclusion_signature(candidate.conclusion) == selected_signature
    )
    confidence = _confidence_from_vote(vote_count, k)
    justification = f"{vote_count}/{k} conclusions concordantes"
    return ReasonedAnswer(
        evidence=selected.evidence,
        analysis=selected.analysis,
        conclusion=selected.conclusion,
        confidence=confidence,
        critic_verdict=selected.critic_verdict,
        vote_count=vote_count,
        total_votes=k,
        confidence_justification=justification,
        input_fields=selected.input_fields,
        missing_information=selected.missing_information,
        candidate_conclusions=[candidate.conclusion for candidate in candidates],
    )


def critic_review(question: str, evidence: list[str], conclusion: str) -> str:
    if not evidence:
        return CriticVerdict.REVISE.value
    if "Niveau de risque probable" not in conclusion:
        return CriticVerdict.REVISE.value
    if len(question.split()) < 4:
        return CriticVerdict.REVISE.value
    return CriticVerdict.APPROVE.value


def format_answer(answer: ReasonedAnswer) -> str:
    evidence = "\n".join(f"- {item}" for item in answer.evidence)
    fields = "\n".join(
        f"- {label}: {answer.input_fields.get(label, 'non renseigné (extraction déterministe limitée)')}"
        for label in REQUIRED_INPUT_FIELDS
    )
    missing = "\n".join(f"- {item}" for item in answer.missing_information) or "- aucun champ manquant détecté"
    return (
        "PREUVES\n"
        f"{evidence}\n\n"
        "CHAMPS D'ENTREE\n"
        f"{fields}\n\n"
        "INFORMATIONS MANQUANTES\n"
        f"{missing}\n\n"
        "ANALYSE\n"
        f"{answer.analysis}\n\n"
        "CONCLUSION\n"
        f"{answer.conclusion}\n\n"
        "CONFIANCE\n"
        f"{answer.confidence:.1f} ({answer.confidence_justification})\n\n"
        f"CRITIC_VERDICT\n{parse_critic_verdict(answer.critic_verdict)}"
    )
