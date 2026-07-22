"""Reasoning and final synthesis utilities."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from enum import Enum
import re
import unicodedata
from typing import Any

import guardrails
import llm_client
from constants import UNKNOWN_DATE
from retrieval import SearchResult


def _strip_accents(text: str) -> str:
    """Fold accented characters to their ASCII base so matching is accent-
    and encoding-insensitive (e.g. "credit" written without an accent in
    source strings still matches "crédit" in the keyword list)."""
    normalized = unicodedata.normalize("NFKD", text or "")
    return "".join(ch for ch in normalized if not unicodedata.combining(ch))


# Bilingual keywords: the risk LABELS (dict keys) stay French — they are used
# verbatim in the obligations mapping and in the final conclusion text — but
# the corpus is mostly English (US + UK + many EU guides), so the MATCHING
# vocabulary must cover both languages or English-language questions and
# passages never match the "élevé"/"interdit" categories at all.
RISK_KEYWORDS = {
    "interdit": [
        "manipulation subliminale", "subliminal manipulation",
        "notation sociale", "social scoring",
        "biométrique", "biometric identification",
        "vulnérabilités", "exploit vulnerabilities",
    ],
    "élevé": [
        "emploi", "employment", "recruitment", "hiring",
        "éducation", "education",
        "crédit", "credit", "creditworthiness", "credit scoring", "loan",
        "services essentiels", "essential services",
        "migration",
        "justice", "law enforcement",
        "infrastructures", "critical infrastructure",
    ],
    "limité": [
        "transparence", "transparency",
        "interagit", "interacts with",
        "contenu synthétique", "synthetic content", "deepfake",
        "chatbot",
    ],
    "minimal": [
        "minimal", "faible impact", "low impact",
        "bonnes pratiques", "good practices",
    ],
}
RISK_KEYWORDS_FOLDED = {
    risk: [_strip_accents(keyword).casefold() for keyword in keywords]
    for risk, keywords in RISK_KEYWORDS.items()
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
    folded = _strip_accents(text).casefold()
    scores = {
        risk: sum(1 for keyword in keywords if keyword in folded)
        for risk, keywords in RISK_KEYWORDS_FOLDED.items()
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
    return ReasonedAnswer(
        evidence=evidence,
        analysis=analysis,
        conclusion=conclusion,
        confidence=0.3,
        # Placeholder: the authoritative critic pass runs once, after the
        # self-consistency vote, in agent.py (see _with_visible_critic).
        # Calling it here too would mean k paid LLM critic calls per
        # question instead of one.
        critic_verdict=CriticVerdict.REVISE.value,
        input_fields=input_fields,
        missing_information=missing_information,
        candidate_conclusions=[conclusion],
    )


# ══════════════════════════════════════════════════════════════════════════════
#  LLM-backed synthesis and critic (DeepInfra) — deterministic fallback above
# ══════════════════════════════════════════════════════════════════════════════

_VALID_RISK_LABELS = {"interdit", "élevé", "limité", "minimal"}

SYNTHESIS_SYSTEM_PROMPT = """Tu es un agent d'analyse de conformite IA (UE / US / UK).

A partir de la question et des preuves fournies (chacune avec sa source, sa juridiction,
son statut legal et sa date), produis une reponse structuree en francais dans EXACTEMENT
ce format :

PREUVES:
- [preuve 1 avec sa source citee]
- [preuve 2 avec sa source citee]

ANALYSE:
Etape 1: [premier raisonnement]
Etape 2: [second raisonnement]
Etape 3: [reconciliation des contradictions ou nuances entre juridictions]

CONCLUSION: Niveau de risque probable: {NIVEAU}. Obligation principale: [obligation en une phrase].

Regles absolues:
- {NIVEAU} doit etre EXACTEMENT un des quatre mots suivants : interdit, eleve, limite, minimal.
- N'affirme jamais un fait qui ne provient pas des preuves fournies.
- Ne presente jamais une regle volontaire ou une recommandation comme une obligation
  contraignante : respecte le statut indique pour chaque source (obligatoire / volontaire /
  recommandation).
- Si aucune preuve n'est fournie, reponds avec le niveau "minimal" et signale l'absence de
  preuves dans l'analyse.

Exemple :
Question : Une banque utilise un systeme IA pour evaluer la solvabilite de ses clients.
PREUVES:
- Annexe III du reglement (UE) 2024/1689 (statut: obligatoire) classe les systemes de
  scoring de credit en haut risque.
ANALYSE:
Etape 1: Le cas decrit correspond exactement a un systeme de scoring de credit.
Etape 2: L'Annexe III liste explicitement ce cas comme haut risque.
Etape 3: Aucune contradiction entre juridictions sur ce point precis.
CONCLUSION: Niveau de risque probable: eleve. Obligation principale: mettre en place la
gestion des risques, la documentation technique et la supervision humaine prevues aux
articles 9 a 15.
"""

CRITIC_SYSTEM_PROMPT = """Tu es un critique independant qui verifie une analyse de
conformite IA avant publication.

Verifie strictement :
1. Chaque preuve listee est-elle plausible au regard du contexte fourni (pas de numero
   d'article ou de juridiction invente) ?
2. La conclusion suit-elle logiquement les preuves listees (pas de saut logique injustifie) ?
3. Le niveau de risque conclu est-il coherent avec les preuves (ex: un systeme de
   recrutement ou de scoring de credit ne devrait jamais etre conclu "minimal") ?
4. Un statut volontaire ou une recommandation n'est-il jamais presente comme une obligation
   contraignante ?

Reponds en DEUX lignes exactement :
Ligne 1 : exactement le mot APPROVE ou le mot REVISE (rien d'autre).
Ligne 2 : une phrase courte justifiant la decision.
"""


def _build_context_block(contexts: list[SearchResult]) -> str:
    if not contexts:
        return "(aucune preuve recuperee)"
    lines = []
    for result in contexts[:5]:
        doc = result.document
        safe_text = guardrails.sanitise_tool_result(doc.text[:500], max_chars=500)
        lines.append(
            f"- [{doc.jurisdiction or '?'} | statut: {doc.status or 'inconnu'} | "
            f"date: {doc.date or UNKNOWN_DATE}] {doc.title}: {safe_text}"
        )
    return "\n".join(lines)


def _extract_section(raw: str, header: str, next_headers: list[str]) -> str:
    """Extract text between `header:` and the next known header (or end of string)."""
    boundary = "|".join(next_headers)
    pattern = rf"{header}\s*:?\s*\n?(.*?)(?=\n(?:{boundary})\s*:|\Z)"
    match = re.search(pattern, raw, re.IGNORECASE | re.DOTALL)
    return match.group(1).strip() if match else ""


def _parse_conclusion(raw: str) -> tuple[str, str] | None:
    """Return (risk_label, full_conclusion_sentence), or None if unparsable/invalid."""
    match = re.search(r"Niveau de risque probable\s*:\s*(\S+)\.?\s*(.*)", raw, re.IGNORECASE)
    if not match:
        return None
    risk = _strip_accents(match.group(1).strip().rstrip(".")).casefold()
    # Map the accent-folded word back to the canonical accented label used
    # elsewhere (obligations dict keys, agent.py output) — never invent a
    # 5th category the rest of the pipeline does not know how to handle.
    canonical = {
        "interdit": "interdit",
        "eleve": "élevé",
        "limite": "limité",
        "minimal": "minimal",
    }.get(risk)
    if canonical is None:
        return None
    rest = match.group(2).strip()
    sentence = f"Niveau de risque probable: {canonical}. {rest}".strip()
    return canonical, sentence


def llm_synthesize_once(
    question: str,
    contexts: list[SearchResult],
    input_fields: dict[str, str],
    missing_information: list[str],
    *,
    model: str,
) -> "ReasonedAnswer | None":
    """Real LLM synthesis. Returns None on any failure so the caller falls back
    to the deterministic synthesize_once() for this variant."""
    user_prompt = (
        f"QUESTION: {question}\n\nPREUVES DISPONIBLES:\n{_build_context_block(contexts)}"
    )
    raw = llm_client.chat(
        SYNTHESIS_SYSTEM_PROMPT, user_prompt, model=model, temperature=0.7, max_tokens=700
    )
    if not raw:
        return None
    parsed = _parse_conclusion(raw)
    if parsed is None:
        return None
    _, conclusion = parsed
    evidence_text = _extract_section(raw, "PREUVES", ["ANALYSE", "CONCLUSION"])
    analysis_text = _extract_section(raw, "ANALYSE", ["CONCLUSION"])
    evidence = [
        line.lstrip("-• ").strip() for line in evidence_text.splitlines() if line.strip()
    ] or [
        f"{result.document.title} ({result.document.source} | date: "
        f"{result.document.date or UNKNOWN_DATE} | statut: {result.document.status})"
        for result in contexts[:3]
    ]
    return ReasonedAnswer(
        evidence=evidence,
        analysis=analysis_text or "Analyse LLM non structuree (voir texte brut).",
        conclusion=conclusion,
        confidence=0.3,  # overwritten by the vote-based confidence in self_consistency
        critic_verdict=CriticVerdict.REVISE.value,  # placeholder, see synthesize_once
        input_fields=input_fields,
        missing_information=missing_information,
        candidate_conclusions=[conclusion],
    )


def llm_critic_review(question: str, evidence: list[str], conclusion: str, *, model: str) -> str | None:
    """Independent LLM critic pass. Returns None on failure so the caller falls
    back to the deterministic critic_review() rules."""
    user_prompt = (
        f"QUESTION: {question}\n\n"
        "PREUVES CITEES:\n" + "\n".join(f"- {item}" for item in evidence)
        + f"\n\nCONCLUSION A VERIFIER:\n{conclusion}"
    )
    return llm_client.chat(
        CRITIC_SYSTEM_PROMPT, user_prompt, model=model, temperature=0.0, max_tokens=150
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
    use_llm = llm_client.is_available()
    synthesis_model = llm_client.get_synthesis_model() if use_llm else None
    # Computed once, shared by every variant, so the k candidates only differ
    # by the LLM's own variance (temperature=0.7), not by re-derived fields.
    input_fields = extract_input_fields(question)
    missing_information = missing_information_from_fields(input_fields)

    def _run_variant(index: int) -> ReasonedAnswer:
        if use_llm:
            result = llm_synthesize_once(
                question, contexts, input_fields, missing_information, model=synthesis_model
            )
            if result is not None:
                return result
        return synthesize_once(question, contexts, variant=index)

    candidates: list[ReasonedAnswer] = []
    for index in range(k):
        span_name = f"reasoning.synthesis.{index + 1}"
        span_metadata = {"k": k, "variant": index + 1, "provider": "deepinfra" if use_llm else "deterministic"}
        if tracer is None:
            candidates.append(_run_variant(index))
            continue
        with tracer.span(span_name, span_metadata):
            candidates.append(_run_variant(index))

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
    """Authoritative critic pass — called once per question (plus once more if
    agent.py's revision loop triggers). Tries the stronger LLM critic model
    first; falls back to the deterministic rules on any failure."""
    if llm_client.is_available():
        raw = llm_critic_review(question, evidence, conclusion, model=llm_client.get_critic_model())
        if raw is not None:
            return raw
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
