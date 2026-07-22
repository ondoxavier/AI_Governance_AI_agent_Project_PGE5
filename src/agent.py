"""Main executable loop for the AI governance agent.

The module keeps the public CLI simple while exposing `run_agent` as a
testable orchestration function. External services are optional: retrieval has
its own corpus fallback, reasoning has a deterministic fallback, and tracing
falls back to LocalTracer-compatible console spans when Langfuse is absent.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
import re
import sys
from time import perf_counter
from types import TracebackType
from typing import Any, Iterator, Mapping, Sequence

from constants import DISCLAIMER, UNKNOWN_DATE
from guardrails import SecurityError, TokenBudget, authorize_action, l1_filter
from observability import Tracer, create_tracer
from reasoning import ReasonedAnswer, critic_review, format_answer, parse_critic_verdict, self_consistency
from retrieval import SearchResult, hybrid_search


DEFAULT_QUESTION = (
    "Une banque utilise un modele IA pour preselectionner des candidats a un credit. "
    "Quel est le niveau de risque AI Act et quelle obligation principale faut-il prevoir ?"
)
MAX_TOP_K = 10
DEFAULT_TOP_K = 5
DEFAULT_SELF_CONSISTENCY_K = 3


@dataclass
class AgentResponse:
    """Structured result returned by the agent orchestration layer."""

    answer: str
    conclusion: str | None
    confidence: float | None
    critic_verdict: str
    sources: list[dict[str, Any]] = field(default_factory=list)
    missing_information: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    trace_id: str | None = None
    latency_ms: float = 0.0
    metadata: dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        return self.answer


def _normalise_question(question: str) -> str:
    """Validate and compact user input before guardrails and retrieval."""
    if not isinstance(question, str):
        raise SecurityError("La question doit etre une chaine de caracteres.")
    compacted = re.sub(r"\s+", " ", question).strip()
    if not compacted:
        raise SecurityError("Question vide: decrivez le systeme IA a analyser.")
    if compacted.casefold() in {"quit", "exit", "q"}:
        raise KeyboardInterrupt
    return compacted


def _normalise_jurisdiction(jurisdiction: str | None) -> str | None:
    """Return a retrieval-compatible jurisdiction value."""
    if jurisdiction is None:
        return None
    value = str(jurisdiction).strip()
    if not value or value.casefold() == "all":
        return None
    upper = value.upper()
    if upper not in {"EU", "US", "UK"}:
        raise SecurityError(f"Juridiction non reconnue: {jurisdiction}")
    return upper


def _normalise_top_k(top_k: int) -> int:
    """Keep retrieval volumes bounded for CLI, tests and MCP callers."""
    try:
        value = int(top_k)
    except (TypeError, ValueError) as exc:
        raise SecurityError("top_k doit etre un entier.") from exc
    if value < 1:
        raise SecurityError("top_k doit etre superieur ou egal a 1.")
    return min(value, MAX_TOP_K)


def _source_dict(result: SearchResult) -> dict[str, Any]:
    """Preserve source, jurisdiction and status in the structured response."""
    return {
        "title": result.document.title,
        "source": result.document.source,
        "jurisdiction": result.document.jurisdiction,
        "status": result.document.status,
        "date": result.document.date or UNKNOWN_DATE,
        "score": round(float(result.score), 4),
        "method": result.method,
    }


def _context_for_budget(result: SearchResult) -> str:
    """Use parent context when available so budget checks match reasoning input."""
    return result.document.context or result.document.text


def _consume_or_warn(budget: TokenBudget, text: str, warnings: list[str], label: str) -> bool:
    """Consume budget and convert expected overflows into user-facing warnings."""
    try:
        budget.consume(text)
        return True
    except SecurityError as exc:
        warnings.append(f"TokenBudget limite atteint pendant {label}: {exc}")
        return False


def _reserve_or_warn(
    budget: TokenBudget,
    estimated_tokens: int,
    warnings: list[str],
    label: str,
) -> bool:
    """Reserve budget for planned future steps before starting a branch."""
    if budget.can_reserve(estimated_tokens):
        try:
            budget.reserve(estimated_tokens)
            return True
        except SecurityError as exc:
            warnings.append(f"TokenBudget limite atteint pendant {label}: {exc}")
            return False
    warnings.append(
        f"TokenBudget limite atteint avant {label}: "
        f"{budget.used_tokens + estimated_tokens}/{budget.max_tokens}"
    )
    return False


def _reasoning_budget_estimate(
    budget: TokenBudget,
    question: str,
    contexts: Sequence[SearchResult],
    k: int,
) -> int:
    """Estimate the complete cost of k syntheses over retrieved contexts."""
    retrieval_text = "\n".join(_context_for_budget(result) for result in contexts)
    payload = f"{question}\n{retrieval_text}"
    return budget.estimate(payload) * max(1, int(k))


def _critic_budget_estimate(budget: TokenBudget, question: str, answer: ReasonedAnswer) -> int:
    """Estimate critic cost over question, evidence and conclusion."""
    payload = f"{question}\n{' '.join(answer.evidence)}\n{answer.conclusion}"
    return budget.estimate(payload)


@contextmanager
def _safe_span(
    tracer: Tracer,
    name: str,
    metadata: Mapping[str, Any] | None,
    warnings: list[str],
) -> Iterator[None]:
    """Use tracing when available without letting tracing failures abort work."""
    manager = None
    entered = False
    try:
        manager = tracer.span(name, metadata)
        manager.__enter__()
        entered = True
    except Exception as exc:
        warnings.append(f"Tracing indisponible pendant {name}: {exc.__class__.__name__}: {exc}")
    exc_info: tuple[type[BaseException] | None, BaseException | None, TracebackType | None] = (None, None, None)
    try:
        yield
    except BaseException:
        exc_info = sys.exc_info()
        raise
    finally:
        if entered and manager is not None:
            try:
                manager.__exit__(*exc_info)
            except Exception as exc:
                warnings.append(f"Fin de trace indisponible pendant {name}: {exc.__class__.__name__}: {exc}")


class _SafeTracerProxy:
    """Tracer facade passed to components that should not see tracing failures."""

    def __init__(self, tracer: Tracer, warnings: list[str]):
        self._tracer = tracer
        self._warnings = warnings

    def span(self, name: str, metadata: Mapping[str, Any] | None = None):
        return _safe_span(self._tracer, name, metadata, self._warnings)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._tracer, name)


def _confidence_to_float(confidence: float | int | str | None) -> float | None:
    """Return the numeric self-consistency confidence without label heuristics."""
    if not confidence:
        return None
    if isinstance(confidence, (float, int)):
        return float(confidence)
    match = re.search(r"0(?:\.\d+)?|1(?:\.0+)?", confidence)
    if match:
        return float(match.group(0))
    return None


def _critic_code(raw_verdict: str) -> str:
    """Map critic output to exact APPROVE/REVISE codes."""
    return parse_critic_verdict(raw_verdict)


def _with_visible_critic(answer: ReasonedAnswer, raw_verdict: str) -> ReasonedAnswer:
    """Attach the normalized critic code without changing reasoning internals."""
    code = _critic_code(raw_verdict)
    return ReasonedAnswer(
        evidence=answer.evidence,
        analysis=answer.analysis,
        conclusion=answer.conclusion,
        confidence=answer.confidence,
        critic_verdict=code,
        vote_count=answer.vote_count,
        total_votes=answer.total_votes,
        confidence_justification=answer.confidence_justification,
        input_fields=answer.input_fields,
        missing_information=answer.missing_information,
        candidate_conclusions=answer.candidate_conclusions,
    )


def _format_final_answer(answer: ReasonedAnswer, warnings: Sequence[str]) -> str:
    """Render the answer, warnings and mandatory legal-validation disclaimer."""
    rendered = format_answer(answer)
    if warnings:
        warning_block = "\n".join(f"- {warning}" for warning in warnings)
        rendered += f"\n\nAVERTISSEMENTS\n{warning_block}"
    rendered += f"\n\nDISCLAIMER\n{DISCLAIMER}"
    return rendered


def _safe_error_response(
    message: str,
    *,
    tracer: Tracer | None,
    start: float,
    warnings: Sequence[str] | None = None,
) -> AgentResponse:
    """Build a controlled response for expected validation/security failures."""
    all_warnings = list(warnings or [])
    all_warnings.append(message)
    latency_ms = (perf_counter() - start) * 1000
    answer = (
        "REQUETE BLOQUEE OU INCOMPLETE\n"
        f"{message}\n\n"
        f"DISCLAIMER\n{DISCLAIMER}"
    )
    return AgentResponse(
        answer=answer,
        conclusion=None,
        confidence=None,
        critic_verdict="REVISE",
        warnings=all_warnings,
        trace_id=tracer.trace_id if tracer else None,
        latency_ms=latency_ms,
        metadata={"fallback_mode": True},
    )


def run_agent(
    question: str,
    *,
    jurisdiction: str = "all",
    top_k: int = DEFAULT_TOP_K,
    data_dir: str | Path | None = None,
    self_consistency_k: int = DEFAULT_SELF_CONSISTENCY_K,
    token_budget: TokenBudget | None = None,
    tracer: Tracer | None = None,
) -> AgentResponse:
    """Run the complete L1 -> retrieval -> reasoning -> critic pipeline.

    Expected failures such as blocked input, retrieval errors, or budget
    exhaustion return a structured degraded response instead of an uncaught
    stack trace. Unexpected programmer errors still surface during tests.
    """
    start = perf_counter()
    warnings: list[str] = []
    contexts: list[SearchResult] = []
    sources: list[dict[str, Any]] = []
    budget = token_budget or TokenBudget(max_tokens=3500)
    active_tracer = tracer or create_tracer(
        "agent.run",
        {"jurisdiction": jurisdiction, "top_k": top_k},
    )
    safe_tracer = _SafeTracerProxy(active_tracer, warnings)
    cleaned_question = ""
    selected_jurisdiction: str | None = None
    selected_top_k = DEFAULT_TOP_K
    reasoned: ReasonedAnswer | None = None
    rendered = ""

    try:
        try:
            cleaned_question = _normalise_question(question)
            selected_jurisdiction = _normalise_jurisdiction(jurisdiction)
            selected_top_k = _normalise_top_k(top_k)
        except KeyboardInterrupt:
            raise
        except SecurityError as exc:
            with safe_tracer.span("agent.validation_refusal", {"error_type": exc.__class__.__name__}):
                return _safe_error_response(str(exc), tracer=active_tracer, start=start)

        with safe_tracer.span(
            "agent.run",
            {"jurisdiction": selected_jurisdiction or "all", "top_k": selected_top_k},
        ):
            try:
                with safe_tracer.span("guardrail.l1"):
                    guarded_question = l1_filter(cleaned_question)
            except SecurityError as exc:
                return _safe_error_response(str(exc), tracer=active_tracer, start=start, warnings=warnings)

            if not _consume_or_warn(budget, guarded_question, warnings, "question"):
                return _safe_error_response(warnings[-1], tracer=active_tracer, start=start, warnings=warnings)

            with safe_tracer.span("guardrail.l4", {"tool_name": "hybrid_search"}):
                try:
                    authorize_action("hybrid_search")
                except SecurityError as exc:
                    return _safe_error_response(str(exc), tracer=active_tracer, start=start, warnings=warnings)

            with safe_tracer.span(
                "retrieval.search",
                {"jurisdiction": selected_jurisdiction or "all", "top_k": selected_top_k},
            ):
                try:
                    contexts = hybrid_search(
                        guarded_question,
                        top_k=selected_top_k,
                        data_dir=data_dir or "data",
                        jurisdiction=selected_jurisdiction,
                    )
                except Exception as exc:
                    contexts = []
                    warnings.append(f"Retrieval indisponible: {exc.__class__.__name__}: {exc}")

            budgeted_contexts: list[SearchResult] = []
            for result in contexts:
                if _consume_or_warn(budget, _context_for_budget(result), warnings, "contexte retrieval"):
                    budgeted_contexts.append(result)
                else:
                    break
            contexts = budgeted_contexts
            sources = [_source_dict(result) for result in contexts]
            if any(source["date"] == UNKNOWN_DATE for source in sources):
                warnings.append("Certaines sources n'ont pas de date renseignee dans le corpus.")
            if not contexts:
                warnings.append("Aucun contexte documentaire exploitable; reponse en mode degrade.")

            reasoning_estimate = _reasoning_budget_estimate(
                budget,
                guarded_question,
                contexts,
                self_consistency_k,
            )
            with safe_tracer.span(
                "reasoning.self_consistency",
                {"k": self_consistency_k, "retrieved_documents": len(contexts)},
            ):
                try:
                    if not _reserve_or_warn(
                        budget,
                        reasoning_estimate,
                        warnings,
                        f"synthese self-consistency k={self_consistency_k}",
                    ):
                        raise SecurityError(warnings[-1])
                    reasoned = self_consistency(
                        guarded_question,
                        contexts,
                        k=self_consistency_k,
                        tracer=safe_tracer,
                    )
                except Exception as exc:
                    warnings.append(f"Reasoning indisponible: {exc.__class__.__name__}: {exc}")
                    reasoned = ReasonedAnswer(
                        evidence=[source["title"] for source in sources[:3]],
                        analysis="Synthese degradee: le composant de raisonnement a echoue.",
                        conclusion="Conclusion indisponible; validation humaine requise.",
                        confidence=0.3,
                        critic_verdict="REVISE",
                    )

            with safe_tracer.span("critic.review"):
                try:
                    if not _reserve_or_warn(
                        budget,
                        _critic_budget_estimate(budget, guarded_question, reasoned),
                        warnings,
                        "critique",
                    ):
                        raise SecurityError(warnings[-1])
                    raw_critic = critic_review(guarded_question, reasoned.evidence, reasoned.conclusion)
                except Exception as exc:
                    raw_critic = "REVISE"
                    warnings.append(f"Critique indisponible: {exc.__class__.__name__}: {exc}")
                critic_code = _critic_code(raw_critic)

            if critic_code == "REVISE" and contexts:
                revision_question = guarded_question + " Reponds uniquement avec les preuves recuperees."
                revision_estimate = _reasoning_budget_estimate(
                    budget,
                    revision_question,
                    contexts,
                    self_consistency_k,
                )
                revision_estimate += _critic_budget_estimate(budget, revision_question, reasoned)
                if _reserve_or_warn(budget, revision_estimate, warnings, "revision"):
                    warnings.append("Le critique a demande une revision; une seule tentative a ete executee.")
                    with safe_tracer.span("reasoning.revision", {"max_revision": 1}):
                        try:
                            revised = self_consistency(
                                revision_question,
                                contexts,
                                k=self_consistency_k,
                                tracer=safe_tracer,
                            )
                        except Exception as exc:
                            revised = reasoned
                            warnings.append(f"Revision indisponible: {exc.__class__.__name__}: {exc}")
                    with safe_tracer.span("critic.review", {"revision": 1}):
                        try:
                            revised_critic = critic_review(
                                guarded_question,
                                revised.evidence,
                                revised.conclusion,
                            )
                        except Exception as exc:
                            revised_critic = "REVISE"
                            warnings.append(f"Critique revision indisponible: {exc.__class__.__name__}: {exc}")
                    reasoned = _with_visible_critic(revised, revised_critic)
                else:
                    with safe_tracer.span("budget.revision_skipped", {"reason": "insufficient_budget"}):
                        warnings.append("Revision non executee: budget insuffisant.")
                    reasoned = _with_visible_critic(reasoned, raw_critic)
            else:
                reasoned = _with_visible_critic(reasoned, raw_critic)

            final_estimate = budget.estimate(
                f"{reasoned.conclusion}\n{' '.join(reasoned.evidence)}\n{' '.join(warnings)}\n{DISCLAIMER}"
            )
            _reserve_or_warn(budget, final_estimate, warnings, "reponse finale")
            try:
                rendered = _format_final_answer(reasoned, warnings)
            except Exception as exc:
                warnings.append(f"Formatage final indisponible: {exc.__class__.__name__}: {exc}")
                warning_block = "\n".join(f"- {warning}" for warning in warnings)
                rendered = (
                    "CONCLUSION\n"
                    f"{reasoned.conclusion}\n\n"
                    "AVERTISSEMENTS\n"
                    f"{warning_block}\n\n"
                    f"DISCLAIMER\n{DISCLAIMER}"
                )

        latency_ms = (perf_counter() - start) * 1000
        return AgentResponse(
            answer=rendered,
            conclusion=reasoned.conclusion,
            confidence=_confidence_to_float(reasoned.confidence),
            critic_verdict=_critic_code(reasoned.critic_verdict),
            sources=sources,
            missing_information=reasoned.missing_information,
            warnings=warnings,
            trace_id=active_tracer.trace_id,
            latency_ms=latency_ms,
            metadata={
                "agent_version": active_tracer.agent_version,
                "jurisdiction": selected_jurisdiction or "all",
                "top_k": selected_top_k,
                "retrieved_documents": len(contexts),
                "fallback_mode": any("(fallback-demo)" in source.get("method", "") for source in sources),
                "tracer_provider": active_tracer.provider,
                "token_budget_remaining": budget.remaining,
                "vote_count": reasoned.vote_count,
                "total_votes": reasoned.total_votes,
                "confidence_justification": reasoned.confidence_justification,
            },
        )
    finally:
        try:
            active_tracer.flush()
        except Exception:
            pass


def _question_from_cli(argv: Sequence[str]) -> str:
    """Read a CLI question from args, piped stdin, or the deterministic default."""
    if argv:
        return " ".join(argv).strip()
    if not sys.stdin.isatty():
        piped = sys.stdin.read().strip()
        if piped:
            return piped
    return DEFAULT_QUESTION


def main(argv: Sequence[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    try:
        question = _question_from_cli(args)
        if question.casefold() in {"quit", "exit", "q"}:
            print("Arret demande.")
            return 0
        tracer = create_tracer("agent.cli", {"entrypoint": "cli"})
        response = run_agent(question, tracer=tracer)
        print(response.answer)
        print(f"\nMETADATA trace_id={response.trace_id} latency_ms={response.latency_ms:.1f}")
        tracer.export_jsonl("observability/latest_trace.jsonl")
        return 0
    except KeyboardInterrupt:
        print("Arret demande.")
        return 0
    except Exception as exc:
        print(f"ERREUR CONTROLEE\n{exc.__class__.__name__}: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
