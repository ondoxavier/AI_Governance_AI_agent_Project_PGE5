"""Main executable loop for the AI governance agent."""

from __future__ import annotations

import sys

from guardrails import SecurityError, TokenBudget, authorize_action, l1_filter
from observability import ObservabilityTracer
from reasoning import format_answer, self_consistency
from retrieval import hybrid_search


DEFAULT_QUESTION = (
    "Une banque utilise un modèle IA pour présélectionner des candidats à un crédit. "
    "Quel est le niveau de risque AI Act et quelle obligation principale faut-il prévoir ?"
)


def run_agent(
    question: str,
    tracer: ObservabilityTracer | None = None,
    verbose: bool = True,
) -> str:
    tracer = tracer or ObservabilityTracer(verbose=verbose)
    budget = TokenBudget(max_tokens=3500)

    with tracer.span("agent"):
        with tracer.span("guardrails.l1"):
            normalized_question = l1_filter(question)
            budget.consume(normalized_question)

        with tracer.span("tool.hybrid_search"):
            authorize_action("hybrid_search")
            contexts = hybrid_search(normalized_question, top_k=4)
            for result in contexts:
                budget.consume(result.document.text)

        with tracer.span("llm.synthesis.self_consistency_k3"):
            answer = self_consistency(normalized_question, contexts, k=3)

        with tracer.span("agent.critic"):
            rendered = format_answer(answer)
            budget.consume(rendered)

    return rendered


def main() -> int:
    question = " ".join(sys.argv[1:]).strip() or DEFAULT_QUESTION
    try:
        tracer = ObservabilityTracer(verbose=True)
        print(run_agent(question, tracer=tracer))
        tracer.export_jsonl("observability/latest_trace.jsonl")
        return 0
    except SecurityError as exc:
        print(f"REQUÊTE BLOQUÉE\n{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
