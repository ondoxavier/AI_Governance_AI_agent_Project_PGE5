"""Main executable loop for the AI governance agent."""

from __future__ import annotations

from contextlib import contextmanager
import os
import sys
import time

from guardrails import SecurityError, TokenBudget, authorize_action, l1_filter
from reasoning import format_answer, self_consistency
from retrieval import hybrid_search


DEFAULT_QUESTION = (
    "Une banque utilise un modèle IA pour présélectionner des candidats à un crédit. "
    "Quel est le niveau de risque AI Act et quelle obligation principale faut-il prévoir ?"
)


class LocalTracer:
    """Tiny span logger compatible with local execution and Langfuse-style naming."""

    def __init__(self) -> None:
        self.agent_version = os.getenv("AGENT_VERSION", "local-dev")

    @contextmanager
    def span(self, name: str):
        start = time.perf_counter()
        print(f"[span:start] {name} version={self.agent_version}")
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            print(f"[span:end] {name} duration_s={duration:.3f}")


def run_agent(question: str) -> str:
    tracer = LocalTracer()
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
        print(run_agent(question))
        return 0
    except SecurityError as exc:
        print(f"REQUÊTE BLOQUÉE\n{exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
