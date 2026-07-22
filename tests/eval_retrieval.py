"""Retrieval evaluation: baseline (naive cosine) vs full pipeline (BM25+dense+RRF+CE).

Deterministic, no API key needed — complements RAGAS with retrieval-only metrics.
Requires the index (python src/ingest.py) — exits early otherwise.

Usage:
    python tests/eval_retrieval.py

Metrics per mode:
    hit@3 — the expected document appears in the top-3 results
    MRR   — mean reciprocal rank of the expected document (top-10 window)

The gold set maps each question to the source document(s) that contain the
answer (doc_id prefixes, several accepted when the answer legitimately lives
in more than one document). Copy the output table into REPORT.md §3.

Résultat d'ablation (12 questions, 2026-07-22) — à lire honnêtement :
- RRF améliore le MRR vs cosine seul (tolérant : 0.958 -> 1.000).
- Le cross-encoder brut DÉGRADAIT le classement niveau document ; deux
  correctifs (titre du document préfixé au passage + blend 0.6*CE + 0.4*RRF)
  récupèrent l'essentiel (tolérant : 0.792 -> 0.917) sans le dépasser.
- Limite de cette éval : elle juge au niveau DOCUMENT ; le cross-encoder
  optimise la pertinence au niveau PASSAGE, que seul RAGAS
  (context_precision) mesure. Ne pas conclure "CE inutile" sur cette seule
  table — la conclusion appartient au rapport, chiffres RAGAS à l'appui.
- Ne PAS tuner le poids du blend sur ce jeu de 12 questions (overfitting).
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from retrieval import hybrid_search, _IndexRetriever

# question, jurisdiction filter (None = all), accepted doc_id prefixes
GOLD: list[tuple[str, str | None, list[str]]] = [
    # — EU / AI Act —
    ("Which AI systems are classified as high-risk under the AI Act?",
     None, ["Article6_AIAct", "decoding-eu-ai-act", "en-pdf-file-ai-act-guide",
            "European Union Artificial Intelligence Act Guide", "AI_act_articles"]),
    ("What transparency obligations apply to AI systems that interact with people?",
     None, ["Article13_AIAct", "AI_act_articles", "decoding-eu-ai-act",
            "en-pdf-file-ai-act-guide", "European Union Artificial Intelligence Act Guide"]),
    ("What does the AI Act require in terms of human oversight of high-risk systems?",
     None, ["Article14_AIAct", "AI_act_articles", "decoding-eu-ai-act",
            "en-pdf-file-ai-act-guide", "European Union Artificial Intelligence Act Guide"]),
    ("What are the data governance requirements for training data of high-risk AI?",
     None, ["Article10_AIAct", "AI_act_articles", "decoding-eu-ai-act",
            "en-pdf-file-ai-act-guide"]),
    # — EU / GDPR —
    ("When is a data protection impact assessment required under the GDPR?",
     None, ["Article35_GDPR", "GDPR_recitals"]),
    ("Under which conditions must an organisation designate a data protection officer?",
     None, ["Article37_GDPR", "GDPR_recitals"]),
    ("What safeguards apply to automated individual decision-making under the GDPR?",
     None, ["Article22_GDPR", "GDPR_recitals",
            "artificial-intelligence-systems-and-the-gdpr---a-data-protection-perspective-december"]),
    # — US —
    ("What are the four functions of the NIST AI risk management framework?",
     "US", ["NIST_AI_RMF_1.0", "NIST_AI_RMF_Playbook"]),
    ("What does the Colorado law require from deployers of high-risk AI systems?",
     "US", ["Colorado_SB24-205_Signed_Act"]),
    ("Which executive order revoked the previous administration's AI policy framework?",
     "US", ["EO_14179_Trump_Deregulation", "EO_14365_National_Policy_Framework"]),
    # — UK —
    ("What are the five cross-sectoral principles of the UK approach to AI regulation?",
     "UK", ["UK_White_Paper_Mar2023", "UK_Government_Response_Feb2024"]),
    ("Does the United Kingdom plan to create a new dedicated AI regulator?",
     "UK", ["UK_White_Paper_Mar2023", "UK_Government_Response_Feb2024"]),
]

EVAL_TOP_K = 10  # MRR window; hit@3 uses the first 3 of the same list


def doc_matches(result_parent_id: str, accepted: list[str]) -> bool:
    return any(result_parent_id.startswith(prefix) for prefix in accepted)


def evaluate(mode: str, strict: bool) -> dict:
    """strict=True: only the primary source document counts (first prefix).
    strict=False: secondary sources (guides quoting the answer) also count."""
    hits, rr_sum, latencies = 0, 0.0, []
    misses: list[str] = []
    for question, jurisdiction, accepted in GOLD:
        targets = accepted[:1] if strict else accepted
        t0 = time.perf_counter()
        results = hybrid_search(question, top_k=EVAL_TOP_K, jurisdiction=jurisdiction, mode=mode)
        latencies.append(time.perf_counter() - t0)

        rank = next((i for i, r in enumerate(results, start=1)
                     if doc_matches(r.document.parent_id or "", targets)), None)
        if rank is not None and rank <= 3:
            hits += 1
        else:
            misses.append(f"{question[:60]}… (rank={rank})")
        if rank is not None:
            rr_sum += 1.0 / rank

    n = len(GOLD)
    return {
        "hit@3": hits / n,
        "MRR": rr_sum / n,
        "avg_latency_s": sum(latencies) / n,
        "misses": misses,
    }


def main() -> int:
    if _IndexRetriever.get() is None:
        print("Index absent — lancer d'abord: python src/ingest.py")
        return 1

    print(f"Jeu d'évaluation : {len(GOLD)} questions "
          f"(EU: 7, US: 3, UK: 2) · hit@3 et MRR@{EVAL_TOP_K}")
    print("strict = seul le document-source primaire compte "
          "(ex. le PDF de l'article officiel lui-même)")
    print("tolérant = les documents secondaires (guides) comptent aussi\n")

    labels = {"baseline": "Baseline (cosine seul)",
              "full": "Final (BM25+dense+RRF+cross-encoder)"}
    modes = ("baseline", "full")
    rows = {}
    for mode in modes:
        print(f"→ mode {mode}…")
        rows[(mode, True)] = evaluate(mode, strict=True)
        rows[(mode, False)] = evaluate(mode, strict=False)

    print()
    print(f"{'Mode':<34}{'hit@3':>8}{'MRR':>8}{'latence':>10}")
    print("-" * 62)
    for strict, tag in ((True, "STRICT"), (False, "TOLÉRANT")):
        print(f"[{tag}]")
        for mode in modes:
            r = rows[(mode, strict)]
            print(f"  {labels[mode]:<32}{r['hit@3']:>8.2f}{r['MRR']:>8.3f}"
                  f"{r['avg_latency_s']:>9.2f}s")

    strict_misses = rows[("full", True)]["misses"]
    if strict_misses:
        print("\nÉchecs hit@3 STRICT en mode complet :")
        for m in strict_misses:
            print(f"  - {m}")

    print("\nCe tableau alimente REPORT.md §3 (complément retrieval-only de RAGAS).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
