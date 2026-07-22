"""Hybrid retrieval: BM25 + dense embeddings + RRF fusion + cross-encoder reranking.

Two operating modes, selected automatically:

1. INDEX mode (production) — used when `index_data/` exists (built by
   `python src/ingest.py`) and the ML dependencies are installed.
   BM25 (rank_bm25) + dense (sentence-transformers, cosine over a
   pre-computed normalised matrix) + RRF + cross-encoder reranking.
   Parent/child: children are indexed and matched; each result carries its
   parent legal block in `document.context` for richer LLM context.

2. FALLBACK mode (offline demo) — pure-python BM25 + hashed vectors over the
   .md/.txt files in data/. No dependency, no index. Keeps a fresh clone
   runnable before ingestion; clearly labelled in `SearchResult.method`.

Public API (stable — used by agent.py, mcp_server.py, reasoning.py, tests):
    Document, SearchResult, hybrid_search(query, top_k=4, data_dir="data",
    jurisdiction=None)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from hashlib import blake2b
from math import log, sqrt
from pathlib import Path
import re

TOKEN_RE = re.compile(r"[a-zA-ZÀ-ÿ0-9_]+")
ROOT = Path(__file__).resolve().parents[1]
INDEX_DIR = ROOT / "index_data"

EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"
INITIAL_TOP_K = 24   # candidates fetched per ranking before fusion/rerank

FALLBACK_JURISDICTION_BY_DIR = {
    "ai_act_corpus": "EU",
    "gdpr_corpus": "EU",
    "us_ai_regulation_corpus": "US",
    "uk_ai_regulation_corpus": "UK",
}

FALLBACK_STATUS_BY_DIR = {
    "ai_act_corpus": "obligatoire",
    "gdpr_corpus": "obligatoire",
    "us_ai_regulation_corpus": "variable",
    "uk_ai_regulation_corpus": "recommandation",
}


@dataclass(frozen=True)
class Document:
    doc_id: str
    title: str
    text: str
    source: str
    parent_id: str | None = None
    context: str = ""          # parent legal block (INDEX mode)
    jurisdiction: str = ""     # EU / US / UK
    status: str = ""           # obligatoire / volontaire / recommandation / ...


@dataclass(frozen=True)
class SearchResult:
    document: Document
    score: float
    method: str


def tokenize(text: str) -> list[str]:
    return [token.casefold() for token in TOKEN_RE.findall(text or "")]


# ══════════════════════════════════════════════════════════════════════════════
#  INDEX mode — real hybrid retrieval over the ingested corpus
# ══════════════════════════════════════════════════════════════════════════════

class _IndexRetriever:
    """Lazy singleton holding the index, BM25, embedder and reranker in memory."""

    _instance: "_IndexRetriever | None" = None
    _failed = False

    def __init__(self):
        import numpy as np
        from rank_bm25 import BM25Okapi
        from sentence_transformers import SentenceTransformer

        self.np = np
        self.chunks: list[dict] = json.loads(
            (INDEX_DIR / "chunks.json").read_text(encoding="utf-8"))
        self.embeddings = np.load(INDEX_DIR / "embeddings.npy")  # (N, d) L2-normalised
        self.bm25 = BM25Okapi([tokenize(c["text"]) for c in self.chunks])
        self.embedder = SentenceTransformer(EMBED_MODEL)

        try:
            from sentence_transformers import CrossEncoder
            self.reranker = CrossEncoder(RERANK_MODEL)
        except Exception:
            self.reranker = None  # rerank falls back to fused order

    @classmethod
    def get(cls) -> "_IndexRetriever | None":
        if cls._instance is not None:
            return cls._instance
        if cls._failed:
            return None
        if not (INDEX_DIR / "chunks.json").exists() or not (INDEX_DIR / "embeddings.npy").exists():
            cls._failed = True
            return None
        try:
            cls._instance = cls()
        except Exception:
            cls._failed = True   # missing ML deps -> fallback mode
            return None
        return cls._instance

    # ── rankings ─────────────────────────────────────────────────────────────

    def _candidate_ids(self, jurisdiction: str | None) -> list[int]:
        if not jurisdiction or jurisdiction.lower() == "all":
            return list(range(len(self.chunks)))
        jur = jurisdiction.upper()
        return [i for i, c in enumerate(self.chunks) if c["jurisdiction"] == jur]

    def search(self, query: str, top_k: int, jurisdiction: str | None) -> list[SearchResult]:
        np = self.np
        ids = self._candidate_ids(jurisdiction)
        if not ids:
            return []

        # BM25 ranking (restricted to candidates)
        bm25_all = self.bm25.get_scores(tokenize(query))
        bm25_rank = sorted(ids, key=lambda i: bm25_all[i], reverse=True)[:INITIAL_TOP_K]

        # Dense ranking: cosine = dot product (embeddings are L2-normalised)
        q = self.embedder.encode([query], convert_to_numpy=True, normalize_embeddings=True)[0]
        sims = self.embeddings @ q
        dense_rank = sorted(ids, key=lambda i: float(sims[i]), reverse=True)[:INITIAL_TOP_K]

        # RRF fusion (k=60)
        rrf: dict[int, float] = {}
        for ranking in (bm25_rank, dense_rank):
            for rank, idx in enumerate(ranking, start=1):
                rrf[idx] = rrf.get(idx, 0.0) + 1.0 / (60 + rank)
        fused = sorted(rrf, key=rrf.get, reverse=True)[:INITIAL_TOP_K]

        # Cross-encoder reranking of the fused pool
        method = "bm25+dense+rrf"
        if self.reranker is not None:
            pairs = [(query, self.chunks[i]["text"]) for i in fused]
            ce_scores = self.reranker.predict(pairs)
            order = sorted(zip(fused, ce_scores), key=lambda t: float(t[1]), reverse=True)
            method += "+cross-encoder"
        else:
            order = [(i, rrf[i]) for i in fused]

        results = []
        for idx, score in order[:top_k]:
            c = self.chunks[idx]
            title_bits = [c["doc_id"]]
            if c["article"]:
                title_bits.append(c["article"])
            elif c["chapter"]:
                title_bits.append(c["chapter"])
            results.append(SearchResult(
                Document(
                    doc_id=c["chunk_id"],
                    parent_id=c["doc_id"],
                    title=" — ".join(title_bits),
                    text=c["text"],
                    context=c.get("parent_text", ""),
                    source=f"{c['jurisdiction']} · {c['corpus']} · statut: {c['status']}",
                    jurisdiction=c["jurisdiction"],
                    status=c["status"],
                ),
                float(score),
                method,
            ))
        return results


# ══════════════════════════════════════════════════════════════════════════════
#  FALLBACK mode — dependency-free demo retrieval (pre-ingestion clones)
# ══════════════════════════════════════════════════════════════════════════════

DEFAULT_CORPUS = [
    Document(
        doc_id="ai-act-high-risk", title="AI Act - Haut risque", source="corpus intégré",
        jurisdiction="EU", status="obligatoire",
        text=(
            "Les systèmes IA utilisés dans l'emploi, l'éducation, le crédit, "
            "les services essentiels, la migration, la justice ou les infrastructures "
            "critiques peuvent relever du haut risque. Les obligations incluent "
            "gestion des risques, gouvernance des données, documentation, traçabilité, "
            "supervision humaine, robustesse et surveillance post-déploiement."
        ),
    ),
    Document(
        doc_id="ai-act-limited-risk", title="AI Act - Risque limité", source="corpus intégré",
        jurisdiction="EU", status="obligatoire",
        text=(
            "Les systèmes IA qui interagissent avec des personnes ou produisent "
            "du contenu synthétique imposent des obligations de transparence. "
            "L'utilisateur doit être informé lorsqu'il interagit avec une IA."
        ),
    ),
    Document(
        doc_id="ai-act-prohibited", title="AI Act - Usages interdits", source="corpus intégré",
        jurisdiction="EU", status="obligatoire",
        text=(
            "Les usages interdits comprennent la manipulation subliminale, "
            "l'exploitation de vulnérabilités, certaines notations sociales et "
            "certaines identifications biométriques à distance en temps réel."
        ),
    ),
]


def load_corpus(data_dir: str | Path = "data") -> list[Document]:
    """Load Markdown and text files from data; fallback to the built-in corpus."""
    root = Path(data_dir)
    documents: list[Document] = []
    if root.exists():
        for path in sorted(root.rglob("*")):
            if path.suffix.casefold() not in {".md", ".txt"}:
                continue
            if path.name.casefold() == "readme.md" and path.parent == root:
                continue
            text = path.read_text(encoding="utf-8")
            documents.extend(split_parent_child(path, text))
    return documents or DEFAULT_CORPUS


def split_parent_child(path: Path, text: str, chunk_words: int = 120) -> list[Document]:
    """Create child chunks while preserving a parent document identifier."""
    words = text.split()
    parent_id = path.stem
    heading = next(
        (line.lstrip("#").strip() for line in text.splitlines() if line.startswith("#")),
        "",
    )
    title = heading or path.stem
    corpus_name = path.parent.name
    jurisdiction = FALLBACK_JURISDICTION_BY_DIR.get(corpus_name, "EU")
    status = FALLBACK_STATUS_BY_DIR.get(corpus_name, "resume non officiel")
    chunks: list[Document] = []
    for index in range(0, len(words), chunk_words):
        chunk = " ".join(words[index : index + chunk_words])
        if chunk.strip():
            chunks.append(
                Document(
                    doc_id=f"{parent_id}-{index // chunk_words}",
                    parent_id=parent_id,
                    title=title,
                    text=chunk,
                    source=str(path),
                    jurisdiction=jurisdiction,
                    status=status,
                )
            )
    return chunks


def bm25_rank(query: str, documents: list[Document]) -> list[SearchResult]:
    query_terms = tokenize(query)
    doc_terms = [tokenize(doc.text) for doc in documents]
    avg_len = sum(len(terms) for terms in doc_terms) / max(1, len(doc_terms))
    k1 = 1.5
    b = 0.75
    results: list[SearchResult] = []
    for doc, terms in zip(documents, doc_terms):
        score = 0.0
        term_counts = {term: terms.count(term) for term in set(terms)}
        for term in query_terms:
            containing = sum(1 for candidate in doc_terms if term in candidate)
            if containing == 0:
                continue
            idf = log(1 + (len(documents) - containing + 0.5) / (containing + 0.5))
            tf = term_counts.get(term, 0)
            denom = tf + k1 * (1 - b + b * len(terms) / max(1, avg_len))
            score += idf * ((tf * (k1 + 1)) / max(denom, 1e-9))
        results.append(SearchResult(doc, score, "bm25"))
    return sorted(results, key=lambda item: item.score, reverse=True)


def dense_vector(text: str, dimensions: int = 64) -> list[float]:
    vector = [0.0] * dimensions
    for token in tokenize(text):
        digest = blake2b(token.encode("utf-8"), digest_size=4).digest()
        bucket = int.from_bytes(digest, "big") % dimensions
        vector[bucket] += 1.0
    norm = sqrt(sum(value * value for value in vector)) or 1.0
    return [value / norm for value in vector]


def cosine(left: list[float], right: list[float]) -> float:
    return sum(a * b for a, b in zip(left, right))


def dense_rank(query: str, documents: list[Document]) -> list[SearchResult]:
    query_vector = dense_vector(query)
    return sorted(
        [
            SearchResult(doc, cosine(query_vector, dense_vector(doc.text)), "dense")
            for doc in documents
        ],
        key=lambda item: item.score,
        reverse=True,
    )


def rrf_fusion(rankings: list[list[SearchResult]], k: int = 60) -> list[SearchResult]:
    scores: dict[str, float] = {}
    docs: dict[str, Document] = {}
    methods: dict[str, list[str]] = {}
    for ranking in rankings:
        for rank, result in enumerate(ranking, start=1):
            docs[result.document.doc_id] = result.document
            methods.setdefault(result.document.doc_id, []).append(result.method)
            scores[result.document.doc_id] = scores.get(result.document.doc_id, 0.0) + 1 / (k + rank)
    return sorted(
        [
            SearchResult(docs[doc_id], score, "+".join(sorted(set(methods[doc_id]))))
            for doc_id, score in scores.items()
        ],
        key=lambda item: item.score,
        reverse=True,
    )


def cross_encoder_rerank(query: str, results: list[SearchResult]) -> list[SearchResult]:
    """Deterministic reranker approximating cross-encoder relevance locally."""
    query_terms = set(tokenize(query))
    reranked: list[SearchResult] = []
    for result in results:
        doc_terms = set(tokenize(result.document.text))
        overlap = len(query_terms & doc_terms) / max(1, len(query_terms))
        phrase_bonus = 0.1 if query.casefold()[:20] in result.document.text.casefold() else 0.0
        reranked.append(
            SearchResult(result.document, result.score + overlap + phrase_bonus, result.method + "+rerank")
        )
    return sorted(reranked, key=lambda item: item.score, reverse=True)


# ══════════════════════════════════════════════════════════════════════════════
#  Entry point
# ══════════════════════════════════════════════════════════════════════════════

def baseline_search(query: str, top_k: int = 4, data_dir: str | Path = "data",
                    jurisdiction: str | None = None) -> list[SearchResult]:
    """Dense-only retrieval baseline used by the evaluation script."""
    documents = load_corpus(data_dir)
    if jurisdiction and jurisdiction.lower() != "all":
        jur = jurisdiction.upper()
        filtered = [d for d in documents if (d.jurisdiction or "EU") == jur]
        documents = filtered or documents
    return [
        SearchResult(result.document, result.score, result.method + " (baseline)")
        for result in dense_rank(query, documents)[:top_k]
    ]


def hybrid_search(query: str, top_k: int = 4, data_dir: str | Path = "data",
                  jurisdiction: str | None = None, mode: str = "final") -> list[SearchResult]:
    """Hybrid search over the ingested index; falls back to the local demo corpus.

    jurisdiction: optional filter — "EU", "US", "UK" or None/"all".
    """
    if mode == "baseline":
        return baseline_search(query, top_k=top_k, data_dir=data_dir, jurisdiction=jurisdiction)

    retriever = _IndexRetriever.get()
    if retriever is not None:
        return retriever.search(query, top_k=top_k, jurisdiction=jurisdiction)

    # FALLBACK mode (no index yet): pure-python pipeline, clearly labelled
    documents = load_corpus(data_dir)
    if jurisdiction and jurisdiction.lower() != "all":
        jur = jurisdiction.upper()
        filtered = [d for d in documents if (d.jurisdiction or "EU") == jur]
        documents = filtered or documents
    fused = rrf_fusion([bm25_rank(query, documents), dense_rank(query, documents)])
    results = cross_encoder_rerank(query, fused)[:top_k]
    return [SearchResult(r.document, r.score, r.method + " (fallback-demo)") for r in results]
